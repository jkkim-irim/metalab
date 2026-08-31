from collections import OrderedDict, UserDict, defaultdict
from collections.abc import Iterable, MutableMapping
import contextlib as _contextlib
import copy
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import functools
from functools import partial, wraps
import glob
import importlib.util
import inspect
import json
import logging as _logging
import os
from typing import Any, Callable, Optional, TypedDict
import warnings

import numpy as np
from packaging import version as _version
import packaging.version
import torch
import torch.nn as nn
import torch.nn.functional as F

# the transformers release this vendored subset tracks (read by the deprecate_kwarg version checks)
__version__ = "4.57.3"


class GenerationMixin:  # GR00T uses the backbone as a feature extractor; generation is unused.
    pass


def model_addition_debugger_context(*args, **kwargs):  # debug no-op
    return _contextlib.nullcontext()


def cached_file(path_or_repo_id, filename, **kwargs):
    """Resolve a file inside a LOCAL checkpoint dir (no hub download). Returns the path, or None if
    absent and errors are suppressed (matching transformers' _raise_exceptions_for_missing_entries)."""
    full = os.path.join(path_or_repo_id, filename)
    if os.path.exists(full):
        return full
    if kwargs.get("_raise_exceptions_for_missing_entries", True):
        raise FileNotFoundError(f"{filename} not found in {path_or_repo_id}")
    return None


def resolve_pretrained_path(name_or_path: str) -> str:
    """Resolve a HF repo id to a LOCAL path, offline (no huggingface_hub, no network). A local dir is
    returned as-is; otherwise we read the HF cache layout
    ``$HF_HOME/hub/models--<org>--<name>/snapshots/<ref>`` (populated from S3 by train_groot.sh)."""
    if os.path.isdir(name_or_path):
        return name_or_path
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "huggingface"
    )
    base = os.path.join(hf_home, "hub", "models--" + name_or_path.replace("/", "--"))
    ref = os.path.join(base, "refs", "main")
    if os.path.exists(ref):
        with open(ref, encoding="utf-8") as f:
            rev = f.read().strip()
        snap = os.path.join(base, "snapshots", rev)
        if os.path.isdir(snap):
            return snap
    snaps = os.path.join(base, "snapshots")
    if os.path.isdir(snaps):
        subs = [d for d in os.listdir(snaps) if os.path.isdir(os.path.join(snaps, d))]
        if subs:
            return os.path.join(snaps, subs[0])
    raise FileNotFoundError(
        f"cannot resolve '{name_or_path}' to a local path (not a dir; not in HF cache under {hf_home}/hub)"
    )


def auto_docstring(obj=None, *args, **kwargs):
    """No-op stand-in for the docstring-decorator (`@auto_docstring` and `@auto_docstring(...)`)."""
    if callable(obj):
        return obj

    def deco(fn):
        return fn

    return deco

"""Tiny stand-in for `transformers.utils.logging` — `get_logger` plus the `warning_once` /
verbosity no-ops the vendored files call."""


_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _warning_once(self, *args, **kwargs):  # transformers adds this method to its loggers
    self.warning(*args, **kwargs)


@functools.lru_cache(None)
def _configure(name: str) -> _logging.Logger:
    logger = _logging.getLogger(name)
    logger.warning_once = functools.partial(_warning_once, logger)  # type: ignore[attr-defined]
    logger.warning_once = functools.lru_cache(None)(logger.warning_once)  # dedup like transformers
    return logger


def get_logger(name: str | None = None) -> _logging.Logger:
    return _configure(name or "transformers")


"""Backend-availability flags — the subset the vendored transformers files query. We target one
environment (CUDA x86_64, torch 2.7.1, flash-attn 2.7.4), so most are hard constants; the TF/JAX/MLX
flags are False so those (guarded) code paths in generic.py never import those libraries."""


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def is_torch_available() -> bool:
    return True


def is_tf_available() -> bool:
    return False


def is_flax_available() -> bool:
    return False


def is_mlx_available() -> bool:
    return False


def is_torch_xpu_available(check_device: bool = False) -> bool:
    return False


def is_torch_npu_available(check_device: bool = False) -> bool:
    return False


def is_torchdynamo_compiling() -> bool:
    try:
        import torch

        return torch.compiler.is_compiling()
    except Exception:
        return False


def is_torch_fx_proxy(x) -> bool:
    return False


def is_torch_flex_attn_available() -> bool:
    return False


def is_flash_attn_2_available() -> bool:
    return _installed("flash_attn")


def is_flash_attn_3_available() -> bool:
    return _installed("flash_attn_3")


def is_flash_attn_greater_or_equal_2_10() -> bool:
    if not _installed("flash_attn"):
        return False
    import flash_attn

    return _version.parse(flash_attn.__version__) >= _version.parse("2.1.0")


def is_torch_greater_or_equal(library_version: str, accept_dev: bool = False) -> bool:
    import torch

    tv = torch.__version__.split("+")[0]
    if accept_dev:
        tv = tv.split(".dev")[0]
    return _version.parse(tv) >= _version.parse(library_version)


def requires(*args, **kwargs):
    """transformers marks a class/fn's backend requirements; here it's a no-op passthrough."""
    def deco(obj):
        return obj

    return deco


"""
Generic utilities
"""


_CAN_RECORD_REGISTRY = {}


logger = get_logger(__name__)


if is_torch_available():
    # required for @can_return_tuple decorator to work with torchdynamo
    import torch


def infer_framework_from_repr(x):
    """
    Tries to guess the framework of an object `x` from its repr (brittle but will help in `is_tensor` to try the
    frameworks in a smart order, without the need to import the frameworks).
    """
    representation = str(type(x))
    if representation.startswith("<class 'torch."):
        return "pt"
    elif representation.startswith("<class 'tensorflow."):
        return "tf"
    elif representation.startswith("<class 'jax"):
        return "jax"
    elif representation.startswith("<class 'numpy."):
        return "np"
    elif representation.startswith("<class 'mlx."):
        return "mlx"


def _get_frameworks_and_test_func(x):
    """
    Returns an (ordered since we are in Python 3.7+) dictionary framework to test function, which places the framework
    we can guess from the repr first, then Numpy, then the others.
    """
    framework_to_test = {
        "pt": is_torch_tensor,
        "tf": is_tf_tensor,
        "jax": is_jax_tensor,
        "np": is_numpy_array,
        "mlx": is_mlx_array,
    }
    preferred_framework = infer_framework_from_repr(x)
    # We will test this one first, then numpy, then the others.
    frameworks = [] if preferred_framework is None else [preferred_framework]
    if preferred_framework != "np":
        frameworks.append("np")
    frameworks.extend([f for f in framework_to_test if f not in [preferred_framework, "np"]])
    return {f: framework_to_test[f] for f in frameworks}


def is_tensor(x):
    """
    Tests if `x` is a `torch.Tensor`, `tf.Tensor`, `jaxlib.xla_extension.DeviceArray`, `np.ndarray` or `mlx.array`
    in the order defined by `infer_framework_from_repr`
    """
    # This gives us a smart order to test the frameworks with the corresponding tests.
    framework_to_test_func = _get_frameworks_and_test_func(x)
    for test_func in framework_to_test_func.values():
        if test_func(x):
            return True

    # Tracers
    if is_torch_fx_proxy(x):
        return True

    if is_flax_available():
        from jax.core import Tracer

        if isinstance(x, Tracer):
            return True

    return False


def _is_numpy(x):
    return isinstance(x, np.ndarray)


def is_numpy_array(x):
    """
    Tests if `x` is a numpy array or not.
    """
    return _is_numpy(x)


def _is_torch(x):
    import torch

    return isinstance(x, torch.Tensor)


def is_torch_tensor(x):
    """
    Tests if `x` is a torch tensor or not. Safe to call even if torch is not installed.
    """
    return False if not is_torch_available() else _is_torch(x)


def _is_tensorflow(x):
    import tensorflow as tf

    return isinstance(x, tf.Tensor)


def is_tf_tensor(x):
    """
    Tests if `x` is a tensorflow tensor or not. Safe to call even if tensorflow is not installed.
    """
    return False if not is_tf_available() else _is_tensorflow(x)


def _is_jax(x):
    import jax.numpy as jnp  # noqa: F811

    return isinstance(x, jnp.ndarray)


def is_jax_tensor(x):
    """
    Tests if `x` is a Jax tensor or not. Safe to call even if jax is not installed.
    """
    return False if not is_flax_available() else _is_jax(x)


def _is_mlx(x):
    import mlx.core as mx

    return isinstance(x, mx.array)


def is_mlx_array(x):
    """
    Tests if `x` is a mlx array or not. Safe to call even when mlx is not installed.
    """
    return False if not is_mlx_available() else _is_mlx(x)


class ModelOutput(OrderedDict):
    """
    Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
    tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
    python dictionary.

    <Tip warning={true}>

    You can't unpack a `ModelOutput` directly. Use the [`~utils.ModelOutput.to_tuple`] method to convert it to a tuple
    before.

    </Tip>
    """

    def __init_subclass__(cls) -> None:
        """Register subclasses as pytree nodes.

        This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
        `static_graph=True` with modules that output `ModelOutput` subclasses.
        """
        if is_torch_available():
            from torch.utils._pytree import register_pytree_node

            register_pytree_node(
                cls,
                _model_output_flatten,
                partial(_model_output_unflatten, output_type=cls),
                serialized_type_name=f"{cls.__module__}.{cls.__name__}",
            )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Subclasses of ModelOutput must use the @dataclass decorator
        # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
        # issubclass() would return True for issubclass(ModelOutput, ModelOutput) when False is needed
        # Just need to check that the current class is not ModelOutput
        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclass."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )

    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        class_fields = fields(self)

        # Safety and consistency checks
        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                # reset first field to None
                setattr(self, class_fields[0].name, None)
                for idx, element in enumerate(iterator):
                    if not isinstance(element, (list, tuple)) or len(element) != 2 or not isinstance(element[0], str):
                        if idx == 0:
                            # If we do not have an iterator of key/values, set it as attribute
                            self[class_fields[0].name] = first_field
                        else:
                            # If we have a mixed iterator, raise an error
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> tuple:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


if is_torch_available():
    import torch.utils._pytree as _torch_pytree

    def _model_output_flatten(output: ModelOutput) -> tuple[list[Any], "_torch_pytree.Context"]:
        return list(output.values()), list(output.keys())

    def _model_output_unflatten(
        values: Iterable[Any],
        context: "_torch_pytree.Context",
        output_type=None,
    ) -> ModelOutput:
        return output_type(**dict(zip(context, values)))

    _torch_pytree.register_pytree_node(
        ModelOutput,
        _model_output_flatten,
        partial(_model_output_unflatten, output_type=ModelOutput),
        serialized_type_name=f"{ModelOutput.__module__}.{ModelOutput.__name__}",
    )


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


def transpose(array, axes=None):
    """
    Framework-agnostic version of `numpy.transpose` that will work on torch/TensorFlow/Jax tensors as well as NumPy
    arrays.
    """
    if is_numpy_array(array):
        return np.transpose(array, axes=axes)
    elif is_torch_tensor(array):
        return array.T if axes is None else array.permute(*axes)
    elif is_tf_tensor(array):
        import tensorflow as tf

        return tf.transpose(array, perm=axes)
    elif is_jax_tensor(array):
        import jax.numpy as jnp

        return jnp.transpose(array, axes=axes)
    else:
        raise ValueError(f"Type not supported for transpose: {type(array)}.")


def reshape(array, newshape):
    """
    Framework-agnostic version of `numpy.reshape` that will work on torch/TensorFlow/Jax tensors as well as NumPy
    arrays.
    """
    if is_numpy_array(array):
        return np.reshape(array, newshape)
    elif is_torch_tensor(array):
        return array.reshape(*newshape)
    elif is_tf_tensor(array):
        import tensorflow as tf

        return tf.reshape(array, newshape)
    elif is_jax_tensor(array):
        import jax.numpy as jnp

        return jnp.reshape(array, newshape)
    else:
        raise ValueError(f"Type not supported for reshape: {type(array)}.")


def squeeze(array, axis=None):
    """
    Framework-agnostic version of `numpy.squeeze` that will work on torch/TensorFlow/Jax tensors as well as NumPy
    arrays.
    """
    if is_numpy_array(array):
        return np.squeeze(array, axis=axis)
    elif is_torch_tensor(array):
        return array.squeeze() if axis is None else array.squeeze(dim=axis)
    elif is_tf_tensor(array):
        import tensorflow as tf

        return tf.squeeze(array, axis=axis)
    elif is_jax_tensor(array):
        import jax.numpy as jnp

        return jnp.squeeze(array, axis=axis)
    else:
        raise ValueError(f"Type not supported for squeeze: {type(array)}.")


class TransformersKwargs(TypedDict, total=False):
    """
    Keyword arguments to be passed to the forward pass of a `PreTrainedModel`.

    Attributes:
        num_items_in_batch (`Optional[torch.Tensor]`, *optional*):
            Number of items in the batch. It is recommended to pass it when you are doing gradient accumulation.
        output_hidden_states (`Optional[bool]`, *optional*):
            Most of the models support outputting all hidden states computed during the forward pass.
        output_attentions (`Optional[bool]`, *optional*):
            Turn this on to return the intermediary attention scores.
        output_router_logits (`Optional[bool]`, *optional*):
            For MoE models, this allows returning the router logits to compute the loss.
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    num_items_in_batch: Optional["torch.Tensor"]
    output_hidden_states: Optional[bool]
    output_attentions: Optional[bool]
    output_router_logits: Optional[bool]
    cu_seq_lens_q: Optional["torch.LongTensor"]
    cu_seq_lens_k: Optional["torch.LongTensor"]
    max_length_q: Optional[int]
    max_length_k: Optional[int]


def can_return_tuple(func):
    """
    Decorator to wrap model method, to call output.to_tuple() if return_dict=False passed as a kwarg or
    use_return_dict=False is set in the config.

    Note:
        output.to_tuple() convert output to tuple skipping all `None` values.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return_dict = self.config.return_dict if hasattr(self, "config") else True
        return_dict_passed = kwargs.pop("return_dict", return_dict)
        if return_dict_passed is not None:
            return_dict = return_dict_passed
        output = func(self, *args, **kwargs)
        if not return_dict and not isinstance(output, tuple):
            output = output.to_tuple()
        return output

    return wrapper


@dataclass
@requires(backends=("torch",))
class OutputRecorder:
    """
    Configuration for recording outputs from a model via hooks.

    Attributes:
        target_class (Type): The class (e.g., nn.Module) to which the hook will be attached.
        index (Optional[int]): If the output is a tuple/list, optionally record only at a specific index.
        layer_name (Optional[str]): Name of the submodule to target (if needed), e.g., "transformer.layer.3.attn".
        class_name (Optional[str]): Name of the class to which the hook will be attached. Could be the suffix of class name in some cases.
    """

    target_class: "type[torch.nn.Module]"
    index: int = 0
    layer_name: Optional[str] = None
    class_name: Optional[str] = None


def check_model_inputs(tie_last_hidden_states=True):
    """
    Decorator to intercept specific layer outputs without using hooks.
    Compatible with torch.compile (Dynamo tracing).

    Args:
        tie_last_hidden_states (`bool`, *optional*, defaults to `True`):
            Whether to overwrite `out.hidden_states[-1]` with the `out.last_hidden_state`.
            This is true for all language models and should be toggled off only if
            `out.hidden_states[-1]` has to be the hidden state before last layer norm, which
            is needed for some vision models (e.g. CLIP, SigLIP)
    """

    def wrapped_fn(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            use_cache = (
                kwargs["use_cache"] if kwargs.get("use_cache") is not None else getattr(self.config, "use_cache", None)
            )
            if use_cache is not None:
                if getattr(self, "gradient_checkpointing", False) and self.training and use_cache:
                    logger.warning_once(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                    )
                    use_cache = False

                kwargs["use_cache"] = use_cache

            return_dict = kwargs.pop("return_dict", None)
            if return_dict is None:
                return_dict = getattr(self.config, "return_dict", True)

            all_args = kwargs.copy()
            if "kwargs" in all_args:
                for k, v in all_args["kwargs"].items():
                    all_args[k] = v

            capture_flags = _CAN_RECORD_REGISTRY.get(str(self.__class__), {})  # there is a weak ref for executorch
            recordable_keys = {
                f"output_{k}": all_args.get(
                    f"output_{k}",
                    getattr(
                        self.config,
                        f"output_{k}",
                        all_args.get("output_attentions", getattr(self.config, "output_attentions", False)),
                    ),
                )
                for k in capture_flags
            }

            # We let cross attentions to be saved separately because some models add `cross-attn` layer
            # when certain condtions are met. Let's output cross attention if attentions are requested (for BC)
            if "output_attentions" in recordable_keys:
                recordable_keys["output_cross_attentions"] = recordable_keys["output_attentions"]

            collected_outputs = defaultdict(tuple)
            monkey_patched_layers = []

            # Check attention implementation is properly set for capturing attention outputs
            if recordable_keys.get("output_attentions", False):
                supported_attn = ["eager", "eager_paged", "flex_attention"]
                config_attn = getattr(self.config, "_attn_implementation", None)
                sub_configs = [getattr(self.config, key, None) for key in self.config.sub_configs]
                sub_configs_attn = [
                    getattr(config, "_attn_implementation", None) for config in sub_configs if config is not None
                ]
                if config_attn not in supported_attn or any(attn not in supported_attn for attn in sub_configs_attn):
                    warnings.warn(
                        f"`output_attentions=True` is not supported with `attn_implementation` other than {supported_attn}. "
                        "Please use `model.set_attn_implementation('eager')` to enable capturing attention outputs.",
                        UserWarning,
                    )

            def make_capture_wrapper(module, orig_forward, key, index):
                @wraps(orig_forward)
                def wrapped_forward(*args, **kwargs):
                    if key == "hidden_states" and len(collected_outputs[key]) == 0:
                        collected_outputs[key] += (args[0],)
                    if kwargs.get("debug_io", False):
                        with model_addition_debugger_context(
                            module, kwargs.get("debug_io_dir", "~/model_debug"), kwargs.get("prune_layers")
                        ):
                            output = orig_forward(*args, **kwargs)
                    else:
                        output = orig_forward(*args, **kwargs)
                    if not isinstance(output, tuple):
                        collected_outputs[key] += (output,)
                    elif output[index] is not None:
                        if key not in collected_outputs:
                            collected_outputs[key] = (output[index],)
                        else:
                            collected_outputs[key] += (output[index],)
                    return output

                return wrapped_forward

            if any(recordable_keys.values()):
                capture_tasks = []
                for key, layer_specs in capture_flags.items():
                    if not recordable_keys.get(f"output_{key}", False):
                        continue
                    if not isinstance(layer_specs, list):
                        layer_specs = [layer_specs]
                    for specs in layer_specs:
                        if not isinstance(specs, OutputRecorder):
                            index = 0 if "hidden_states" in key else 1
                            class_name = None if not isinstance(specs, str) else specs
                            target_class = specs if not isinstance(specs, str) else None
                            specs = OutputRecorder(target_class=target_class, index=index, class_name=class_name)
                        capture_tasks.append((key, specs))

                for name, module in self.named_modules():
                    for key, specs in capture_tasks:
                        # The second check is for multimodals where only backbone layer suffix is available
                        if (specs.target_class is not None and isinstance(module, specs.target_class)) or (
                            specs.class_name is not None and name.endswith(specs.class_name)
                        ):
                            if specs.layer_name is not None and specs.layer_name not in name:
                                continue
                            # Monkey patch forward
                            original_forward = module.forward
                            module.forward = make_capture_wrapper(module, original_forward, key, specs.index)
                            monkey_patched_layers.append((module, original_forward))

            try:
                outputs = func(self, *args, **kwargs)
            except TypeError as original_exception:
                # If we get a TypeError, it's possible that the model is not receiving the recordable kwargs correctly.
                # Get a TypeError even after removing the recordable kwargs -> re-raise the original exception
                # Otherwise -> we're probably missing `**kwargs` in the decorated function
                kwargs_without_recordable = {k: v for k, v in kwargs.items() if k not in recordable_keys}
                try:
                    outputs = func(self, *args, **kwargs_without_recordable)
                except TypeError:
                    raise original_exception
                raise TypeError(
                    "Missing `**kwargs` in the signature of the `@check_model_inputs`-decorated function "
                    f"({func.__qualname__})"
                )

            # Restore original forward methods
            for module, original_forward in monkey_patched_layers:
                module.forward = original_forward

            # Inject collected outputs into model output
            for key in collected_outputs:
                if key == "hidden_states":
                    if not tie_last_hidden_states:
                        pass
                    elif hasattr(outputs, "vision_hidden_states"):
                        collected_outputs[key] = collected_outputs[key][:-1]
                        collected_outputs[key] += (outputs.vision_hidden_states,)
                    elif hasattr(outputs, "last_hidden_state"):
                        collected_outputs[key] = collected_outputs[key][:-1]
                        collected_outputs[key] += (outputs.last_hidden_state,)

                    outputs[key] = collected_outputs[key]
                elif key == "attentions":
                    if isinstance(capture_flags[key], list) and len(capture_flags[key]) == 2:
                        outputs[key] = collected_outputs[key][0::2]
                        outputs["cross_" + key] = collected_outputs[key][1::2]
                    else:
                        outputs[key] = collected_outputs[key]
                else:
                    outputs[key] = collected_outputs[key]
            if return_dict is False:
                outputs = outputs.to_tuple()
            return outputs

        return wrapper

    return wrapped_fn


class GeneralInterface(MutableMapping):
    """
    Dict-like object keeping track of a class-wide mapping, as well as a local one. Allows to have library-wide
    modifications though the class mapping, as well as local modifications in a single file with the local mapping.
    """

    # Class instance object, so that a call to `register` can be reflected into all other files correctly, even if
    # a new instance is created (in order to locally override a given function)
    _global_mapping = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key):
        # First check if instance has a local override
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key, value):
        # Allow local update of the default functions without impacting other instances
        self._local_mapping.update({key: value})

    def __delitem__(self, key):
        del self._local_mapping[key]

    def __iter__(self):
        # Ensure we use all keys, with the overwritten ones on top
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self):
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    @classmethod
    def register(cls, key: str, value: Callable):
        cls._global_mapping.update({key: value})

    def valid_keys(self) -> list[str]:
        return list(self.keys())


if is_torch_available():
    import torch  # noqa: F401


class Action(ExplicitEnum):
    NONE = "none"
    NOTIFY = "notify"
    NOTIFY_ALWAYS = "notify_always"
    RAISE = "raise"


def deprecate_kwarg(
    old_name: str,
    version: str,
    new_name: Optional[str] = None,
    warn_if_greater_or_equal_version: bool = False,
    raise_if_greater_or_equal_version: bool = False,
    raise_if_both_names: bool = False,
    additional_message: Optional[str] = None,
):
    """
    Function or method decorator to notify users about deprecated keyword arguments, replacing them with a new name if specified.
    Note that is decorator is `torch.compile`-safe, i.e. it will not cause graph breaks (but no warning will be displayed if compiling).

    This decorator allows you to:
    - Notify users when a keyword argument is deprecated.
    - Automatically replace deprecated keyword arguments with new ones.
    - Raise an error if deprecated arguments are used, depending on the specified conditions.

    By default, the decorator notifies the user about the deprecated argument while the `transformers.__version__` < specified `version`
    in the decorator. To keep notifications with any version `warn_if_greater_or_equal_version=True` can be set.

    Parameters:
        old_name (`str`):
            Name of the deprecated keyword argument.
        version (`str`):
            The version in which the keyword argument was (or will be) deprecated.
        new_name (`Optional[str]`, *optional*):
            The new name for the deprecated keyword argument. If specified, the deprecated keyword argument will be replaced with this new name.
        warn_if_greater_or_equal_version (`bool`, *optional*, defaults to `False`):
            Whether to show warning if current `transformers` version is greater or equal to the deprecated version.
        raise_if_greater_or_equal_version (`bool`, *optional*, defaults to `False`):
            Whether to raise `ValueError` if current `transformers` version is greater or equal to the deprecated version.
        raise_if_both_names (`bool`, *optional*, defaults to `False`):
            Whether to raise `ValueError` if both deprecated and new keyword arguments are set.
        additional_message (`Optional[str]`, *optional*):
            An additional message to append to the default deprecation message.

    Raises:
        ValueError:
            If raise_if_greater_or_equal_version is True and the current version is greater than or equal to the deprecated version, or if raise_if_both_names is True and both old and new keyword arguments are provided.

    Returns:
        Callable:
            A wrapped function that handles the deprecated keyword arguments according to the specified parameters.

    Example usage with renaming argument:

        ```python
        @deprecate_kwarg("reduce_labels", new_name="do_reduce_labels", version="6.0.0")
        def my_function(do_reduce_labels):
            print(do_reduce_labels)

        my_function(reduce_labels=True)  # Will show a deprecation warning and use do_reduce_labels=True
        ```

    Example usage without renaming argument:

        ```python
        @deprecate_kwarg("max_size", version="6.0.0")
        def my_function(max_size):
            print(max_size)

        my_function(max_size=1333)  # Will show a deprecation warning
        ```

    """

    deprecated_version = packaging.version.parse(version)
    current_version = packaging.version.parse(__version__)
    is_greater_or_equal_version = current_version >= deprecated_version

    if is_greater_or_equal_version:
        version_message = f"and removed starting from version {version}"
    else:
        version_message = f"and will be removed in version {version}"

    def wrapper(func):
        # Required for better warning message
        sig = inspect.signature(func)
        function_named_args = set(sig.parameters.keys())
        is_instance_method = "self" in function_named_args
        is_class_method = "cls" in function_named_args

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            # Get class + function name (just for better warning message)
            func_name = func.__name__
            if is_instance_method:
                func_name = f"{args[0].__class__.__name__}.{func_name}"
            elif is_class_method:
                func_name = f"{args[0].__name__}.{func_name}"

            minimum_action = Action.NONE
            message = None

            # deprecated kwarg and its new version are set for function call -> replace it with new name
            if old_name in kwargs and new_name in kwargs:
                minimum_action = Action.RAISE if raise_if_both_names else Action.NOTIFY_ALWAYS
                message = f"Both `{old_name}` and `{new_name}` are set for `{func_name}`. Using `{new_name}={kwargs[new_name]}` and ignoring deprecated `{old_name}={kwargs[old_name]}`."
                kwargs.pop(old_name)

            # only deprecated kwarg is set for function call -> replace it with new name
            elif old_name in kwargs and new_name is not None and new_name not in kwargs:
                minimum_action = Action.NOTIFY
                message = f"`{old_name}` is deprecated {version_message} for `{func_name}`. Use `{new_name}` instead."
                kwargs[new_name] = kwargs.pop(old_name)

            # deprecated kwarg is not set for function call and new name is not specified -> just notify
            elif old_name in kwargs:
                minimum_action = Action.NOTIFY
                message = f"`{old_name}` is deprecated {version_message} for `{func_name}`."

            if message is not None and additional_message is not None:
                message = f"{message} {additional_message}"

            # update minimum_action if argument is ALREADY deprecated (current version >= deprecated version)
            if is_greater_or_equal_version:
                # change to (NOTIFY, NOTIFY_ALWAYS) -> RAISE if specified
                # in case we want to raise error for already deprecated arguments
                if raise_if_greater_or_equal_version and minimum_action != Action.NONE:
                    minimum_action = Action.RAISE

                # change to NOTIFY -> NONE if specified (NOTIFY_ALWAYS can't be changed to NONE)
                # in case we want to ignore notifications for already deprecated arguments
                elif not warn_if_greater_or_equal_version and minimum_action == Action.NOTIFY:
                    minimum_action = Action.NONE

            # raise error or notify user
            if minimum_action == Action.RAISE:
                raise ValueError(message)
            # If we are compiling, we do not raise the warning as it would break compilation
            elif minimum_action in (Action.NOTIFY, Action.NOTIFY_ALWAYS) and not is_torchdynamo_compiling():
                # DeprecationWarning is ignored by default, so we use FutureWarning instead
                warnings.warn(message, FutureWarning, stacklevel=2)

            return func(*args, **kwargs)

        return wrapped_func

    return wrapper


"""Minimal KV cache for the single-forward feature-extractor path (no incremental generation).

GR00T runs one `forward`: during training gradient-checkpointing forces `use_cache=False` (no cache
at all); during inference `use_cache=True` builds a cache that is written once and never read across
steps. So `DynamicCache` only needs to accumulate per-layer K/V within one forward and hand it back."""


class Cache:
    """Base marker type (vendored code does `isinstance(x, Cache)` / type hints)."""


class EncoderDecoderCache(Cache):
    """Stub — the vendored modeling_outputs imports this name; Qwen3-VL is decoder-only, so it's unused."""

    def __init__(self, *args, **kwargs):
        pass


class DynamicCache(Cache):
    def __init__(self, config=None, *args, **kwargs):
        self.key_cache: list[Optional[torch.Tensor]] = []
        self.value_cache: list[Optional[torch.Tensor]] = []

    def _ensure(self, layer_idx: int):
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self._ensure(layer_idx)
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:  # only hit if a caller ever runs >1 step; concat on the sequence axis
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_mask_sizes(self, cache_position, layer_idx: int = 0):
        # (kv_length, kv_offset) for mask construction — single forward: offset 0, length = seq len
        kv_length = cache_position.shape[0] + self.get_seq_length(layer_idx)
        return kv_length, 0

    def __len__(self):
        return len(self.key_cache)

    def __getitem__(self, layer_idx):
        return (self.key_cache[layer_idx], self.value_cache[layer_idx])


__all__ = ["Cache", "DynamicCache", "EncoderDecoderCache"]


"""Minimal `PretrainedConfig` — enough for the vendored Qwen3-VL configs (and GR00T's own config) to
construct, round-trip config.json, carry `_attn_implementation`, and expose the attrs the modeling code
reads. No generation-config, no auto-mapping, no hub — construction + local json only."""


class PretrainedConfig:
    model_type: str = ""
    base_config_key: str = ""
    sub_configs: dict[str, type] = {}
    attribute_map: dict[str, str] = {}
    keys_to_ignore_at_inference: list[str] = []

    def __init__(self, **kwargs):
        # common flags the modeling code / trainer read
        self.return_dict = kwargs.pop("return_dict", True)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.torch_dtype = kwargs.pop("torch_dtype", None)
        self.tie_word_embeddings = kwargs.pop("tie_word_embeddings", True)
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)
        # special-token ids the modeling code reads (transformers' PretrainedConfig defaults these)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.sep_token_id = kwargs.pop("sep_token_id", None)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)
        # attn implementation (concrete value gets set by the loader; may be overridden per sub-config)
        self._attn_implementation = kwargs.pop("attn_implementation", None)
        self._attn_implementation_autoset = False
        # keep transformers-only bookkeeping keys from leaking in as surprises, but don't fail on them
        for k in ("architectures", "transformers_version", "model_type", "_name_or_path"):
            kwargs.pop(k, None)
        # everything else becomes an attribute (model-specific fields the subclass didn't name)
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError:
                pass

    def __setattr__(self, key, value):
        if key in type(self).attribute_map:
            key = type(self).attribute_map[key]
        super().__setattr__(key, value)

    def __getattr__(self, key):  # support attribute_map aliases on read
        amap = type(self).__dict__.get("attribute_map", {})
        if key in amap:
            return getattr(self, amap[key])
        raise AttributeError(key)

    # ---- serialization -------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") and key not in ("_attn_implementation",):
                continue
            if isinstance(value, PretrainedConfig):
                out[key] = value.to_dict()
            else:
                out[key] = value
        out["model_type"] = type(self).model_type
        return copy.deepcopy(out)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs) -> "PretrainedConfig":
        config_dict = dict(config_dict)
        config_dict.pop("_attn_implementation_autoset", None)
        attn = kwargs.pop("attn_implementation", None)
        config = cls(**config_dict)
        if attn is not None:
            config._attn_implementation = attn
        for k, v in kwargs.items():
            setattr(config, k, v)
        return config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> "PretrainedConfig":

        pretrained_model_name_or_path = resolve_pretrained_path(pretrained_model_name_or_path)
        path = os.path.join(pretrained_model_name_or_path, "config.json")
        with open(path, encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict, **kwargs)

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    def get_text_config(self, decoder: bool = False):
        return getattr(self, "text_config", self)

    def to_diff_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def __repr__(self):
        return f"{type(self).__name__}({json.dumps(self.to_dict(), default=str)[:200]}...)"


__all__ = ["PretrainedConfig"]


"""`BatchFeature` — the dict-like container transformers processors/models return. Supports attribute
access, `.to(device/dtype)`, and the mapping API GR00T uses (`.items()`, `dict(...)`, `[key]`)."""


class BatchFeature(UserDict):
    def __init__(self, data=None, tensor_type=None):
        super().__init__(data or {})

    def __getattr__(self, item):
        try:
            return self.data[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def to(self, *args, **kwargs) -> "BatchFeature":
        new = {}
        for k, v in self.data.items():
            if isinstance(v, torch.Tensor):
                # only move floating tensors on dtype casts; always allow device moves
                new[k] = v.to(*args, **kwargs)
            else:
                new[k] = v
        self.data = new
        return self


__all__ = ["BatchFeature"]


"""`Unpack` for `**kwargs: Unpack[TransformersKwargs]` annotations (typing-only)."""


try:
    from typing import Unpack  # py3.11+
except ImportError:  # py3.10
    try:
        from typing_extensions import Unpack
    except ImportError:  # last-resort: subscriptable placeholder (annotations only)
        class _Unpack:
            def __class_getitem__(cls, item):
                return item

        Unpack = _Unpack  # type: ignore[assignment,misc]


class ProcessorMixin:
    """Minimal base for GR00T's own processors (they subclass this). No auto-map / hub machinery;
    subclasses implement their own from_pretrained/save_pretrained where needed."""

    attributes: list = []

    @classmethod
    def register_for_auto_class(cls, *args, **kwargs):
        return None


__all__ = ["Unpack", "ProcessorMixin"]


logger = get_logger(__name__)


def flash_attn_supports_top_left_mask():
    if is_flash_attn_3_available():
        return False
    if is_flash_attn_2_available():
        return not is_flash_attn_greater_or_equal_2_10()

    return False


_flash_fn = None


_flash_varlen_fn = None


_pad_fn = None


_unpad_fn = None


_process_flash_kwargs_fn = None


_hf_api_to_flash_mapping = {
    "dropout": "dropout_p",
    "sliding_window": "window_size",
}


def _lazy_imports(implementation: Optional[str]):
    """
    Lazy loads the respective flash attention implementations.

    Return:
        flash_attn_func: The base flash attention function.
        flash_attn_varlen_func: The flash attention function supporting variable sequence lengths,
                                e.g. for padding-free training.
        pad_input: The function to pad inputs into one sequence and returning the respective kwargs.
        unpad_input: The function to unpad outputs based on the kwargs (from pad_input).
    """
    is_fa2 = is_flash_attn_2_available()
    is_fa3 = is_flash_attn_3_available()

    pad_input, unpad_input = _pad_input, _unpad_input

    if (implementation == "flash_attention_2" and is_fa2) or (implementation is None and is_fa2 and not is_fa3):
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import pad_input, unpad_input
    else:
        if implementation == "flash_attention_3" or (implementation is None and is_fa3):
            from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        # Kernels fallback
        else:
            flash_attn_func = getattr(implementation, "flash_attn_func", None)
            flash_attn_varlen_func = getattr(implementation, "flash_attn_varlen_func", None)
            if flash_attn_varlen_func is None or flash_attn_func is None:
                raise ValueError(
                    f"Could not find the currently requested flash attention implementation at `{implementation}`."
                    f"Make sure that you request a valid kernel from the hub, e.g. `kernels-community/flash-attn`."
                )

    return flash_attn_func, flash_attn_varlen_func, pad_input, unpad_input


def _lazy_define_process_function(flash_function):
    """
    Depending on the version and kernel some features are not supported. Due to limitations in
    `torch.compile`, we opt to statically type which (optional) kwarg parameters are supported
    within `_process_flash_attention_kwargs`.

    NOTE: While all supported kwargs are marked as `True`, everything else is marked as `False`.
          This might be confusing for kwargs that we use in any case, e.g. `is_causal`.
    """

    flash_parameters = inspect.signature(flash_function).parameters
    process_parameters = inspect.signature(_process_flash_attention_kwargs).parameters

    supports_mapping = {}
    for param in process_parameters:
        fa_param = _hf_api_to_flash_mapping.get(param, param)
        supports_mapping[fa_param] = fa_param in flash_parameters

    return partial(_process_flash_attention_kwargs, supports_mapping=supports_mapping)


def lazy_import_flash_attention(implementation: Optional[str], force_import: Optional[bool] = False):
    """
    Lazily import flash attention and return the respective functions + flags.

    NOTE: For fullgraph, this needs to be called before compile, while no fullgraph can
    work without preloading. See `load_and_register_kernel` in `integrations.hub_kernels`.
    """
    global _flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn
    if force_import or any(k is None for k in [_flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn]):
        _flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn = _lazy_imports(implementation)

    global _process_flash_kwargs_fn
    if force_import or _process_flash_kwargs_fn is None:
        _process_flash_kwargs_fn = _lazy_define_process_function(_flash_varlen_fn)

    return (_flash_fn, _flash_varlen_fn, _pad_fn, _unpad_fn), _process_flash_kwargs_fn


def _index_first_axis(tensor, indices):
    """
    A local implementation of the PyTorch indexing operation `tensor[indices]` on the first axis,
    after flattening the first two dimensions of the tensor. This is functionally equivalent to
    FA2's `index_first_axis` and replaces the need to import it.
    """
    # The input tensor is expected to be of shape (batch, seq_len, ...). We flatten the first
    # two dimensions to get (total_tokens, ...) before indexing.
    reshaped_tensor = tensor.reshape(-1, *tensor.shape[2:])
    return reshaped_tensor[indices]


def _unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    unpad_input function for flash attention variants that do not have them within their pkg themselves, e.g. fa3.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.

    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    return (
        _index_first_axis(hidden_states, indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def _pad_input(hidden_states, indices, batch, seqlen):
    """
    pad_input function for flash attention variants that do not have them within their pkg themselves, e.g. fa3.

    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.

    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[1:]
    output = torch.zeros((batch * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.view(batch, seqlen, *dim)


def _get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Arguments:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # NOTE: Similar to the `.item()` in prepare_fa2_from_position_ids, with torch compile,
    # this might cause a graph break
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    unpad_input_func,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens even though they belong to different batches.
    This function is used instead of `flash_attn.bert_padding.unpad_input` in order to avoid the recomputation of the same intermediary
    tensors for query, key, value tensors.

    Arguments:
        query_layer (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        query_length (`int`):
            Target length.
        unpad_input_func:
            The function to use for unpadding the input tensors.

    Return:
        query_layer (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

    # With static caches, the k/v states may be larger than the mask -> we need to slice them to avoid generating garbage
    # It's a bit of an anti-pattern, but otherwise we silently compute wrong attentions scores
    if key_layer.shape[1] > (seq_len := attention_mask.shape[-1]):
        key_layer, value_layer = key_layer[:, :seq_len, :, :], value_layer[:, :seq_len, :, :]

    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = _index_first_axis(key_layer, indices_k)
    value_layer = _index_first_axis(value_layer, indices_k)
    if query_length == kv_seq_len:
        query_layer = _index_first_axis(query_layer, indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, *_ = unpad_input_func(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def prepare_fa_kwargs_from_position_ids(position_ids):
    """
    This function returns all the necessary kwargs to call `flash_attn_varlen_func` extracted from position_ids.

    Arguments:
        position_ids (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into
            ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query,
            `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    tensor_kwargs = {"dtype": torch.int32, "device": position_ids.device}

    position_ids = position_ids.view(-1)
    indices_q = (position_ids == 0).nonzero().view(-1)

    cu_seq_lens_q = torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )
    cu_seq_lens_k = cu_seq_lens_q

    # https://github.com/Dao-AILab/flash-attention/blob/2dd8078adc1d9b74e315ee99718c0dea0de8eeb6/flash_attn/flash_attn_interface.py#L1423-L1424
    # We should use cu_seq_lens instead of position_ids to get the max length since position_ids is not always increasing
    # for some models (e.g. qwen2-vl).
    max_length_q = cu_seq_lens_q.diff().max()
    # NOTE: With torch compile, this will cause a graph break if you don't set
    # `TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1` in the environment or call
    # `torch._dynamo.config.capture_scalar_outputs = True` before doing the forward pass.
    # This is a limitation of flash attention API, as the function `flash_attn_varlen_func`
    # requires `max_length_q`, `max_length_k` to be passed as `int` and not `torch.Tensor`.
    max_length_q = max_length_q.item()
    max_length_k = max_length_q

    return (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k)


def _prepare_from_posids(query, key, value, position_ids):
    """
    This function returns necessary arguments to call `flash_attn_varlen_func`.
    All three query, key, value states will be flattened.
    Cumulative lengths of each examples in the batch will be extracted from position_ids.
    NOTE: ideally cumulative lengths should be prepared at the data collator stage

    Arguments:
        query (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        position_ids (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        query (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    query = query.contiguous().view(-1, query.size(-2), query.size(-1))
    key = key.contiguous().view(-1, key.size(-2), key.size(-1))
    value = value.contiguous().view(-1, value.size(-2), value.size(-1))

    (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(position_ids)

    return (query, key, value, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k))


def _is_packed_sequence(position_ids, batch_size):
    """
    Check the position ids whether packed sequences are indicated or not
        1. Position ids exist
        2. Flattened sequences only are supported
        3. Compile-friendly `not (torch.diff(position_ids, dim=-1) >= 0).all()`, i.e. we have multiple increasing sequences
    """
    if position_ids is None:
        return False

    increasing_position_sequences = (
        torch.arange(position_ids.shape[1], device=position_ids.device) + position_ids.min()
    )
    return batch_size == 1 and (increasing_position_sequences - position_ids).abs().sum().bool()


def fa_peft_integration_check(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    target_dtype: Optional[torch.dtype] = None,
):
    """
    PEFT usually casts the layer norms in float32 for training stability reasons
    therefore the input hidden states gets silently casted in float32. Hence, we need
    cast them back in float16 / bfloat16 just to be sure everything works as expected.
    This might slowdown training & inference so it is recommended to not cast the LayerNorms!
    """
    if target_dtype and q.dtype == torch.float32:
        logger.warning_once(f"Casting fp32 inputs back to {target_dtype} for flash-attn compatibility.")
        q, k, v = q.to(target_dtype), k.to(target_dtype), v.to(target_dtype)
    return q, k, v


class FlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cu_seq_lens_q: Optional[torch.LongTensor]
    cu_seq_lens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]


def _process_flash_attention_kwargs(
    query_length: int,
    key_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    s_aux: Optional[torch.Tensor] = None,
    supports_mapping: Optional[dict[str, bool]] = None,
    **kwargs,
):
    """
    Returns a set of kwargs that are passed down to the according flash attention function based on
    requested features and whether it is supported - depends on the version and kernel implementation
    which is dynamically configured at `lazy_import_flash_attention`. The (un)supported features can be
    inspected in `supports_mapping`, see `_lazy_define_process_function` for more details.

    Args:
        query_length (`int`):
            Length of the query states
        key_length (`int`):
            Length of the key states
        is_causal (`bool`):
            Whether we perform causal (decoder) attention or full attention.
        dropout (`float`):
            Attention dropout.
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to `1 / sqrt(head_dim)`.
        sliding_window (`int`, *optional*):
            The size of the sliding window, i.e. we look at a max of `sliding_window` tokens back.
        use_top_left_mask (`bool`):
            Deprecated behavior of older versions of flash attention requiring different masking.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
        s_aux (`torch.Tensor`, *optional*):
            Attention sink auxiliary that adds a `bias` to the attention calculation via an additional head.
    Return:
        flash_kwargs (`dict`):
            A dict of kwargs that are requested and supported.
    """
    flash_kwargs = {
        "causal": is_causal and not (use_top_left_mask and query_length == 1),
        "softmax_scale": softmax_scale,
    }

    if supports_mapping["dropout_p"]:
        flash_kwargs["dropout_p"] = dropout

    if supports_mapping["window_size"] and sliding_window is not None and key_length > sliding_window:
        # The flash attention API sets inclusive boundaries, i.e. (4, 0) would take 4 tokens to the left
        # and the current token for a total size of 5. However, we usually define our window sizes by
        # their total window size (when causal). Encoder models as of now seldom use SWA and when they
        # do, they have a custom workaround (e.g. ModernBERT) which would align with this symmetric logic, i.e.
        # for a total of `2*sliding_window + 1`.
        flash_kwargs["window_size"] = (sliding_window - 1, sliding_window - 1)

    if supports_mapping["deterministic"]:
        flash_kwargs["deterministic"] = (
            deterministic if deterministic is not None else os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
        )

    if supports_mapping["softcap"] and softcap is not None:
        flash_kwargs["softcap"] = softcap

    # Only within kernel implementation atm
    if supports_mapping["s_aux"] and s_aux is not None:
        flash_kwargs["s_aux"] = s_aux

    return flash_kwargs


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    implementation: Optional[str] = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    (Optional) kwargs are described further in `_process_flash_attention_kwargs` and `FlashAttentionKwargs`.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`, *optional*):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        implementation (`str`, *optional*):
            The attention implementation to use. If None, will default to the one based on the environment.
    """
    (flash_fn, flash_varlen_fn, pad_fn, unpad_fn), process_flash_kwargs_fn = lazy_import_flash_attention(
        implementation
    )

    # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )

    # Extract the flash attention kwargs that have been requested (and are supported by the implementation)
    flash_kwargs = process_flash_kwargs_fn(
        query_length=query_length,
        key_length=key_states.size(1),
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        use_top_left_mask=use_top_left_mask,
        softcap=softcap,
        deterministic=deterministic,
        **kwargs,
    )

    # We will use `flash_varlen_fn` to prevent cross-example attention and also allow padding free approach under two cases:
    # Case 1. If position ids is provided and the position ids indicate packed sequences, see `_is_packed_sequence`.
    # Case 2. Some models pass directly pre-computed `cu_seqlens` so we don't need to infer it from position ids. It is safe to
    # use `flash_varlen_fn` knowing we already have all necessary the kwargs.
    #
    # NOTE: it is user's responsibility to take care of flattening `position_ids` if that's needed by the model.
    # See #39121 for more information.
    is_fa_with_position_ids = _is_packed_sequence(position_ids, batch_size=query_states.size(0))
    is_fa_with_varlen_kwargs = all(
        kwarg is not None for kwarg in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    )

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length, unpad_fn
        )

        # TODO for now this is required to work with
        # https://huggingface.co/kernels-community/metal-flash-sdpa/blob/main/torch-ext/metal_flash_sdpa/__init__.py
        if "mps" in str(q.device):
            cu_seq_lens_k = cu_seq_lens_k.clone()

        out_unpad = flash_varlen_fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            **flash_kwargs,
        )
        if isinstance(out_unpad, tuple):
            out_unpad = out_unpad[0]

        out = pad_fn(out_unpad, indices_q, query_states.size(0), query_length)

    # Padding free, i.e. sequences flattened into one total sequence
    elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:
        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            q, k, v, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _prepare_from_posids(
                query_states, key_states, value_states, position_ids
            )
        else:
            q = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
            k = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
            v = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))

        # TODO for now this is required to work with
        # https://huggingface.co/kernels-community/metal-flash-sdpa/blob/main/torch-ext/metal_flash_sdpa/__init__.py
        if "mps" in str(q.device):
            cu_seq_lens_k = cu_seq_lens_k.clone()

        out = flash_varlen_fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            max_seqlen_q=max_length_q,
            max_seqlen_k=max_length_k,
            **flash_kwargs,
        )
        if isinstance(out, tuple):
            out = out[0]

        out = out.view(query_states.size(0), -1, out.size(-2), out.size(-1))

    # No padding
    else:
        out = flash_fn(query_states, key_states, value_states, **flash_kwargs)
        if isinstance(out, tuple):
            out = out[0]

    return out


logger = get_logger(__name__)


_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)


_is_torch_greater_or_equal_than_2_8 = is_torch_greater_or_equal("2.8", accept_dev=True)


_is_torch_xpu_available = is_torch_xpu_available()


_is_torch_npu_available = is_torch_npu_available()


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def use_gqa_in_sdpa(attention_mask: Optional[torch.Tensor], key: torch.Tensor) -> bool:
    # GQA can only be used under the following conditions
    # 1.cuda
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    #   - key is not a torch.fx.Proxy (otherwise it will fail with a tracing error)
    # 2.xpu
    #   - torch version >= 2.8
    #   - key is not a torch.fx.Proxy (otherwise it will fail with a tracing error)
    # 3.npu
    #   - npu is not supported gqa currently
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8 and not isinstance(key, torch.fx.Proxy)
    if _is_torch_npu_available:
        return False
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None and not isinstance(key, torch.fx.Proxy)


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key.shape[-2]]

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # When `is_causal = False` and the `attention_mask` is not of boolean type, the Ascend NPU's SDPA interface cannot utilize the FlashAttentionScore operator，
    # and falls back to small-operator concatenation. To invoke the FlashAttentionScore, the attention_mask must be converted to boolean type.
    # This adaptation ensures the `attention_mask` meets the requirement for using FlashAttentionScore.
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Convert to boolean type, making sdpa to force call FlashAttentionScore to improve performance.
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None


logger = get_logger(__name__)


_use_top_left_mask = flash_attn_supports_top_left_mask()


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # This is before the transpose
    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape  with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )
    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = module.is_causal

    attn_output = _flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=target_dtype,
        attn_implementation=module.config._attn_implementation,
        layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
        **kwargs,
    )

    return attn_output, None


"""`use_kernel_forward_from_hub` swaps a module's forward for an optional HF-Hub kernel. We don't use
hub kernels, so it's a no-op class decorator (works as `@use_kernel_forward_from_hub("RMSNorm")`)."""


def use_kernel_forward_from_hub(*args, **kwargs):
    def deco(cls):
        return cls

    return deco


__all__ = ["use_kernel_forward_from_hub"]


"""Minimal `PreTrainedModel` + attention-function registry for the vendored Qwen3-VL backbone (and
GR00T's own model). Covers exactly the feature-extractor path: construct from config, load weights
from local safetensors, dispatch attention, enable gradient checkpointing. NO generation, device_map,
quantization, peft, tensor-parallel, or hub download."""


ALL_ATTENTION_FUNCTIONS = {
    "flash_attention_2": flash_attention_forward,
    "flash_attention_3": flash_attention_forward,
    "sdpa": sdpa_attention_forward,
}


def _set_attn_implementation(config, impl: str) -> None:
    """Set the concrete attn impl on the config and every sub-config (the attention modules read it off
    their own sub-config at forward time)."""
    config._attn_implementation = impl
    for name in getattr(config, "sub_configs", {}):
        sub = getattr(config, name, None)
        if sub is not None:
            sub._attn_implementation = impl


def _load_local_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Read weights from a local checkpoint dir: sharded (model.safetensors.index.json) or single
    (model.safetensors), falling back to pytorch_model*.bin."""
    from safetensors.torch import load_file

    index = os.path.join(path, "model.safetensors.index.json")
    single = os.path.join(path, "model.safetensors")
    state: dict[str, torch.Tensor] = {}
    if os.path.exists(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
        for shard in shards:
            state.update(load_file(os.path.join(path, shard)))
    elif os.path.exists(single):
        state.update(load_file(single))
    else:
        bins = sorted(glob.glob(os.path.join(path, "pytorch_model*.bin")))
        if not bins:
            raise FileNotFoundError(f"no safetensors/bin weights under {path}")
        for b in bins:
            state.update(torch.load(b, map_location="cpu", weights_only=True))
    return state


class PreTrainedModel(nn.Module):
    config_class = None
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules: list = []
    _skip_keys_device_placement = None
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = False
    _tied_weights_keys: list = []
    _keep_in_fp32_modules = None
    _can_record_outputs: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Register the class's output-capture spec so `@check_model_inputs` can collect hidden_states /
        # attentions for this exact class (keyed by str(cls), read from _CAN_RECORD_REGISTRY at forward).
        can_record = getattr(cls, "_can_record_outputs", None)
        if can_record:

            _CAN_RECORD_REGISTRY[str(cls)] = can_record

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config

    # ---- construction hooks (weights come from the checkpoint, so these are light) --------------
    def post_init(self):
        pass

    def _init_weights(self, module):
        pass

    def init_weights(self):
        pass

    def tie_weights(self):
        pass

    def _backward_compatibility_gradient_checkpointing(self):
        pass

    # ---- embeddings (subclasses override; provide safe fallbacks) --------------------------------
    def get_input_embeddings(self):
        if hasattr(self, "embed_tokens"):
            return self.embed_tokens
        base = getattr(self, self.base_model_prefix, None)
        if base is not None and base is not self:
            return base.get_input_embeddings()
        raise NotImplementedError

    def set_input_embeddings(self, value):
        if hasattr(self, "embed_tokens"):
            self.embed_tokens = value
            return
        base = getattr(self, self.base_model_prefix, None)
        if base is not None and base is not self:
            base.set_input_embeddings(value)
        else:
            raise NotImplementedError

    def get_output_embeddings(self):
        return None

    # ---- gradient checkpointing ------------------------------------------------------------------
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        import functools

        import torch.utils.checkpoint as _cp

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        func = functools.partial(_cp.checkpoint, **gradient_checkpointing_kwargs)
        # A GradientCheckpointingLayer checks self.gradient_checkpointing AND calls
        # self._gradient_checkpointing_func — transformers sets both here; so must we.
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = True
                module._gradient_checkpointing_func = func

    def gradient_checkpointing_disable(self):
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = False

    # ---- misc used by callers --------------------------------------------------------------------
    def can_generate(self) -> bool:
        return False

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def num_parameters(self, only_trainable: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not only_trainable))

    # ---- loading / saving (local only) -----------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        config=None,
        torch_dtype: Optional[torch.dtype] = None,
        dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
        **kwargs,
    ):

        torch_dtype = torch_dtype or dtype
        pretrained_model_name_or_path = resolve_pretrained_path(pretrained_model_name_or_path)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        impl = attn_implementation or getattr(config, "_attn_implementation", None) or "sdpa"
        _set_attn_implementation(config, impl)
        if torch_dtype not in (None, "auto"):
            config.torch_dtype = torch_dtype

        model = cls(config, *model_args)
        state_dict = _load_local_state_dict(pretrained_model_name_or_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Non-persistent buffers (e.g. rotary inv_freq) are recomputed in __init__ -> expected "missing".
        real_missing = [k for k in missing if "inv_freq" not in k and "rotary_emb" not in k]
        if real_missing:
            import logging as _l

            _l.getLogger(__name__).warning(
                "from_pretrained: %d missing keys (e.g. %s)", len(real_missing), real_missing[:5]
            )
        if torch_dtype not in (None, "auto"):
            # Cast PARAMETERS only — leave buffers (notably the RoPE `inv_freq`, computed in fp32) alone,
            # matching transformers. A blanket model.to(bf16) would coarsen inv_freq and skew RoPE.
            for p in model.parameters():
                p.data = p.data.to(torch_dtype)
        model.eval()
        return model

    @classmethod
    def _from_config(cls, config, **kwargs):
        impl = kwargs.get("attn_implementation") or getattr(config, "_attn_implementation", None) or "sdpa"
        _set_attn_implementation(config, impl)
        return cls(config)

    def save_pretrained(self, save_directory: str, **kwargs):
        from safetensors.torch import save_file

        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        sd = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(sd, os.path.join(save_directory, "model.safetensors"), metadata={"format": "pt"})


__all__ = ["PreTrainedModel", "ALL_ATTENTION_FUNCTIONS"]


"""Auto* stubs. GR00T only uses these to `.register(...)` its classes into the transformers auto-map
(so `AutoModel.from_pretrained` could find them) — but we always load via explicit classes, so
registration is a no-op and `from_pretrained`/`from_config` raise if actually called."""


class _AutoStub:
    @classmethod
    def register(cls, *args, **kwargs):
        return None  # no-op: we load via explicit classes, not the auto-map

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise RuntimeError(f"{cls.__name__} is not supported in the vendored subset; load via explicit classes.")

    @classmethod
    def from_config(cls, *args, **kwargs):
        raise RuntimeError(f"{cls.__name__} is not supported in the vendored subset; load via explicit classes.")


class AutoConfig(_AutoStub):
    pass


class AutoModel(_AutoStub):
    pass


class AutoProcessor(_AutoStub):
    pass


__all__ = ["AutoConfig", "AutoModel", "AutoProcessor"]
