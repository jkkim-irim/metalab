import json
import math
import os
from typing import Optional

from tokenizers import Tokenizer
import torch
from torchvision.transforms.v2 import functional as F

from .pretrained_base import BatchFeature

# ---- from models.qwen3_vl.image_processing ----
"""Compact, transformers-free Qwen2VL/Qwen3-VL image processor — a faithful reimplementation of the
`Qwen2VLImageProcessorFast` path GR00T uses (smart-resize to a patch multiple + rescale/normalize +
patchify). Deterministic torch/torchvision math; equivalence-gated against the stock processor.

Cosmos-Reason2-2B preprocessor_config: patch_size=16, temporal_patch_size=2, merge_size=2,
image_mean=image_std=0.5, pixel bounds [65536, 16777216]. Output: pixel_values
(num_patches, C*temporal_patch_size*patch_size*patch_size) + image_grid_thw (num_images, 3=[t,h,w])."""


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Verbatim Qwen2-VL logic: H,W divisible by `factor`, total pixels in [min,max], aspect kept."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be < 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen3VLImageProcessor:
    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        min_pixels: int = 65536,
        max_pixels: int = 16777216,
        rescale_factor: float = 1 / 255,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.rescale_factor = rescale_factor

    @classmethod
    def from_dict(cls, d: dict):
        size = d.get("size", {})
        return cls(
            patch_size=d.get("patch_size", 16),
            temporal_patch_size=d.get("temporal_patch_size", 2),
            merge_size=d.get("merge_size", 2),
            image_mean=d.get("image_mean", (0.5, 0.5, 0.5)),
            image_std=d.get("image_std", (0.5, 0.5, 0.5)),
            min_pixels=size.get("shortest_edge", d.get("min_pixels", 65536)),
            max_pixels=size.get("longest_edge", d.get("max_pixels", 16777216)),
        )

    def _to_chw(self, image) -> torch.Tensor:
        """Accept a PIL image / HWC array / CHW tensor -> CHW tensor, dtype preserved (uint8 from PIL).
        The fast processor resizes in the ORIGINAL dtype (uint8) before rescaling, so we keep uint8."""
        if isinstance(image, torch.Tensor):
            t = image
        else:
            import numpy as np

            t = torch.from_numpy(np.array(image))  # PIL -> HWC uint8
        if t.ndim == 2:
            t = t.unsqueeze(-1).repeat(1, 1, 3)
        if t.shape[-1] in (1, 3) and t.shape[0] not in (1, 3):  # HWC -> CHW
            t = t.permute(2, 0, 1)
        return t

    @torch.no_grad()
    def preprocess(self, images: list, return_grid: bool = True):
        factor = self.patch_size * self.merge_size
        mean = torch.tensor(self.image_mean).view(-1, 1, 1)
        std = torch.tensor(self.image_std).view(-1, 1, 1)
        all_patches, grids = [], []
        for image in images:
            img = self._to_chw(image)  # (C,H,W), uint8 [0,255]
            c, h, w = img.shape
            rh, rw = smart_resize(h, w, factor, self.min_pixels, self.max_pixels)
            img = F.resize(img, [rh, rw], interpolation=F.InterpolationMode.BICUBIC, antialias=True)
            img = (img.float() * self.rescale_factor - mean) / std  # resize on uint8, THEN rescale+normalize
            patches = img.unsqueeze(0).unsqueeze(0)  # (1, T=1, C, H, W)
            if patches.shape[1] % self.temporal_patch_size != 0:
                repeats = patches[:, -1:].repeat(1, self.temporal_patch_size - 1, 1, 1, 1)
                patches = torch.cat([patches, repeats], dim=1)
            bsz, gt, ch = patches.shape[:3]
            gt = gt // self.temporal_patch_size
            gh, gw = rh // self.patch_size, rw // self.patch_size
            ms, ps = self.merge_size, self.patch_size
            patches = patches.view(bsz, gt, self.temporal_patch_size, ch, gh // ms, ms, ps, gw // ms, ms, ps)
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flat = patches.reshape(bsz, gt * gh * gw, ch * self.temporal_patch_size * ps * ps)
            all_patches.append(flat.reshape(gt * gh * gw, -1))
            grids.append([gt, gh, gw])
        pixel_values = torch.cat(all_patches, dim=0)
        image_grid_thw = torch.tensor(grids, dtype=torch.long)
        return pixel_values, image_grid_thw


# ---- from models.qwen3_vl.tokenization ----
"""Compact, transformers-free Qwen2 tokenizer — wraps the `tokenizers` (Rust) backend loaded from
tokenizer.json and renders the jinja chat template, exposing exactly the API GR00T's processor uses:
__call__ (encode + pad + attention_mask), apply_chat_template, convert_tokens_to_ids, padding_side,
and the special-token ids. Equivalence-gated against the stock Qwen2TokenizerFast."""


class Qwen2TokenizerFast:
    def __init__(self, tokenizer_file: str, config: dict):
        self._tok = Tokenizer.from_file(tokenizer_file)
        self.chat_template = config.get("chat_template")
        self.padding_side = config.get("padding_side", "right")
        self.model_input_names = ["input_ids", "attention_mask"]
        self.init_kwargs: dict = {}

        def _tok_str(x):
            if isinstance(x, dict):
                return x.get("content")
            return x

        self.pad_token = _tok_str(config.get("pad_token"))
        self.eos_token = _tok_str(config.get("eos_token"))
        # Qwen has no dedicated pad token in some configs -> fall back to eos
        pad = self.pad_token or self.eos_token
        self.pad_token_id = self._tok.token_to_id(pad) if pad else 0
        self.eos_token_id = self._tok.token_to_id(self.eos_token) if self.eos_token else None
        self._jinja = None  # lazily compiled chat template

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        with open(os.path.join(path, "tokenizer_config.json"), encoding="utf-8") as f:
            config = json.load(f)
        ct = os.path.join(path, "chat_template.json")  # some repos split the template out
        if os.path.exists(ct) and not config.get("chat_template"):
            with open(ct, encoding="utf-8") as f:
                config["chat_template"] = json.load(f).get("chat_template")
        return cls(os.path.join(path, "tokenizer.json"), config)

    def convert_tokens_to_ids(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)

    def _pad(self, seqs: list[list[int]]) -> dict:
        maxlen = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:
            n = maxlen - len(s)
            padseq = [self.pad_token_id] * n
            m1, m0 = [1] * len(s), [0] * n
            if self.padding_side == "left":
                ids.append(padseq + s)
                mask.append(m0 + m1)
            else:
                ids.append(s + padseq)
                mask.append(m1 + m0)
        return {"input_ids": ids, "attention_mask": mask}

    def __call__(self, text, padding=False, return_tensors=None, add_special_tokens=False, **kwargs):
        if isinstance(text, str):
            text = [text]
        encs = self._tok.encode_batch(text, add_special_tokens=add_special_tokens)
        seqs = [e.ids for e in encs]
        if padding:
            out = self._pad(seqs)
        else:
            out = {"input_ids": seqs, "attention_mask": [[1] * len(s) for s in seqs]}
        if return_tensors == "pt":
            out = {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
        return out

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        if self._jinja is None:
            import jinja2

            env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
            env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
            self._jinja = env.from_string(self.chat_template)
        # transformers renders one conversation at a time; support a single conversation or a batch
        is_batch = messages and isinstance(messages[0], list)
        convs = messages if is_batch else [messages]
        rendered = [
            self._jinja.render(messages=c, add_generation_prompt=add_generation_prompt, **kwargs)
            for c in convs
        ]
        if not tokenize:
            return rendered if is_batch else rendered[0]
        out = self.__call__(rendered, **kwargs)
        return out["input_ids"]


__all__ = ["Qwen2TokenizerFast"]


# ---- from models.qwen3_vl.processing ----
"""Compact, transformers-free `Qwen3VLProcessor` — wraps the vendored image processor + tokenizer and
reproduces the `__call__` GR00T uses: image preprocessing, per-image `<|image_pad|>` expansion by grid,
then tokenization. Equivalence-gated against the stock Qwen3VLProcessor (input_ids + pixel_values)."""


class Qwen3VLProcessor:
    def __init__(self, image_processor: Qwen3VLImageProcessor, tokenizer: Qwen2TokenizerFast):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        from .pretrained_base import resolve_pretrained_path

        path = resolve_pretrained_path(path)
        with open(os.path.join(path, "preprocessor_config.json"), encoding="utf-8") as f:
            pp = json.load(f)
        return cls(Qwen3VLImageProcessor.from_dict(pp), Qwen2TokenizerFast.from_pretrained(path))

    def apply_chat_template(self, messages, **kwargs):
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(self, text=None, images=None, videos=None, return_tensors=None, padding=False, **kwargs):
        data = {}
        image_grid_thw = None
        if images is not None:
            pixel_values, image_grid_thw = self.image_processor.preprocess(images)
            data["pixel_values"] = pixel_values
            data["image_grid_thw"] = image_grid_thw

        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text = list(text)
            if image_grid_thw is not None:
                merge_length = self.image_processor.merge_size**2
                index = 0
                for i in range(len(text)):
                    while self.image_token in text[i]:
                        n = int(image_grid_thw[index].prod() // merge_length)
                        text[i] = text[i].replace(self.image_token, "<|placeholder|>" * n, 1)
                        index += 1
                    text[i] = text[i].replace("<|placeholder|>", self.image_token)
            tok = self.tokenizer(text, padding=padding, return_tensors=return_tensors)
            data["input_ids"] = tok["input_ids"]
            data["attention_mask"] = tok["attention_mask"]

        return BatchFeature(data=data)


__all__ = ["Qwen3VLProcessor"]
