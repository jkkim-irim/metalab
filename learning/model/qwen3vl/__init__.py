"""Vendored Qwen3-VL — the minimal ``transformers`` subset the GR00T backbone needs, with the
``transformers`` package itself removed. Not a 1:1 upstream mirror (dead code pruned); re-sync against
the pristine ``allex_groot/Isaac-GR00T`` reference is manual.

  * ``pretrained_base`` — PreTrainedModel + the attention backends, PretrainedConfig, Cache,
                          BatchFeature, ProcessorMixin, the Auto* factories, utils, and the offline
                          HF-cache path resolver.
  * ``modeling``        — the Qwen3-VL backbone: config + modeling (activations/rope/masking/layers).
  * ``processing``      — the Qwen3-VL image processor + tokenizer + processor.

GR00T uses the backbone purely as a feature extractor (``model(**inputs, output_hidden_states=True)``
-> ``hidden_states[-1]``); this targets exactly that path. Vendored from transformers 4.57.3.
"""
