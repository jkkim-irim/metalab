# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Obsolete after the transformers-package removal.

These were HuggingFace-Hub workarounds (local-first `from_pretrained`, mistral-regex network-error
suppression) that patched `transformers.*`. With the Qwen3-VL backbone vendored under
`learning.model.qwen3vl` and weights loaded from local safetensors, there is no `transformers` hub
path to patch — so this module is now a no-op, kept only so existing `import ... hf_compat` sites
(applied on `groot_policy` import) keep working."""


def _patch_hf_local_first() -> None:
    pass


def _patch_mistral() -> None:
    pass
