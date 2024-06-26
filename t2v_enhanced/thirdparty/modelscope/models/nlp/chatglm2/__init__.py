# These codes are copied from modelscope revision c58451baead80d83281f063d12fb377fad415257 
# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .configuration import ChatGLM2Config
    from .tokenization import ChatGLM2Tokenizer
    from .text_generation import ChatGLM2ForConditionalGeneration
    from .quantization import (
        quantize, )

else:
    _import_structure = {
        'configuration': ['ChatGLM2Config'],
        'text_generation': ['ChatGLM2ForConditionalGeneration'],
        'quantization': ['quantize'],
        'tokenization': [
            'ChatGLM2Tokenizer',
        ],
    }
    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__)
