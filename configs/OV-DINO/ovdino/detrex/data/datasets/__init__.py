# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
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
# ------------------------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# ------------------------------------------------------------------------------------------------

from . import (
    register_coco_ovd,
    # register_custom_ovd
    register_lvis_ovd,
    register_o365_ovd,
)
from .coco_ovd import register_coco_ovd_instances
from .imagenet_template import template_meta
from .o365_ovd import register_objects365_ovd_instances
# from .custom_ovd import register_custom_ovd_instances
from .utils import clean_caption, clean_words_or_phrase
