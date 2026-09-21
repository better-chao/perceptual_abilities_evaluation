# -*- coding: utf-8 -*-
"""Visual Prompt Fine-tuning for YOLO-World.

This is the unified feature-space visual prompt described in the paper: a
50x50 learnable tensor is added to every level of the multi-scale image
feature map, and nothing else is updated.  It replaces the per-architecture
image-space variants used in the first submission, so that the
*Visual Prompt Fine-tuning* column is one mechanism across all six detectors.

YOLO-World returns ``(img_feats, txt_feats)`` from ``extract_feat`` and its
neck is text-guided, so the prompt is applied to the image features only,
after the neck has already fused them with the text embeddings.

Usage: referenced from yolo_world_visual_prompt_tuning.py through ``custom_imports``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmyolo.registry import MODELS
from yolo_world.models.detectors import SimpleYOLOWorldDetector


class MultiScaleVisualPrompt(nn.Module):
    """The visual prompt used by *Visual Prompt Fine-tuning* in this study.

    One ``prompt_size x prompt_size`` learnable tensor is held per level of the
    multi-scale image feature map.  At every forward pass each tensor is resized
    to the spatial size of its level with bilinear interpolation and added to the
    feature map.  The tensors are the only parameters that receive gradients;
    every backbone, neck, head and text-encoder parameter is frozen by a zero
    learning-rate multiplier in the optimizer configuration.

    The prompt is initialised at zero, so the model starts from exactly the
    frozen model's behaviour and the run is directly comparable with
    *Zero-shot Inference*.
    """

    def __init__(self, channels, prompt_size=50):
        super().__init__()
        self.prompt_size = prompt_size
        self.prompts = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, c, prompt_size, prompt_size)) for c in channels])

    def forward(self, feats):
        if len(feats) != len(self.prompts):
            raise ValueError(
                'the model produced %d feature levels but the visual prompt was '
                'built for %d; set prompt_channels to match the neck output'
                % (len(feats), len(self.prompts)))
        out = []
        for feat, prompt in zip(feats, self.prompts):
            p = F.interpolate(prompt.to(feat.dtype), size=feat.shape[-2:],
                              mode='bilinear', align_corners=False)
            out.append(feat + p)
        # keep the container type the detector expects downstream
        return tuple(out) if isinstance(feats, tuple) else out


@MODELS.register_module()
class SimpleYOLOWorldDetector_VisualPrompt(SimpleYOLOWorldDetector):
    """YOLO-World with the feature-space visual prompt inserted after the neck.

    The base class matches the one used by Textual Prompt Fine-tuning
    (SimpleYOLOWorldDetector), so the two paradigms differ only in what trains.

    ``prompt_channels`` must match ``model.neck.out_channels``; the default is
    the YOLOv8-L setting used in this study.
    """

    def __init__(self, *args, prompt_size=50,
                 prompt_channels=(256, 512, 512), **kwargs):
        super().__init__(*args, **kwargs)
        self.visual_prompt = MultiScaleVisualPrompt(prompt_channels, prompt_size)

    def extract_feat(self, batch_inputs, batch_data_samples=None):
        img_feats, txt_feats = super().extract_feat(batch_inputs, batch_data_samples)
        return self.visual_prompt(img_feats), txt_feats
