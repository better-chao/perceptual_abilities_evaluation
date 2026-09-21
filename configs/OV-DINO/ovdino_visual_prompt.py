# -*- coding: utf-8 -*-
"""Visual Prompt Fine-tuning for OV-DINO.

This is the unified feature-space visual prompt described in the paper: a
50x50 learnable tensor is added to every level of the multi-scale image feature
map, and nothing else is updated.  It replaces the per-architecture image-space
variants used in the first submission, so that the *Visual Prompt Fine-tuning*
column is one mechanism across all six detectors.

OV-DINO (detrex) maps the Swin-T stages to four 256-channel levels through a
ChannelMapper-style neck.  Attach the prompt to that neck rather than to the
raw backbone by passing ``module_name="neck"``.

Usage
-----
In the training entry point, right after the model is built and the pre-trained
weights are loaded::

    from ovdino_visual_prompt import attach_visual_prompt
    prompt = attach_visual_prompt(model, channels=[256, 256, 256, 256], prompt_size=50)
    optimizer = torch.optim.AdamW(prompt.parameters(), lr=..., weight_decay=...)

``channels`` must list the channel count of each level the backbone returns;
the default above is the configuration used in this study.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleVisualPrompt(nn.Module):
    """The visual prompt used by *Visual Prompt Fine-tuning* in this study.

    One ``prompt_size x prompt_size`` learnable tensor is held per level of the
    multi-scale image feature map.  At every forward pass each tensor is resized
    to the spatial size of its level with bilinear interpolation and added to the
    feature map.  The tensors are the only parameters that receive gradients.

    The prompt is initialised at zero, so training starts from exactly the frozen
    model's behaviour and the run is directly comparable with *Zero-shot Inference*.
    """

    def __init__(self, channels, prompt_size=50):
        super().__init__()
        self.prompt_size = prompt_size
        self.prompts = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, c, prompt_size, prompt_size)) for c in channels])

    def _add(self, feat, prompt):
        p = F.interpolate(prompt.to(feat.dtype), size=feat.shape[-2:],
                          mode='bilinear', align_corners=False)
        return feat + p

    def forward(self, feats):
        """``feats`` may be a dict (detectron2 / detrex) or a list/tuple."""
        if isinstance(feats, dict):
            keys = list(feats.keys())
            if len(keys) != len(self.prompts):
                raise ValueError('backbone returned %d levels, prompt built for %d'
                                 % (len(keys), len(self.prompts)))
            return {k: self._add(feats[k], p) for k, p in zip(keys, self.prompts)}
        if len(feats) != len(self.prompts):
            raise ValueError('backbone returned %d levels, prompt built for %d'
                             % (len(feats), len(self.prompts)))
        out = [self._add(f, p) for f, p in zip(feats, self.prompts)]
        return type(feats)(out) if isinstance(feats, tuple) else out


def attach_visual_prompt(model, channels, prompt_size=50, module_name='backbone'):
    """Insert the prompt on the output of ``model.<module_name>`` and freeze the rest.

    Works through a forward hook, so no framework-internal class has to be
    subclassed and the released model definition is left untouched.

    Returns the prompt module; pass ``prompt.parameters()`` to the optimizer.
    """
    target = model
    for part in module_name.split('.'):
        target = getattr(target, part)

    prompt = MultiScaleVisualPrompt(channels, prompt_size)
    device = next(model.parameters()).device
    prompt.to(device)

    def _hook(module, inputs, output):
        return prompt(output)

    target.register_forward_hook(_hook)
    model.add_module('visual_prompt', prompt)

    # every backbone, neck, head and text-encoder parameter is frozen
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith('visual_prompt'))

    trainable = sum(p.numel() for p in prompt.parameters())
    total = sum(p.numel() for p in model.parameters())
    print('[visual prompt] trainable %d / %d parameters (%.4f%%)'
          % (trainable, total, 100.0 * trainable / max(total, 1)))
    return prompt
