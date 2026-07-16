"""Weighted-average focus stacking (SPEC §7.3).

Softmax over per-frame sharpness with a stack-global scale, computed in one
streaming pass using the online-softmax trick (numerically stable at any T).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from planefuse.backend import Device, ops
from planefuse.stack.base import CancelFn, FrameSource, ParamSpec, ProgressFn, StackResult, register


def _sharp(frame_np: np.ndarray, device: Device, radius: int) -> torch.Tensor:
    t = ops.to_tensor(frame_np, device)
    return ops.sharpness_map(ops.rgb_to_luminance(t), radius).squeeze(0)  # (H, W)


@register("weighted")
class Weighted:
    @staticmethod
    def params() -> list[ParamSpec]:
        return [
            ParamSpec("temperature", "Temperature", "float", default=0.05, min=0.001, max=1.0,
                      tooltip="Softmax temperature; lower = closer to hard max (sharper, "
                              "more halo-prone). Default 0.05."),
            ParamSpec("sharpness_radius", "Sharpness radius", "int", default=8, min=2, max=40,
                      tooltip="Radius (px) of the local sharpness window."),
        ]

    def run(self, source: FrameSource, device: Device, params: dict[str, Any],
            progress: ProgressFn | None = None, cancel: CancelFn | None = None,
            masks: FrameSource | None = None) -> StackResult:
        n = len(source)
        temp = float(params.get("temperature", 0.05))
        radius = int(params.get("sharpness_radius", 8))

        ref = n // 2
        ref_sharp = _sharp(source.read(ref), device, radius)
        scale = torch.quantile(ref_sharp.flatten(), 0.999).clamp_min(1e-8)

        dev = device.torch_device
        first = ops.to_tensor(source.read(0), device)
        c, h, w = first.shape
        m_run = torch.full((h, w), float("-inf"), device=dev)
        num = torch.zeros((c, h, w), device=dev)
        den = torch.zeros((h, w), device=dev)

        for i in range(n):
            if cancel is not None and cancel():
                raise InterruptedError("weighted stack cancelled")
            if progress is not None:
                progress(f"weighted {i + 1}/{n}", i / n)
            frame = first if i == 0 else ops.to_tensor(source.read(i), device)
            s = ops.sharpness_map(ops.rgb_to_luminance(frame), radius).squeeze(0)
            if masks is not None:
                mk = torch.from_numpy(np.ascontiguousarray(masks.read(i))).to(dev).bool()
                s = torch.where(mk, s, torch.zeros_like(s))
            logit = s / (scale * temp)
            new_m = torch.maximum(m_run, logit)
            rescale = torch.exp(m_run - new_m)
            rescale = torch.nan_to_num(rescale, nan=0.0, posinf=0.0)
            wt = torch.exp(logit - new_m)
            num = num * rescale.unsqueeze(0) + frame * wt.unsqueeze(0)
            den = den * rescale + wt
            m_run = new_m

        image = num / den.clamp_min(1e-8).unsqueeze(0)
        if progress is not None:
            progress("done", 1.0)
        return StackResult(image=ops.to_numpy(image))
