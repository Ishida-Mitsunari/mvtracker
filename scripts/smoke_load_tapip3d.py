#!/usr/bin/env python
"""Load TAPIP3D on the currently visible GPU and print image_size / VRAM."""
import os
import sys

print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
import torch

print("visible", torch.cuda.device_count(), "name", torch.cuda.get_device_name(0), flush=True)
torch.cuda.set_device(0)

sys.path.insert(0, "/share/tgp/yangyi/mvtracker")
from mvtracker.models.core.monocular_baselines import TAPIP3DWrapper

print("constructing wrapper...", flush=True)
w = TAPIP3DWrapper(
    ckpt="/share/tgp/yangyi/mvtracker/checkpoints/tapip3d_final.pth",
    num_iters=6,
    grid_size=8,
    resolution_factor=1,
)
print("ok", type(w.model), "image_size", getattr(w.model, "image_size", None), flush=True)
print("allocated GB", torch.cuda.memory_allocated() / 1024**3, flush=True)
