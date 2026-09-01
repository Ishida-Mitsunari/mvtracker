#!/usr/bin/env python
"""Prefetch SpaTrackerV2-Offline weights into HF_HOME, then construct the wrapper."""
import os
import sys

print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
print("HF_HOME", os.environ.get("HF_HOME"), flush=True)

import torch

print("torch", torch.__version__, "visible", torch.cuda.device_count(), flush=True)
torch.cuda.set_device(0)
print("name", torch.cuda.get_device_name(0), flush=True)

sys.path.insert(0, "/share/tgp/yangyi/mvtracker")
from mvtracker.models.core.monocular_baselines import SpaTrackerV2Wrapper

print("constructing wrapper...", flush=True)
w = SpaTrackerV2Wrapper(model_type="offline", vo_points=756)
print("ok", type(w.model), flush=True)
print("allocated GB", torch.cuda.memory_allocated() / 1024**3, flush=True)
