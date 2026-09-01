#!/usr/bin/env python
"""Download SpatialTrackerV2-Offline into HF_HOME (CPU)."""
import os
import sys

os.environ.setdefault("HF_HOME", "/share/tgp/yangyi/envs/hf-home")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
print("HF_HOME", os.environ["HF_HOME"], flush=True)
sys.path.insert(0, "/share/tgp/yangyi/spatialtrackerv2")
from models.SpaTrackV2.models.predictor import Predictor

print("from_pretrained...", flush=True)
model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
model.eval()
print("ok", type(model), flush=True)
