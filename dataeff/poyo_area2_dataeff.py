"""Data-efficiency variant of PoyoArea2Dataset.

Identical to transfer/poyo_area2.py, but the TRAIN split sampling intervals are truncated to a
fraction of total duration, set by the env var POYO_TRAIN_FRACTION (default 1.0). This drives the
"foundation model pays off" curve: R2(hand.vel) vs amount of new-session training data, for
POYO-finetune (freeze core) vs from-scratch. valid/test domains are NEVER truncated -> every point
on the curve is evaluated on the same held-out data.

Drop into the poyo repo as src/datasets/poyo_area2_dataeff.py.
"""
import os
from typing import Callable, Optional, Literal
from copy import deepcopy
from pathlib import Path

import numpy as np
import torchmetrics
from temporaldata import Data, Interval
from torch_brain.dataset import Dataset, SpikingDatasetMixin

TRAIN_RECORDING_IDS = ["han_area2_bump_train"]


def _truncate_interval(iv: Interval, frac: float) -> Interval:
    """Keep the first `frac` of total interval duration (clip the boundary interval)."""
    if frac >= 1.0:
        return iv
    start = np.asarray(iv.start, dtype=float)
    end = np.asarray(iv.end, dtype=float)
    dur = end - start
    target = float(dur.sum()) * frac
    keep_s, keep_e, cum = [], [], 0.0
    for s, e, d in zip(start, end, dur):
        if cum + d <= target:
            keep_s.append(s); keep_e.append(e); cum += d
        else:
            rem = target - cum
            if rem > 1e-6:               # keep a partial boundary window
                keep_s.append(s); keep_e.append(s + rem)
            break
    if not keep_s:                       # frac so small nothing survives -> keep first interval
        keep_s, keep_e = [start[0]], [end[0]]
    return Interval(start=np.array(keep_s, dtype=float), end=np.array(keep_e, dtype=float))


class PoyoArea2DataEffDataset(SpikingDatasetMixin, Dataset):
    READOUT_CONFIG = {
        "readout": {
            "readout_id": "cursor_velocity_2d",
            "timestamp_key": "hand.timestamps",
            "value_key": "hand.vel",
            "normalize_mean": 0.0,
            "normalize_std": 20.0,
            "metrics": [{"metric": torchmetrics.R2Score()}],
            "eval_interval": "nlb_eval_intervals",
        }
    }

    def __init__(self, root: str, recording_ids: Optional[list[str]] = None,
                 transform: Optional[Callable] = None, dirname: str = "area2_bump", **kwargs):
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids or TRAIN_RECORDING_IDS,
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "units.id"],
            **kwargs,
        )
        self.spiking_dataset_mixin_uniquify_unit_ids = True

    def get_recording_hook(self, data: Data):
        data.config = deepcopy(self.READOUT_CONFIG)
        super().get_recording_hook(data)

    def get_sampling_intervals(self, split: Optional[Literal["train", "valid", "test"]] = None):
        domain_key = "domain" if split is None else f"{split}_domain"
        out = {rid: getattr(self.get_recording(rid), domain_key) for rid in self.recording_ids}
        if split == "train":
            frac = float(os.environ.get("POYO_TRAIN_FRACTION", "1.0"))
            if frac < 1.0:
                out = {rid: _truncate_interval(iv, frac) for rid, iv in out.items()}
        return out
