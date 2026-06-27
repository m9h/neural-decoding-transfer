"""POYO dataset for the rodent (Vollan/Moser 2025) cross-species spatial transfer.

The farthest-domain transfer test in this project: monkey motor-cortex base -> rat MEC/hippocampus,
decoding 2D allocentric position (the grid/place-cell readout) via the NEW position_2d head
(registered by importing register_position_modality). Single open-field session via env
POYO_SESSION (e.g. of_24365_2). Generic torch_brain Dataset over the prepped brainset dir
(mirrors transfer/poyo_dmfc.py). Drop into the poyo repo as src/datasets/poyo_vollan_session.py.

Behavior: the brainset's `samples` IrregularTimeSeries carries x, y (100 Hz arena position,
normalized ~[-0.75, 0.75], no NaN, spanning the full recording). The recording hook stacks them
into a fresh `position` channel over the same full domain -- so every 1 s window intersects it and
there is no out-of-window lazy-slice pathology (cf. the area2 hand channel; unlike the dmfc timing
channel which only spanned Set->Go).
"""
import os
from typing import Callable, Optional, Literal
from copy import deepcopy
from pathlib import Path

import numpy as np
import torchmetrics
from temporaldata import Data, IrregularTimeSeries
from torch_brain.dataset import Dataset, SpikingDatasetMixin

from . import register_position_modality  # noqa: F401  -> registers position_2d on import


class PoyoVollanSessionDataset(SpikingDatasetMixin, Dataset):
    READOUT_CONFIG = {
        "readout": {
            "readout_id": "position_2d",          # NEW 2D position readout (dim 2, MSE)
            "timestamp_key": "position.timestamps",
            "value_key": "position.pos",
            "normalize_mean": 0.0,                 # arena coords ~[-0.75, 0.75], mean ~0
            "normalize_std": 0.4,                  # measured std ~0.40 across of_* sessions
            "metrics": [{"metric": torchmetrics.R2Score()}],
            "eval_interval": "domain",             # continuous navigation: score every in-window
                                                   # step; the split is restricted by
                                                   # get_sampling_intervals (test_domain at eval).
        }
    }

    def __init__(self, root: str, recording_ids: Optional[list[str]] = None,
                 transform: Optional[Callable] = None,
                 dirname: str = "vollan_moser_alternating_2025", **kwargs):
        session = os.environ.get("POYO_SESSION")
        super().__init__(
            dataset_dir=Path(root) / dirname,
            recording_ids=recording_ids or ([session] if session else None),
            transform=transform,
            namespace_attributes=["session.id", "subject.id", "units.id"],
            **kwargs,
        )
        self.spiking_dataset_mixin_uniquify_unit_ids = True

    def get_recording_hook(self, data: Data):
        s = data.samples
        x = np.asarray(s.x, dtype=np.float32)
        y = np.asarray(s.y, dtype=np.float32)
        ts = np.asarray(s.timestamps, dtype=np.float64)
        data.position = IrregularTimeSeries(
            timestamps=ts,
            pos=np.stack([x, y], axis=-1).astype(np.float32),
            domain=s.domain,
        )
        data.config = deepcopy(self.READOUT_CONFIG)
        super().get_recording_hook(data)

    def get_sampling_intervals(self, split: Optional[Literal["train", "valid", "test"]] = None):
        domain_key = "domain" if split is None else f"{split}_domain"
        return {rid: getattr(self.get_recording(rid), domain_key) for rid in self.recording_ids}
