"""POYO dataset for the dmfc_rsg timing transfer (Stage B: the Merchant-aligned target).

Drop into the poyo repo as src/datasets/poyo_dmfc.py. NEW interval_timing_1d readout (registered
by importing register_timing_modality). Decode target = produced interval `tp` (per trial, ms),
exposed as a timestamped `timing` channel and read within the Set->Go window. Base classes/__init__
verified against brainsets@4ccee58 + torch_brain@ca3cfb (see poyo_area2.py).

The dmfc_rsg pipeline emits `data.timing` (tp broadcast over each [set_time, go_time] window) +
`timing_eval_intervals` -- see dmfc_rsg_pipeline/pipeline.py (timing-channel addition).
"""
from typing import Callable, Optional, Literal
from copy import deepcopy
from pathlib import Path

import torchmetrics
from temporaldata import Data
from torch_brain.dataset import Dataset, SpikingDatasetMixin

from . import register_timing_modality  # noqa: F401  -> registers interval_timing_1d on import

TRAIN_RECORDING_IDS = ["haydn_dmfc_rsg_train"]


class PoyoDmfcDataset(SpikingDatasetMixin, Dataset):
    READOUT_CONFIG = {
        "readout": {
            "readout_id": "interval_timing_1d",    # NEW scalar timing readout (dim 1, MSE)
            "timestamp_key": "timing.timestamps",
            "value_key": "timing.tp",
            "normalize_mean": 750.0,               # tp in ms (~short 480-800 / long 800-1200)
            "normalize_std": 150.0,
            "metrics": [{"metric": torchmetrics.R2Score()}],
            "eval_interval": "timing_eval_intervals",
        }
    }

    def __init__(self, root: str, recording_ids: Optional[list[str]] = None,
                 transform: Optional[Callable] = None, dirname: str = "dmfc_rsg", **kwargs):
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
        return {rid: getattr(self.get_recording(rid), domain_key) for rid in self.recording_ids}
