"""POYO dataset for the ELAPSED-TIME (time-to-go ramp) dmfc_rsg transfer — the dynamics-engaging
reframing of the cognitive-timing decode. Target = data.timing.tp, which the (edited) dmfc_rsg
pipeline now fills as a DESCENDING RAMP over each [set,go] epoch (tp ms at Set -> 0 at Go), 0
elsewhere. Same readout (interval_timing_1d, dim1, MSE) as the constant-tp version; only the
target VALUES and the normalization differ. tp is recoverable as the predicted ramp value at Set.

Drop into the poyo repo as src/datasets/poyo_dmfc_elapsed.py.
"""
from typing import Callable, Optional, Literal
from copy import deepcopy
from pathlib import Path

import torchmetrics
from temporaldata import Data
from torch_brain.dataset import Dataset, SpikingDatasetMixin

from . import register_timing_modality  # noqa: F401  -> registers interval_timing_1d on import

TRAIN_RECORDING_IDS = ["haydn_dmfc_rsg_train"]


class PoyoDmfcElapsedDataset(SpikingDatasetMixin, Dataset):
    READOUT_CONFIG = {
        "readout": {
            "readout_id": "interval_timing_1d",
            "timestamp_key": "timing.timestamps",
            "value_key": "timing.tp",            # now holds time-to-go (ms), a ramp not a constant
            "normalize_mean": 300.0,             # ramp 0..~1200ms (+zeros outside epochs) -> ~300 mean
            "normalize_std": 300.0,
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
