"""POYO dataset for area2_bump finetune (Stage A: reuses cursor_velocity_2d -> isolates the
§2.5 transfer machinery from any new-readout code).

Drop into the poyo repo as src/datasets/poyo_area2.py. Base classes + __init__ pattern verified
against brainsets@4ccee58 (PerichMillerPopulation2018) + torch_brain@ca3cfb. Decode target =
data.hand.vel (the area2_bump pipeline emits a `hand` RegularTimeSeries + `nlb_eval_intervals`).
"""
from typing import Callable, Optional, Literal
from copy import deepcopy
from pathlib import Path

import torchmetrics
from temporaldata import Data
from torch_brain.dataset import Dataset, SpikingDatasetMixin

# .h5 stem the area2_bump pipeline writes (process(): "<subject>_area2_bump_train")
TRAIN_RECORDING_IDS = ["han_area2_bump_train"]


class PoyoArea2Dataset(SpikingDatasetMixin, Dataset):
    READOUT_CONFIG = {
        "readout": {
            "readout_id": "cursor_velocity_2d",   # REUSE -> no new modality registration
            "timestamp_key": "hand.timestamps",
            "value_key": "hand.vel",
            "normalize_mean": 0.0,
            "normalize_std": 20.0,                 # TODO: set from measured hand.vel std
            "metrics": [{"metric": torchmetrics.R2Score()}],
            "eval_interval": "nlb_eval_intervals",
        }
    }

    def __init__(self, root: str, recording_ids: Optional[list[str]] = None,
                 transform: Optional[Callable] = None, dirname: str = "area2_bump", **kwargs):
        super().__init__(
            dataset_dir=Path(root) / dirname,           # root/<brainset_id>/<rid>.h5
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
