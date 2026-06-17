"""Single-session Perich/Miller dataset for the chronic-stability eval.

recording_ids = [ os.environ["POYO_SESSION"] ] so run_chronic.sh can sweep one session at a
time. READOUT matches PoyoMPDataset's center-out config exactly (cursor_velocity_2d, reach-period
eval) so the frozen base model's session/unit embeddings line up with how it was trained.

Drop into the poyo repo as src/datasets/poyo_perich_session.py.
"""
import os
from copy import deepcopy

import numpy as np
import torchmetrics
from brainsets.datasets import PerichMillerPopulation2018
from temporaldata import Data, Interval


def _truncate_interval(iv, frac):
    """Keep the first `frac` of total interval duration (for low-data generalization tests)."""
    if frac >= 1.0:
        return iv
    start = np.asarray(iv.start, dtype=float); end = np.asarray(iv.end, dtype=float)
    dur = end - start; target = float(dur.sum()) * frac
    ks, ke, cum = [], [], 0.0
    for s, e, d in zip(start, end, dur):
        if cum + d <= target:
            ks.append(s); ke.append(e); cum += d
        else:
            rem = target - cum
            if rem > 1e-6:
                ks.append(s); ke.append(s + rem)
            break
    if not ks:
        ks, ke = [start[0]], [end[0]]
    return Interval(start=np.array(ks, dtype=float), end=np.array(ke, dtype=float))

CO_READOUT_CONFIG = {
    "readout": {
        "readout_id": "cursor_velocity_2d",
        "normalize_mean": 0.0,
        "normalize_std": 20.0,
        "weights": {
            "movement_phases.random_period": 1.0,
            "movement_phases.hold_period": 0.1,
            "movement_phases.reach_period": 5.0,
            "movement_phases.return_period": 1.0,
            "movement_phases.invalid": 0.1,
            "cursor_outlier_segments": 0.0,
        },
        "metrics": [{"metric": torchmetrics.R2Score()}],
        "eval_interval": "movement_phases.reach_period",
    }
}


class PoyoPerichSessionDataset(PerichMillerPopulation2018):
    def __init__(self, root, transform=None, **kwargs):
        session = os.environ["POYO_SESSION"]
        super().__init__(root, recording_ids=[session], transform=transform, **kwargs)

    def get_recording_hook(self, data: Data):
        data.config = deepcopy(CO_READOUT_CONFIG)
        super().get_recording_hook(data)

    def get_sampling_intervals(self, split=None):
        # backward-compatible: default frac=1.0 -> no-op (chronic/xsub unaffected). For low-data
        # generalization tests, POYO_TRAIN_FRACTION<1.0 truncates ONLY the train split; valid/test
        # untouched so every point is evaluated on the same held-out data.
        out = super().get_sampling_intervals(split)
        if split == "train":
            frac = float(os.environ.get("POYO_TRAIN_FRACTION", "1.0"))
            if frac < 1.0:
                out = {rid: _truncate_interval(iv, frac) for rid, iv in out.items()}
        return out
