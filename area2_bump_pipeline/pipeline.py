# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["dandi==0.74.0"]
# ///
#
# area2_bump (DANDI:000127) — Neural Latents Benchmark '21 dataset:
# macaque somatosensory area 2 spiking during a center-out reaching task with
# mechanical BUMP perturbations (Chowdhury, Glaser, Miller). Same NLB NWB schema
# as pei_pandarinath_nlb_2021 (mc_maze, DANDI:000140) — this is that pipeline
# adapted to area2_bump. Run as a LOCAL brainset:
#     brainsets prepare ./area2_bump_pipeline --local \
#         --raw-dir ./data/raw --processed-dir ./data/processed
#
# Demonstrates onboarding a NEW DANDI dataset into POYO. The decoding target is
# 2D hand velocity (data.hand.vel) — directly analogous to POYO's
# cursor_velocity_2d readout — but the units are NEW (area 2, not M1) and the
# session/animal are new, so it exercises POYO "unit identification" transfer.
#
# VALIDATED against the DANDI:000127/0.220113.0359 train NWB (streamed 2026-06-12):
#   [V1] DONE: no eye tracking (behavior = force/hand_pos/hand_vel/joint_*/muscle_*, no
#        eye_pos) -> eye is always None.
#   [V1b] FIXED: hand_pos/hand_vel are RATE-based (has_timestamps=False), already uniform
#        -> RegularTimeSeries from .rate/.starting_time (old code read .timestamps[:] = None).
#   [V2] DONE: move_onset_time is NaN on passive/bump trials (split=='none') -> excluded
#        from nlb_eval_intervals; window [-0.05,+0.65] kept as the NLB convention.
#   [V3] DONE: `ctr_hold_bump` (bool) + bump_dir/bump_time present; carries over via
#        Interval.from_dataframe. split values: train/val/none. Device: 96-ch Utah array.
from argparse import ArgumentParser
import datetime

import numpy as np
import h5py
from pynwb import NWBHDF5IO
from temporaldata import Data, Interval, RegularTimeSeries
import pandas as pd

from brainsets.descriptions import (
    BrainsetDescription,
    SessionDescription,
    DeviceDescription,
)
from brainsets.utils.dandi_utils import (
    extract_spikes_from_nwbfile,
    extract_subject_from_nwb,
    get_nwb_asset_list,
    download_file,
)
from brainsets import serialize_fn_map
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import Task, RecordingTech  # enum members, NOT raw strings

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")

# area2_bump version, from the DANDI API (api.dandiarchive.org/api/dandisets/000127)
DANDISET = "DANDI:000127/0.220113.0359"


class Pipeline(BrainsetPipeline):
    brainset_id = "area2_bump"
    dandiset_id = DANDISET
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        asset_list = get_nwb_asset_list(cls.dandiset_id)
        manifest_list = [{"path": x.path, "url": x.download_url} for x in asset_list]
        for m in manifest_list:
            # NLB releases ship a train asset and a held-out test asset
            m["id"] = "area2_bump_test" if "test" in m["path"] else "area2_bump_train"
        return pd.DataFrame(manifest_list).set_index("id")

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        return download_file(
            manifest_item.path,
            manifest_item.url,
            self.raw_dir,
            overwrite=self.args.redownload,
        )

    def process(self, fpath):
        brainset_description = BrainsetDescription(
            id=self.brainset_id,
            origin_version="dandi/000127/0.220113.0359",
            derived_version="1.0.0",
            source="https://dandiarchive.org/dandiset/000127",
            description="Sorted unit spiking from macaque somatosensory area 2 with "
            "simultaneous 2D hand kinematics, during a center-out reaching task that "
            "includes mechanical BUMP perturbations (active reaches + passive bumps). "
            "Neural Latents Benchmark '21 (Chowdhury, Glaser, Miller).",
        )

        self.update_status("Loading NWB")
        io = NWBHDF5IO(fpath, "r")
        nwbfile = io.read()

        self.update_status("Extracting Metadata")
        subject = extract_subject_from_nwb(nwbfile)
        recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
        device_id = f"{subject.id}_{recording_date}"
        session_id = f"{subject.id}_area2_bump"
        session_id += "_test" if "test" in str(fpath) else "_train"

        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        session_description = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
            task=Task.REACHING,
        )
        device_description = DeviceDescription(
            id=device_id,
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
        )

        self.update_status("Extracting Spikes")
        spikes, units = extract_spikes_from_nwbfile(
            nwbfile, recording_tech=RecordingTech.UTAH_ARRAY_SPIKES
        )

        self.update_status("Extracting Trials")
        trials = extract_trials(nwbfile)

        data = Data(
            brainset=brainset_description,
            session=session_description,
            device=device_description,
            spikes=spikes,
            units=units,
            trials=trials,
            domain="auto",
        )

        # The held-out test asset has no public behavior; splits/behavior only on train.
        if "test" not in str(fpath):
            self.update_status("Creating Splits")
            hand, eye = extract_behavior(nwbfile, trials)
            data.hand = hand
            if eye is not None:  # [V1] area2_bump may lack eye tracking
                data.eye = eye

            # [V2] eval window around move onset. VERIFIED against DANDI:000127: passive/bump
            # trials have NaN move_onset_time (split=='none'), so exclude them from eval.
            mo = np.asarray(trials.move_onset_time, dtype=float)
            has_move = ~np.isnan(mo)
            data.nlb_eval_intervals = Interval(
                start=mo[has_move] - 0.05,
                end=mo[has_move] + 0.65,
            )

            # NLB-defined train/val masks (val used as our test), same as mc_maze
            train_trials, valid_trials = trials.select_by_mask(
                trials.train_mask_nwb
            ).split([0.8, 0.2], shuffle=True, random_seed=42)
            test_trials = trials.select_by_mask(trials.test_mask_nwb)
            data.train_domain = train_trials
            data.valid_domain = valid_trials
            data.test_domain = test_trials

        io.close()
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def extract_trials(nwbfile):
    r"""Trial table -> Interval. NLB split column ('train'/'val') becomes the
    train/test masks. Any extra columns (e.g. [V3] `ctr_hold_bump` marking passive
    bump trials, `move_onset_time`) carry over onto the Interval automatically."""
    trial_table = nwbfile.trials.to_dataframe().rename(
        columns={"start_time": "start", "stop_time": "end", "split": "split_indicator"}
    )
    trials = Interval.from_dataframe(trial_table)
    split = trial_table.split_indicator.to_numpy()
    trials.train_mask_nwb = split == "train"  # "_" suffix: train_mask is reserved
    trials.test_mask_nwb = split == "val"
    return trials


def extract_behavior(nwbfile, trials):
    """Extract 2D hand kinematics. hand.vel is the POYO cursor_velocity_2d target.

    VERIFIED against the DANDI:000127 train NWB: processing['behavior'] holds
    force/hand_pos/hand_vel/joint_*/muscle_*; hand_pos & hand_vel are RATE-based
    TimeSeries (has_timestamps=False), already uniformly sampled — so build the
    RegularTimeSeries directly from .rate/.starting_time (NO interpolation/regularize).
    [V1] No eye tracking present (no eye_pos interface) -> eye is always None here."""
    beh = nwbfile.processing["behavior"]
    hv = beh["hand_vel"]
    rate = float(hv.rate)                    # Hz; rate-based, not per-sample timestamps
    start_time = float(hv.starting_time)

    hand = RegularTimeSeries(
        sampling_rate=rate,
        pos=beh["hand_pos"].data[:],
        vel=hv.data[:],
        domain="auto",
        domain_start=start_time,
    )

    eye = None  # [V1] CONFIRMED absent in area2_bump (no eye_pos interface in 000127)
    if "eye_pos" in beh.data_interfaces:     # defensive; not present in the validated asset
        ep = beh["eye_pos"]
        eye = RegularTimeSeries(
            sampling_rate=float(ep.rate), pos=ep.data[:],
            domain="auto", domain_start=float(ep.starting_time),
        )

    return hand, eye
