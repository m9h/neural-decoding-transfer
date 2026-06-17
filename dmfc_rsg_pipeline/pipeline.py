# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["dandi==0.74.0"]
# ///
#
# dmfc_rsg (DANDI:000130) — Neural Latents Benchmark '21 cognitive-timing dataset:
# macaque DORSOMEDIAL FRONTAL CORTEX (DMFC) spiking during a Ready-Set-Go (RSG)
# time-interval REPRODUCTION task (Sohn, Narain, Meirhaeghe, Jazayeri). This is the
# Merchant-aligned target: medial/dorsomedial premotor timing, the same scientific
# question (population clocks for interval timing) as the Merchant/UNAM rhythmic-
# tapping work — but on a PUBLIC, raw-spike, NLB-benchmarked DANDI deposit that POYO
# can actually ingest.
#
# *** IMPORTANT — this is NOT the area2_bump pipeline with a swapped DANDISET. ***
# area2_bump decodes CONTINUOUS 2D hand velocity (-> POYO's cursor_velocity_2d head).
# dmfc_rsg has NO continuous kinematic trace to regress: the behavioral output is a
# single produced interval `tp` PER TRIAL (the monkey reports an estimated duration by
# an eye or hand movement). So the decoding target here is a SCALAR timing readout, and
# POYO needs a NEW readout head (interval-/timing-regression), not cursor_velocity_2d.
# That makes this a more ambitious but more scientifically pointed demo: does POYO's
# pretrained motor-cortex representation transfer to a COGNITIVE TIMING readout in a
# different (frontal) area via unit identification?
#
# Run as a LOCAL brainset:
#     brainsets prepare ./dmfc_rsg_pipeline --local \
#         --raw-dir ./data/raw --processed-dir ./data/processed
#
# VALIDATED against the DANDI:000130/0.220113.0407 train NWB (streamed 2026-06-12) — ALL CONFIRMED:
#   [V1] DONE: 3x 'Linear probe with 24 recording channels' (NOT Utah). MULTI_ELECTRODE_SPIKES
#        is the right family; units have spike_times (n=54).
#   [V2] DONE: trials.colnames has EXACTLY tp, ts, theta, is_eye, is_short, set_time, go_time,
#        split (+ ready_time/target_acq_time/reward_time/is_outlier...). NOTE: tp & ts are in
#        MILLISECONDS (~500-1000) -> normalize the timing readout accordingly.
#   [V3] DONE: set_time & go_time present -> timing_eval_intervals = [set_time, go_time] valid.
#   [V4] DONE: split column present (values 'train'/'val'). is_eye/is_short are booleans.
#   NOTE: processing modules are EMPTY (no 'behavior') -> confirms NO continuous kinematics;
#        the decode target tp is strictly per-trial (as this pipeline assumes). Correct.
from argparse import ArgumentParser
import datetime

import numpy as np
import h5py
from pynwb import NWBHDF5IO
from temporaldata import Data, Interval, RegularTimeSeries, IrregularTimeSeries, ArrayDict
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
from brainsets.taxonomy import RecordingTech  # enum members, NOT raw strings (Task has no
                                              # timing member -> SessionDescription.task=None)

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")

# DMFC_RSG version. Pin the NLB '21 benchmark release (0.220113.0407) so the train/val
# split matches the published benchmark; a newer 0.241017.1448 also exists.
DANDISET = "DANDI:000130/0.220113.0407"


class Pipeline(BrainsetPipeline):
    brainset_id = "dmfc_rsg"
    dandiset_id = DANDISET
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        asset_list = get_nwb_asset_list(cls.dandiset_id)
        manifest_list = [{"path": x.path, "url": x.download_url} for x in asset_list]
        for m in manifest_list:
            m["id"] = "dmfc_rsg_test" if "test" in m["path"] else "dmfc_rsg_train"
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
            origin_version="dandi/000130/0.220113.0407",
            derived_version="1.0.0",
            source="https://dandiarchive.org/dandiset/000130",
            description="Sorted unit spiking from macaque dorsomedial frontal cortex "
            "(DMFC) during a Ready-Set-Go time-interval reproduction task (eye and hand "
            "effectors, short/long prior contexts). Behavioral target is the produced "
            "interval tp per trial — a cognitive-timing readout, not continuous "
            "kinematics. Neural Latents Benchmark '21 (Sohn, Narain, Meirhaeghe, "
            "Jazayeri).",
        )

        self.update_status("Loading NWB")
        io = NWBHDF5IO(fpath, "r")
        nwbfile = io.read()

        self.update_status("Extracting Metadata")
        subject = extract_subject_from_nwb(nwbfile)
        recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
        device_id = f"{subject.id}_{recording_date}"
        session_id = f"{subject.id}_dmfc_rsg"
        session_id += "_test" if "test" in str(fpath) else "_train"

        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        session_description = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
            task=None,  # RSG time-interval reproduction: no matching Task taxonomy member
        )
        device_description = DeviceDescription(
            id=device_id,
            # DMFC_RSG is linear/V-probe, but brainsets' extractor + taxonomy only support
            # UTAH_ARRAY_SPIKES as the "sorted spike-times" extraction MODE -> use it as the
            # mode label (not a probe-type assertion).
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
        )

        self.update_status("Extracting Spikes")
        # dmfc_rsg units have NO `electrodes` column, so brainsets' extract_spikes_from_nwbfile
        # (reads nwbfile.units.electrodes.table) AttributeErrors. Use a minimal local extractor.
        spikes, units = extract_dmfc_spikes(nwbfile)

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

        # Held-out test asset has no public behavior; splits only on train.
        if "test" not in str(fpath):
            self.update_status("Creating Splits")

            set_t = np.asarray(trials.set_time, dtype=float)
            go_t = np.asarray(trials.go_time, dtype=float)
            tp_v = np.asarray(trials.tp, dtype=float)
            ok = ~(np.isnan(set_t) | np.isnan(go_t) | np.isnan(tp_v))

            # [V3] eval window = the Set->Go production epoch per VALID trial. Some trials have
            # NaN set/go/tp (no Set/Go event) -> excluded (Interval rejects NaN starts).
            data.timing_eval_intervals = Interval(start=set_t[ok], end=go_t[ok])

            # ELAPSED-TIME (proper cognitive-timing) TARGET: emit `data.timing` = TIME-TO-GO, a
            # DESCENDING RAMP within each [set,go] production epoch -- tp(ms) at Set down to 0 at Go
            # -- instead of a constant tp. This is the canonical ramping-to-threshold signal of
            # dmfc/preSMA: its value at every timestep is trial-specific (steeper for short tp), so
            # the model must TRACK INTERVAL-SPECIFIC DYNAMICS, not read a static scalar. tp is
            # recoverable as the ramp value at Set. Units: ms (==(go-t)*1000), so at Set == tp(ms),
            # matching the readout's normalization. Outside [set,go] = 0 (not a production epoch);
            # 0-fill (no NaN) keeps POYO's unmasked MSELoss finite. Metric still restricted to
            # Set->Go via eval_interval=timing_eval_intervals.
            # CRITICAL: span the FULL data domain (POYO samples windows across the whole session; a
            # lazy RegularTimeSeries sliced outside its domain returns negative length -> the
            # "__len__() should return >= 0" train-time crash, reproduced+fixed locally).
            rate = 50.0
            t0 = float(data.domain.start[0])
            t1 = float(data.domain.end[0])
            n = int(round((t1 - t0) * rate)) + 1
            times = t0 + np.arange(n) / rate
            ttg = np.zeros((n, 1), dtype=np.float32)   # time-to-go (ms); 0 outside production epochs
            for s, g in zip(set_t[ok], go_t[ok]):
                i0 = max(0, int(round((s - t0) * rate)))
                i1 = min(n - 1, int(round((g - t0) * rate)))
                if i1 >= i0:
                    ttg[i0:i1 + 1, 0] = ((g - times[i0:i1 + 1]) * 1000.0).astype(np.float32)
            data.timing = RegularTimeSeries(
                sampling_rate=rate, tp=ttg, domain="auto", domain_start=t0,
            )

            # RE-ANALYSIS (Set->Go-focused): sample a 1s window ANCHORED AT THE SET CUE per trial
            # ([set, set+W]) instead of the full trial window. The produced-interval signal lives
            # in [set,go]; full-trial sampling diluted it with pre-Set/ITI epochs. W=1.05 >= POYO's
            # 1s window (the sampler DROPS intervals shorter than the window), and tp (forward-filled)
            # holds the current trial's value across [set, set+W]. Metric still restricted to Set->Go
            # via eval_interval=timing_eval_intervals.
            W = 1.05
            end = np.minimum(set_t[ok] + W, t1)            # clamp to data domain end
            prod = Interval(start=set_t[ok], end=end)
            tr_mask = np.asarray(trials.train_mask_nwb)[ok]
            te_mask = np.asarray(trials.test_mask_nwb)[ok]
            train_dom, valid_dom = prod.select_by_mask(tr_mask).split(
                [0.8, 0.2], shuffle=True, random_seed=42
            )
            data.train_domain = train_dom
            data.valid_domain = valid_dom
            data.test_domain = prod.select_by_mask(te_mask)

        io.close()
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)


def extract_dmfc_spikes(nwbfile):
    """Minimal spike/unit extraction for dmfc_rsg (units lack an `electrodes` column, which
    brainsets' extract_spikes_from_nwbfile requires). Mirrors that util but labels units
    `unit_<i>` (no electrode/group). Returns (spikes IrregularTimeSeries, units ArrayDict)."""
    spike_times = nwbfile.units.spike_times_index[:]
    timestamps, unit_index, unit_meta = [], [], []
    for i in range(len(spike_times)):
        st = spike_times[i]
        timestamps.append(st)
        if len(st) > 0:
            unit_index.append([i] * len(st))
        unit_meta.append({
            "id": f"unit_{i}", "unit_number": i, "count": len(st),
            "type": int(RecordingTech.UTAH_ARRAY_SPIKES),
        })
    units = ArrayDict.from_dataframe(pd.DataFrame(unit_meta), unsigned_to_long=True)
    spikes = IrregularTimeSeries(
        timestamps=np.concatenate(timestamps),
        unit_index=np.concatenate(unit_index),
        domain="auto",
    )
    spikes.sort()
    return spikes, units


def extract_trials(nwbfile):
    r"""Trial table -> Interval carrying the RSG timing variables.

    [V2] Column names follow NLB dmfc_rsg conventions; verify against
    nwbfile.trials.colnames before a real run:
      - tp        : produced interval (Set->Go), the DECODING TARGET (scalar/trial)
      - ts/theta  : target/sample interval the monkey is reproducing
      - is_eye    : effector (eye vs hand)        -> condition label
      - is_short  : prior context (short vs long) -> condition label
      - set_time  : time of the Set cue   (eval-window start, [V3])
      - go_time   : time of the Go response (eval-window end,  [V3])
      - split     : NLB train/val mask    ([V4])
    Any present columns carry over onto the Interval automatically via from_dataframe.
    """
    trial_table = nwbfile.trials.to_dataframe().rename(
        columns={"start_time": "start", "stop_time": "end", "split": "split_indicator"}
    )
    trials = Interval.from_dataframe(trial_table)
    split = trial_table.split_indicator.to_numpy()
    trials.train_mask_nwb = split == "train"  # "_" suffix: train_mask is reserved
    trials.test_mask_nwb = split == "val"
    return trials
