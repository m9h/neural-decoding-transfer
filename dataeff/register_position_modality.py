"""Register the NEW position_2d readout for the rodent (Vollan/Moser 2025) spatial transfer.

Importing this module registers `position_2d` into torch_brain's MODALITY_REGISTRY (mirrors the
built-in cursor_velocity_2d and our interval_timing_1d). PoyoVollanSessionDataset's READOUT_CONFIG
references this readout_id; the DataModule then looks it up to build the readout head. The dataset
hook constructs the `position` channel (2D allocentric arena position) from the brainset's
samples.x / samples.y. Drop into the poyo repo as src/datasets/register_position_modality.py and
import it from poyo_vollan_session.py so registration runs before the DataModule resolves the readout.

Mirrors transfer/register_timing_modality.py; verified against the same register_modality signature.
"""
import torch_brain
from torch_brain.registry import register_modality, DataType, MODALITY_REGISTRY

# 2D allocentric position regression (x, y), CONTINUOUS, MSE -- the entorhinal/hippocampal
# spatial-coding analog of cursor_velocity_2d. dim=2. The dataset's READOUT_CONFIG points
# timestamp_key/value_key at the constructed `position` channel.
if "position_2d" not in MODALITY_REGISTRY:
    register_modality(
        "position_2d",
        dim=2,
        type=DataType.CONTINUOUS,
        timestamp_key="position.timestamps",
        value_key="position.pos",
        loss_fn=torch_brain.nn.loss.MSELoss(),
    )
