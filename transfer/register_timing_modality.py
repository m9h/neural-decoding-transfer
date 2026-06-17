"""Register the NEW readout modality for the Merchant-aligned dmfc_rsg timing transfer.

Importing this module registers `interval_timing_1d` into torch_brain's MODALITY_REGISTRY
(mirrors the built-in cursor_velocity_2d registration). The dmfc_rsg POYO dataset's
READOUT_CONFIG references this readout_id; the DataModule then looks it up in the registry
to build the readout head. Drop this file into the poyo repo (e.g. src/datasets/) and import
it from poyo_dmfc.py so the registration runs before the DataModule resolves the readout.

Verified against /tmp/tb_src/torch_brain/registry.py: register_modality(name, dim, type,
timestamp_key, value_key, loss_fn) and the existing cursor_velocity_2d/arm_velocity_2d defs.
"""
import torch_brain
from torch_brain.registry import register_modality, DataType, MODALITY_REGISTRY

# scalar produced-interval regression (one tp per trial, MSE), distinct from the 2D
# continuous cursor_velocity_2d. dim=1, CONTINUOUS, MSE loss. The dataset's READOUT_CONFIG
# overrides timestamp_key/value_key to point at the dmfc Data object's timing channel.
if "interval_timing_1d" not in MODALITY_REGISTRY:
    register_modality(
        "interval_timing_1d",
        dim=1,
        type=DataType.CONTINUOUS,
        timestamp_key="timing.timestamps",
        value_key="timing.tp",
        loss_fn=torch_brain.nn.loss.MSELoss(),
    )
