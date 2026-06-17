# Galaxy-brain / POYO invasive-neurophys model stack for the DGX Spark (GB10, sm_121, aarch64).
#
# Base: latest NGC PyTorch (torch built for sm_121; PyPI cu wheels stop at sm_120).
FROM nvcr.io/nvidia/pytorch:26.05-py3

# Protect the NGC sm_121 torch build from pip's resolver for every pip call.
COPY constraints.txt /opt/constraints.txt
ENV PIP_CONSTRAINT=/opt/constraints.txt \
    PIP_NO_CACHE_DIR=1

# neuro-galaxy stack:
#   torch_brain  -> POYO / POYO+ models (torch_brain.models.poyo)
#   brainsets    -> data ingestion/processing pipeline + CLI
#   temporaldata -> underlying temporal data structures (pulled transitively, pinned explicit)
# plus the training/eval stack the POYO examples use.
RUN pip install \
        temporaldata \
        torch_brain \
        brainsets \
        lightning \
        hydra-core \
        hydra-submitit-launcher \
        wandb \
        torchmetrics \
        einops \
        omegaconf \
        scikit-learn

# brainsets writes a config telling it where raw/processed data live.
# Point it at the NAS mount (bind-mounted at runtime, see run.sh).
ENV BRAINSETS_RAW_DIR=/data/datasets/brainsets/raw \
    BRAINSETS_PROCESSED_DIR=/data/datasets/brainsets/processed

# Fail the build if the stack can't import or torch lost its CUDA build.
RUN python -c "import torch, torch_brain, brainsets, temporaldata, lightning; \
import torch_brain.models.poyo as _p; \
print('torch', torch.__version__); \
print('torch_brain', torch_brain.__version__); \
print('brainsets', brainsets.__version__); \
print('temporaldata', temporaldata.__version__); \
assert torch.version.cuda is not None, 'torch lost its CUDA build!'"

WORKDIR /workspace
