"""Frozen-base eval: load the pretrained POYO-MP checkpoint and run trainer.test on ONE session
WITHOUT any fit (no optimizer, no OneCycleLR -> avoids the total_steps>0 requirement). Reuses the
repo's DataModule / TrainWrapper / DecodingStitchEvaluator so windowing, stitching, eval-mask and
the R2 metric are identical to the authors' test path.

Drop into the poyo repo as eval_session.py. Run via run_chronic.sh.
"""
import hydra
import lightning as L
import torch
from omegaconf import DictConfig

from torch_brain.models.poyo import POYO
from train import DataModule, TrainWrapper
from src.callbacks import DecodingStitchEvaluator


@hydra.main(version_base="1.3", config_path="./configs")
def main(cfg: DictConfig):
    data_module = DataModule(cfg=cfg)
    readout_spec = data_module.readout_spec

    # load_pretrained restores trained vocab + weights; link_model's extend_vocab(exist_ok=True)
    # keeps the trained rows for an in-vocab session -> we evaluate the model AS TRAINED.
    model = POYO.load_pretrained(
        cfg.finetune.pretrained_ckpt, readout_spec=readout_spec, skip_readout=False
    )
    data_module.link_model(model)

    wrapper = TrainWrapper(cfg=cfg, model=model, modality_spec=readout_spec)
    stitch_evaluator = DecodingStitchEvaluator(
        session_ids=data_module.get_session_ids(), modality_spec=readout_spec
    )

    trainer = L.Trainer(
        logger=False,
        default_root_dir=cfg.log_dir,
        callbacks=[stitch_evaluator],
        precision=cfg.precision,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.gpus,
        num_nodes=cfg.nodes,
    )
    trainer.test(wrapper, data_module)  # ckpt_path=None -> uses the loaded (frozen) weights


if __name__ == "__main__":
    main()
