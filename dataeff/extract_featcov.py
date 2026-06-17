"""Feature-covariance spectrum of POYO's readout-input representation (output_latents, the input
to self.readout = the features the linear predictor regresses from -- the deep-model analog of the
DATA COVARIANCE in Lin/Kaushik/Dyer/Muthukumar JMLR'24 "implicit spectral regularization"). Used
to test whether UnitDropout (= their random-mask augmentation) reshapes this spectrum (eigenvalue-
proportion reweighting + ridge-like boost) and whether that tracks generalization.

Loads a FINETUNED Lightning ckpt onto the vocab-extended model (same build path as eval_session.py),
hooks model.readout, runs the authors' trainer.test forward over the test set, accumulates the
covariance INCREMENTALLY (dim x dim, no per-token storage), and dumps eigenvalues + summary metrics.

Run inside the poyo repo:
  POYO_SESSION=<sess> python extract_featcov.py --config-name finetune_perich_xsub.yaml \
     data_root=<dir> finetune.pretrained_ckpt=<base.ckpt> ++ft_ckpt=<finetuned_best.ckpt> \
     ++featcov_out=<out.json> gpus=1
"""
import json
import hydra
import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig

from torch_brain.models.poyo import POYO
from train import DataModule, TrainWrapper
from src.callbacks import DecodingStitchEvaluator


@hydra.main(version_base="1.3", config_path="./configs")
def main(cfg: DictConfig):
    dm = DataModule(cfg=cfg)
    readout_spec = dm.readout_spec
    # build base arch + vocab, then extend vocab to this session's units (matches finetune shapes)
    model = POYO.load_pretrained(cfg.finetune.pretrained_ckpt, readout_spec=readout_spec,
                                 skip_readout=cfg.finetune.get("skip_readout", False))
    dm.link_model(model)
    wrapper = TrainWrapper(cfg=cfg, model=model, modality_spec=readout_spec)

    # overlay the FINETUNED weights (t-adapted embeddings + readout)
    ft_ckpt = cfg.get("ft_ckpt", None)
    if ft_ckpt:
        sd = torch.load(ft_ckpt, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = wrapper.load_state_dict(sd, strict=False)
        print(f"[featcov] loaded {ft_ckpt}; missing={len(missing)} unexpected={len(unexpected)}")

    # incremental covariance accumulators over output_latents (input to self.readout)
    state = {"n": 0, "sum": None, "outer": None, "dim": None}

    def hook(module, inp):
        x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double().cpu()
        if state["sum"] is None:
            d = x.shape[1]; state["dim"] = d
            state["sum"] = torch.zeros(d, dtype=torch.float64)
            state["outer"] = torch.zeros(d, d, dtype=torch.float64)
        state["n"] += x.shape[0]
        state["sum"] += x.sum(0)
        state["outer"] += x.t() @ x

    h = wrapper.model.readout.register_forward_pre_hook(hook)
    stitch = DecodingStitchEvaluator(session_ids=dm.get_session_ids(), modality_spec=readout_spec)
    trainer = L.Trainer(logger=False, default_root_dir=cfg.log_dir, callbacks=[stitch],
                        precision=cfg.precision, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                        devices=cfg.gpus, num_nodes=cfg.nodes)
    trainer.test(wrapper, dm)
    h.remove()

    n = state["n"]
    mean = state["sum"] / n
    cov = state["outer"] / n - torch.outer(mean, mean)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0).numpy()[::-1]  # descending
    eig = eig[eig > 0]
    p = eig / eig.sum()
    spectral_entropy = float(-(p * np.log(p)).sum())
    eff_rank = float(np.exp(spectral_entropy))                 # participation ratio (entropy form)
    pr = float((eig.sum() ** 2) / (eig ** 2).sum())            # participation ratio (Tr^2/||.||^2)
    out = {
        "n_tokens": int(n), "dim": int(state["dim"]),
        "trace": float(eig.sum()),                              # overall scale (their "ridge boost")
        "top1_frac": float(p[0]), "top5_frac": float(p[:5].sum()),
        "spectral_entropy": spectral_entropy,
        "eff_rank_entropy": eff_rank, "participation_ratio": pr,
        "cond_number": float(eig[0] / eig[-1]),
        "eigvals_top50": [float(v) for v in eig[:50]],
    }
    dest = cfg.get("featcov_out", "featcov.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print("[featcov]", json.dumps({k: v for k, v in out.items() if k != "eigvals_top50"}))


if __name__ == "__main__":
    main()
