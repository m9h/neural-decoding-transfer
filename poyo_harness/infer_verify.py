"""Does the on-disk epoch=49 checkpoint (reported R2=0.37) actually decode?

Loads the checkpoint weights into POYO and measures R2 directly on valid windows.
- R2 ~0.3+  -> on-disk weights are functional (the 'zero layers' reading is wrong)
- R2 ~0     -> checkpoint is corrupt on save (bf16-mixed save bug); R2=0.37 was in-memory only
"""
import sys
import torch, hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_brain.registry import MODALITY_REGISTRY
from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.transforms import Compose
import poyo_datasets.poyo_mp as pm
from poyo_datasets.poyo_mp import PoyoMPDataset

CKPT = sys.argv[1]
pm.TRAIN_RECORDING_IDS[:] = pm.TRAIN_RECORDING_IDS[:5]  # few sessions for fast load
spec = MODALITY_REGISTRY["cursor_velocity_2d"]
model = hydra.utils.instantiate(OmegaConf.load("configs/model/poyo_11.8M.yaml"), readout_spec=spec)

ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"load: {len(missing)} missing, {len(unexpected)} unexpected keys", flush=True)

# Confirm we loaded the on-disk weights (match what we saw: some alive, some 0)
for nm, p in [("dec_atn.to_q", model.dec_atn.to_q.weight),
              ("enc_atn.to_q", model.enc_atn.to_q.weight),
              ("unit_emb", model.unit_emb.weight)]:
    print(f"  loaded {nm}: |w|mean={p.abs().mean().item():.6f}", flush=True)

model = model.cuda().eval()
ds = PoyoMPDataset("/data/datasets/brainsets/processed")
ds.transform = Compose([model.tokenize])
sampler = RandomFixedWindowSampler(sampling_intervals=ds.get_sampling_intervals("valid"),
                                   window_length=model.sequence_length,
                                   generator=torch.Generator().manual_seed(0))
loader = DataLoader(ds, sampler=sampler, collate_fn=collate, batch_size=64, num_workers=4, drop_last=True)

P, T = [], []
with torch.no_grad():
    for i, b in enumerate(loader):
        if i >= 30:
            break
        mi = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b["model_inputs"].items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**mi)
        m = mi["output_mask"]
        mc = m.cpu()
        P.append(out[m].float().cpu()); T.append(b["target_values"][mc].float())

P, T = torch.cat(P), torch.cat(T)
ss_res = ((P - T) ** 2).sum(0); ss_tot = ((T - T.mean(0)) ** 2).sum(0).clamp_min(1e-8)
r2 = (1 - ss_res / ss_tot).mean().item()
print(f"\n=== VERIFY: per-window R2 on {P.shape[0]} valid samples = {r2:.4f} ===", flush=True)
print("on-disk weights ARE functional" if r2 > 0.15 else "on-disk weights are DEAD -> checkpoint corrupt on save", flush=True)
