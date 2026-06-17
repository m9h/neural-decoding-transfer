"""A/B validation of the embedding-weight-decay fix.

Overfits POYO on 2 perich sessions for a few hundred steps and tracks the unit
embedding norm + batch R2. Run twice:
  python fix_validate.py 0.0     # FIX  : no weight decay on embeddings
  python fix_validate.py 1e-4    # BUG  : weight decay on embeddings (collapses)
"""
import sys
import torch, hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch_brain.registry import MODALITY_REGISTRY
from torch_brain.optim import SparseLamb
from torch_brain.data import collate
from torch_brain.data.sampler import RandomFixedWindowSampler
from torch_brain.transforms import Compose
import poyo_datasets.poyo_mp as pm
from poyo_datasets.poyo_mp import PoyoMPDataset

torch.set_float32_matmul_precision("medium")
EMB_WD = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
print(f"=== embedding weight_decay = {EMB_WD}  ({'FIX' if EMB_WD==0 else 'BUG'}) ===", flush=True)

pm.TRAIN_RECORDING_IDS[:] = pm.TRAIN_RECORDING_IDS[:2]
ds = PoyoMPDataset("/data/datasets/brainsets/processed")
spec = MODALITY_REGISTRY["cursor_velocity_2d"]
model = hydra.utils.instantiate(OmegaConf.load("configs/model/poyo_11.8M.yaml"), readout_spec=spec).cuda()
model.unit_emb.initialize_vocab(ds.get_unit_ids())
model.session_emb.initialize_vocab(ds.recording_ids)
ds.transform = Compose([model.tokenize])

emb = list(model.unit_emb.parameters()) + list(model.session_emb.parameters())
rest = [p for n, p in model.named_parameters() if "unit_emb" not in n and "session_emb" not in n]
opt = SparseLamb([{"params": emb, "sparse": True, "weight_decay": EMB_WD},
                  {"params": rest}], lr=0.008, weight_decay=1e-4)

sampler = RandomFixedWindowSampler(sampling_intervals=ds.get_sampling_intervals("train"),
                                   window_length=model.sequence_length,
                                   generator=torch.Generator().manual_seed(0))
loader = DataLoader(ds, sampler=sampler, collate_fn=collate, batch_size=64,
                    num_workers=4, drop_last=True, persistent_workers=True)
def cyc(dl):
    while True:
        for b in dl: yield b
it = cyc(loader)
mv = lambda x: x.cuda(non_blocking=True) if torch.is_tensor(x) else x

print(f"{'step':>5} {'loss':>8} {'unit_emb|w|':>12} {'unit_emb_std':>13} {'batch_R2':>9}", flush=True)
for step in range(601):
    b = next(it)
    mi = {k: mv(v) for k, v in b["model_inputs"].items()}
    tgt, w = mv(b["target_values"]), mv(b["target_weights"])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**mi)
    m = mi["output_mask"]
    loss = spec.loss_fn(out[m], tgt[m], w[m])
    loss.backward(); opt.step(); opt.zero_grad()
    if step % 100 == 0:
        with torch.no_grad():
            une = model.unit_emb.weight.float()
            p, t = out[m].float(), tgt[m].float()
            ss_res = ((p - t) ** 2).sum(0); ss_tot = ((t - t.mean(0)) ** 2).sum(0).clamp_min(1e-6)
            r2 = (1 - ss_res / ss_tot).mean().item()
            print(f"{step:>5} {loss.item():>8.4f} {une.abs().mean():>12.5f} {une.std():>13.5f} {r2:>9.3f}", flush=True)
