"""Leg-visibility classifier from joint states, trained on the SAM3 leg labels (leg_label.py), then applied to every frame of
the indexed episodes; leg-free training windows are cut from the result.
Features (prep_index.py): lower body 15 + waist 3 + torso pose 6 + head-camera pose 6 (30-D). The one dataset with the older
36-D schema has its own small model if it has both classes, else its SAM3 rate decides.
Validation: 20 % of the labelled episodes held out. Prediction is made conservative: a frame counts as leg-free only if
p(leg) < THR for it and for the frames within +-PAD (0.5 s).
Windows: 49 frames at 15 fps (every 2nd raw frame) with every frame leg-free; non-overlapping, at most MAX_WIN per episode.
usage: python leg_clf.py   (retarget venv: torch, numpy)  -> leg_free.npz, windows.json, leg_clf_report.json"""
import json, os, random
import numpy as np, torch

ROOT = os.environ.get("G1I_ROOT", "/data/datasets/unitree_g1_inspire")  # dataset work dir
THR, PAD, WIN, MAX_WIN = float(os.environ.get("THR", 0.3)), 15, 49, int(os.environ.get("MAX_WIN", 4))
index = json.load(open(f"{ROOT}/index.json")); labels = json.load(open(f"{ROOT}/leg_labels.json"))
states = {}
for ds in sorted({e["dataset"] for e in index}):
    z = np.load(f"{ROOT}/states/{ds}.npz"); states[ds] = {k: z[k] for k in z.files}
torch.manual_seed(0); random.seed(0)


def feats(ds, ep, fr):
    q = states[ds].get(f"ep{ep:05d}")
    return None if q is None or fr >= len(q) else q[fr]


groups = {}
for l in labels:
    x = feats(l["dataset"], l["episode"], l["frame"])
    if x is not None:
        groups.setdefault(len(x), []).append((x, float(l["leg"]), (l["dataset"], l["episode"])))
report, models = {}, {}
for dim, rows in groups.items():
    X = np.stack([r[0] for r in rows]); y = np.array([r[1] for r in rows], np.float32); grp = [r[2] for r in rows]
    if y.min() == y.max():
        models[dim] = ("const", float(y[0])); report[dim] = dict(n=len(y), constant=float(y[0])); continue
    eps = sorted(set(grp)); random.shuffle(eps); held = set(eps[: max(1, len(eps) // 5)])
    te = np.array([g in held for g in grp]); mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); yt = torch.tensor(y)

    def fit(mask):
        net = torch.nn.Sequential(torch.nn.Linear(dim, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        for _ in range(600):
            opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[mask]).squeeze(1), yt[mask]); loss.backward(); opt.step()
        return net

    net = fit(torch.tensor(~te))
    with torch.no_grad(): p = torch.sigmoid(net(Xt[torch.tensor(te)]).squeeze(1)).numpy()
    yv = y[te]; pred = p >= THR
    report[dim] = dict(n=len(y), held_out=int(te.sum()), leg_rate=float(y.mean()), acc=float((pred == (yv > 0.5)).mean()),
                       leg_recall=float(pred[yv > 0.5].mean()) if (yv > 0.5).any() else None,
                       legfree_precision=float((yv[~pred] < 0.5).mean()) if (~pred).any() else None)
    models[dim] = ("mlp", fit(torch.ones(len(y), dtype=torch.bool)), mu, sd)
print(json.dumps(report, indent=1))

free, windows = {}, []
dups = json.load(open(f"{ROOT}/duplicates.json")) if os.path.exists(f"{ROOT}/duplicates.json") else {}  # identical re-uploads
for e in index:
    ds, ep = e["dataset"], e["episode"]; q = states[ds].get(f"ep{ep:05d}")
    if q is None or ds in dups: continue
    m = models.get(q.shape[1])
    if m is None: continue
    if m[0] == "const": p = np.full(len(q), m[1])
    else:
        with torch.no_grad(): p = torch.sigmoid(m[1](torch.tensor((q - m[2]) / m[3], dtype=torch.float32)).squeeze(1)).numpy()
    leg = p >= THR
    legd = np.convolve(leg.astype(float), np.ones(2 * PAD + 1), "same") > 0  # +-0.5 s margin
    free[f"{ds}/ep{ep:05d}"] = ~legd
    ok = ~legd[::2]  # 15 fps
    s, n = 1, 0
    while s + WIN <= len(ok) and n < MAX_WIN:
        if ok[s - 1:s + WIN].all():
            windows.append(dict(dataset=ds, episode=ep, start15=s, file=e["file"], t0=e["t0"])); n += 1; s += WIN + 1
        else:
            s += 1
np.savez_compressed(f"{ROOT}/leg_free.npz", **free)
json.dump(windows, open(f"{ROOT}/windows.json", "w"))
tot = sum(len(v) for v in free.values()); fr = sum(v.sum() for v in free.values())
print(f"leg-free frames {fr / tot * 100:.0f}% of {tot} ({fr / 30 / 3600:.1f} h); {len(windows)} windows from {len({(w['dataset'], w['episode']) for w in windows})} episodes")
json.dump(dict(report={str(k): v for k, v in report.items()}, legfree_frac=fr / tot, windows=len(windows)), open(f"{ROOT}/leg_clf_report.json", "w"), indent=1)
