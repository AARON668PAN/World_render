"""Tissue clip: human video next to several runs (each scored against its own input npz: grey share under the human-hand
holes the robot does not cover, per 49-frame chunk; PSNR of the visible box / scene against the source).
usage: python compare_runs.py OUT_PREFIX name=mp4=npz [name=mp4=npz ...]
env: E = tissue-clip work dir (run_tissue_chain.sh E), SIM = its letterboxed sim render (default $E/depth_compare/v7_letterbox),
     V = the human video (default the CHORD tissue_box.mp4). The defaults are the paths of the original run."""
import os, sys
import cv2, imageio.v2 as imageio, numpy as np
E = os.environ.get("E", "/data/repos/chord/local/ego/tissue_box"); out = sys.argv[1]
SIM = os.environ.get("SIM", f"{E}/depth_compare/v7_letterbox"); V = os.environ.get("V", "/data/repos/chord/reconstruction/assets/tissue_box.mp4")
runs = [a.split("=") for a in sys.argv[2:]]
lab_all = np.load(f"{SIM}/sim_depth.npz")["label"][::2]
hum = [f for i, f in enumerate(imageio.get_reader(V)) if i % 2 == 0]
ps = lambda a, b: 10 * np.log10(255 ** 2 / max(1e-6, ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()))
gens = {}
for name, mp4, npz in runs:
    g = np.stack([f[:, 896:1344] for f in imageio.get_reader(mp4)]); T = len(g)
    z = np.load(npz); src, hole = z["src"][:T], z["hole"][:T]; lab = lab_all[:T]; box, rob = lab == 2, np.isin(lab, (3, 4))
    thumb = hole & ~np.stack([cv2.dilate(r.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 for r in rob]) & box
    ch = g.max(-1).astype(int) - g.min(-1); vis = box & ~hole
    grey = [np.mean(ch[c * 49:c * 49 + 49][thumb[c * 49:c * 49 + 49]] < 30) if thumb[c * 49:c * 49 + 49].any() else float("nan") for c in range(T // 49)]
    print(f"{name:22s} grey {' '.join(f'{x:.2f}' for x in grey)} | box {np.mean([ps(g[t][vis[t]], src[t][vis[t]]) for t in range(T) if vis[t].sum() > 300]):.2f} | scene {np.mean([ps(g[t][~hole[t]], src[t][~hole[t]]) for t in range(T)]):.2f}")
    gens[name] = g
T = min(len(g) for g in gens.values())
def tag(im, s):
    im = im.copy(); cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(im, s, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA); return im
rows = [np.hstack([tag(cv2.resize(hum[k], (448, 252), interpolation=cv2.INTER_AREA), f"human f{k}")] + [tag(g[k][42:294], n) for n, g in gens.items()]) for k in (20, 60, 100, 140)]
cv2.imwrite(f"{out}.jpg", np.vstack(rows)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 84])
w = imageio.get_writer(f"{out}.mp4", fps=15, codec="libx264", quality=7, macro_block_size=4)
for k in range(T):
    w.append_data(np.hstack([tag(cv2.resize(hum[k], (448, 252), interpolation=cv2.INTER_AREA), "human")] + [tag(g[k][42:294], n) for n, g in gens.items()]))
w.close(); print("wrote", out)
