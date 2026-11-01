"""Tissue clip v3 / v4 / v5 outputs: prints grey share under the thumb holes per chunk, box PSNR, scene PSNR; writes page
images img/inf_output.jpg (v5 step 5000, frame 60), img/inf_relay.jpg (frame 48), img/compare.jpg and
/data/models/roborender_g1src/compare_v3v4v5/compare_5000.mp4. The page images show v5, not v6; for v6 use
tools/eval/compare_runs.py. Ran inline on 10-31; recovered from the session transcript.
usage: python page_outputs_v5.py PAGE_DIR   (writes PAGE_DIR/img/...)"""
import sys
import numpy as np, cv2, imageio.v2 as imageio
M = "/data/models/roborender_g1src"; E = "/data/repos/chord/local/ego/tissue_box"; P = sys.argv[1]
runs = {"v3 (5000)": f"{M}/g1i_obj2_2g/out_05000_realcol/human_human_g1i_letterbox_obj_realcol.mp4",
        "v4 depth (5000)": f"{M}/g1i_obj3_2g/out_05000_c1realcol/human_human_g1i_letterbox_c1_realcol.mp4",
        "v5 info (3000)": f"{M}/g1i_obj4_2g/out_03000_info3/human_human_g1i_letterbox_obj_realcol.mp4",
        "v5 info (5000)": f"{M}/g1i_obj4_2g/out_05000_info3/human_human_g1i_letterbox_obj_realcol.mp4"}
g = {n: np.stack([f[:, 896:1344] for f in imageio.get_reader(p)]) for n, p in runs.items()}
T = min(len(v) for v in g.values())
z = np.load(f"{E}/human_g1i_letterbox_obj_realcol.npz"); src, hole = z["src"][:T], z["hole"][:T]
lab = np.load(f"{E}/depth_compare/v7_letterbox/sim_depth.npz")["label"][::2][:T]; box, rob = lab == 2, np.isin(lab, (3, 4))
thumb = hole & ~np.stack([cv2.dilate(r.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0 for r in rob]) & box
ps = lambda a, b: 10 * np.log10(255 ** 2 / max(1e-6, ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()))
for n, v in g.items():
    ch = v[:T].max(-1).astype(int) - v[:T].min(-1); vis = box & ~hole
    grey = [np.mean(ch[c * 49:c * 49 + 49][thumb[c * 49:c * 49 + 49]] < 30) for c in range(T // 49)]
    print(f"{n:16s} grey {' '.join(f'{x:.2f}' for x in grey)} | box {np.mean([ps(v[t][vis[t]], src[t][vis[t]]) for t in range(T) if vis[t].sum() > 300]):.2f} | scene {np.mean([ps(v[t][~hole[t]], src[t][~hole[t]]) for t in range(T)]):.2f}")
v5 = g["v5 info (5000)"]
cv2.imwrite(f"{P}/img/inf_output.jpg", v5[60][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 86])
cv2.imwrite(f"{P}/img/inf_relay.jpg", v5[48][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 86])
hum = [f for i, f in enumerate(imageio.get_reader("/data/repos/chord/reconstruction/assets/tissue_box.mp4")) if i % 2 == 0]
def tag(im, s):
    im = im.copy(); cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(im, s, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA); return im
rows = []
for k in (20, 60, 100, 140):
    rows.append(np.hstack([tag(cv2.resize(hum[k], (448, 252), interpolation=cv2.INTER_AREA), f"human f{k}")] + [tag(g[n][k][42:294], n) for n in ["v3 (5000)", "v4 depth (5000)", "v5 info (5000)"]]))
cv2.imwrite(f"{P}/img/compare.jpg", np.vstack(rows)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 84])
w = imageio.get_writer(f"{M}/compare_v3v4v5/compare_5000.mp4", fps=15, codec="libx264", quality=7, macro_block_size=4)
for k in range(T):
    w.append_data(np.hstack([tag(cv2.resize(hum[k], (448, 252), interpolation=cv2.INTER_AREA), "human")] + [tag(g[n][k][42:294], n) for n in ["v3 (5000)", "v4 depth (5000)", "v5 info (5000)"]]))
w.close()
