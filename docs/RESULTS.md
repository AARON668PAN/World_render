# Results (as of 11-01)

Best model overall: **v6a at step 3000**. v6b is best on the grey patches under the human thumbs.

## Versions

| Version | Run name | Change | Training |
|---|---|---|---|
| v3 | `g1i_obj2_2g` | source at the same time as the target with p .5 (robot holed), object holes p .15, handled object as one flat colour | base Wan + new LoRA, 5000 steps |
| v4 | `g1i_obj3_2g` | v3 + object drawn with its depth (C1, `--obj_enc c1`) | base Wan + new LoRA, 5000 steps; not continued |
| v5 | `g1i_obj4_2g` | v3 + 3 scene (info) frames as extra tokens + holes glued to the object (`--obj_stick 0.4`) | base Wan + new LoRA, 5000 steps, about 9 h |
| v6a | `g1i_v6a` | v5 + 1 px robot holes, no random hand stamps, ghost holes p .5, glued holes p .2, hole channel = share of hole pixels per 8x8 cell | from v5 step 5000, 3000 steps |
| v6b | `g1i_v6b` | v6a + control B (whole-frame VDA depth, turbo x shading) in the base model's first-frame slot (dropped p .2), no object colour p .3 | from v5 step 5000, 3000 steps |

All runs: Isambard-AI, 2 GH200, LoRA rank 128, global batch 16, lr 1e-4 (LoRA) / 2e-5 (input layer, ref_conv). Only
the v5, v6a and v6b job scripts are in this repo (`cluster/train_*.sbatch`, path-adapted copies; see
`CHANGES_FROM_WORKING_COPY.md`).

## Tissue clip (human video)

CHORD's `tissue_box` sample, 147 frames at 15 fps = 3 chunks of 49 frames, letterboxed 16:9 into 448x336.
Measured by `tools/eval/compare_runs.py`; each run is scored against the input npz it was given.

- **grey under the thumbs**: the human thumbs leave holes on the box that the robot does not cover. Share of those
  pixels that come out grey (channel spread < 30), per chunk. The box itself is greyish: its visible parts score
  .35-.44, so values in that range mean "normal box surface".
- **box / scene PSNR**: generated frame vs the human video on the visible box pixels / everywhere outside the holes.
  Only a reference: v5 and v6 use different holes (5 px vs 1 px margin), so the pixel sets differ.

| Version, step | grey under thumbs (chunk 1 / 2 / 3) | box PSNR | scene PSNR | source of the numbers |
|---|---|---|---|---|
| v3, 5000 | .41 / .60 / .70 | 17.5 | 24.3 | 10-31, `tools/figures/page_outputs_v5.py` |
| v4, 5000 | .72 / .59 / .53 | 16.3 | 23.5 | 10-31, `page_outputs_v5.py` |
| v5, 3000 | .47 / .38 / .44 | 19.0 | 26.0 | 10-31, `page_outputs_v5.py` |
| v5, 5000 | .46 / .38 / .42 | 20.0 | 26.7 | re-run 11-01 |
| **v6a, 3000** | .44 / .40 / .37 | **21.0** | **27.4** | re-run 11-01 |
| v6b, 3000 | **.40 / .39 / .35** | 20.4 | 26.4 | re-run 11-01 |

The 11-01 re-run (CPU, from the stored videos) printed 19.98 / 26.72, 21.01 / 27.38 and 20.44 / 26.42.

## Held-out WBT segments

Three held-out segments (`data_pipeline/meta/val_segments.json`), one 49-frame chunk each, source at the same time as
the target (`--shift 0`), 3 info frames, no reference frame for the chunk. PSNR against the real frames; "robot" =
the robot mask dilated by 5 px. Read from `/data/models/roborender_g1src/g1i_*/out_*_val/results_*.json`.

| Segment | v5 @5000: PSNR / robot / LPIPS | v6a @3000 | v6b @3000 |
|---|---|---|---|
| Pick_Up_Drinks_ep00036_s0001 | 27.71 / 18.19 / .0401 | 28.62 / 19.45 / .0349 | 28.13 / 19.56 / .0384 |
| Put_Vegetables_Into_Basket_ep00009_s0159 | 24.24 / 19.41 / .0448 | 25.70 / 21.18 / .0356 | 25.15 / 20.72 / .0382 |
| Tidy_Living_Room_Slippers_ep00003_s0095 | 28.48 / 19.92 / .0395 | 29.69 / 21.17 / .0326 | 29.08 / 20.65 / .0354 |
| LPIPS range | .040-.045 | .033-.036 | .035-.038 |

Read these with care:

- 3 segments and one chunk each. `val_segments.json` lists 36 segments, 18 of them leg-free; the other 15 were not run.
- v5 was run with its training holes (5 px, `max` pooling), v6 with 1 px holes and `mean` pooling. With 1 px holes the
  source shows real pixels 2-5 px from the robot, which fall inside the "robot" region of the metric. So part of v6's
  robot-PSNR gain comes from the evaluation setting, not the model.

## Commands

Paths of the original runs: `E=/data/repos/chord/local/ego/tissue_box`,
`V=/data/repos/chord/reconstruction/assets/tissue_box.mp4`, `M=/data/models/roborender_g1src`. `$NEWTON_PY` is the
Newton env (any env with numpy, cv2 and imageio works for `prep_human_g1i.py`), `$SAM3_PY` the SAM3/VDA venv,
`$RR_PY` the renderer venv. Run from the repo root with `ROBORENDER_ROOT` set to the RoboRender clone.

Tissue-clip inputs (after `sim_bridge/run_tissue_chain.sh`):

```bash
# v6 input: thin holes, then control B added to the same npz
$NEWTON_PY renderer/prep_human_g1i.py $V $E/align/human_table_masks.npz $E/depth_compare/v7_letterbox_realcol $E/human_g1i_v6_realcol.npz --hole_px 1
$SAM3_PY renderer/prep_human_depth.py $V $E/align/human_table_masks.npz $E/align/box_masks.npz $E/depth_compare/v7_letterbox $E/human_g1i_v6_realcol.npz
# v5 input: default 5 px holes
$NEWTON_PY renderer/prep_human_g1i.py $V $E/align/human_table_masks.npz $E/depth_compare/v7_letterbox_realcol $E/human_g1i_letterbox_obj_realcol.npz
```

Inference (one GPU each):

```bash
$RR_PY renderer/infer_g1i.py $M/g1i_obj4_2g/ckpt_05000.pt $M/g1i_obj4_2g/out_05000_info3 --human $E/human_g1i_letterbox_obj_realcol.npz --info 3
$RR_PY renderer/infer_g1i.py $M/g1i_v6a/ckpt_03000.pt $M/g1i_v6a/out_03000 --human $E/human_g1i_v6_realcol.npz --info 3 --mask_pool mean
$RR_PY renderer/infer_g1i.py $M/g1i_v6b/ckpt_03000.pt $M/g1i_v6b/out_03000 --human $E/human_g1i_v6_realcol.npz --info 3 --depth --mask_pool mean
for v in Pick_Up_Drinks_ep00036_s0001 Put_Vegetables_Into_Basket_ep00009_s0159 Tidy_Living_Room_Slippers_ep00003_s0095; do
  $RR_PY renderer/infer_g1i.py $M/g1i_obj4_2g/ckpt_05000.pt $M/g1i_obj4_2g/out_05000_val --val $v --shift 0 --info 3 --max_chunks 1
  $RR_PY renderer/infer_g1i.py $M/g1i_v6a/ckpt_03000.pt $M/g1i_v6a/out_03000_val --val $v --shift 0 --info 3 --mask_pool mean --hole_px 1 --max_chunks 1
  $RR_PY renderer/infer_g1i.py $M/g1i_v6b/ckpt_03000.pt $M/g1i_v6b/out_03000_val --val $v --shift 0 --info 3 --depth --mask_pool mean --hole_px 1 --max_chunks 1
done
```

These are the command lines of the 10-31 runs (from the session log). `--val` reads the dataset dir through `--root`
(default `/data/datasets/unitree_g1_inspire`).

Comparison:

```bash
E=$E $RR_PY tools/eval/compare_runs.py /tmp/v5_v6 \
  "v5@5000=$M/g1i_obj4_2g/out_05000_info3/human_human_g1i_letterbox_obj_realcol.mp4=$E/human_g1i_letterbox_obj_realcol.npz" \
  "v6a@3000=$M/g1i_v6a/out_03000/human_human_g1i_v6_realcol.mp4=$E/human_g1i_v6_realcol.npz" \
  "v6b@3000=$M/g1i_v6b/out_03000/human_human_g1i_v6_realcol.mp4=$E/human_g1i_v6_realcol.npz"
```
