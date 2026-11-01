# Training data: Unitree G1 + Inspire segments

Turns Unitree's official `G1_WBT_Inspire_*` datasets (LeRobot v3, Hugging Face) into what `renderer/cache_g1i.py`
reads: 448x336 15 fps segments where the robot's legs are not visible, a SAM3 mask of the robot, the ONE handled object
(boxed by Claude, tracked by SAM3), and VDA depth.

All scripts work in one dataset dir, `G1I_ROOT` (default `/data/datasets/unitree_g1_inspire`, the original location).
They write their outputs there too. The commands below run from this folder (`data_pipeline/`); `$SAM3_PY` is the SAM3/VDA venv
(`env/sam3_vda_reqs.txt`), `$RT_PY` the retarget venv (`sim_bridge/env/retarget_reqs.txt`; torch, pyarrow).

## Before you start

```bash
export G1I_ROOT=/data/datasets/unitree_g1_inspire
mkdir -p $G1I_ROOT/obj_queue
cp meta/*.json meta/leg_free.npz $G1I_ROOT/          # pinned lists, meta/README.md
cp -r labeling/{README.md,DISPATCH.md,check_results.py,fix_clips.json,batches,results} $G1I_ROOT/obj_queue/
```

The labelling rules, the Claude boxes and `check_results.py` expect to sit in `$G1I_ROOT/obj_queue/`, and
`leg_clf.py` reads `$G1I_ROOT/duplicates.json`. Without this staging they are silently not found.
On the original machine `G1I_ROOT` already holds all of this (the working copy), so skip the copy there.

## Order (what was run, 10-28 .. 10-31)

| # | Step | Command | Output in `$G1I_ROOT` |
|---|---|---|---|
| 1 | download (hf-mirror, local proxy) | `python3 fetch_all.py <workers> 4` (video files 000-003 per dataset = 104 files; the worker count was not logged) | `official/`, `community/` (community sets are downloaded but unused) |
| 2 | index + joint states | `$RT_PY prep_index.py` (default 4 files) | `index.json`, `states/` |
| 3 | SAM3 leg labels | `$SAM3_PY leg_label.py` (defaults: file 0, every 90th frame; 3308 labels) | `leg_labels.json` |
| 4 | leg classifier + windows | `$RT_PY leg_clf.py` (THR 0.3, MAX_WIN 4) | `leg_free.npz`, `windows.json`, `leg_clf_report.json` |
| 5 | cut segments | `$SAM3_PY extract_windows.py 1500` | `segments/*.mp4`, `segments.json` |
| 6 | SAM3 leg check | `$SAM3_PY leg_check.py` (457 of 1500 rejected) | `leg_check.json` |
| 7 | job list + val split | `python3 make_jobs.py` (1043 jobs) | `jobs_robot.json`, `val_segments.json` |
| 8 | robot masks | `for K in 0 1; do JOB_RANK=$K JOB_WORLD=2 EDGE=1 $SAM3_PY track_prompt.py $G1I_ROOT/jobs_robot.json $G1I_ROOT robot & done; wait` | `robot/*.npz` (1043) |
| 9 | labelling sheets | `python3 labeling/obj_sheets.py` | `obj_queue/sheets/`, `obj_queue/batches/` |
| 10 | Claude boxes the handled object | a Claude session follows `labeling/DISPATCH.md`, with `labeling/obj-labeler.md` in `.claude/agents/` | `obj_queue/results/*.jsonl` |
| 11 | check the boxes | `cd $G1I_ROOT/obj_queue && python3 check_results.py` | prints `0 to do` when done |
| 12 | SAM3 tracks each box | `$SAM3_PY box_sam3.py --follow` (can run during step 10) | `obj1/*.npz` (1008 maps) |
| 13 | object map checks | `python3 obj_qc.py` (report only, the cache does not read it) | `check/obj_qc.json` |
| 14 | object depth | `$SAM3_PY obj_depth_batch.py` | `obj_depth/*.npz` (1008) |
| 15 | whole-frame depth (v6b) | `$SAM3_PY full_depth_batch.py` | `full_depth/*.npz` (1043) |
| 16 | upload | `cluster/upload_isambard.sh` | Isambard `$D/data/` |

With the pinned lists from `meta/` you can skip steps 3, 4 and 6 (see `meta/README.md`). Steps 1, 2 and 5 still have
to run to get the videos. The shipped boxes in `labeling/results/` replace step 10.

`chain_g1i.sh` is the 10-29 pilot chain kept as a record (working-copy paths, 1-GPU pilot training). Do not use it.

## Labelling (steps 9-12)

- `labeling/README.md`: the rules the labelling agents follow (one box per sheet, which object counts).
- `labeling/DISPATCH.md`: how the main Claude session ran the agents (3 at a time, one batch each).
- `labeling/obj-labeler.md`: the agent definition. It only works from a `.claude/agents/` folder.
- `labeling/batches/`, `labeling/results/`: the record of the run. `b000`-`b051` = 1023 clips (6-frame sheets);
  `fix000`-`fix004` = 82 clips relabelled with 9-frame sheets (`fix_clips.json`), whose generator ran inline and is not
  shipped; `pilot.jsonl` = the first 20 clips. Over all 1043 clips the final labels are 1008 boxes and 35 "none"
  (a later file overrides an earlier one), and SAM3 tracking gave the 1008 object maps in `obj1/`.

## Environment variables

| Variable | Default | Used by |
|---|---|---|
| `G1I_ROOT` | `/data/datasets/unitree_g1_inspire` | every script here |
| `SAM3_CKPT` | `/data/models/sam3/sam3/sam3.pt` | leg_label, leg_check, track_prompt, box_sam3 |
| `SAM3_TMP` | `$G1I_ROOT` (box_sam3), `/data/datasets/unitree_g1_dex3` (track_prompt) | frame dump dirs |
| `VDA_ROOT`, `VDA_CKPT` | `/data/repos/video_depth_anything`, `.../large/video_depth_anything_vitl.pth` | obj_depth_batch, full_depth_batch |
| `FETCH_MIRROR`, `FETCH_PROXY` | `https://hf-mirror.com`, `http://127.0.0.1:10900` (`""` = no proxy) | fetch_all |
| `JOB_RANK`, `JOB_WORLD`, `EDGE`, `MIN_FRAC`, `INSTANCES` | 0, 1, off, 0.1, off | track_prompt |
| `THR`, `MAX_WIN` | 0.3, 4 | leg_clf |
