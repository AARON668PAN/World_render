# Pinned lists of the training data

Copies of the lists in the dataset work dir (`/data/datasets/unitree_g1_inspire`, env `G1I_ROOT`) as they were when
v5, v6a and v6b were trained. They fix which segments exist and what their ids are, so the Claude boxes in
`../labeling/results/` and the held-out split line up. Put them into `$G1I_ROOT` before running the pipeline.

| File | Made by | What it is |
|---|---|---|
| `index.json` | `prep_index.py` (4 video files per dataset) | episodes of the 26 official datasets that lie in the downloaded video files |
| `leg_free.npz` | `leg_clf.py` | per-frame leg-free flags from the joint-state classifier |
| `duplicates.json` | written 10-29 (how the pairs were found is not recorded) | 4 datasets that are identical re-uploads of another; `leg_clf.py` drops their windows |
| `windows.json` | `leg_clf.py` | 2384 leg-free 49-frame windows (15 fps) after removing duplicates |
| `segments.json` | `extract_windows.py 1500` | the 1500 cut segments: id, dataset, episode, window start, per-frame leg-free flags |
| `leg_check.json` | `leg_check.py` (SAM3) | per segment: legs in the window (457 of 1500), per-frame leg flags |
| `jobs_robot.json` | `make_jobs.py` | the 1043 leg-free segments, SAM3 prompt "robotic arm" (paths point into `/data/datasets/unitree_g1_inspire`; rerun `make_jobs.py` for another `G1I_ROOT`) |
| `val_segments.json` | `make_jobs.py` | 36 segments of one random episode per dataset; 18 of them are in `jobs_robot.json` and are never trained on |

With these in place you can skip `leg_label.py` and `leg_clf.py`. `extract_windows.py 1500` then cuts the same
segments again (it rewrites `segments.json` from `windows.json`, `leg_free.npz` and `index.json`), and `leg_check.py`
can be skipped too. Checked: `make_jobs.py` on these files gives the same `jobs_robot.json` clip list and a
byte-identical `val_segments.json`.
