# Changes from the working copy

This repo was assembled on 11-01 from the working copies on the training machine. Nothing in the working copies was
changed. This file lists every file whose content differs from its source, and why.

## Where the files came from

| Repo folder | Working copy |
|---|---|
| `renderer/` | `/data/repos/roborender/local/g1src/*.py`, `/data/repos/roborender/local/inst_model.py` |
| `cluster/` | `/data/datasets/unitree_g1_inspire/{fetch_steps.sh, upload_isambard.sh}`; `train_v5.sbatch` <- `g1i_obj4_2g.sbatch`, `train_v6a.sbatch` <- `g1i_v6a.sbatch`, `train_v6b.sbatch` <- `g1i_v6b.sbatch` |
| `data_pipeline/` | `/data/datasets/unitree_g1_inspire/*.py, chain_g1i.sh`; `track_prompt.py` <- `/data/repos/sam3/local/` |
| `data_pipeline/labeling/` | `/data/datasets/unitree_g1_inspire/obj_queue/{README.md, DISPATCH.md, check_results.py, fix_clips.json, batches/, results/}`, `../obj_sheets.py`; `obj-labeler.md` <- `/home/yan/WorldModel/.claude/agents/` |
| `data_pipeline/meta/` | `/data/datasets/unitree_g1_inspire/*.json, leg_free.npz` (unchanged copies) |
| `sim_bridge/` | `/data/repos/chord/local/*.py`; `env/retarget_reqs.txt` <- `chord/local/retarget_reqs.txt`; `assets/head_cam_calib.json` <- `/data/datasets/unitree_g1_dex3/pick_tissue/calib/` |
| `tools/` | `chord/local/*` (chord_recon, depth_check, head_cam_calib, sim_bridge), `chord/local/recon_patches/`, `chord/local/logs/*.sh` (chord_recon/setup), `unitree_g1_inspire/*.py` (data_checks, figures, previews), `roborender/local/g1src/diag_gap.py`, `/data/models/roborender_g1src/compare_runs.py` |
| `docs/` | `chord_recon_native.md` <- `chord/local/RECON_SETUP.md`; `pipeline/` <- `unitree_g1_inspire/pipeline_page/` |

Renamed but byte-identical: `tools/data_checks/leg_prompt_test.py` (<- `unitree_g1_inspire/check/prompt_test.py`),
`sim_bridge/assets/head_cam_calib.json`, `tools/chord_recon/setup/*.sh`, `data_pipeline/meta/*`.
Every other copied file not named below is byte-identical to its source (checked with `cmp`), among them
`renderer/inst_model.py, objdepth_enc.py, fulldepth_enc.py, ghost_holes.py`, `data_pipeline/track_prompt.py`,
`sim_bridge/{detect_hands_2d, make_static_camera, recolor_robot, render_sim_depth}.py` and all `tools/` files not listed.

## 1. Path changes (behaviour the same at the old locations)

The defaults are the old absolute paths, so on the original machine every script behaves as before.

| File | Change |
|---|---|
| `renderer/cache_g1i.py`, `train_src.py` | `ROOT` = `$ROBORENDER_ROOT` or `/data/repos/roborender` (was: two folders above the script); the script's own folder goes on `sys.path` for `inst_model` and the sibling modules (was: `ROOT/local`) |
| `renderer/infer_g1i.py`, `merge_lora_g1i.py`, `tools/eval/diag_gap.py` | same; `diag_gap.py` finds `inst_model` in `../../renderer` |
| `renderer/prep_human_depth.py` | `fulldepth_enc` from the script's folder (was `/data/repos/roborender/local/g1src`); `VDA_ROOT`, `VDA_CKPT` env with the old defaults |
| `tools/figures/fulldepth_steps.py`, `page_images.py`, `tools/previews/fulldepth_preview.py` | `objdepth_enc` / `ghost_holes` from `../../renderer` (was `/data/repos/roborender/local/g1src`) |
| `cluster/fetch_steps.sh` | merges with `$REPO/renderer/merge_lora_g1i.py` and `$ROBORENDER_ROOT/.venv` (was `cd /data/repos/roborender && local/g1src/...`) |
| `data_pipeline/*.py`, `labeling/obj_sheets.py`, `labeling/check_results.py` | dataset dir = `$G1I_ROOT` (default `/data/datasets/unitree_g1_inspire`); `leg_label.py`, `leg_check.py`: `SAM3_CKPT` env; `obj_depth_batch.py`, `full_depth_batch.py`: `VDA_ROOT`, `VDA_CKPT` env; `fetch_all.py`: `FETCH_MIRROR`, `FETCH_PROXY` env (`""` turns the proxy off); `make_jobs.py`: `import os` added |
| `sim_bridge/{ego_to_dex3, estimate_ego_camera, refine_ego_camera, fit_box_shape, ik_camera_track, inspire_contact_ik}.py` | the CHORD repo for mesh paths (and `inspire_contact_ik.py`'s `robotic_grounding` import) = `$CHORD` (default `/data/repos/chord`); `ego_to_dex3.py`: `import os` added |
| `sim_bridge/mano_to_inspire.py` | `XR_TELEOP` (default `/data/repos/unitree_assets/xr_teleoperate`) and `UNITREE_ROS` (default `/data/repos/unitree_assets/unitree_ros`) env |
| `sim_bridge/sam3_union_masks.py`, `sam3_video_masks.py` | `SAM3_CKPT` env; `import os` added |
| `sim_bridge/run_tissue_chain.sh` | adds `XR_TELEOP` and exports `CHORD UNITREE_ROS XR_TELEOP`, so steps 8-10 see the same paths as the shell steps; the closing `echo` lines also print the infer and compare commands |
| `tools/eval/compare_runs.py` | tissue-clip dir `E`, sim render `SIM` and human video `V` from env, old paths as defaults |
| `tools/chord_recon/patches/trtexec_py.sh` | runs the `trtexec_py.py` next to itself (was the absolute path in `chord/local/recon_patches`), venv from `RECON_VENV` |
| `cluster/upload_isambard.sh` | `G1I_ROOT` and `HANDS` (tissue-clip hand masks) env, old paths as defaults |

## 2. Changes that do something different

- `cluster/upload_isambard.sh`: new code layout on Isambard. Upstream `core/` goes to `$D/roborender/`, this repo's
  `renderer/` and `cluster/*.sbatch` to `$D/World_render/`. The working copy put everything in `$D/code/{core,
  local/inst_model.py, local/g1src}`. It also does the four data uploads that were separate glue scripts before
  (`handoff2.sh` robot/, `handoff_obj1.sh` obj1/, `upload_v4.sh` obj_depth/, the `upload-fulldepth` unit full_depth/)
  and adds `mkdir -p seg_up`. This layout has never been used; run the script once before any sbatch.
- `cluster/train_v5.sbatch`, `train_v6a.sbatch`, `train_v6b.sbatch`: `cd $D/World_render/renderer` with
  `ROBORENDER_ROOT=$D/roborender`. The runs that made the checkpoints used snapshots `code5/` (v5) and `code6/` (v6)
  with `ROBORENDER_ROOT` there. So these are path-adapted copies, not the exact files that ran. v5 now runs the current
  renderer code; the v6 additions sit behind flags whose defaults keep the v5 path (read against the v4 snapshot of
  `cache_g1i.py`; `code5/` itself is not available locally, so bit-identity is unverified; noted in the script).
- `cluster/train_v6a.sbatch`, `train_v6b.sbatch`: `--init_from` falls back to `runs/g1i_obj4_2g/resume.pt` when
  `lora_05000.pt` is missing. `train_src.py` never writes `lora_*.pt`; that file only exists if
  `cluster/fetch_steps.sh g1i_obj4_2g 5000` exported it. `resume.pt` of the finished v5 run holds the same step-5000
  weights in fp32 (the lora file has LoRA A/B in bf16), so the start differs by bf16 rounding.
- `renderer/cache_g1i.py`: the tissue-clip hand masks (`--hands`) are loaded on first use instead of at start-up.
  v6 caches (`--hand_stamps 0`, ring off) no longer need the file. Random numbers are drawn in the same order, so a
  cache with the same flags and seed is unchanged.

## 3. Docs and comments only

- `renderer/cache_g1i.py`: `--obj_dir` help and a comment say SAM3 (`data_pipeline/box_sam3.py`), not SAM2
  `box_sam.py`; docstring points to `data_pipeline/extract_windows.py`; `--hands` got a help text.
- `renderer/train_src.py`: first lines say G1 + Inspire (`cache_g1i.py`); the usage line points to the sbatch flags.
- `renderer/merge_lora_g1i.py`: `fetch_g1i.sh` -> `cluster/fetch_steps.sh`.
- `renderer/prep_human_g1i.py`: usage line lists `--hole_px` (default 5, v6: 1).
- `cluster/upload_isambard.sh`: header rewritten for the new layout.
- `data_pipeline/chain_g1i.sh`: header marks it as the historical 10-29 pilot chain; the promise of an eval step
  (the script has none) is gone.
- `data_pipeline/labeling/obj_sheets.py`: SAM3 instead of SAM2 in the docstring; notes that the 9-frame fix sheets came
  from a variant that is not shipped.
- `data_pipeline/labeling/obj-labeler.md`: "6- or 9-frame" sheets. `DISPATCH.md`: mentions the fix batches.
- `sim_bridge/env/retarget_reqs.txt`: `mediapipe==0.10.14` added (used by `detect_hands_2d.py`, present in the venv).
- `docs/chord_recon_native.md`: repo note at the top (path map), MANO and FoundationPose weights now present, full run
  still not done.
- `docs/pipeline/g1_renderer_v6.html`: a note under the title that the page predates v6 training and points to
  `RESULTS.md`; "30 % no object mask" now says v6b only (v6a has `--mask_drop 0`); the ghost-hole jitter is described
  as normal with sigma 8 deg / 6 px (`ghost_holes.py` draws `rng.normal`), not as hard bounds.

## 4. New or rebuilt files

| File | What |
|---|---|
| `renderer/cache_empty_prompt.py` | trimmed from `roborender/local/cache_prompts.py` (that one also encoded four instruction files); not run in this pass |
| `data_pipeline/make_jobs.py` | the inline Python of `chain_g1i.sh`; checked: gives the same 1043-clip job list and a byte-identical `val_segments.json` |
| `sim_bridge/make_inspire_nohead_urdf.py` | rebuilds `chord/local/assets/g1_inspire/g1_inspire_ftp_nohead.urdf` (made by hand on 10-29) from unitree_ros; checked: byte-identical output |
| `sim_bridge/run_tissue_chain.sh` | written afterwards from the outputs; the steps were run one by one, never as this script (its header says so) |
| `tools/figures/{channels_figure, ghost_holes_figure, hole_mask_cells, hole_mask_explain, page_outputs_v5}.py` | page-figure code that ran inline on 10-31, recovered from the session log; not re-run in this pass |
| `tools/chord_recon/patches/foundationpose_mycpp/CMakeLists{,.orig}.txt`, `foundationpose_mycuda/torch28_scalar_type.patch` | copied from `chord/local/recon_patches/` (the rebuild steps in `chord_recon_native.md` need them) |
| `data_pipeline/meta/{segments, leg_check, jobs_robot, windows, index}.json`, `leg_free.npz` | unchanged copies that pin the training set (`meta/README.md`) |
| `env/*.txt` | package lists read from the local envs |
| `README.md`, `docs/RESULTS.md`, this file, `data_pipeline/README.md`, `data_pipeline/meta/README.md`, `tools/chord_recon/README.md`, `.gitignore` | new |

## 5. What was checked

- All 67 Python files compile (`py_compile`); all shell scripts pass `bash -n`.
- `tools/eval/compare_runs.py` (after the env change) re-run on the stored tissue-clip videos: same numbers as
  `docs/RESULTS.md`.
- The val command lines in `RESULTS.md` come from the 10-31 session log; the hole width they imply (5 px for v5, 1 px
  for v6) matches the grey pixels in the stored output videos.
- `make_jobs.py` and `make_inspire_nohead_urdf.py`: see section 4.

Not checked: a GPU run of the repo's `infer_g1i.py` against the stored v6a output (the GPU was busy with another job
the whole time; the only code change on that path is the `sys.path` setup, and the modules it imports are
byte-identical); bit-identity of `train_v5.sbatch` with `code5/`; `cache_empty_prompt.py`; the new Isambard layout;
`run_tissue_chain.sh` as a whole.
