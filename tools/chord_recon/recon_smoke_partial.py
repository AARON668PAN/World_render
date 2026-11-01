"""Partial native smoke test of the CHORD ego reconstruction stages that need
neither MANO nor FoundationPose weights. Calls the same v2d.*.docker wrappers
run_ego_wilor.py uses (they go through the native run_in_container shim).

  source /data/repos/chord/local/recon_env.sh
  cd /data/repos/chord/reconstruction
  python /data/repos/chord/local/recon_smoke_partial.py \
      --video /data/repos/chord/local/recon_test/airplane_60f.mp4 \
      --out /data/repos/chord/local/recon_test/partial --prompt "A toy airplane"

Stages: AnyCalib undistort -> extract frames -> MoGe depth+intrinsics ->
stabilise intrinsics -> Grounding DINO (frame 0) -> SAM2 propagation of the
object box. Writes overlay.png (box + mask on frame 0 and last frame).
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

from v2d.anycalib.docker.run_video_to_calibration import run_video_to_calibration
from v2d.common.datatypes import BoundingBox, Sam2Prompt, Sam2Prompts
from v2d.common.utils import extract_images
from v2d.depth.lib.stabilize_intrinsics import stabilize_intrinsics
from v2d.grounding_dino.docker.run_image_to_object_bboxes import run_image_to_object_bboxes
from v2d.moge.docker.run_video_to_depth import run_video_to_depth
from v2d.sam2.docker.run_video_to_masks import run_video_to_masks

W = "/data/repos/chord/reconstruction/data/weights"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--prompt", default="A toy airplane")
    a = p.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    und = f"{out}/anycalib/undistorted.mp4"
    if not os.path.exists(und):
        run_video_to_calibration(
            video_path=a.video, intrinsics_path=f"{out}/anycalib/intrinsics.json",
            distortion_path=f"{out}/anycalib/distortion.json", weights_path=f"{W}/anycalib",
            undistorted_video_path=und, undistorted_intrinsics_path=f"{out}/anycalib/undistorted_intrinsics.json")
    frames = f"{out}/frames"
    if not glob.glob(f"{frames}/*"):
        extract_images(und, frames)
    if not glob.glob(f"{out}/depth/*"):
        run_video_to_depth(video_path=und, depth_folder=f"{out}/depth", intrinsics_folder=f"{out}/intrinsics",
                           weights_path=f"{W}/moge",
                           input_intrinsics_path=f"{out}/anycalib/undistorted_intrinsics.json")
    stabilize_intrinsics(f"{out}/intrinsics", f"{out}/intrinsics_stable.json")

    ref = sorted(glob.glob(f"{frames}/*"))[0]
    dets_path = f"{out}/dino_detections.json"
    run_image_to_object_bboxes(image_path=ref, output_path=dets_path, prompt=a.prompt,
                               model_dir=f"{W}/grounding_dino")
    dets = json.load(open(dets_path))
    print("Grounding DINO:", dets[:2])
    box = BoundingBox.from_dict(dets[0]["box"])
    prompts_path = f"{out}/sam2_prompts.json"
    Sam2Prompts(prompts=[Sam2Prompt(frame_index=0, object_id=1, box=box)]).save(prompts_path)
    run_video_to_masks(video_path=und, prompts_path=prompts_path, masks_dir=f"{out}/masks",
                       weights_dir=f"{W}/sam2")

    masks = sorted(glob.glob(f"{out}/masks/1/*.png"))
    print(f"SAM2 masks for object 1: {len(masks)} frames")
    tiles = []
    for m in (masks[0], masks[-1]):
        idx = os.path.basename(m)
        rgb = np.asarray(Image.open(f"{frames}/{os.path.splitext(idx)[0]}{os.path.splitext(ref)[1]}").convert("RGB")).astype(float)
        mk = np.asarray(Image.open(m)) > 0
        rgb[mk] = 0.5 * rgb[mk] + 0.5 * np.array([255, 0, 0])
        tiles.append(rgb.astype(np.uint8))
        print(f"  {idx}: mask area {int(mk.sum())} px")
    d0 = np.asarray(Image.open(sorted(glob.glob(f"{out}/depth/*"))[0]))
    print(f"MoGe depth[0]: shape {d0.shape} dtype {d0.dtype} min {d0.min()} max {d0.max()}")
    print("intrinsics_stable:", json.load(open(f"{out}/intrinsics_stable.json")))
    Image.fromarray(np.concatenate(tiles, 1)).resize((1280, 400)).save(f"{out}/overlay.png")

    # Mask-prompt path (what run_ego_wilor does with rendered WiLoR hand masks):
    # the sam2 wrapper rewrites mask paths to container mounts via extra_volumes,
    # which the native shim must map back.
    os.makedirs(f"{out}/prompt_masks/2", exist_ok=True)
    Image.open(masks[0]).save(f"{out}/prompt_masks/2/000000.png")
    mp = f"{out}/sam2_prompts_mask.json"
    Sam2Prompts(prompts=[Sam2Prompt(frame_index=0, object_id=2, mask_path="prompt_masks/2/000000.png")]).save(mp)
    run_video_to_masks(video_path=und, prompts_path=mp, masks_dir=f"{out}/masks_from_mask_prompt",
                       weights_dir=f"{W}/sam2")
    m2 = sorted(glob.glob(f"{out}/masks_from_mask_prompt/2/*.png"))
    a_box = np.asarray(Image.open(masks[-1])) > 0
    a_msk = np.asarray(Image.open(m2[-1])) > 0
    iou = (a_box & a_msk).sum() / max(1, (a_box | a_msk).sum())
    print(f"SAM2 mask-prompt run: {len(m2)} frames, last-frame IoU vs box-prompt run {iou:.3f}")


if __name__ == "__main__":
    main()
