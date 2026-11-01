#!/bin/bash
# Human ego video -> sim render inputs of the renderer (G1 + Inspire), the chain that made the tissue-clip inputs of v5/v6
# (10-27 .. 10-31). Written afterwards from the outputs (parquet contents, mesh paths stored in them, sim_depth.npz
# metadata); the steps were run one by one, never as this script. The env values of step 11 were not logged for the v7
# run: they come from the earlier notes, so compare the result with the existing motion.parquet (390 x 53).
# Only the CHORD tissue_box sample has gone through it; a new video first needs CHORD reconstruction + its load workflow
# (processed parquet), see docs/chord_recon_native.md.
# usage: E=<out dir> bash run_tissue_chain.sh
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
E=${E:?set E to an output directory}
CHORD=${CHORD:-/data/repos/chord}                                     # CHORD repo (scripts map /workspace/video_to_data -> /data/repos/chord)
V=${V:-$CHORD/reconstruction/assets/tissue_box.mp4}                   # human ego video, 1280x720, 30 fps, 390 frames
RECON=${RECON:-$CHORD/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed/sequence_id=tissue_box_simple/robot_name=sharpa_wave}
UNITREE_ROS=${UNITREE_ROS:-/data/repos/unitree_assets/unitree_ros}    # g1_description (FTP URDF + meshes)
XR_TELEOP=${XR_TELEOP:-/data/repos/unitree_assets/xr_teleoperate}    # DexPilot config assets/inspire_hand/inspire_hand.yml
export CHORD UNITREE_ROS XR_TELEOP                                   # read by the Python steps (mesh paths, steps 9-10)
PY_RT=${PY_RT:-$CHORD/local/.venv-retarget/bin/python}                # pin 3.7, pink 4.2, mediapipe 0.10.14, pyarrow, trimesh
PY_SAM3=${PY_SAM3:-/data/repos/sam3/.venv/bin/python}                 # SAM3
PY_NEWTON=${PY_NEWTON:-/home/yan/miniconda3/envs/isaaclab_newton/bin/python}  # dex_retargeting 0.5, pinocchio, newton
S=0.75                                                                 # world scale (human scene -> G1 size)
mkdir -p $E/align $E/staged/sequence_id=tissue_box_simple
ln -sfn $RECON $E/staged/sequence_id=tissue_box_simple/robot_name=sharpa_wave
P=$E/staged
URDF=$E/g1_inspire_ftp_nohead.urdf
$PY_RT $HERE/make_inspire_nohead_urdf.py $UNITREE_ROS/robots/g1_description $URDF
# 1-3: 2D hand key points, object mask, hand/forearm/table masks
$PY_RT $HERE/detect_hands_2d.py $V $E/align/hands2d.json
$PY_SAM3 $HERE/sam3_video_masks.py $V "tissue box" $E/align/box_masks.npz          # prompt string not logged; output keys match
$PY_SAM3 $HERE/sam3_union_masks.py $V $E/align/human_table_masks.npz hand forearm table
# 4-7: ego camera (PnP, refine with FOCAL=950), object shape fit, static camera
$PY_RT $HERE/estimate_ego_camera.py $P $E/align/hands2d.json $V $E/align
FOCAL=950 $PY_RT $HERE/refine_ego_camera.py $P $E/align/hands2d.json $E/align/box_masks.npz $V $E/align
$PY_RT $HERE/fit_box_shape.py $P $E/align/ego_camera_refined.npz $E/align/box_masks.npz $E/align/human_table_masks.npz $E/box_fit
$PY_RT $HERE/make_static_camera.py $E/box_fit/staged $E/align/ego_camera_refined.npz $E/static_cam
# 8: scale the scene by 0.75 about the mean static-camera xy, push MANO points out of the object (CHORD retarget code)
PYTHONPATH=$CHORD/robotic_grounding/source/robotic_grounding $PY_RT $HERE/ego_to_dex3.py --input_dir $E/static_cam/staged \
  --output_dir $E/static_cam/dex3 --save --world_scale $S --scale_center 0.10647 -0.42966
# 9-10: Inspire fingers (DexPilot as Unitree xr_teleoperate), then contact-aware IK on the box
$PY_NEWTON $HERE/mano_to_inspire.py $E/static_cam/dex3 $E/static_cam/inspire $S
$PY_NEWTON $HERE/inspire_contact_ik.py $E/static_cam/inspire $E/static_cam/inspire_contact $S
# 11: camera-consistent whole-body IK (G1 head camera follows the scaled human camera)
G1_URDF=$URDF PALM_LINKS=left_base_link,right_base_link PRESCALED_HANDS=1 SMOOTH=0.05 IK_ITERS=150 STANCE_DX=0.10 PALM_ORI=0.1 \
  $PY_RT $HERE/ik_camera_track.py $E/static_cam/inspire_contact $E/static_cam/ego_camera_static.npz $HERE/assets/head_cam_calib.json $S $E/camtrack_v7_inspire_contact
# 12: Newton render, 16:9 video letterboxed into 448x336 (focal 332.5 = 950 * 448 / 1280)
G1_URDF=$URDF CAM_WHF=448,336,332.5 $PY_NEWTON $HERE/render_sim_depth.py $E/camtrack_v7_inspire_contact $E/depth_compare/v7_letterbox
# 13: robot repainted with the real G1 + Inspire look
$PY_NEWTON $HERE/recolor_robot.py $E/depth_compare/v7_letterbox $E/depth_compare/v7_letterbox_realcol
echo "next (renderer env; the full commands with the v6 flags are in docs/RESULTS.md):"
echo "  renderer/prep_human_g1i.py $V $E/align/human_table_masks.npz $E/depth_compare/v7_letterbox_realcol <out.npz> --hole_px 1"
echo "  renderer/prep_human_depth.py $V $E/align/human_table_masks.npz $E/align/box_masks.npz $E/depth_compare/v7_letterbox <out.npz>  (v6b, adds ctrlB to the same npz)"
echo "  renderer/infer_g1i.py <ckpt_03000.pt> <out dir> --human <out.npz> --info 3 --mask_pool mean [--depth for v6b]"
echo "  E=$E tools/eval/compare_runs.py <out prefix> v6a=<out dir>/human_<npz name>.mp4=<out.npz>"
