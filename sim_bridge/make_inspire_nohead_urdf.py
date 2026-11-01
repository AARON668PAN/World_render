"""Build the G1 + Inspire (FTP) URDF the render chain uses (ik_camera_track.py, render_sim_depth.py, G1_URDF=...):
Unitree's g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf (unitree_ros, robots/g1_description) with
  1. relative mesh paths ("meshes/...") made absolute, so the file can live anywhere;
  2. the head_link <visual> and <collision> removed (the head camera sits inside the head; with the head mesh the
     rendered view is blocked).
This reproduces assets/g1_inspire/g1_inspire_ftp_nohead.urdf of the working copy, which was made by hand (10-29).
usage: make_inspire_nohead_urdf.py <unitree_ros/robots/g1_description> <out.urdf>"""
import os
import re
import sys

desc, out = os.path.abspath(sys.argv[1]), sys.argv[2]
s = open(os.path.join(desc, "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf")).read()
s = s.replace('filename="meshes/', f'filename="{desc}/meshes/')
i = s.index('<link name="head_link">'); j = s.index("</link>", i)
head = re.sub(r"<visual>.*?</visual>", "", s[i:j], flags=re.S)
head = re.sub(r"<collision>.*?</collision>", "", head, flags=re.S)
open(out, "w").write(s[:i] + head + s[j:])
print(f"wrote {out}")
