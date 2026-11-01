"""Per-frame joint jumps of a motion parquet (arm and finger joints).

usage: check_jumps.py <motion.parquet>
"""
import sys

import numpy as np
import pyarrow.parquet as pq

t = pq.read_table(sys.argv[1])
names = t.column("robot_joint_names")[0].as_py()
P = np.array(t.column("robot_joint_positions")[0].as_py())
for label, keys in (("arm", ("shoulder", "elbow", "wrist")), ("finger", ("thumb", "index", "middle")), ("waist", ("waist",))):
    cols = [i for i, n in enumerate(names) if any(k in n for k in keys)]
    jump = np.abs(np.diff(P[:, cols], axis=0)).max(axis=1)
    print(f"{label:6s} joints: per-frame jump median {np.median(jump):.3f} rad, 95% {np.percentile(jump, 95):.3f}, max {jump.max():.3f}; frames >0.2 rad: {(jump > 0.2).sum()}")
