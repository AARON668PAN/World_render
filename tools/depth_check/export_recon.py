"""Dump the ego reconstruction geometry (MANO joints, object pose, object surface samples) to one npz, so scripts in
venvs without pyarrow/trimesh (e.g. the SAM3/VDA one) can use it.

usage: export_recon.py <ego parquet dir> <out.npz>
"""

import glob
import sys

import numpy as np
import pyarrow.parquet as pq
import trimesh

parquet_dir, out = sys.argv[1:3]
table = pq.read_table(glob.glob(f"{parquet_dir}/**/*.parquet", recursive=True)[0])
col = lambda n: table.column(n)[0].as_py()
mesh = trimesh.load(col("object_mesh_paths")[0].replace("/workspace/video_to_data", "/data/repos/chord"), force="mesh")
np.savez_compressed(out, mano=np.concatenate([np.asarray(col("mano_right_joints")), np.asarray(col("mano_left_joints"))], 1),
                    obj_p=np.asarray(col("object_body_position"))[:, 0], obj_q=np.asarray(col("object_body_wxyz"))[:, 0],
                    surf=np.asarray(mesh.sample(200000, return_index=False)))
print("wrote", out)
