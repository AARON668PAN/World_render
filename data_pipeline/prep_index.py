"""Index the downloaded G1 WBT Inspire episodes: for every official dataset, the episodes whose left-head video lies in a
downloaded video file (file_index < MAX_FILE): video file, from/to timestamp, length, task text; plus each episode's joint
states for the leg-visibility classifier: lower body (15) + waist (3) + torso pose (6) + head camera (d435) pose (6) = 30-D,
or robot_q_current (36-D) for the one dataset with the older schema (Take_Plates_Into_Dishwasher). Run with the retarget venv (pyarrow).
writes index.json and states/<dataset>.npz (one array per episode: key ep<NNNNN>)"""
import glob, json, os, sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.environ.get("G1I_ROOT", "/data/datasets/unitree_g1_inspire")  # dataset work dir
MAX_FILE = int(sys.argv[1]) if len(sys.argv) > 1 else 4
KEY = "videos/observation.images.head_stereo_left"
os.makedirs(f"{ROOT}/states", exist_ok=True)
index = []
for d in sorted(glob.glob(f"{ROOT}/official/*")):
    name = os.path.basename(d)
    m = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(f"{d}/meta/episodes/**/*.parquet", recursive=True))]).to_pydict()
    tasks = {}
    tf = glob.glob(f"{d}/meta/tasks*.parquet") + glob.glob(f"{d}/meta/tasks/**/*.parquet", recursive=True)
    if tf:
        tt = pq.read_table(tf[0]).to_pandas()
        tasks = {int(r["task_index"]): str(i) for i, r in tt.iterrows()} if "task_index" in tt else {}
    eps = [dict(dataset=name, episode=int(e), chunk=int(c), file=int(f), t0=float(a), t1=float(b), length=int(n),
                tasks=list(tk) if isinstance(tk, (list, tuple)) else tk)
           for e, c, f, a, b, n, tk in zip(m["episode_index"], m[f"{KEY}/chunk_index"], m[f"{KEY}/file_index"], m[f"{KEY}/from_timestamp"],
                                            m[f"{KEY}/to_timestamp"], m["length"], m.get("tasks", [None] * len(m["episode_index"]))) if f < MAX_FILE]
    want = {e["episode"] for e in eps}
    states = {}
    for f in sorted(glob.glob(f"{d}/data/**/*.parquet", recursive=True)):
        names = pq.read_schema(f).names
        cols = (["observation.state.robot_q_current"] if "observation.state.robot_q_current" in names else
                ["observation.state.lower_body", "observation.state.waist_state_joint", "observation.state.state_torso", "observation.state.state_d435"])
        t = pq.read_table(f, columns=["episode_index", "frame_index"] + cols)
        ep = t.column("episode_index").to_numpy()
        sel = np.isin(ep, list(want))
        if not sel.any():
            continue
        q = np.concatenate([np.array(t.column(c).to_pylist(), np.float32) for c in cols], 1)[sel]
        fi = t.column("frame_index").to_numpy()[sel]
        for e in np.unique(ep[sel]):
            k = ep[sel] == e
            states[f"ep{int(e):05d}"] = q[k][np.argsort(fi[k])]
    np.savez_compressed(f"{ROOT}/states/{name}.npz", **states)
    index += eps
    print(f"{name}: {len(eps)} episodes in files < {MAX_FILE}, states for {len(states)}", flush=True)
json.dump(index, open(f"{ROOT}/index.json", "w"), indent=0)
print(f"{len(index)} episodes, {sum(e['length'] for e in index) / 30 / 3600:.1f} h")
