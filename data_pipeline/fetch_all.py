"""Download the G1 + Inspire data for the source-video renderer through hf-mirror (proxy), N parallel workers.
Official: unitreerobotics/G1_WBT_Inspire_* (LeRobot v3): meta/, data/ (joint states, for the leg-visibility check) and
the left head video; video files are interleaved across datasets (file-000 of every dataset first, then file-001, ...)
so that a partial download already covers every task. Community real-robot tabletop sets: their head/ego video, meta, data.
Files are written as .part and renamed when the size matches; reruns skip finished files.
usage: python fetch_all.py [workers] [video files per official dataset]"""
import json, os, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get("G1I_ROOT", "/data/datasets/unitree_g1_inspire")  # dataset work dir
MIRROR = os.environ.get("FETCH_MIRROR", "https://hf-mirror.com")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MAX_FILES = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 6  # per official dataset: only video files file-000 .. file-(MAX_FILES-1)
COMMUNITY = {  # repo: camera key (real robot only; simulated sets and H1-2 left out)
    "datamentorshf/G1_Teleop_PickApple_Inspire": "cam_left_high",
    "NbRobot/G1_inspire_29": "agent_view_pad_res256_freq20",
    "MLeggiero/g1-inspire-nut-in-box-twist2": "head_rgb",
    "MLeggiero/g1-inspire-pipette-tip": "ego_view",
    "birbirll/g1-inspire-red-ball-success-waist": "ego_view",
    "MLeggiero/g1-inspire-turn-page-twist2": "head_rgb",
}


PROXY = os.environ.get("FETCH_PROXY", "http://127.0.0.1:10900")  # "" = no proxy; downloads go through the local proxy (3 MB/s per stream); API calls go direct


def api(url):
    for _ in range(5):
        r = subprocess.run(["curl", "-s", "--noproxy", "*", "--max-time", "60", url], capture_output=True, text=True)
        try:
            return json.loads(r.stdout)
        except Exception:
            time.sleep(3)
    print("api failed", url, r.stdout[:200], flush=True); return []


def tree(repo, path):
    d = api(f"{MIRROR}/api/datasets/{repo}/tree/main/{path}?recursive=true")
    return [x for x in d if isinstance(x, dict) and x.get("type") == "file"]


def official_repos():
    ids = [x["id"] for x in api(f"{MIRROR}/api/datasets?author=unitreerobotics&limit=500")]
    return sorted(i for i in ids if "WBT_Inspire" in i and "MainCamOnly" not in i)


jobs_meta, video_lists, jobs_comm = [], [], []
for repo in official_repos():
    name = repo.split("/")[1]
    for sub in ("meta", "data"):
        jobs_meta += [(repo, f["path"], f.get("size", 0), f"{ROOT}/official/{name}/{f['path']}") for f in tree(repo, sub)]
    vids = sorted(tree(repo, "videos/observation.images.head_stereo_left"), key=lambda f: f["path"])[:MAX_FILES]
    video_lists.append([(repo, f["path"], f.get("size", 0), f"{ROOT}/official/{name}/{f['path']}") for f in vids])
for repo, cam in COMMUNITY.items():
    name = repo.replace("/", "__")
    for sub in ("meta", "data", f"videos"):
        for f in tree(repo, sub):
            if sub == "videos" and cam not in f["path"]:
                continue
            jobs_comm.append((repo, f["path"], f.get("size", 0), f"{ROOT}/community/{name}/{f['path']}"))
interleaved = [v[i] for i in range(max(len(v) for v in video_lists)) for v in video_lists if i < len(v)]
jobs = jobs_meta + interleaved + jobs_comm  # community sets last: many small files, slow per file
total = sum(j[2] for j in jobs)
print(f"{len(jobs)} files, {total / 1e9:.1f} GB ({len(video_lists)} official datasets, {len(COMMUNITY)} community)", flush=True)
json.dump([j[1:] for j in jobs], open(f"{ROOT}/fetch_plan.json", "w"))


def get(job):
    repo, path, size, dst = job
    if os.path.exists(dst) and (size == 0 or os.path.getsize(dst) == size):
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(6):
        r = subprocess.run(["curl", "-sL", "--fail", "-x", PROXY, "-C", "-", "--max-time", "1800", "-o", dst + ".part", f"{MIRROR}/datasets/{repo}/resolve/main/{path}"])
        if os.path.exists(dst + ".part") and (size == 0 or os.path.getsize(dst + ".part") == size):
            os.replace(dst + ".part", dst); return size
        time.sleep(5 * (attempt + 1))
    print("FAILED", repo, path, flush=True); return 0


done, t0 = 0, time.time()
with ThreadPoolExecutor(N) as ex:
    for k, s in enumerate(ex.map(get, jobs)):
        done += s
        if k % 20 == 0:
            el = time.time() - t0
            print(f"{k + 1}/{len(jobs)} files, {done / 1e9:.1f} GB new, {done / 1e6 / max(el, 1):.1f} MB/s", flush=True)
print("FETCH_DONE", flush=True)
