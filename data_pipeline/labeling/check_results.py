"""Check the labelling results against the batch lists: which batches are done and valid, which are missing or bad.
A batch is OK when results/bNNN.jsonl has exactly one valid line per clip of batches/bNNN.txt (box inside 448x336 with
x0 < x1, y0 < y1, frame an integer; or none). Prints a summary and the batches still to (re)do, one per line, after
"TODO:". usage: python3 check_results.py"""
import glob, json, os

Q = os.environ.get("G1I_ROOT", "/data/datasets/unitree_g1_inspire") + "/obj_queue"  # staged copy of data_pipeline/labeling
ok, todo, n_box, n_none, problems = [], [], 0, 0, []
for bf in sorted(glob.glob(f"{Q}/batches/*.txt")):  # bNNN + fixNNN (relabel of approach-phase none, 10-29)
    b = os.path.basename(bf)[:-4]; clips = [l.strip() for l in open(bf) if l.strip()]
    rf = f"{Q}/results/{b}.jsonl"
    if not os.path.exists(rf):
        todo.append(b); continue
    try:
        rows = [json.loads(l) for l in open(rf) if l.strip()]
    except json.JSONDecodeError as e:
        problems.append(f"{b}: bad JSON ({e})"); todo.append(b); continue
    got = {r.get("clip"): r for r in rows}
    bad = [c for c in clips if c not in got]
    for c in clips:
        r = got.get(c)
        if r is None or r.get("none"):
            continue
        bx, fr = r.get("box"), r.get("frame")
        if not (isinstance(fr, int) and isinstance(bx, list) and len(bx) == 4 and 0 <= bx[0] < bx[2] <= 448 and 0 <= bx[1] < bx[3] <= 336):
            bad.append(c)
    if bad or len(rows) != len(clips):
        problems.append(f"{b}: {len(rows)} lines for {len(clips)} clips, bad/missing {bad[:3]}{'...' if len(bad) > 3 else ''}"); todo.append(b); continue
    ok.append(b); n_box += sum(not r.get("none") for r in rows); n_none += sum(bool(r.get("none")) for r in rows)
print(f"{len(ok)} batches OK ({n_box} boxes, {n_none} none), {len(todo)} to do")
for p in problems:
    print("  " + p)
print("TODO:")
print("\n".join(todo))
