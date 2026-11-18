"""11-18: checks on what the (fake) robot side received from g1_client.py (fake_robot.py log):
  1. pose messages arrive only between an accepted instruction (command planner=0) and the hand-back (command planner=1)
  2. inside an episode the stream is continuous at 50 Hz
  3. every token is finite and within +-0.6, every hand target inside the Dex3 limits (motor order of --hand-order)
  4. the first token of an episode is close to the controller's own token (no jump at the mode switch)
usage: check_protocol.py <robot log npz> [hand order=unitree]"""
import sys
import numpy as np
import wire
z = np.load(sys.argv[1]); order = sys.argv[2] if len(sys.argv) > 2 else "unitree"; t, kind, cmd = z["t"], z["kind"], z["cmd"]; pose = kind == "pose"
spans = []; start = None
for i in range(len(t)):
    if kind[i] == "command":
        s, p, pl = cmd[i]
        if s and not pl and start is None: start = float(t[i])
        elif pl and start is not None: spans.append((float(start), float(t[i]))); start = None
inside = np.zeros(len(t), bool)
for a, b in spans: inside |= (t >= a - 1e-3) & (t <= b + 1e-3)
print(f"commands: {[(round(float(t[i]), 2), tuple(int(x) for x in cmd[i])) for i in range(len(t)) if kind[i] == 'command' and (i == 0 or kind[i - 1] != 'command')]}  (start, stop, planner)")
print(f"episodes (streamed-token spans): {[(round(a, 2), round(b, 2)) for a, b in spans]}")
out = int((pose & ~inside).sum()); print(f"1. pose messages outside an episode: {out}  {'OK' if out == 0 else 'FAIL'}")
ok2 = True
for a, b in spans:
    tt = t[pose & (t >= a) & (t <= b)]; d = np.diff(tt); print(f"2. episode {a:.1f}-{b:.1f} s: {len(tt)} pose messages, interval median {1000 * np.median(d):.1f} ms, max {1000 * d.max():.1f} ms"); ok2 &= d.max() < 0.08 and abs(np.median(d) - 0.02) < 0.004
print("   ", "OK" if ok2 else "FAIL")
tok, L, R = z["token"][pose], z["left"][pose], z["right"][pose]
lim = lambda side: np.array([wire.DEX3_LIMITS[side][j] for j in wire.DEX3_ORDER[order][side]])
ll, rl = lim("left"), lim("right"); ok3 = np.all(np.isfinite(tok)) and np.abs(tok).max() <= 0.6 + 1e-6 and np.all(L >= ll[:, 0] - 1e-6) and np.all(L <= ll[:, 1] + 1e-6) and np.all(R >= rl[:, 0] - 1e-6) and np.all(R <= rl[:, 1] + 1e-6)
print(f"3. token range {tok.min():+.3f} .. {tok.max():+.3f}; left hand {L.min():+.2f} .. {L.max():+.2f}, right hand {R.min():+.2f} .. {R.max():+.2f}; finger step per message max {max(np.abs(np.diff(L, axis=0)).max(), np.abs(np.diff(R, axis=0)).max()):.3f} rad  {'OK' if ok3 else 'FAIL'}")
for a, b in spans:
    i0 = np.flatnonzero(pose & (t >= a))[0]; d0 = np.abs(z["token"][i0] - z["planner_token"]).max(); print(f"4. first token of the episode at {a:.1f} s differs from the controller's own token by at most {d0:.3f}  {'OK' if d0 < 0.05 else 'FAIL'}")
