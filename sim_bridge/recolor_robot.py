"""Re-colour the rendered G1 + Inspire (render_sim_depth.py output with per-pixel body indices) to the real robot's look,
so a renderer that copies the control's appearance draws a real-looking robot: Inspire hand black with a white back cover,
the wrist block under the hand black, arm modules silver-white aluminium, shoulder / torso dark grey; the shading (and highlights) of the original render
kept. Writes <out dir>/sim_color.mp4 + a symlink to sim_depth.npz, so prep_human_g1i.py can use it as a sim dir.
usage: recolor_robot.py <sim dir> <out dir>"""
import os, re, sys
import imageio.v2 as imageio, numpy as np

src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
z = np.load(f"{src}/sim_depth.npz"); body, names, lab = z["body"], [str(n) for n in z["body_names"]], z["label"]
back = z["hand_back"] if "hand_back" in z.files else np.zeros_like(lab, bool)
ALBEDO = [  # (regex on the body name, RGB albedo), from close-ups of the real G1 + Inspire FTP (WBT videos)
    (r"_(thumb|index|middle|ring|little)_|_palm_force_sensor|_base_link$", (26, 26, 29)),   # Inspire hand: black (back cover: below)
    (r"_wrist_yaw_link$", (32, 32, 35)),                                                  # wrist block right under the hand: black
    (r"_wrist_(pitch|roll)_link$|_elbow_link$|_shoulder_(roll|yaw)_link$", (206, 208, 213)),  # arm modules: silver-white aluminium
    (r"_shoulder_pitch_link$|torso_link|logo_link|waist", (72, 72, 76)),                  # shoulder / torso: dark grey
]
BACK = (228, 228, 230)  # the white back cover of the Inspire hand (render_sim_depth.py hand_back mask)
alb = np.full((len(names) + 1, 3), 120, np.float32)  # last row: no body
for i, n in enumerate(names):
    for pat, col in ALBEDO:
        if re.search(pat, n):
            alb[i] = col; break
rob = np.isin(lab, (3, 4))
w = imageio.get_writer(f"{out}/sim_color.mp4", fps=30, codec="libx264", quality=9, macro_block_size=8)
for t, fr in enumerate(imageio.get_reader(f"{src}/sim_color.mp4")):
    f = fr.astype(np.float32); m = rob[t]
    shade = f.mean(-1, keepdims=True) / 255.0  # the CG render is near-white: its brightness is the shading
    a = alb[np.where(body[t] >= 0, body[t], len(names))]
    a[back[t]] = BACK
    col = a * (0.30 + 0.95 * shade) + 255.0 * np.clip(shade - 0.88, 0, None) * 2.0  # albedo x shading + a little specular
    f[m] = col[m]
    w.append_data(np.clip(f, 0, 255).astype(np.uint8))
w.close()
dst = f"{out}/sim_depth.npz"
if not os.path.exists(dst):
    os.symlink(os.path.abspath(f"{src}/sim_depth.npz"), dst)
print("wrote", out)
