"""Render skeleton-animation GIFs of gesture evolution across generations
   from the Motamedi et al. MediaPipe tracking CSVs (no source video needed).

   Body CSVs hold MediaPipe *world* landmarks (metres, hip-centred); hand CSVs
   hold image-normalised coordinates, so each hand is rigidly re-attached to its
   body wrist and rescaled to a plausible metric hand size before drawing.
"""
import glob, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import gaussian_filter1d

TS  = "../data/ValidationDataMotamedi/data_tracking_mediapipe/"
OUT = "Images/"

ARM_EDGES = [("LEFT_SHOULDER", "RIGHT_SHOULDER"), ("LEFT_SHOULDER", "LEFT_ELBOW"),
             ("LEFT_ELBOW", "LEFT_WRIST"), ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
             ("RIGHT_ELBOW", "RIGHT_WRIST")]
FINGERS = {"THUMB": ["CMC", "MCP", "IP", "TIP"],
           "INDEX_FINGER":  ["MCP", "PIP", "DIP", "TIP"],
           "MIDDLE_FINGER": ["MCP", "PIP", "DIP", "TIP"],
           "RING_FINGER":   ["MCP", "PIP", "DIP", "TIP"],
           "PINKY_FINGER":  ["MCP", "PIP", "DIP", "TIP"]}
HAND_LEN_M = 0.09   # wrist -> middle-finger MCP, metres

def col_xy(df, name):
    cx, cy = f"X_{name}", f"Y_{name}"
    if cx not in df.columns or cy not in df.columns:
        return None
    a = df[[cx, cy]].astype(float).values
    return np.column_stack([
        gaussian_filter1d(pd.Series(a[:, k]).interpolate().ffill().bfill().fillna(0).values, 2)
        for k in range(2)])

def hand_points(h, side):
    """dict name -> (n,2) in image coords, or None if the hand is not tracked."""
    names = [f"{side}_WRIST"] + [f"{side}_{f}_{j}" for f, js in FINGERS.items() for j in js]
    pts = {n: col_xy(h, n) for n in names}
    if any(v is None for v in pts.values()):
        return None
    return pts

def build_panel(body, hands):
    """Return per-frame drawing data in metres, centred on the shoulder midpoint."""
    b = {n: col_xy(body, n) for n in
         ["NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
          "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST"]}
    mid = (b["LEFT_SHOULDER"] + b["RIGHT_SHOULDER"]) / 2
    for k, v in b.items():
        if v is not None:
            b[k] = v - mid
    # head marker: eye midpoint if available, else nose
    head = ((b["LEFT_EYE"] + b["RIGHT_EYE"]) / 2
            if b["LEFT_EYE"] is not None and b["RIGHT_EYE"] is not None else b["NOSE"])

    hand_chains = {}
    if hands is not None:
        n = min(len(body), len(hands))
        for side in ("LEFT", "RIGHT"):
            hp = hand_points(hands, side)
            if hp is None:
                continue
            w_img   = hp[f"{side}_WRIST"][:n]
            mcp_img = hp[f"{side}_MIDDLE_FINGER_MCP"][:n]
            scale   = HAND_LEN_M / np.maximum(np.linalg.norm(mcp_img - w_img, axis=1), 1e-4)
            w_body  = b[f"{side}_WRIST"][:n]
            chains  = []
            for finger, joints in FINGERS.items():
                seq = [w_img] + [hp[f"{side}_{finger}_{j}"][:n] for j in joints]
                chains.append([w_body + (p - w_img) * scale[:, None] for p in seq])
            hand_chains[side] = chains
    return b, head, hand_chains

def render(concept, chain, title, outfile, trail=18):
    gens = []
    for g in range(1, 6):
        files = sorted(glob.glob(f"{TS}{concept}_exp1_ch{chain}_g{g}_f*_body.csv"))
        if not files:
            continue
        stem = files[0][:-len("_body.csv")]
        body = pd.read_csv(stem + "_body.csv")
        try:
            hands = pd.read_csv(stem + "_hands.csv")
        except FileNotFoundError:
            hands = None
        gens.append((g, build_panel(body, hands), len(body)))
    if not gens:
        print("no data for", concept, chain)
        return

    nmax = max(n for _, _, n in gens)
    fig, axes = plt.subplots(1, len(gens), figsize=(2.5 * len(gens), 3.0), dpi=110)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor("#0f141b")

    panels = []
    for ax, (g, (b, head, hand_chains), n) in zip(axes, gens):
        ax.set_facecolor("#0f141b")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"Gen {g}", color="#dfe3e8", fontsize=10, pad=4)
        ax.set_xlim(0.62, -0.62); ax.set_ylim(0.42, -0.42)   # x flipped -> mirror view

        arm_lines = [ax.plot([], [], "-", lw=3.2, color="#dfe3e8",
                             solid_capstyle="round")[0] for _ in ARM_EDGES]
        neck_line = ax.plot([], [], "-", lw=2.4, color="#dfe3e8",
                            solid_capstyle="round")[0]
        head_dot  = ax.plot([], [], "o", ms=13, color="#dfe3e8")[0]
        trails    = {s: ax.plot([], [], "-", lw=1.6, alpha=.55, color=c)[0]
                     for s, c in (("LEFT", "#4da3ff"), ("RIGHT", "#ff7a59"))}
        finger_lines = {}
        for side, c in (("LEFT", "#4da3ff"), ("RIGHT", "#ff7a59")):
            finger_lines[side] = [ax.plot([], [], "-", lw=1.8, color=c,
                                          solid_capstyle="round")[0]
                                  for _ in hand_chains.get(side, [])]
        panels.append((b, head, hand_chains, n, arm_lines, neck_line, head_dot,
                       trails, finger_lines))

    def update(f):
        for (b, head, hand_chains, n, arm_lines, neck_line, head_dot,
             trails, finger_lines) in panels:
            i = min(f, n - 1)
            for ln, (a, c) in zip(arm_lines, ARM_EDGES):
                ln.set_data([b[a][i, 0], b[c][i, 0]], [b[a][i, 1], b[c][i, 1]])
            neck_line.set_data([0, head[i, 0]], [0, head[i, 1]])
            head_dot.set_data([head[i, 0]], [head[i, 1]])
            for side, ln in trails.items():
                s = max(0, i - trail)
                w = b[f"{side}_WRIST"]
                ln.set_data(w[s:i + 1, 0], w[s:i + 1, 1])
            for side, lines in finger_lines.items():
                for ln, seq in zip(lines, hand_chains.get(side, [])):
                    j = min(i, len(seq[0]) - 1)
                    ln.set_data([p[j, 0] for p in seq], [p[j, 1] for p in seq])
        return []

    fig.suptitle(title, color="#dfe3e8", fontsize=11.5, y=0.985)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    FuncAnimation(fig, update, frames=range(0, nmax, 2)).save(
        OUT + outfile, writer=PillowWriter(fps=14))
    plt.close(fig)
    print("wrote", OUT + outfile, os.path.getsize(OUT + outfile) // 1024, "KB")



if __name__ == "__main__":
    render("handcuffs", 1, 'Motamedi et al., "handcuffs", chain 1', "handcuffs_evolution_ch1.gif")
    render("haircut",   2, 'Motamedi et al., "to give a haircut", chain 2', "haircut_evolution_ch2.gif")
    render("camera",    3, 'Motamedi et al., "camera", chain 3', "camera_evolution_ch3.gif")
