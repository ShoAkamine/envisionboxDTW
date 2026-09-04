import numpy as np  # basic data operations
import pandas as pd # dataframe operations
import pickle 
from scipy.ndimage import gaussian_filter1d  # for smoothing
from dtw import dtw
from shapedtw.shapedtw import shape_dtw
from shapedtw.shapeDescriptors import RawSubsequenceDescriptor, SlopeDescriptor, DerivativeShapeDescriptor

MOTAMEDI_DIR    = "../data/ValidationDataMotamedi/"
MOTAMEDI_PROC_DIR        = MOTAMEDI_DIR + "processed/"


def minmax(s):
    return (s - s.min()) / (s.max() - s.min())
    

def mirror_ts(ts, cols):
    """
    Mirror the time series by (i) swapping LEFT and RIGHT columns for partner keypoints, and (ii) negating the X coordinates of all keypoints.
    The function only properly mirrors the time series if the LEFT and RIGHT keypoints are centered around the same vertical axis.
    """
    ts_m = ts.copy()
    cols = [c.upper() for c in cols]  # ensure consistent case
    for col in cols:
        if "LEFT" in col:
            partner = col.replace("LEFT", "RIGHT")
        elif "RIGHT" in col:
            partner = col.replace("RIGHT", "LEFT")
        else:
            continue
        if partner in cols:
            ci, cp = cols.index(col), cols.index(partner)
            ts_m[:, ci], ts_m[:, cp] = ts[:, cp].copy(), ts[:, ci].copy()
            # negate X coordinates
            if col.startswith("X_"):
                ts_m[:, ci], ts_m[:, cp] = -ts_m[:, ci], -ts_m[:, cp]
    return ts_m


def get_landmarks_motamedi(cols):
    """
    Given a list of column names, return a dictionary mapping keypoint names to their X and Y indices.
    """
    landmarks = {}
    # making things homogenous across files with different feature sets
    for idx, col in enumerate(cols):
        if col.startswith("X_"):
            landmarks.setdefault(col[2:], {})["x"] = idx
        elif col.startswith("Y_"):
            landmarks.setdefault(col[2:], {})["y"] = idx

    return landmarks


def get_landmarks(cols):
    """
    Given a list of column names, return a dictionary mapping keypoint names to their X and Y indices.
    """
    left_landmarks = {}
    right_landmarks = {}

    cols = [c.upper() for c in cols]  # ensure consistent case
    for idx, col in enumerate(cols):
        if col.startswith(f"X_LEFT_"):
            left_landmarks.setdefault(col[2:], {})["x"] = idx
        elif col.startswith(f"Y_LEFT_"):
            left_landmarks.setdefault(col[2:], {})["y"] = idx
        elif col.startswith(f"X_RIGHT_"):
            right_landmarks.setdefault(col[2:], {})["x"] = idx
        elif col.startswith(f"Y_RIGHT_"):
            right_landmarks.setdefault(col[2:], {})["y"] = idx

    return left_landmarks, right_landmarks


def select_distance(dtw_left, dtw_right, dtw_mean, 
                    dtw_left_mirror, dtw_right_mirror, dtw_mean_mirror,
                    hand):
    """
    Select the appropriate DTW distance based on the specified hand.
    """
    if hand == "left":
        distance = dtw_left
    elif hand == "right":
        distance = dtw_right
    elif hand == "left_right":
        distance = min(dtw_mean, dtw_left_mirror)
    elif hand == "right_left":
        distance = min(dtw_mean, dtw_right_mirror)
    else:  # both
        distance = min(dtw_mean, dtw_mean_mirror)

    return distance


def calculate_window_size(ts1, ts2, window_size=20):
    """
    Calculate the window size for DTW based on the lengths of the two time series.
    The window size is set to a fraction of the length of the longer time series, with a minimum of 1.
    """
    len1, len2 = ts1.shape[0], ts2.shape[0]
    len_diff = abs(len1 - len2)
    if len_diff >= window_size:
        # widen the window dynamically so it can't fully exclude the last column
        window_size = len_diff + 5
    return window_size


################################## DTW variants ##################################
def obe_dtw_distance(ts1, ts2, landmarks):
    """
    Compute the distance between two time series ts1 and ts2, using the specified columns.
    This function computes the dependent DTW distance per keypoint (X, Y) and returns the mean distance across all keypoints.
    """
    distances = []
    for name, idxs in landmarks.items():
        if "x" not in idxs or "y" not in idxs:
            continue
        ts1_sliced = ts1[:, [idxs["x"], idxs["y"]]]
        ts2_sliced = ts2[:, [idxs["x"], idxs["y"]]]
        window_size = calculate_window_size(ts1_sliced, ts2_sliced, window_size=20)

        res = dtw(
            ts1_sliced, ts2_sliced,
            step_pattern = "asymmetric",
            open_begin = True, 
            open_end = True,
            window_type="sakoechiba", window_args={"window_size": window_size}
            )
        distances.append(res.normalizedDistance)

    return float(np.mean(distances))


def shape_dtw_distance(ts1, ts2, landmarks):
    """
    Compute the shapeDTW distance between two time series ts1 and ts2, using the specified columns.
    This function computes the dependent DTW distance per keypoint (X, Y) and returns the mean distance across all keypoints.
    """        
    distances = []
    for name, idxs in landmarks.items():
        if "x" not in idxs or "y" not in idxs:
            continue
        ts1_sliced = ts1[:, [idxs["x"], idxs["y"]]]
        ts2_sliced = ts2[:, [idxs["x"], idxs["y"]]]
        window_size = calculate_window_size(ts1_sliced, ts2_sliced, window_size=20)

        res = shape_dtw(
            ts1_sliced, ts2_sliced,
            subsequence_width=10,
            step_pattern="asymmetric",
            open_begin=True, open_end=True,
            shape_descriptor=DerivativeShapeDescriptor(),
            window_type="sakoechiba", window_args={"window_size": window_size},
            multivariate_version="dependent"
            )
        distances.append(res.normalized_distance)

    return float(np.mean(distances))

##################################################################################



################################## Motamedi dataset ##################################
# - Body keypoints (shoulders kept for centering, dropped after)
KP_BODY = [
    "X_NOSE",        "Y_NOSE",
    "X_LEFT_ELBOW",  "Y_LEFT_ELBOW",
    "X_RIGHT_ELBOW", "Y_RIGHT_ELBOW",
    "X_LEFT_WRIST",  "Y_LEFT_WRIST",
    "X_RIGHT_WRIST", "Y_RIGHT_WRIST",
    "X_LEFT_SHOULDER",  "Y_LEFT_SHOULDER",
    "X_RIGHT_SHOULDER", "Y_RIGHT_SHOULDER",
]

# - Hand keypoints (all fingers, X/Y only)
FINGERS = ["THUMB", "INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY_FINGER"]
THUMB_JOINTS  = ["CMC", "IP", "TIP"]
FINGER_JOINTS = ["MCP", "DIP", "TIP"]

KP_HANDS = []
for side in ["LEFT", "RIGHT"]:
    KP_HANDS += [f"X_{side}_WRIST", f"Y_{side}_WRIST"]
    for joint in THUMB_JOINTS:
        KP_HANDS += [f"X_{side}_THUMB_{joint}", f"Y_{side}_THUMB_{joint}"]
    for finger in ["INDEX_FINGER", "MIDDLE_FINGER", "RING_FINGER", "PINKY_FINGER"]:
        for joint in FINGER_JOINTS:
            KP_HANDS += [f"X_{side}_{finger}_{joint}", f"Y_{side}_{finger}_{joint}"]

ts_cache   = {}
cols_cache = {}


def load_ts(path):
    """Load body + hands CSVs, center by shoulder midpoint, add finger extension ratios, smooth."""
    if path not in ts_cache:
        # - Load body
        df_body = pd.read_csv(path)
        body_cols = [c for c in KP_BODY if c in df_body.columns]
        df_body = df_body[body_cols].copy()

        # - Load hands
        hand_path = path.replace("_body.csv", "_hands.csv")
        try:
            df_hand = pd.read_csv(hand_path)
            hand_cols = [c for c in KP_HANDS if c in df_hand.columns
                            and c not in df_body.columns]
            df_hand = df_hand[hand_cols].copy()
            n = min(len(df_body), len(df_hand))
            df_body = df_body.iloc[:n].reset_index(drop=True)
            df_hand = df_hand.iloc[:n].reset_index(drop=True)
            df_ts = pd.concat([df_body, df_hand], axis=1).reset_index(drop=True) # we need to check if this min n is needed
        except FileNotFoundError:
            df_ts = df_body.copy().reset_index(drop=True)

        # - Shoulder midpoint centering
        if all(c in df_ts.columns for c in 
                ["X_LEFT_SHOULDER", "X_RIGHT_SHOULDER",
                "Y_LEFT_SHOULDER", "Y_RIGHT_SHOULDER"]):
            mid_x = (df_ts["X_LEFT_SHOULDER"] + df_ts["X_RIGHT_SHOULDER"]).values / 2
            mid_y = (df_ts["Y_LEFT_SHOULDER"] + df_ts["Y_RIGHT_SHOULDER"]).values / 2
            for col in df_ts.columns:
                if col.startswith("X_"):   df_ts[col] = df_ts[col].values - mid_x
                elif col.startswith("Y_"): df_ts[col] = df_ts[col].values - mid_y
            df_ts = df_ts.drop(columns=["X_LEFT_SHOULDER", "X_RIGHT_SHOULDER",
                                        "Y_LEFT_SHOULDER", "Y_RIGHT_SHOULDER"])

        # - Per-finger extension ratio (more features)
        for side in ["LEFT", "RIGHT"]:
            wx, wy = f"X_{side}_WRIST", f"Y_{side}_WRIST"
            if wx not in df_ts.columns:
                continue
            for finger in FINGERS:
                if finger == "THUMB":
                    mcp_x, mcp_y = f"X_{side}_THUMB_MCP", f"Y_{side}_THUMB_MCP"
                    tip_x, tip_y = f"X_{side}_THUMB_TIP", f"Y_{side}_THUMB_TIP"
                else:
                    mcp_x, mcp_y = f"X_{side}_{finger}_MCP", f"Y_{side}_{finger}_MCP"
                    tip_x, tip_y = f"X_{side}_{finger}_TIP", f"Y_{side}_{finger}_TIP"
                if not all(c in df_ts.columns for c in [mcp_x, mcp_y, tip_x, tip_y]):
                    continue
                mcp_dist = np.sqrt((df_ts[mcp_x].values - df_ts[wx].values)**2 +
                                   (df_ts[mcp_y].values - df_ts[wy].values)**2) + 1e-6
                tip_dist = np.sqrt((df_ts[tip_x].values - df_ts[wx].values)**2 +
                                   (df_ts[tip_y].values - df_ts[wy].values)**2)
                df_ts[f"EXT_{side}_{finger}"] = tip_dist / mcp_dist

        # - Interpolate, smooth
        df_ts = df_ts.interpolate(method="linear", axis=0).ffill().bfill().fillna(0)
        df_ts = df_ts.apply(lambda x: gaussian_filter1d(x, sigma=2))

        ts_cache[path]   = df_ts.values
        cols_cache[path] = list(df_ts.columns)
    return ts_cache[path]

def normalise_ts(ts):
    # get global mean and std from pickle file
    with open(MOTAMEDI_PROC_DIR + "global_stats.pkl", "rb") as f:
        global_stats = pickle.load(f)

    global_mean = global_stats["mean"]
    global_std = global_stats["std"]

    return (ts - global_mean) / global_std