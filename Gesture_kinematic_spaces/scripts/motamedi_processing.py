# importing some basic packages
import numpy as np                  # basic data operations
import pandas as pd                 # data wrangling
import matplotlib.pyplot as plt     # for plotting
import plotly.graph_objects as go   # for plotting
import html                         # for html-related add-ons
import os                           # for foldering        
import pickle                       # for saving and loading data
from IPython.display import Image   # for showing videos
from scipy.ndimage import gaussian_filter1d         # for smoothing
from scipy.stats import pearsonr, spearmanr, zscore # for correlation and z-score
from tqdm import tqdm               # for progress bars

# define directories
# get directory of the current script
DIR = os.path.dirname(os.path.realpath(__file__)) + "/"
MOTAMEDI_DIR    = DIR + "../data/ValidationDataMotamedi/"
MOTAMEDI_TS_DIR    = MOTAMEDI_DIR + "data_tracking_mediapipe/"
MOTAMEDI_PROC_DIR        = MOTAMEDI_DIR + "processed/"



####### Step 1: Load annotation data and precomputed distance matrices #######
# - Annotation target -> tracking-file prefix
# The tracking filenames start with a short concept key, not the full target name.
# Targets with no tracking files
# are simply omitted — they will be filtered out automatically below.
TARGET_TO_PREFIX = {'chef':'chef', 'frying pan':'fryingpan','to cook':'cook',
    'church': 'church','bible':  'bible', 'to preach':  'preach','photographer':'photographer','darkroom':'darkroom',
    'camera': 'camera','concert hall':'concerthall','microphone':'microphone','hairdresser':'hairdresser',
    'hair salon': 'hairsalon','to give a haircut': 'haircut','police officer':'policeofficer','prison': 'prison',
    'handcuffs':  'handcuffs','to make an arrest': 'arrest',
}


# - Load annotation: all concepts from experiment 1
# gen 0 = seed gesture (no tracking file), so we keep only generation > 0.
# participant = "full{N}" maps to tracking filename suffix "f{N}".
anno = pd.read_csv(MOTAMEDI_DIR + "annotation_data/ex1.csv")
df   = (anno[(anno["generation"] > 0) & # we could keep the seed data!
   anno["participant"].str.startswith("full", na=False)] 
        .copy())

df["chain_num"] = df["chain"].str.extract(r"(\d+)").astype(int)
df["f_num"]     = df["participant"].str.extract(r"full(\d+)").astype(int)
df["prefix"]    = df["target"].map(TARGET_TO_PREFIX)

# Build path to the body tracking CSV (contains nose, elbows, wrists, index fingers)
df["filepath"] = (MOTAMEDI_TS_DIR + df["prefix"].fillna("__missing__")
       + "_exp1_ch" + df["chain_num"].astype(str)
       + "_g"       + df["generation"].astype(str)
       + "_f"       + df["f_num"].astype(str)
       + "_body.csv")

# gesture_id must match the row/column labels in the precomputed distance matrices
# (spaces in target name → underscores, same as compute_annotation_distances.py)
df["gesture_id"] = ("ex1_ch" + df["chain_num"].astype(str)
  + "_g"   + df["generation"].astype(str)
  + "_"    + df["target"].str.replace(" ", "_", regex=False)
  + "_"    + df["director"].astype(str)
  + "_TI")

# - Load precomputed annotation distance matrices --------------─
lev_mat  = pd.read_csv(MOTAMEDI_PROC_DIR + "ex1_levenshtein.csv", index_col=0)
jacc_mat = pd.read_csv(MOTAMEDI_PROC_DIR + "ex1_jaccard.csv",     index_col=0)

# Keep only rows that have a tracking file AND an entry in the distance matrix
gesture_ok = set(lev_mat.index)
df = (df[df["filepath"].apply(os.path.exists) &
  df["gesture_id"].isin(gesture_ok)]
      .reset_index(drop=True))

print(f"Usable gesture instances : {len(df)}")
print(f"Concepts covered  : {df['target'].nunique()} / {anno['target'].nunique()}")
print(f"Distance matrix size     : {lev_mat.shape[0]} x {lev_mat.shape[1]}")



####### Step 2: Load and process tracking data (body + hands) #######
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

# - Step 2a.5: Global z-score statistics
print("Computing global statistics...")
for path in df["filepath"].unique():
    load_ts(path)

# only use files with full feature set (i.e. hands data present)
n_features_max = max(v.shape[1] for v in ts_cache.values())
valid_paths    = {k for k, v in ts_cache.items() if v.shape[1] == n_features_max}
skipped        = len(ts_cache) - len(valid_paths)
print(f"Skipped {skipped} files without hands data, using {len(valid_paths)} files")

all_data    = np.concatenate([ts_cache[p] for p in valid_paths], axis=0)
global_mean = all_data.mean(axis=0)
global_std  = all_data.std(axis=0) + 1e-8

def normalise_ts(ts):
    return (ts - global_mean) / global_std

print(f"Global stats computed over {all_data.shape[0]} frames, {all_data.shape[1]} features")

# - Step 2b: Sample pairs — same-target only (only from files with full feature set)
all_pairs = [
    (i, j)
    for i in range(len(df))
    for j in range(i + 1, len(df))
    if df.iloc[i]["target"] == df.iloc[j]["target"]
    and df.iloc[i]["filepath"] in valid_paths
    and df.iloc[j]["filepath"] in valid_paths
]

print(f"Targets represented      : {df[df['filepath'].isin(valid_paths)]['target'].nunique()}")



####### Step 3. Save processed data and selected pairs for later use #######
# save the processed dataframe to a CSV file
df.to_csv(MOTAMEDI_PROC_DIR + "ex1_processed.csv", index=False)

# save the all_pairs to a CSV file
pairs_df = pd.DataFrame(all_pairs, columns=["idx1", "idx2"])
pairs_df.to_csv(MOTAMEDI_PROC_DIR + "ex1_all_pairs.csv", index=False)

# save cols_cache dict to a CSV file 
cols_cache_path = MOTAMEDI_PROC_DIR + "ex1_cols_cache.pkl"
with open(cols_cache_path, "wb") as f:
    pickle.dump(cols_cache, f)

# export global mean and std for later use as a pickle file
with open(MOTAMEDI_PROC_DIR + "global_stats.pkl", "wb") as f:
    pickle.dump({"mean": global_mean, "std": global_std}, f)