"""
compute_annotation_distances.py
================================
Computes pairwise gesture-code distances for the Motamedi et al. dataset
(experiments 1, 2, 3).  Two metrics are calculated from the `code_string`
column of each experiment CSV:

  * Token-level Levenshtein distance
        code_string is tokenised by splitting on "," and stripping whitespace.
        Edit distance is computed on the resulting token sequence.

  * Token-level Jaccard distance  (= 1 - Jaccard similarity)
        Jaccard similarity = |intersection| / |union| of the token sets.

Output (written to ../processed/):
    ex{1,2,3}_gesture_index.csv   — one row per gesture: id + full metadata
    annotation_distances.npz      — compressed numpy archive containing:
          ex1_levenshtein, ex1_jaccard,
          ex2_levenshtein, ex2_jaccard,
          ex3_levenshtein, ex3_jaccard
        each matrix is (n x n), dtype float32.
        NaN cells = missing code_string for that gesture.
        Rows/cols correspond to the gesture_index rows (same order).

contact: w.pouw@tilburguniversity.edu
"""

import numpy as np
import pandas as pd
import os
import time

# ── paths ──────────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = HERE
OUT_DIR  = os.path.join(HERE, '..', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

EX_FILES = {
    'ex1': os.path.join(DATA_DIR, 'ex1.csv'),
    'ex2': os.path.join(DATA_DIR, 'ex2.csv'),
    'ex3': os.path.join(DATA_DIR, 'ex3.csv'),
}

COND_ABBREV = {
    'Transmission + Interaction': 'TI',
    'Transmission only':          'TO',
    'Interaction only':           'IO',
}


# ── helpers ────────────────────────────────────────────────────────────────

def build_gesture_id(df, ex):
    """Construct a unique, human-readable gesture_id for every row."""
    chain_num  = df['chain'].str.extract(r'(\d+)')[0].astype(int).astype(str)
    gen_str    = df['generation'].astype(int).astype(str)
    target_str = df['target'].fillna('NA').str.replace(' ', '_', regex=False)
    dir_str    = df['director'].fillna('NA').astype(str)
    cond_str   = df['condition'].map(COND_ABBREV).fillna('UNK')

    base = (ex + '_ch' + chain_num + '_g' + gen_str + '_'
            + target_str + '_' + dir_str + '_' + cond_str)

    # ex3: same (chain,gen,target,director,condition) can have multiple
    # participants -> append participant to keep IDs unique
    if ex == 'ex3':
        base = base + '_' + df['participant'].fillna('NA').astype(str)

    return base


def tokenise(code_string):
    """Split a code_string into a list of gesture tokens."""
    if not isinstance(code_string, str):
        return []
    return [t.strip() for t in code_string.split(',') if t.strip()]


def levenshtein_tokens(seq_a, seq_b):
    """Wagner-Fischer token-level Levenshtein distance."""
    m, n = len(seq_a), len(seq_b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(prev[j] + 1,
                          curr[j - 1] + 1,
                          prev[j - 1] + cost)
        prev = curr
    return prev[n]


def jaccard_distance_tokens(seq_a, seq_b):
    """Jaccard distance (1 - Jaccard similarity) on token sets."""
    set_a, set_b = set(seq_a), set(seq_b)
    union = set_a | set_b
    if not union:
        return 0.0
    return 1.0 - len(set_a & set_b) / len(union)


# ── per-experiment computation ─────────────────────────────────────────────

def compute_matrices(ex, df):
    """Return (lev_matrix float32, jacc_matrix float32)."""
    n = len(df)
    lev_mat  = np.zeros((n, n), dtype=np.float32)
    jacc_mat = np.zeros((n, n), dtype=np.float32)

    tokens   = [tokenise(cs) for cs in df['code_string']]
    nan_mask = df['code_string'].isna().values

    t0 = time.time()
    for i in range(n):
        if i % 200 == 0 and i > 0:
            elapsed = time.time() - t0
            pairs_done = i * n - i * (i + 1) // 2
            pairs_total = n * (n - 1) // 2
            pct = 100 * pairs_done / pairs_total
            eta = elapsed / pct * (100 - pct) if pct > 0 else 0
            print(f'  {ex}: row {i:4d}/{n}  {pct:5.1f}% done  '
                  f'{elapsed:.0f}s elapsed  ETA {eta:.0f}s')

        for j in range(i + 1, n):
            if nan_mask[i] or nan_mask[j]:
                lev_mat[i, j] = lev_mat[j, i] = float('nan')
                jacc_mat[i, j] = jacc_mat[j, i] = float('nan')
            else:
                lev  = levenshtein_tokens(tokens[i], tokens[j])
                jacc = jaccard_distance_tokens(tokens[i], tokens[j])
                lev_mat[i, j]  = lev_mat[j, i]  = lev
                jacc_mat[i, j] = jacc_mat[j, i] = jacc

    print(f'  {ex}: finished in {time.time() - t0:.0f}s')
    return lev_mat, jacc_mat


# ── main ───────────────────────────────────────────────────────────────────

def main():
    archive = {}

    for ex, path in EX_FILES.items():
        print(f'\n=== {ex} ===')
        df = pd.read_csv(path)
        df['gesture_id'] = build_gesture_id(df, ex)
        assert df['gesture_id'].nunique() == len(df), \
            f'{ex}: gesture_ids are not unique!'

        # ---- index CSV ----
        META_COLS = ['gesture_id', 'chain', 'generation', 'target', 'director',
                     'condition', 'participant', 'ent_type', 'code_string',
                     'code_len', 'num_reps', 'vid_len', 'phase', 'acc']
        META_COLS = [c for c in META_COLS if c in df.columns]
        idx_path = os.path.join(OUT_DIR, f'{ex}_gesture_index.csv')
        df[META_COLS].to_csv(idx_path, index=True, index_label='matrix_idx')
        print(f'  Saved {idx_path}  ({len(df)} rows)')

        # ---- distance matrices ----
        lev, jacc = compute_matrices(ex, df)
        archive[f'{ex}_levenshtein'] = lev
        archive[f'{ex}_jaccard']     = jacc

        print(f'  Levenshtein: min={np.nanmin(lev):.0f}  '
              f'max={np.nanmax(lev):.0f}  mean={np.nanmean(lev):.2f}')
        print(f'  Jaccard:     min={np.nanmin(jacc):.3f}  '
              f'max={np.nanmax(jacc):.3f}  mean={np.nanmean(jacc):.3f}')

    # ---- save compressed archive ----
    npz_path = os.path.join(OUT_DIR, 'annotation_distances.npz')
    print(f'\nSaving compressed archive to {npz_path} ...')
    np.savez_compressed(npz_path, **archive)
    size_mb = os.path.getsize(npz_path) / 1e6
    print(f'Saved  {npz_path}  ({size_mb:.1f} MB)')
    print('\nTo load:')
    print("  import numpy as np, pandas as pd")
    print("  d   = np.load('processed/annotation_distances.npz')")
    print("  idx = pd.read_csv('processed/ex1_gesture_index.csv', index_col='matrix_idx')")
    print("  lev  = d['ex1_levenshtein']  # (1320, 1320) float32")
    print("  jacc = d['ex1_jaccard']      # (1320, 1320) float32")
    print("  # look up distance for two gesture_ids:")
    print("  i = idx.index[idx['gesture_id'] == 'ex1_ch1_g0_chef_B_TI'][0]")
    print("  j = idx.index[idx['gesture_id'] == 'ex1_ch1_g1_chef_B_TI'][0]")
    print("  print(lev[i, j], jacc[i, j])")


if __name__ == '__main__':
    main()
