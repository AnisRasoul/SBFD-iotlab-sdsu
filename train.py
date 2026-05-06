# =====================
# Fall Detection — Training Script (SBFD-iotlab-sdsu)
# =====================
# This script is the FULL-IMU companion to repo 1
# (Sensor-Based-Fall-detection). It trains the same three models on the
# same SDSU IoTLab Globecom-2019 dataset, but uses every IMU placement
# (16 sensors × 9 raw channels + 16 derived acc_mag = 160 channels)
# instead of head-only.
#
# Compared to repo 1's train.py the ONLY thing that changes is the
# dataset layer (Steps 1–4 below):
#   * DATA_DIR points at data/processed/separate_trials/trial_XX/trial_XX_full.csv
#   * SENSOR_COLS are built from a PLACEMENTS table that covers all 16
#     IMUs with friendly names (head_acc_x, lthigh_rot_y, ...).
#   * acc_mag is derived for every placement, not just the head.
# The model architectures, training logic, evaluation logic, threshold
# calibration, hard-negative mining, event-level metrics, scaler/JSON/H
# exports and TFLite/micromlgen export are preserved verbatim from
# repo 1 so the metrics produced here are directly comparable.

import os
import glob
import json
import warnings
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for CI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, fbeta_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# For model export (optional – used only in Step 10; each call is already
# wrapped in try/except so missing packages are handled gracefully)
try:
    from micromlgen import port as mlport
except ImportError:
    mlport = None

try:
    import m2cgen as m2c
except ImportError:
    m2c = None

try:
    import emlearn
except ImportError:
    emlearn = None


warnings.filterwarnings("ignore")
print(f"TensorFlow {tf.__version__} | NumPy {np.__version__}")

# ── Paths ────────────────────────────────────────────────────────────────────
# Per-trial CSVs produced by scripts/process_falls.py (147 cols each).
DATA_DIR = "data/processed/separate_trials"
OUTPUTS_DIR = "."          # write outputs to the repo root for CML to pick up

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 100          # Hz
WINDOW_SIZE = 100          # 1 s at 100 Hz
STEP_SIZE   = 50           # 50 % overlap
THRESHOLD   = 0.55         # base threshold (precision/recall trade-off)
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FALL_DURATION_SECS = 15   # cap unclosed fall segments at this many seconds
FALL_START_MARKERS = {'startOfFall', 'FallStart'}
FALL_END_MARKERS   = {'endOfFall',   'FallEnd'}

# ── Placement table ──────────────────────────────────────────────────────────
# 16 IMU sensor placements in the SDSU body suit. Used to build:
#   1. The CSV column rename map (Step 2).
#   2. SENSOR_COLS, the list of channels fed into the windowed feature
#      extractor (Step 4).
#
# Each row is (clean_prefix, csv_base_name, side_token).
#   * clean_prefix → underscore-style column name used everywhere downstream.
#   * csv_base_name → matches the SDSU header tokens for course/pitch/roll
#     and Accel Sensor X|Y|Z columns.
#   * side_token → " LT" / " RT" / "" — appended to course/pitch/roll and
#     accel column names. The Rot columns use this same token as a *prefix*
#     ("LT Upper arm Rot X,") instead of a suffix, so the build code
#     handles that asymmetry explicitly.
_PLACEMENTS = [
    ("head",     "Head",        ""),
    ("uspine",   "Upper spine", ""),
    ("lspine",   "Lower spine", ""),
    ("pelvis",   "Pelvis",      ""),
    ("larm_ua",  "Upper arm",   " LT"),
    ("rarm_ua",  "Upper arm",   " RT"),
    ("larm_fa",  "Forearm",     " LT"),
    ("rarm_fa",  "Forearm",     " RT"),
    ("lhand",    "Hand",        " LT"),
    ("rhand",    "Hand",        " RT"),
    ("lthigh",   "Thigh",       " LT"),
    ("rthigh",   "Thigh",       " RT"),
    ("lshank",   "Shank",       " LT"),
    ("rshank",   "Shank",       " RT"),
    ("lfoot",    "Foot",        " LT"),
    ("rfoot",    "Foot",        " RT"),
]

# Per-placement raw channel suffixes (excluding the derived acc_mag).
_RAW_PER_PLACEMENT = ["yaw", "pitch", "roll",
                      "acc_x", "acc_y", "acc_z",
                      "rot_x", "rot_y", "rot_z"]


def _build_col_map():
    """Build the SDSU-CSV → clean-name rename dict from _PLACEMENTS."""
    m = {"Time,s": "time"}
    for clean, base, side in _PLACEMENTS:
        # course/pitch/roll  e.g. "Upper arm course LT,deg" → "larm_ua_yaw"
        m[f"{base} course{side},deg"] = f"{clean}_yaw"
        m[f"{base} pitch{side},deg"]  = f"{clean}_pitch"
        m[f"{base} roll{side},deg"]   = f"{clean}_roll"
        # accel  e.g. "Upper arm Accel Sensor X LT,mG" → "larm_ua_acc_x"
        for axis in ("X", "Y", "Z"):
            m[f"{base} Accel Sensor {axis}{side},mG"] = f"{clean}_acc_{axis.lower()}"
        # rot — LT/RT is a *prefix* in SDSU rotation column names
        side_prefix = side.strip() + " " if side else ""  # "LT " / "RT " / ""
        for axis in ("X", "Y", "Z"):
            m[f"{side_prefix}{base} Rot {axis},"] = f"{clean}_rot_{axis.lower()}"
    return m


COL_MAP = _build_col_map()
# 1 (time) + 16 placements × 9 raw channels = 145 source CSV columns;
# Markers / MarkerNames pass through without a rename.
assert len(COL_MAP) == 1 + 16 * 9, f"unexpected COL_MAP size: {len(COL_MAP)}"

# The "raw" sensor channels (no derived acc_mag) — used for numeric coercion
# and NaN-drop. Order matches per-placement, then per-channel.
RAW_SENSOR_COLS = [f"{clean}_{ch}"
                   for clean, _, _ in _PLACEMENTS
                   for ch in _RAW_PER_PLACEMENT]

# Full feature set (default): 144 raw channels + 16 derived acc_mag = 160.
INITIAL_SENSOR_COLS = [
    f"{clean}_{ch}"
    for clean, _, _ in _PLACEMENTS
    for ch in _RAW_PER_PLACEMENT + ["acc_mag"]
]

# Restricted: 6 head channels (mirrors repo 1's RESTRICTED_SENSOR_COLS so
# ablation runs in this repo are comparable to ablation runs in repo 1).
RESTRICTED_SENSOR_COLS = [
    "head_acc_x", "head_acc_y", "head_acc_z",
    "head_pitch", "head_roll", "head_yaw",
]

# Head-only: exact 10 channels used by repo 1 — useful for sanity-checking
# that repo-2 reproduces repo-1 numbers when restricted to the same data.
HEAD_ONLY_SENSOR_COLS = [
    "head_acc_x", "head_acc_y", "head_acc_z",
    "head_rot_x", "head_rot_y", "head_rot_z",
    "head_pitch", "head_roll", "head_yaw", "head_acc_mag",
]


def _env_int(name, default, min_value=None):
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(f"Invalid integer for {name}: '{raw}'") from e
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def _env_float(name, default, min_value=None, max_value=None):
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid float for {name}: '{raw}'") from e
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    return value


FEATURE_SET = os.getenv("FEATURE_SET", "initial").strip().lower()
SPLIT_STRATEGY = os.getenv("SPLIT_STRATEGY", "group").strip().lower()
RF_TOP_K = _env_int("RF_TOP_K", 40, min_value=1)
PATTERN_CLUSTERS = _env_int("PATTERN_CLUSTERS", 4, min_value=2)
THRESHOLD = _env_float("THRESHOLD", THRESHOLD, min_value=0.0, max_value=1.0)
CALIBRATION_SIZE = _env_float("CALIBRATION_SIZE", 0.20, min_value=0.05, max_value=0.50)
THRESHOLD_BETA = _env_float("THRESHOLD_BETA", 0.5, min_value=0.1, max_value=2.0)
TARGET_PRECISION = _env_float("TARGET_PRECISION", 0.40, min_value=0.0, max_value=1.0)
MIN_RECALL_AT_TARGET = _env_float("MIN_RECALL_AT_TARGET", 0.20, min_value=0.0, max_value=1.0)


def _env_bool(name, default):
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in {"true", "1", "yes", "on"}


EXTENDED_FEATURES = _env_bool("EXTENDED_FEATURES", True)
HARD_NEG_MINING = _env_bool("HARD_NEG_MINING", True)
HARD_NEG_REPEATS = _env_int("HARD_NEG_REPEATS", 3, min_value=1)
EVENT_MIN_CONSEC = _env_int("EVENT_MIN_CONSEC", 2, min_value=1)
EVENT_COOLDOWN_WINDOWS = _env_int("EVENT_COOLDOWN_WINDOWS", 5, min_value=0)
EXPORT_TFLITE = _env_bool("EXPORT_TFLITE", True)

if FEATURE_SET == "initial":
    SENSOR_COLS = list(INITIAL_SENSOR_COLS)
elif FEATURE_SET == "restricted":
    SENSOR_COLS = list(RESTRICTED_SENSOR_COLS)
elif FEATURE_SET == "head_only":
    SENSOR_COLS = list(HEAD_ONLY_SENSOR_COLS)
elif FEATURE_SET == "rf_importance":
    # Feature-importance selection is applied later to the engineered window features.
    SENSOR_COLS = list(INITIAL_SENSOR_COLS)
else:
    raise ValueError(
        f"Unsupported FEATURE_SET='{FEATURE_SET}'. "
        "Use one of: initial, restricted, head_only, rf_importance."
    )

if SPLIT_STRATEGY not in {"group", "temporal", "pattern", "random"}:
    raise ValueError(
        f"Unsupported SPLIT_STRATEGY='{SPLIT_STRATEGY}'. "
        "Use one of: group, temporal, pattern, random."
    )

N_FEATURES = len(SENSOR_COLS)
print(
    "Experiment config:"
    f" feature_set={FEATURE_SET}, split_strategy={SPLIT_STRATEGY},"
    f" rf_top_k={RF_TOP_K}, pattern_clusters={PATTERN_CLUSTERS},"
    f" base_threshold={THRESHOLD:.3f}, calibration_size={CALIBRATION_SIZE:.2f},"
    f" threshold_beta={THRESHOLD_BETA:.2f}, target_precision={TARGET_PRECISION:.2f},"
    f" min_recall_at_target={MIN_RECALL_AT_TARGET:.2f},"
    f" extended_features={EXTENDED_FEATURES}, hard_neg_mining={HARD_NEG_MINING},"
    f" event_min_consec={EVENT_MIN_CONSEC}, event_cooldown={EVENT_COOLDOWN_WINDOWS},"
    f" export_tflite={EXPORT_TFLITE}"
)
print(f"Sensor channels ({N_FEATURES}): "
      f"{SENSOR_COLS if N_FEATURES <= 12 else SENSOR_COLS[:6] + ['...'] + SENSOR_COLS[-6:]}")

# ── Step 1: Load processed trial CSVs ────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "trial_*", "*_full.csv")))
if not csv_files:
    raise FileNotFoundError(
        f"No *_full.csv files found under {DATA_DIR}. "
        "Did you run scripts/process_falls.py?"
    )

all_data = []
for fp in csv_files:
    df = pd.read_csv(fp, low_memory=False)
    # Use trial name from folder instead of filename for clarity
    df["trial_id"] = os.path.basename(os.path.dirname(fp))
    if 'MarkerNames' not in df.columns:
        print(f"  ⚠️  WARNING: {os.path.basename(fp)} has no MarkerNames column — all rows treated as non-fall")
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
print(f"Loaded {len(csv_files)} trial files  →  {len(data):,} total rows")
print(f"  Detected {data['trial_id'].nunique()} trials")


# ── Step 2: Rename & clean columns ───────────────────────────────────────────
print("\nSTEP 2: Cleaning data and deriving features")
data.rename(columns=COL_MAP, inplace=True)

# Coerce all expected raw IMU channels (and time) to numeric; drop rows with
# any NaN in that block (matches repo 1's strict policy).
present_raw = [c for c in (["time"] + RAW_SENSOR_COLS) if c in data.columns]
missing_raw = [c for c in (["time"] + RAW_SENSOR_COLS) if c not in data.columns]
if missing_raw:
    print(f"  ⚠️  WARNING: {len(missing_raw)} expected channels missing after rename "
          f"(first 3: {missing_raw[:3]})")

for c in present_raw:
    data[c] = pd.to_numeric(data[c], errors='coerce')

before = len(data)
data.dropna(subset=present_raw, inplace=True)
data.reset_index(drop=True, inplace=True)
print(f"  After NaN drop: {data.shape[0]:,} rows (dropped {before - data.shape[0]:,})")

# ---- derived features: per-placement acceleration magnitude ----
# acc_mag for placement P is sqrt(P_acc_x² + P_acc_y² + P_acc_z²). One per
# IMU → 16 derived channels added to the dataframe. Whether they are
# *used* downstream depends on FEATURE_SET / SENSOR_COLS.
# Build all 16 derived channels in one pd.concat to avoid the
# per-insert fragmentation warning pandas emits when 16 columns are
# appended individually.
_acc_mag_cols = {}
for clean, _, _ in _PLACEMENTS:
    ax = f"{clean}_acc_x"; ay = f"{clean}_acc_y"; az = f"{clean}_acc_z"
    if ax in data.columns and ay in data.columns and az in data.columns:
        _acc_mag_cols[f"{clean}_acc_mag"] = np.sqrt(
            data[ax].to_numpy() ** 2 + data[ay].to_numpy() ** 2 + data[az].to_numpy() ** 2
        )
if _acc_mag_cols:
    data = pd.concat([data, pd.DataFrame(_acc_mag_cols, index=data.index)], axis=1)


# ── Step 3: Label falls using MarkerNames ────────────────────────────────────
print("\nSTEP 3: Labelling falls from MarkerNames")
# Ensure the column exists even if some trials lacked it (filled with NaN by concat)
if 'MarkerNames' not in data.columns:
    data['MarkerNames'] = ''
data['MarkerNames'] = data['MarkerNames'].fillna('').astype(str).str.strip()
data['label'] = 0

MAX_FALL_SAMPLES = int(MAX_FALL_DURATION_SECS * SAMPLE_RATE)

def _label_trial(names):
    """State-machine labeler that handles marker aliases, double starts, and unclosed falls.

    - Recognises both 'startOfFall'/'FallStart' and 'endOfFall'/'FallEnd'.
    - When two consecutive start markers appear without an intervening end (e.g.
      trial_01 at t=178 s), the first segment is capped at MAX_FALL_SAMPLES and a
      warning is emitted instead of mislabelling the rest of the trial as a fall.
    - Any fall still open at the end of the trial is likewise capped and flagged.
    """
    labels = np.zeros(len(names), dtype=np.int8)
    in_fall = False
    fall_start_i = None
    msgs = []

    for i, name in enumerate(names):
        if name in FALL_START_MARKERS:
            if in_fall:
                cap = min(fall_start_i + MAX_FALL_SAMPLES, i)
                labels[fall_start_i:cap] = 1
                msgs.append(
                    f"double fall-start at row {i} (prev at {fall_start_i}); "
                    f"capped previous segment at {cap - fall_start_i} samples"
                )
            in_fall = True
            fall_start_i = i
        elif name in FALL_END_MARKERS:
            if in_fall:
                labels[fall_start_i : i + 1] = 1
                in_fall = False
                fall_start_i = None
            # else: spurious end marker — ignore

    # Unclosed fall at end of trial
    if in_fall and fall_start_i is not None:
        cap = min(fall_start_i + MAX_FALL_SAMPLES, len(names))
        labels[fall_start_i:cap] = 1
        msgs.append(
            f"unclosed fall at row {fall_start_i}; "
            f"capped at {cap - fall_start_i} samples "
            f"({(cap - fall_start_i) / SAMPLE_RATE:.1f} s)"
        )

    return labels, msgs

for tid, g in data.groupby('trial_id', sort=False):
    idx = g.index.to_numpy()
    names = g['MarkerNames'].to_numpy()
    labels, msgs = _label_trial(names)
    for m in msgs:
        print(f"  [{tid}] {m}")
    data.loc[idx, 'label'] = labels

fall_n   = int(data["label"].sum())
normal_n = len(data) - fall_n
print(f"  Fall samples  : {fall_n:>8,} ({fall_n / len(data) * 100:.1f}%)")
print(f"  Normal samples: {normal_n:>8,} ({normal_n / len(data) * 100:.1f}%)")
non_empty_markers = data[data['MarkerNames'] != '']['MarkerNames']
print(f"  Marker events : {non_empty_markers.value_counts().to_dict()}")


# ── Step 4: Sliding-window feature extraction ─────────────────────────────────
print("\nSTEP 4: Extracting sliding-window features")

_BASE_STATS = ['mean', 'std', 'min', 'max', 'range', 'peak', 'rms', 'delta']
_EXT_STATS = ['zero_cross', 'spec_energy', 'first_diff_rms']
STAT_NAMES = _BASE_STATS + (_EXT_STATS if EXTENDED_FEATURES else [])
feature_names = [f'{col}_{stat}' for col in SENSOR_COLS for stat in STAT_NAMES]
print(f"  Feature vector size : {len(feature_names)} (extended_features={EXTENDED_FEATURES})")
print(f"  Sequence shape      : ({WINDOW_SIZE}, {N_FEATURES})")

def _stats(v):
    base = [v.mean(), v.std(), v.min(), v.max(),
            v.max()-v.min(), np.abs(v).max(),
            np.sqrt((v**2).mean()), v[-1]-v[0]]
    if not EXTENDED_FEATURES:
        return base
    n = len(v)
    centered = v - v.mean()
    # Zero-crossing rate of the centered signal (captures oscillation),
    # normalised by number of intervals.
    zc = float(np.sum(np.diff(np.sign(centered)) != 0)) / max(1, n - 1)
    # Spectral energy: sum of squared FFT magnitudes excluding DC, normalised.
    fft_vals = np.abs(np.fft.rfft(centered)) ** 2
    spec_energy = float(fft_vals[1:].sum()) / max(1, n)
    # RMS of first differences = jerk proxy for accel channels,
    # angular acceleration proxy for rotation channels.
    diff_v = np.diff(v)
    first_diff_rms = float(np.sqrt((diff_v ** 2).mean())) if len(diff_v) > 0 else 0.0
    return base + [zc, spec_energy, first_diff_rms]

def extract_windows(trial_df):
    """Return feature matrix X_f, sequence tensor X_s, labels y, and window starts."""
    arr   = trial_df[SENSOR_COLS].values.astype(np.float32)
    labs  = trial_df['label'].values
    n     = len(arr)

    X_f, X_s, y, starts = [], [], [], []
    for s in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        e    = s + WINDOW_SIZE
        win  = arr[s:e]
        lab  = int(labs[s:e].mean() >= 0.5) # majority vote

        feats = []
        for ci in range(N_FEATURES):
            feats.extend(_stats(win[:, ci]))

        X_f.append(feats)
        X_s.append(win)
        y.append(lab)
        starts.append(s)

    return (
        np.array(X_f, np.float32),
        np.array(X_s, np.float32),
        np.array(y, np.int32),
        np.array(starts, np.int32),
    )

X_feat_list, X_seq_list, y_list, group_list = [], [], [], []
window_trial_order, window_start_idx = [], []
trial_signatures = {}
for trial_order, (tid, g) in enumerate(data.groupby('trial_id', sort=False)):
    g = g.reset_index(drop=True)
    if len(g) < WINDOW_SIZE:
        continue
    xf, xs, yy, starts = extract_windows(g)
    X_feat_list.append(xf)
    X_seq_list.append(xs)
    y_list.append(yy)
    group_list.extend([tid] * len(yy))
    window_trial_order.extend([trial_order] * len(yy))
    window_start_idx.extend(starts.tolist())

    trial_arr = g[SENSOR_COLS].to_numpy(dtype=np.float32)
    trial_sig = np.concatenate([
        trial_arr.mean(axis=0),
        trial_arr.std(axis=0),
        np.percentile(trial_arr, [10, 50, 90], axis=0).reshape(-1),
    ])
    trial_signatures[tid] = trial_sig.astype(np.float32)

X_feat  = np.concatenate(X_feat_list, axis=0)
X_seq   = np.concatenate(X_seq_list,  axis=0)
y       = np.concatenate(y_list,      axis=0)
groups  = np.array(group_list)
window_trial_order = np.array(window_trial_order, dtype=np.int32)
window_start_idx = np.array(window_start_idx, dtype=np.int32)

print(f"\n  Feature matrix : {X_feat.shape}")
print(f"  Sequence tensor: {X_seq.shape}")
print(f"  Labels         : {np.bincount(y)}  (normal={np.bincount(y)[0]}, fall={np.bincount(y)[1]})")
print(f"  Groups (trials): {len(np.unique(groups))} trials")


# ── Step 5: Prepare train / test splits ──────────────────────────────────────
print(f"\nSTEP 5: Preparing train/test splits ({SPLIT_STRATEGY}), scaling, and SMOTE")

if SPLIT_STRATEGY == "group":
    # Whole trials go entirely to train or test (no subject leakage).
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_feat, y, groups=groups))
elif SPLIT_STRATEGY == "temporal":
    # Chronological split over extracted windows:
    # first (1 - TEST_SIZE) fraction for training, last TEST_SIZE for testing.
    order = np.lexsort((window_start_idx, window_trial_order))
    split_at = int((1.0 - TEST_SIZE) * len(order))
    split_at = max(1, min(split_at, len(order) - 1))
    train_idx = order[:split_at]
    test_idx = order[split_at:]
elif SPLIT_STRATEGY == "pattern":
    # Cluster trials by movement signatures, then stratify train/test by cluster.
    trial_ids = np.array(sorted(np.unique(groups)))
    sig_matrix = np.vstack([trial_signatures[tid] for tid in trial_ids])
    n_clusters = min(PATTERN_CLUSTERS, len(trial_ids))

    if n_clusters < 2:
        raise RuntimeError("Pattern-based split requires at least 2 trials.")

    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = km.fit_predict(sig_matrix)

    cluster_counts = np.bincount(cluster_labels, minlength=n_clusters)
    if np.any(cluster_counts < 2):
        print("  Pattern split fallback: some clusters have <2 trials; using group split.")
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_idx, test_idx = next(gss.split(X_feat, y, groups=groups))
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        tr_trials_i, te_trials_i = next(sss.split(np.zeros(len(trial_ids)), cluster_labels))
        tr_trials = trial_ids[tr_trials_i]
        te_trials = trial_ids[te_trials_i]
        train_idx = np.where(np.isin(groups, tr_trials))[0]
        test_idx = np.where(np.isin(groups, te_trials))[0]
        print(f"  Pattern clusters: {dict(zip(trial_ids, cluster_labels))}")
elif SPLIT_STRATEGY == "random":
    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
else:
    raise RuntimeError(f"Unexpected split strategy: {SPLIT_STRATEGY}")

X_feat_tr, X_feat_te = X_feat[train_idx], X_feat[test_idx]
X_seq_tr,  X_seq_te  = X_seq[train_idx],  X_seq[test_idx]
y_tr,      y_te       = y[train_idx],      y[test_idx]
test_trial_order  = window_trial_order[test_idx]
test_window_start = window_start_idx[test_idx]

print(f"  Train windows : {len(y_tr):,}  (falls: {y_tr.sum()})  — {len(np.unique(groups[train_idx]))} trials: {np.unique(groups[train_idx])}")
print(f"  Test  windows : {len(y_te):,}  (falls: {y_te.sum()})  — {len(np.unique(groups[test_idx]))} trials: {np.unique(groups[test_idx])}")

if y_te.sum() == 0:
    raise RuntimeError("Test set has 0 fall windows — try a different RANDOM_STATE.")

# Hold out a calibration split from train for threshold tuning (no test leakage).
fit_idx = np.arange(len(y_tr), dtype=np.int32)
cal_idx = np.array([], dtype=np.int32)
if np.unique(y_tr).size == 2:
    try:
        sss_cal = StratifiedShuffleSplit(n_splits=1, test_size=CALIBRATION_SIZE, random_state=RANDOM_STATE)
        fit_idx, cal_idx = next(sss_cal.split(np.zeros(len(y_tr)), y_tr))
    except ValueError:
        cal_idx = np.array([], dtype=np.int32)

X_feat_fit, y_fit = X_feat_tr[fit_idx], y_tr[fit_idx]
X_seq_fit = X_seq_tr[fit_idx]

if int(y_fit.sum()) < 2:
    fit_idx = np.arange(len(y_tr), dtype=np.int32)
    cal_idx = np.array([], dtype=np.int32)
    X_feat_fit, y_fit = X_feat_tr, y_tr
    X_seq_fit = X_seq_tr
    print("  Calibration fallback: insufficient fall samples after split; using full train set.")

if len(cal_idx) > 0:
    X_feat_cal, y_cal = X_feat_tr[cal_idx], y_tr[cal_idx]
    X_seq_cal = X_seq_tr[cal_idx]
    print(f"  Calibration windows: {len(y_cal):,} (falls: {int(y_cal.sum())})")
else:
    X_feat_cal = np.empty((0, X_feat_tr.shape[1]), dtype=np.float32)
    y_cal = np.empty((0,), dtype=np.int32)
    X_seq_cal = np.empty((0, X_seq_tr.shape[1], X_seq_tr.shape[2]), dtype=np.float32)
    print("  Calibration windows: 0 (threshold tuning fallback to base threshold)")

# ---- feature scaler (for RF + NN) ----
feat_scaler = StandardScaler()
X_feat_fit_sc = feat_scaler.fit_transform(X_feat_fit)
X_feat_cal_sc = feat_scaler.transform(X_feat_cal) if len(y_cal) > 0 else np.empty((0, X_feat_fit_sc.shape[1]), dtype=np.float32)
X_feat_te_sc = feat_scaler.transform(X_feat_te)

# ---- sequence scaler (for LSTM) ----
n_tr, W, F = X_seq_fit.shape
seq_scaler  = StandardScaler()
X_seq_fit_sc = seq_scaler.fit_transform(X_seq_fit.reshape(-1, F)).reshape(n_tr, W, F)
X_seq_cal_sc = (
    seq_scaler.transform(X_seq_cal.reshape(-1, F)).reshape(len(y_cal), W, F)
    if len(y_cal) > 0 else np.empty((0, W, F), dtype=np.float32)
)
X_seq_te_sc = seq_scaler.transform(X_seq_te.reshape(-1, F)).reshape(len(y_te), W, F)

# ---- SMOTE on 2-D feature data (RF + NN training) ----
# LSTM uses class_weight on the original imbalanced sequences instead.
print("  Applying SMOTE to feature training set ...")
minority_falls = int(y_fit.sum())
if minority_falls < 2:
    raise RuntimeError("Train set has fewer than 2 fall windows — cannot apply SMOTE.")
sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=max(1, min(5, minority_falls - 1)))
X_feat_fit_sm, y_fit_sm = sm.fit_resample(X_feat_fit_sc, y_fit)
print(f"  After SMOTE : {np.bincount(y_fit_sm)}")

selected_feature_idx = np.arange(X_feat_fit_sc.shape[1], dtype=np.int32)
if FEATURE_SET == "rf_importance":
    print(f"  Selecting top-{RF_TOP_K} engineered features with RF importance ...")
    selector_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    selector_rf.fit(X_feat_fit_sm, y_fit_sm)
    top_k = min(RF_TOP_K, X_feat_fit_sm.shape[1])
    selected_feature_idx = np.argsort(selector_rf.feature_importances_)[::-1][:top_k].astype(np.int32)
    X_feat_fit_sc = X_feat_fit_sc[:, selected_feature_idx]
    X_feat_cal_sc = X_feat_cal_sc[:, selected_feature_idx] if len(y_cal) > 0 else X_feat_cal_sc
    X_feat_te_sc = X_feat_te_sc[:, selected_feature_idx]
    X_feat_fit_sm = X_feat_fit_sm[:, selected_feature_idx]
    feature_names = [feature_names[i] for i in selected_feature_idx]
    print(f"  Selected feature count: {len(feature_names)}")

# ---- class weights for LSTM (computed from the original imbalanced distribution) ----
cw_arr  = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_fit)
class_w = {0: float(cw_arr[0]), 1: float(cw_arr[1])}
print(f"  Class weights (LSTM): {class_w}")


# ── Step 6: Train models ──────────────────────────────────────────────────────
print("\nSTEP 6: Training models")
results = {}


def _event_level_metrics(y_true, y_pred, trial_order, window_starts,
                         min_consec=2, cooldown=5):
    """Event-level precision/recall with consecutive-window gating + cooldown.

    Windows are grouped by trial and sorted chronologically. A detection fires
    when `min_consec` consecutive positive predictions occur; after firing,
    predictions are suppressed for `cooldown` windows (same trial only).
    A ground-truth event is a contiguous run of y_true==1 within a trial.
    A detection is a true positive if it lands within [e_start-cooldown,
    e_end+cooldown] of an unmatched event (at most one detection per event).
    Returns dict with event_precision, event_recall, n_detections, n_events, tp.
    """
    if len(y_true) == 0:
        return {"event_precision": 0.0, "event_recall": 0.0,
                "n_detections": 0, "n_events": 0, "tp": 0}

    order = np.lexsort((window_starts, trial_order))
    yt = np.asarray(y_true)[order]
    yp = np.asarray(y_pred)[order]
    tr = np.asarray(trial_order)[order]

    detections = []
    current_trial = -1
    run_len = 0
    cd_left = 0
    for i in range(len(yp)):
        tid = int(tr[i])
        if tid != current_trial:
            current_trial = tid
            run_len = 0
            cd_left = 0
        if cd_left > 0:
            cd_left -= 1
            run_len = 0
            continue
        if yp[i] == 1:
            run_len += 1
            if run_len >= min_consec:
                detections.append((tid, i))
                cd_left = cooldown
                run_len = 0
        else:
            run_len = 0

    events = []
    current_trial = -1
    in_event = False
    ev_start = 0
    for i in range(len(yt)):
        tid = int(tr[i])
        if tid != current_trial:
            if in_event:
                events.append((current_trial, ev_start, i - 1))
                in_event = False
            current_trial = tid
        if yt[i] == 1 and not in_event:
            ev_start = i
            in_event = True
        elif yt[i] == 0 and in_event:
            events.append((current_trial, ev_start, i - 1))
            in_event = False
    if in_event:
        events.append((current_trial, ev_start, len(yt) - 1))

    matched = set()
    tp = 0
    for d_tid, d_i in detections:
        for idx, (e_tid, e_start, e_end) in enumerate(events):
            if idx in matched or e_tid != d_tid:
                continue
            if e_start - cooldown <= d_i <= e_end + cooldown:
                matched.add(idx)
                tp += 1
                break

    n_det = len(detections)
    n_ev = len(events)
    return {
        "event_precision": (tp / n_det) if n_det > 0 else 0.0,
        "event_recall":    (tp / n_ev) if n_ev > 0 else 0.0,
        "n_detections":    n_det,
        "n_events":        n_ev,
        "tp":              tp,
    }


def _best_fbeta_threshold(y_true, proba, beta=1.0):
    """Return the probability threshold that maximises F-beta on the given split."""
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return float(THRESHOLD), 0.0
    b2 = beta ** 2
    denom = (b2 * precision_vals) + recall_vals
    safe_denom = np.where(denom == 0, 1.0, denom)
    fbeta_vals = np.where(denom == 0, 0.0, (1 + b2) * precision_vals * recall_vals / safe_denom)
    best_idx = int(np.argmax(fbeta_vals[:-1]))   # indices that map to thresholds
    assert best_idx < len(thresholds), "best_idx out of range for thresholds array"
    return float(thresholds[best_idx]), float(fbeta_vals[best_idx])


def _select_operating_threshold(y_true, proba, default_thr):
    """Choose a threshold on calibration data with a precision-oriented objective."""
    if proba is None or y_true is None or len(y_true) == 0 or np.unique(y_true).size < 2:
        return float(default_thr), "default", None

    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, proba)
    if len(thresholds) == 0:
        return float(default_thr), "default", None

    b2 = THRESHOLD_BETA ** 2
    denom = (b2 * precision_vals) + recall_vals
    safe_denom = np.where(denom == 0, 1.0, denom)
    fbeta_vals = np.where(denom == 0, 0.0, (1 + b2) * precision_vals * recall_vals / safe_denom)

    p = precision_vals[:-1]
    r = recall_vals[:-1]
    fbeta = fbeta_vals[:-1]
    feasible = np.where((p >= TARGET_PRECISION) & (r >= MIN_RECALL_AT_TARGET))[0]

    if len(feasible) > 0:
        best_idx = int(feasible[np.argmax(fbeta[feasible])])
        source = "precision-constrained"
    else:
        best_idx = int(np.argmax(fbeta))
        source = "best-fbeta"

    return float(thresholds[best_idx]), source, float(fbeta[best_idx])


# --- Model 1: Random Forest ---
print("\n--- Random Forest ---")
_rf_params = dict(
    n_estimators   = 200,
    max_depth      = 15,
    min_samples_leaf = 3,
    max_features   = 'sqrt',
    class_weight   = 'balanced',
    random_state   = RANDOM_STATE,
    n_jobs         = -1,
)
rf = RandomForestClassifier(**_rf_params)
rf.fit(X_feat_fit_sm, y_fit_sm)

# Hard-negative mining: find false positives on the un-resampled fit set and
# oversample them in a retrain pass. Addresses the "near-fall non-fall" errors
# flagged in the improvement roadmap.
if HARD_NEG_MINING:
    _rf_fit_proba = rf.predict_proba(X_feat_fit_sc)[:, 1]
    _fp_mask = (_rf_fit_proba >= THRESHOLD) & (y_fit == 0)
    _n_fps = int(_fp_mask.sum())
    if _n_fps > 0:
        hard_X = np.repeat(X_feat_fit_sc[_fp_mask], HARD_NEG_REPEATS, axis=0)
        hard_y = np.zeros(len(hard_X), dtype=y_fit_sm.dtype)
        X_aug = np.concatenate([X_feat_fit_sm, hard_X])
        y_aug = np.concatenate([y_fit_sm, hard_y])
        rf = RandomForestClassifier(**_rf_params)
        rf.fit(X_aug, y_aug)
        print(f"  Hard-negative mining: upweighted {_n_fps} FPs ×{HARD_NEG_REPEATS} and retrained.")
    else:
        print("  Hard-negative mining: no FPs in fit set at base threshold; kept original RF.")

rf_cal_proba = rf.predict_proba(X_feat_cal_sc)[:, 1] if len(y_cal) > 0 else None
rf_thr, rf_thr_src, rf_thr_fbeta = _select_operating_threshold(y_cal, rf_cal_proba, THRESHOLD)
rf_proba = rf.predict_proba(X_feat_te_sc)[:, 1]
rf_pred  = (rf_proba >= rf_thr).astype(int)
_rf_best_thr, _rf_best_f1 = _best_fbeta_threshold(y_te, rf_proba, beta=1.0)
_rf_event = _event_level_metrics(y_te, rf_pred, test_trial_order, test_window_start,
                                 min_consec=EVENT_MIN_CONSEC, cooldown=EVENT_COOLDOWN_WINDOWS)
results["Random Forest"] = {
    "proba": rf_proba, "pred": rf_pred,
    "acc": accuracy_score(y_te, rf_pred), "f1": f1_score(y_te, rf_pred),
    "f0_5": fbeta_score(y_te, rf_pred, beta=0.5),
    "recall": recall_score(y_te, rf_pred), "precision": precision_score(y_te, rf_pred),
    "roc_auc": roc_auc_score(y_te, rf_proba),
    "selected_threshold": rf_thr, "threshold_source": rf_thr_src, "threshold_fbeta": rf_thr_fbeta,
    "best_f1_threshold": _rf_best_thr, "best_f1": _rf_best_f1,
    "event_precision": _rf_event["event_precision"], "event_recall": _rf_event["event_recall"],
    "n_detections": _rf_event["n_detections"], "n_events": _rf_event["n_events"],
    "event_tp": _rf_event["tp"],
}
print(f"  Precision {results['Random Forest']['precision']:.4f}, Recall {results['Random Forest']['recall']:.4f}, F1 {results['Random Forest']['f1']:.4f}, ROC-AUC {results['Random Forest']['roc_auc']:.4f}")
print(f"  (Selected threshold: {rf_thr:.3f}, source={rf_thr_src})")
print(f"  (Best-F1 threshold on test set: {_rf_best_thr:.3f} → F1 {_rf_best_f1:.4f})")
print(f"  Event-level (min_consec={EVENT_MIN_CONSEC}, cooldown={EVENT_COOLDOWN_WINDOWS}):"
      f" P={_rf_event['event_precision']:.3f}, R={_rf_event['event_recall']:.3f},"
      f" det={_rf_event['n_detections']}, events={_rf_event['n_events']}, TP={_rf_event['tp']}")

# MCU-friendly Random Forest (small enough for AVR / STM32 Flash ≤ 256 KB)
print("\n--- MCU Random Forest (microcontroller export) ---")
rf_mcu = RandomForestClassifier(
    n_estimators   = 15,
    max_depth      = 8,
    min_samples_leaf = 5,
    max_features   = 'sqrt',
    class_weight   = 'balanced',
    random_state   = RANDOM_STATE,
    n_jobs         = -1,
)
rf_mcu.fit(X_feat_fit_sm, y_fit_sm)
rf_mcu_proba = rf_mcu.predict_proba(X_feat_te_sc)[:, 1]
rf_mcu_pred  = (rf_mcu_proba >= rf_thr).astype(int)
print(f"  Recall {recall_score(y_te, rf_mcu_pred):.4f},  F1 {f1_score(y_te, rf_mcu_pred):.4f}"
      f"  (15 trees × depth-8, suitable for ≤256 KB Flash)")


# --- Model 2: Simple Neural Network (MLP) ---
print("\n--- Simple Neural Network (MLP) ---")
nn_model = keras.Sequential([
    layers.Input(shape=(X_feat_fit_sm.shape[1],)),
    layers.Dense(128, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.30),
    layers.Dense(64, activation='relu'),  layers.BatchNormalization(), layers.Dropout(0.30),
    layers.Dense(32, activation='relu'),  layers.Dropout(0.20),
    layers.Dense(1, activation='sigmoid'),
], name='simple_nn')
nn_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

nn_callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
]
nn_history = nn_model.fit(
    X_feat_fit_sm, y_fit_sm,
    epochs=150, batch_size=64, validation_split=0.15,
    callbacks=nn_callbacks, verbose=0,
)
nn_cal_proba = nn_model.predict(X_feat_cal_sc, verbose=0).flatten() if len(y_cal) > 0 else None
nn_thr, nn_thr_src, nn_thr_fbeta = _select_operating_threshold(y_cal, nn_cal_proba, THRESHOLD)
nn_proba = nn_model.predict(X_feat_te_sc, verbose=0).flatten()
nn_pred = (nn_proba >= nn_thr).astype(int)
_nn_best_thr, _nn_best_f1 = _best_fbeta_threshold(y_te, nn_proba, beta=1.0)
_nn_event = _event_level_metrics(y_te, nn_pred, test_trial_order, test_window_start,
                                 min_consec=EVENT_MIN_CONSEC, cooldown=EVENT_COOLDOWN_WINDOWS)
results["Simple NN"] = {
    "proba": nn_proba, "pred": nn_pred, "history": nn_history,
    "acc": accuracy_score(y_te, nn_pred), "f1": f1_score(y_te, nn_pred),
    "f0_5": fbeta_score(y_te, nn_pred, beta=0.5),
    "recall": recall_score(y_te, nn_pred), "precision": precision_score(y_te, nn_pred),
    "roc_auc": roc_auc_score(y_te, nn_proba),
    "selected_threshold": nn_thr, "threshold_source": nn_thr_src, "threshold_fbeta": nn_thr_fbeta,
    "best_f1_threshold": _nn_best_thr, "best_f1": _nn_best_f1,
    "event_precision": _nn_event["event_precision"], "event_recall": _nn_event["event_recall"],
    "n_detections": _nn_event["n_detections"], "n_events": _nn_event["n_events"],
    "event_tp": _nn_event["tp"],
}
print(f"  Precision {results['Simple NN']['precision']:.4f}, Recall {results['Simple NN']['recall']:.4f}, F1 {results['Simple NN']['f1']:.4f}, ROC-AUC {results['Simple NN']['roc_auc']:.4f}")
print(f"  (Selected threshold: {nn_thr:.3f}, source={nn_thr_src})")
print(f"  (Best-F1 threshold on test set: {_nn_best_thr:.3f} → F1 {_nn_best_f1:.4f})")
print(f"  Event-level: P={_nn_event['event_precision']:.3f}, R={_nn_event['event_recall']:.3f},"
      f" det={_nn_event['n_detections']}, events={_nn_event['n_events']}, TP={_nn_event['tp']}")


# --- Model 3: LSTM ---
print("\n--- LSTM ---")
lstm_model = keras.Sequential([
    layers.Input(shape=(WINDOW_SIZE, N_FEATURES)),
    layers.LSTM(64, return_sequences=True), layers.Dropout(0.30),
    layers.LSTM(32), layers.Dropout(0.30),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
], name='lstm_model')
lstm_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

lstm_callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=0),
]
lstm_history = lstm_model.fit(
    X_seq_fit_sc, y_fit,
    epochs=100, batch_size=64, validation_split=0.15,
    class_weight=class_w, callbacks=lstm_callbacks, verbose=0,
)
lstm_cal_proba = lstm_model.predict(X_seq_cal_sc, verbose=0).flatten() if len(y_cal) > 0 else None
lstm_thr, lstm_thr_src, lstm_thr_fbeta = _select_operating_threshold(y_cal, lstm_cal_proba, THRESHOLD)
lstm_proba = lstm_model.predict(X_seq_te_sc, verbose=0).flatten()
lstm_pred = (lstm_proba >= lstm_thr).astype(int)
_lstm_best_thr, _lstm_best_f1 = _best_fbeta_threshold(y_te, lstm_proba, beta=1.0)
_lstm_event = _event_level_metrics(y_te, lstm_pred, test_trial_order, test_window_start,
                                   min_consec=EVENT_MIN_CONSEC, cooldown=EVENT_COOLDOWN_WINDOWS)
results["LSTM"] = {
    "proba": lstm_proba, "pred": lstm_pred, "history": lstm_history,
    "acc": accuracy_score(y_te, lstm_pred), "f1": f1_score(y_te, lstm_pred),
    "f0_5": fbeta_score(y_te, lstm_pred, beta=0.5),
    "recall": recall_score(y_te, lstm_pred), "precision": precision_score(y_te, lstm_pred),
    "roc_auc": roc_auc_score(y_te, lstm_proba),
    "selected_threshold": lstm_thr, "threshold_source": lstm_thr_src, "threshold_fbeta": lstm_thr_fbeta,
    "best_f1_threshold": _lstm_best_thr, "best_f1": _lstm_best_f1,
    "event_precision": _lstm_event["event_precision"], "event_recall": _lstm_event["event_recall"],
    "n_detections": _lstm_event["n_detections"], "n_events": _lstm_event["n_events"],
    "event_tp": _lstm_event["tp"],
}
print(f"  Precision {results['LSTM']['precision']:.4f}, Recall {results['LSTM']['recall']:.4f}, F1 {results['LSTM']['f1']:.4f}, ROC-AUC {results['LSTM']['roc_auc']:.4f}")
print(f"  (Selected threshold: {lstm_thr:.3f}, source={lstm_thr_src})")
print(f"  (Best-F1 threshold on test set: {_lstm_best_thr:.3f} → F1 {_lstm_best_f1:.4f})")
print(f"  Event-level: P={_lstm_event['event_precision']:.3f}, R={_lstm_event['event_recall']:.3f},"
      f" det={_lstm_event['n_detections']}, events={_lstm_event['n_events']}, TP={_lstm_event['tp']}")


# ── Step 7: Save metrics.txt ─────────────────────────────────────────────────
print("\nSTEP 7: Saving metrics and reports")
best_name = max(results, key=lambda k: (results[k]["f0_5"], results[k]["f1"], results[k]["precision"]))
best_pred = results[best_name]["pred"]

metrics_path = os.path.join(OUTPUTS_DIR, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write("Fall Detection — Training Results\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"{'Model':<20} {'Accuracy':>10} {'F0.5':>8} {'F1':>8} {'Recall':>8} {'Precision':>10} {'ROC-AUC':>8} {'ThrUsed':>8} {'BestF1Thr':>10}\n")
    f.write("-" * 80 + "\n")
    for name, r in results.items():
        # 'best_f1_threshold' is present for all main models; guard against
        # any future model added without calling _best_fbeta_threshold().
        best_thr_str = f"{r['best_f1_threshold']:.3f}" if r.get('best_f1_threshold') is not None else "N/A"
        used_thr_str = f"{r['selected_threshold']:.3f}" if r.get('selected_threshold') is not None else "N/A"
        f.write(
            f"{name:<20} {r['acc']:>10.4f} {r['f0_5']:>8.4f} {r['f1']:>8.4f} "
            f"{r['recall']:>8.4f} {r['precision']:>10.4f} {r['roc_auc']:>8.4f} {used_thr_str:>8} {best_thr_str:>10}\n"
        )
    f.write(f"\n  Note: BestF1Thr = threshold that maximises F1 on the test set (informational).\n")
    f.write(f"  ThrUsed = threshold selected on calibration split using F{THRESHOLD_BETA:.1f} with precision target.\n")
    f.write(f"  Base threshold fallback : {THRESHOLD}\n")
    f.write(f"  Calibration size        : {CALIBRATION_SIZE}\n")

    f.write(f"\nEvent-level metrics (min_consec={EVENT_MIN_CONSEC}, cooldown={EVENT_COOLDOWN_WINDOWS} windows):\n")
    f.write(f"{'Model':<20} {'EventP':>8} {'EventR':>8} {'Detect':>8} {'Events':>8} {'TP':>6}\n")
    f.write("-" * 72 + "\n")
    for name, r in results.items():
        f.write(
            f"{name:<20} {r.get('event_precision', 0.0):>8.4f} {r.get('event_recall', 0.0):>8.4f} "
            f"{r.get('n_detections', 0):>8d} {r.get('n_events', 0):>8d} {r.get('event_tp', 0):>6d}\n"
        )
    f.write(f"  An event is a contiguous run of fall windows in ground truth.\n")
    f.write(f"  A detection fires after {EVENT_MIN_CONSEC} consecutive positives, then suppresses for {EVENT_COOLDOWN_WINDOWS} windows.\n")
    f.write(f"\n★ Best model (highest F0.5): {best_name}\n\n")
    f.write(f"Detailed report — {best_name}:\n")
    f.write(classification_report(y_te, best_pred, target_names=["Normal", "Fall"]))
    f.write(f"\nTraining configuration:\n")
    f.write(f"  Window size  : {WINDOW_SIZE} samples\n")
    f.write(f"  Step size    : {STEP_SIZE} samples\n")
    f.write(f"  Base threshold    : {THRESHOLD}\n")
    f.write(f"  Threshold beta    : {THRESHOLD_BETA}\n")
    f.write(f"  Target precision  : {TARGET_PRECISION}\n")
    f.write(f"  Min recall target : {MIN_RECALL_AT_TARGET}\n")
    f.write(f"  Feature set  : {FEATURE_SET}\n")
    f.write(f"  Split        : {SPLIT_STRATEGY}\n")
    f.write(f"  SMOTE        : enabled\n")
    f.write(f"  Extended features : {EXTENDED_FEATURES} (stats per channel: {len(STAT_NAMES)})\n")
    f.write(f"  Feature count     : {len(feature_names)}\n")
    f.write(f"  Hard-neg mining   : {HARD_NEG_MINING} (×{HARD_NEG_REPEATS})\n")
    f.write(f"  Event post-proc   : min_consec={EVENT_MIN_CONSEC}, cooldown={EVENT_COOLDOWN_WINDOWS}\n")
    f.write(f"  Total windows: {len(y):,} (Fall: {sum(y):,}, {sum(y)/len(y)*100:.1f}%)\n")

print(f"  Saved {metrics_path}")

# Save metrics as JSON for programmatic tracking across CI runs
metrics_json_path = os.path.join(OUTPUTS_DIR, "metrics.json")
with open(metrics_json_path, "w") as f:
    json.dump(
        {name: {"accuracy": r["acc"], "f0_5": r["f0_5"], "f1": r["f1"], "recall": r["recall"],
                "precision": r["precision"], "roc_auc": r["roc_auc"],
                "selected_threshold": r.get("selected_threshold"),
                "threshold_source": r.get("threshold_source"),
                "threshold_fbeta": r.get("threshold_fbeta"),
                "best_f1_threshold": r.get("best_f1_threshold"),
                "best_f1": r.get("best_f1"),
                "event_precision": r.get("event_precision"),
                "event_recall": r.get("event_recall"),
                "n_detections": r.get("n_detections"),
                "n_events": r.get("n_events"),
                "event_tp": r.get("event_tp"),
                "feature_set": FEATURE_SET,
                "split_strategy": SPLIT_STRATEGY,
                "extended_features": EXTENDED_FEATURES,
                "hard_neg_mining": HARD_NEG_MINING,
                "event_min_consec": EVENT_MIN_CONSEC,
                "event_cooldown_windows": EVENT_COOLDOWN_WINDOWS}
         for name, r in results.items()},
        f, indent=2,
    )
print(f"  Saved {metrics_json_path}")

# ── Step 8: Save plots ───────────────────────────────────────────────────────
print("  Plotting and saving visualizations ...")
# --- Confusion matrices ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_te, r["pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Fall"], yticklabels=["Normal", "Fall"])
    ax.set_title(f"{name}\nAcc={r['acc']:.3f}  Recall={r['recall']:.3f}")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrices.png"), dpi=120)
plt.close()

# --- Training histories ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for row, name in enumerate(['Simple NN', 'LSTM']):
    hist = results[name]['history']
    axes[row, 0].plot(hist.history['loss'], label='Train')
    axes[row, 0].plot(hist.history['val_loss'], label='Val')
    axes[row, 0].set_title(f'{name} — Loss'); axes[row, 0].legend(); axes[row, 0].grid(alpha=0.3)
    axes[row, 1].plot(hist.history['accuracy'], label='Train')
    axes[row, 1].plot(hist.history['val_accuracy'], label='Val')
    axes[row, 1].set_title(f'{name} — Accuracy'); axes[row, 1].legend(); axes[row, 1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "training_history.png"), dpi=120)
plt.close()

# --- Feature importance (RF) ---
feat_imp = pd.Series(rf.feature_importances_, index=feature_names)
plt.figure(figsize=(10, 7))
feat_imp.nlargest(20).sort_values().plot(kind='barh', color='steelblue')
plt.title('Top-20 Feature Importances (Random Forest)'); plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "feature_importance.png"), dpi=120)
plt.close()
print("  Saved confusion_matrices.png, training_history.png, feature_importance.png")


# ── Step 9: Save scalers and models ──────────────────────────────────────────
print("\nSTEP 9: Saving scalers and models for deployment")
DEPLOY_THRESHOLD = float(results["Random Forest"]["selected_threshold"])
selected_feat_mean = feat_scaler.mean_[selected_feature_idx]
selected_feat_scale = feat_scaler.scale_[selected_feature_idx]
# Scaler parameters as JSON (for firmware)
scaler_json = {
    'sample_rate': SAMPLE_RATE, 'window_size': WINDOW_SIZE, 'step_size': STEP_SIZE,
    'threshold': DEPLOY_THRESHOLD, 'sensor_cols': SENSOR_COLS, 'feature_names': feature_names,
    'feature_set': FEATURE_SET, 'split_strategy': SPLIT_STRATEGY,
    'feat_mean': selected_feat_mean.tolist(), 'feat_scale': selected_feat_scale.tolist(),
    'seq_mean': seq_scaler.mean_.tolist(), 'seq_scale': seq_scaler.scale_.tolist(),
}
with open(os.path.join(OUTPUTS_DIR, 'scaler_params.json'), 'w') as f:
    json.dump(scaler_json, f, indent=2)
print("  Saved scaler_params.json")

# Python pickle objects
with open(os.path.join(OUTPUTS_DIR, 'fall_detector_rf.pkl'), 'wb') as f: pickle.dump(rf, f)
with open(os.path.join(OUTPUTS_DIR, 'feat_scaler.pkl'), 'wb') as f: pickle.dump(feat_scaler, f)
with open(os.path.join(OUTPUTS_DIR, 'seq_scaler.pkl'), 'wb') as f: pickle.dump(seq_scaler, f)
print("  Saved fall_detector_rf.pkl, feat_scaler.pkl, seq_scaler.pkl")

# Generate a C header so firmware can use the scaler without JSON parsing.
# acc_mag must be computed on-chip as sqrt(acc_x²+acc_y²+acc_z²) before
# feature extraction.  'peak' = abs(v).max(), 'delta' = v[-1]-v[0].
def _c_array(name, values, dtype='float', per_row=8):
    rows = []
    vals = list(values)
    for i in range(0, len(vals), per_row):
        rows.append('    ' + ', '.join(f'{v:.8f}f' for v in vals[i:i + per_row]))
    return f'static const {dtype} {name}[{len(vals)}] = {{\n' + ',\n'.join(rows) + '\n};\n'

h_lines = [
    '// Auto-generated by train.py — do not edit manually.',
    '// Copy next to your firmware sketch and #include it.',
    '#pragma once',
    '#include <stdint.h>',
    '',
    f'#define FALL_WINDOW_SIZE  {WINDOW_SIZE}    // samples per inference window',
    f'#define FALL_SAMPLE_RATE  {SAMPLE_RATE}    // Hz',
    f'#define FALL_STEP_SIZE    {STEP_SIZE}      // hop size in samples ({100 * (1 - STEP_SIZE / WINDOW_SIZE):.0f} % overlap)',
    f'#define FALL_N_FEATURES   {len(feature_names)} // statistical features per window',
    f'#define FALL_THRESHOLD    {DEPLOY_THRESHOLD}f    // classification threshold (RF tuned)',
    '',
    '// Feature-level scaler: apply AFTER extracting the statistical feature vector.',
    _c_array('FEAT_MEAN',  selected_feat_mean),
    _c_array('FEAT_SCALE', selected_feat_scale),
    '// Sequence-level scaler: apply to raw sensor channels BEFORE LSTM inference.',
    _c_array('SEQ_MEAN',  seq_scaler.mean_),
    _c_array('SEQ_SCALE', seq_scaler.scale_),
]
header_path = os.path.join(OUTPUTS_DIR, 'scaler_params.h')
with open(header_path, 'w') as f:
    f.write('\n'.join(h_lines))
print(f"  Saved {header_path}")


# ── Step 10: Export models for microcontrollers ──────────────────────────────
# rf_mcu (15 trees, depth 8) is exported instead of the full 200-tree RF so the
# resulting C code fits within the Flash of AVR / STM32 class devices (≤256 KB).
# NN / LSTM models are exported as int8-quantised TFLite for mid-range MCUs.
print("\nSTEP 10: Exporting MCU-friendly models")

try:
    c_code = mlport(rf_mcu, classmap={0: 'NORMAL', 1: 'FALL'})
    with open(os.path.join(OUTPUTS_DIR, 'fall_detector_micromlgen.h'), 'w') as f: f.write(c_code)
    print(f"  micromlgen → fall_detector_micromlgen.h ({len(c_code)//1024} KB)")
except Exception as e: print(f"  micromlgen → SKIPPED ({e})")

try:
    c_code_m2c = m2c.export_to_c(rf_mcu)
    with open(os.path.join(OUTPUTS_DIR, 'fall_detector_m2cgen.c'), 'w') as f: f.write(c_code_m2c)
    print(f"  m2cgen     → fall_detector_m2cgen.c ({len(c_code_m2c)//1024} KB)")
except Exception as e: print(f"  m2cgen     → SKIPPED ({e})")

try:
    emlearn_model = emlearn.convert(rf_mcu, method='inline')
    emlearn_model.save(file=os.path.join(OUTPUTS_DIR, 'fall_detector_emlearn.h'))
    size = os.path.getsize(os.path.join(OUTPUTS_DIR, 'fall_detector_emlearn.h')) // 1024
    print(f"  emlearn    → fall_detector_emlearn.h ({size} KB)")
except Exception as e: print(f"  emlearn    → SKIPPED ({e})")


# --- TFLite int8 export (NN and LSTM) ---
# int8 quantisation shrinks weights ~4× vs float32 and is required for most
# sub-Cortex-M7 MCUs. The NN converts cleanly; LSTM ops often need a float
# fallback (TFLITE_BUILTINS) since not every LSTM kernel has an int8 version.
if EXPORT_TFLITE:
    def _rep_nn():
        n_samples = min(100, len(X_feat_fit_sc))
        for i in range(n_samples):
            yield [X_feat_fit_sc[i:i + 1].astype(np.float32)]

    def _rep_lstm():
        n_samples = min(100, len(X_seq_fit_sc))
        for i in range(n_samples):
            yield [X_seq_fit_sc[i:i + 1].astype(np.float32)]

    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(nn_model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = _rep_nn
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
        tfl_nn = conv.convert()
        nn_tfl_path = os.path.join(OUTPUTS_DIR, 'fall_detector_nn_int8.tflite')
        with open(nn_tfl_path, 'wb') as f: f.write(tfl_nn)
        print(f"  TFLite NN int8 → fall_detector_nn_int8.tflite ({len(tfl_nn) // 1024} KB)")
    except Exception as e:
        print(f"  TFLite NN int8 → SKIPPED ({e})")

    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(lstm_model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = _rep_lstm
        # Accept float fallback for ops the int8 spec does not cover (LSTM kernels).
        conv.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        tfl_lstm = conv.convert()
        lstm_tfl_path = os.path.join(OUTPUTS_DIR, 'fall_detector_lstm_int8.tflite')
        with open(lstm_tfl_path, 'wb') as f: f.write(tfl_lstm)
        print(f"  TFLite LSTM → fall_detector_lstm_int8.tflite ({len(tfl_lstm) // 1024} KB)")
    except Exception as e:
        print(f"  TFLite LSTM → SKIPPED ({e})")


# ── Done ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Training complete!")
print(f"  Best model        : {best_name}")
print(f"  F0.5 Score        : {results[best_name]['f0_5']:.4f}")
print(f"  Precision         : {results[best_name]['precision']:.4f}")
print(f"  Recall            : {results[best_name]['recall']:.4f}")
print(f"  F1 Score          : {results[best_name]['f1']:.4f}")
print(f"  ROC-AUC           : {results[best_name]['roc_auc']:.4f}")
if results[best_name].get('selected_threshold') is not None:
    print(f"  Selected threshold: {results[best_name]['selected_threshold']:.3f}"
          f"  ({results[best_name]['threshold_source']})")
if results[best_name].get('best_f1_threshold') is not None:
    print(f"  Best-F1 threshold : {results[best_name]['best_f1_threshold']:.3f}"
          f"  (F1={results[best_name]['best_f1']:.4f} on test set)")
print(f"  Event precision   : {results[best_name].get('event_precision', 0.0):.4f}"
      f"  (min_consec={EVENT_MIN_CONSEC}, cooldown={EVENT_COOLDOWN_WINDOWS})")
print(f"  Event recall      : {results[best_name].get('event_recall', 0.0):.4f}"
      f"  ({results[best_name].get('event_tp', 0)}/{results[best_name].get('n_events', 0)} events)")
print("=" * 60)
