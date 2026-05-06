# =====================================================================
# SBFD-iotlab-sdsu — Preprocessing
# =====================================================================
# Reads the raw SDSU IoTLab Globecom-2019 fall-detection CSVs from
# `original_data/` and writes per-trial processed CSVs to
# `data/processed/separate_trials/trial_XX/trial_XX_full.csv`.
#
# Compared to the head-only preprocessing in repo 1
# (Sensor-Based-Fall-detection/scripts/process_falls.py) this script:
#   * Skips the 4 metadata header rows of the raw files.
#   * KEEPS all 16 IMU placements (head, spines, pelvis, arms, hands,
#     legs, feet) — accel X/Y/Z, rot X/Y/Z, course/pitch/roll = 9 raw
#     channels each → 144 raw IMU channels.
#   * DROPS derived joint angles (knee/hip/ankle flexion etc.) and the
#     two Noraxon switch flags, since those are computed from IMUs and
#     would leak redundant information.
#   * Preserves the original column names verbatim — train.py performs
#     the rename so the trial CSVs remain self-describing.
#
# Output column count per trial: 1 (Time) + 144 (IMU) + 2 (Markers,
# MarkerNames) = 147.
# =====================================================================

import os
import glob
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

INPUT_DIR = os.path.join(REPO_ROOT, "original_data")
OUTPUT_SEPARATE_DIR = os.path.join(REPO_ROOT, "data", "processed", "separate_trials")
OUTPUT_COMBINED_DIR = os.path.join(REPO_ROOT, "data", "processed", "combined_data")

os.makedirs(OUTPUT_SEPARATE_DIR, exist_ok=True)
os.makedirs(OUTPUT_COMBINED_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Column whitelist — kept verbatim from the SDSU header (including
# trailing commas in the rotation columns). train.py performs the
# rename so this file stays a thin selection layer.
# ---------------------------------------------------------------------
META_COLS = ["Time,s", "Markers", "MarkerNames"]

# 16 IMU placements x 3 orientation angles (course/pitch/roll) = 48
ANGLE_COLS = [
    "Head course,deg", "Head pitch,deg", "Head roll,deg",
    "Upper spine course,deg", "Upper spine pitch,deg", "Upper spine roll,deg",
    "Upper arm course LT,deg", "Upper arm pitch LT,deg", "Upper arm roll LT,deg",
    "Forearm course LT,deg", "Forearm pitch LT,deg", "Forearm roll LT,deg",
    "Hand course LT,deg", "Hand pitch LT,deg", "Hand roll LT,deg",
    "Upper arm course RT,deg", "Upper arm pitch RT,deg", "Upper arm roll RT,deg",
    "Forearm course RT,deg", "Forearm pitch RT,deg", "Forearm roll RT,deg",
    "Hand course RT,deg", "Hand pitch RT,deg", "Hand roll RT,deg",
    "Lower spine course,deg", "Lower spine pitch,deg", "Lower spine roll,deg",
    "Pelvis course,deg", "Pelvis pitch,deg", "Pelvis roll,deg",
    "Thigh course LT,deg", "Thigh pitch LT,deg", "Thigh roll LT,deg",
    "Shank course LT,deg", "Shank pitch LT,deg", "Shank roll LT,deg",
    "Foot course LT,deg", "Foot pitch LT,deg", "Foot roll LT,deg",
    "Thigh course RT,deg", "Thigh pitch RT,deg", "Thigh roll RT,deg",
    "Shank course RT,deg", "Shank pitch RT,deg", "Shank roll RT,deg",
    "Foot course RT,deg", "Foot pitch RT,deg", "Foot roll RT,deg",
]

# 16 IMU placements x 3 accelerometer axes = 48
ACCEL_COLS = [
    "Head Accel Sensor X,mG", "Head Accel Sensor Y,mG", "Head Accel Sensor Z,mG",
    "Upper spine Accel Sensor X,mG", "Upper spine Accel Sensor Y,mG", "Upper spine Accel Sensor Z,mG",
    "Upper arm Accel Sensor X LT,mG", "Upper arm Accel Sensor Y LT,mG", "Upper arm Accel Sensor Z LT,mG",
    "Forearm Accel Sensor X LT,mG", "Forearm Accel Sensor Y LT,mG", "Forearm Accel Sensor Z LT,mG",
    "Hand Accel Sensor X LT,mG", "Hand Accel Sensor Y LT,mG", "Hand Accel Sensor Z LT,mG",
    "Upper arm Accel Sensor X RT,mG", "Upper arm Accel Sensor Y RT,mG", "Upper arm Accel Sensor Z RT,mG",
    "Forearm Accel Sensor X RT,mG", "Forearm Accel Sensor Y RT,mG", "Forearm Accel Sensor Z RT,mG",
    "Hand Accel Sensor X RT,mG", "Hand Accel Sensor Y RT,mG", "Hand Accel Sensor Z RT,mG",
    "Lower spine Accel Sensor X,mG", "Lower spine Accel Sensor Y,mG", "Lower spine Accel Sensor Z,mG",
    "Pelvis Accel Sensor X,mG", "Pelvis Accel Sensor Y,mG", "Pelvis Accel Sensor Z,mG",
    "Thigh Accel Sensor X LT,mG", "Thigh Accel Sensor Y LT,mG", "Thigh Accel Sensor Z LT,mG",
    "Shank Accel Sensor X LT,mG", "Shank Accel Sensor Y LT,mG", "Shank Accel Sensor Z LT,mG",
    "Foot Accel Sensor X LT,mG", "Foot Accel Sensor Y LT,mG", "Foot Accel Sensor Z LT,mG",
    "Thigh Accel Sensor X RT,mG", "Thigh Accel Sensor Y RT,mG", "Thigh Accel Sensor Z RT,mG",
    "Shank Accel Sensor X RT,mG", "Shank Accel Sensor Y RT,mG", "Shank Accel Sensor Z RT,mG",
    "Foot Accel Sensor X RT,mG", "Foot Accel Sensor Y RT,mG", "Foot Accel Sensor Z RT,mG",
]

# 16 IMU placements x 3 rotation rate axes = 48.
# Note: the SDSU CSV uses "LT/RT" as a *prefix* on the rotation columns
# (e.g. "LT Upper arm Rot X,") even though accel uses it as a *suffix*
# (e.g. "Upper arm Accel Sensor X LT,mG"). All names include a trailing
# comma in the source file — they are kept verbatim here so the
# whitelist matches exactly against pd.read_csv column names.
ROT_COLS = [
    "Head Rot X,", "Head Rot Y,", "Head Rot Z,",
    "Upper spine Rot X,", "Upper spine Rot Y,", "Upper spine Rot Z,",
    "LT Upper arm Rot X,", "LT Upper arm Rot Y,", "LT Upper arm Rot Z,",
    "LT Forearm Rot X,", "LT Forearm Rot Y,", "LT Forearm Rot Z,",
    "LT Hand Rot X,", "LT Hand Rot Y,", "LT Hand Rot Z,",
    "RT Upper arm Rot X,", "RT Upper arm Rot Y,", "RT Upper arm Rot Z,",
    "RT Forearm Rot X,", "RT Forearm Rot Y,", "RT Forearm Rot Z,",
    "RT Hand Rot X,", "RT Hand Rot Y,", "RT Hand Rot Z,",
    "Lower spine Rot X,", "Lower spine Rot Y,", "Lower spine Rot Z,",
    "Pelvis Rot X,", "Pelvis Rot Y,", "Pelvis Rot Z,",
    "LT Thigh Rot X,", "LT Thigh Rot Y,", "LT Thigh Rot Z,",
    "LT Shank Rot X,", "LT Shank Rot Y,", "LT Shank Rot Z,",
    "LT Foot Rot X,", "LT Foot Rot Y,", "LT Foot Rot Z,",
    "RT Thigh Rot X,", "RT Thigh Rot Y,", "RT Thigh Rot Z,",
    "RT Shank Rot X,", "RT Shank Rot Y,", "RT Shank Rot Z,",
    "RT Foot Rot X,", "RT Foot Rot Y,", "RT Foot Rot Z,",
]

TARGET_COLUMNS = META_COLS + ANGLE_COLS + ACCEL_COLS + ROT_COLS
assert len(ANGLE_COLS) == 48 and len(ACCEL_COLS) == 48 and len(ROT_COLS) == 48
assert len(TARGET_COLUMNS) == 147, f"unexpected whitelist size: {len(TARGET_COLUMNS)}"


def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "fall dataset *.csv")))
    if not files:
        raise FileNotFoundError(f"No 'fall dataset *.csv' files found under {INPUT_DIR}")

    print(f"Found {len(files)} raw trial files in {INPUT_DIR}")
    all_combined = []
    fall_durations = []

    for file_idx, filepath in enumerate(files, 1):
        filename = os.path.basename(filepath)
        trial_id = f"trial_{file_idx:02d}"

        # The first 4 rows are metadata: "MR32 CSV", "Name,...",
        # "Frequency,100", "Date,...". Real headers start on row 5.
        df = pd.read_csv(filepath, skiprows=4, low_memory=False)

        # Detect missing columns and warn — skip them rather than crash so a
        # single mislabelled trial does not break the whole pipeline.
        present_cols, missing_cols = [], []
        for col in TARGET_COLUMNS:
            (present_cols if col in df.columns else missing_cols).append(col)

        if missing_cols:
            print(f"  [{trial_id}] WARNING: {len(missing_cols)} expected columns missing "
                  f"(first 3: {missing_cols[:3]})")

        df_sub = df[present_cols].copy()

        # --- fall-duration stats only (does NOT mutate the saved CSV) ---
        if "MarkerNames" in df_sub.columns and "Time,s" in df_sub.columns:
            in_fall = False
            t_start = None
            for marker, t in zip(df_sub["MarkerNames"].astype(str).str.strip(),
                                 df_sub["Time,s"]):
                if marker in {"startOfFall", "FallStart"} and not in_fall:
                    in_fall = True
                    t_start = t
                elif marker in {"endOfFall", "FallEnd"} and in_fall:
                    in_fall = False
                    if t_start is not None:
                        fall_durations.append(float(t) - float(t_start))
                    t_start = None

        # --- save per-trial CSV ---
        trial_dir = os.path.join(OUTPUT_SEPARATE_DIR, trial_id)
        os.makedirs(trial_dir, exist_ok=True)
        out_path = os.path.join(trial_dir, f"{trial_id}_full.csv")
        df_sub.to_csv(out_path, index=False)
        all_combined.append(df_sub.assign(trial_id=trial_id))
        print(f"  [{trial_id}] {filename} -> {out_path}  "
              f"({df_sub.shape[0]:,} rows × {df_sub.shape[1]} cols)")

    # --- save combined CSV (kept for parity with repo 1's combined_data dir) ---
    combined = pd.concat(all_combined, ignore_index=True)
    combined_path = os.path.join(OUTPUT_COMBINED_DIR, "combined_full.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nSaved combined dataset to {combined_path}  "
          f"({combined.shape[0]:,} rows × {combined.shape[1]} cols)")

    # --- summary ---
    if fall_durations:
        avg = sum(fall_durations) / len(fall_durations)
        print(f"\nFall events: {len(fall_durations)}")
        print(f"  Avg duration: {avg:.2f}s   "
              f"(min {min(fall_durations):.2f}s, max {max(fall_durations):.2f}s)")
    else:
        print("\nNo fall events detected — check Marker labelling in raw CSVs.")


if __name__ == "__main__":
    main()
