# scripts/preprocess_freeviewing.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
# If pupil_preprocessing.py is in src/, make it importable:
sys.path.append(str(ROOT / "src"))

from pupil_preprocessing import (
    preprocess_mat_files,
    PreprocessConfig,
    save_mat_list,
)

# === INPUT FILES ===
left_in  = r"C:/Users/charl/OneDrive\Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Continuos/pupilSizeLeftContinuos.mat"
right_in = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Continuos/pupilSizeRightContinuos.mat"

# === OUTPUT FILES ===
left_out  = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Continuos/pupilSizeLeft_cleanedInterpolated.mat"
right_out = r"C:/Users/charl/OneDrive/Dokumente/Uni/7. Semester/Bachelor Thesis/Eye Data Experiment1Task2/Continuos/pupilSizeRight_cleanedInterpolated.mat"

# === CONFIG (same behavior as your original scripts) ===
cfg = PreprocessConfig(
    bad_values=(0, 32700),
    dilate_window=5,
    z_thresh=2.0,
    spike_pad=5,
    global_diff_thresholds=True,
    interpolate_method="linear",
    median_kernel=11,
    verbose=True,
)

# === RUN PIPELINE ===
left_proc, right_proc = preprocess_mat_files(left_in, right_in, cfg=cfg)

# === SAVE ===
save_mat_list(left_out,  "pupilSizeLeft",  left_proc)
save_mat_list(right_out, "pupilSizeRight", right_proc)

print("✅ Free Viewing preprocessing finished and saved.")
