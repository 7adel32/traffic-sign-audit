"""Shared configuration for the Traffic Sign Visibility Audit project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "gtsrb"
MANIFEST_PATH = DATA_DIR / "train_manifest.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = OUTPUTS_DIR / "processed"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Five recognizable GTSRB classes spanning high and low sample counts
SELECTED_CLASSES = {
    2: "Speed limit (50km/h)",
    13: "Yield",
    12: "Priority road",
    14: "Stop",
    27: "Pedestrians",
}

# Scope limit: up to 500 images per selected class with fixed seed for reproducibility
MAX_SAMPLES_PER_CLASS = 500
RANDOM_SEED = 42
ROI_COMPARE_SAMPLE_SIZE = 50

# Project-specific percentile cutoffs (triage choices, not official safety standards)
BASELINE_CUTOFFS = {
    "sharpness_pct": 10.0,
    "brightness_low_pct": 10.0,
    "brightness_high_pct": 90.0,
    "contrast_pct": 10.0,
}

SENSITIVITY_SCENARIOS = {
    "Strict (5/95)": {
        "sharpness_pct": 5.0,
        "brightness_low_pct": 5.0,
        "brightness_high_pct": 95.0,
        "contrast_pct": 5.0,
    },
    "Baseline (10/90)": dict(BASELINE_CUTOFFS),
    "Lenient (15/85)": {
        "sharpness_pct": 15.0,
        "brightness_low_pct": 15.0,
        "brightness_high_pct": 85.0,
        "contrast_pct": 15.0,
    },
}


def ensure_output_dirs(project_root: Path = PROJECT_ROOT) -> tuple[Path, Path]:
    """Create processed and figure output directories if they do not exist."""
    processed_dir = project_root / "outputs" / "processed"
    figures_dir = project_root / "outputs" / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir, figures_dir


if __name__ == "__main__":
    processed_dir, figures_dir = ensure_output_dirs()
    print("=== Traffic Sign Audit Configuration ===")
    print(f"Manifest path : {MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Processed dir : {processed_dir.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Figures dir   : {figures_dir.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Max/class     : {MAX_SAMPLES_PER_CLASS} (seed={RANDOM_SEED})")
    print(f"Selected      : {SELECTED_CLASSES}")
    print(f"Baseline rule : {BASELINE_CUTOFFS}")
