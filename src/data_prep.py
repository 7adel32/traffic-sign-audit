"""Load, validate, inspect, and sample GTSRB annotation data."""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    MANIFEST_PATH,
    MAX_SAMPLES_PER_CLASS,
    PROJECT_ROOT,
    RANDOM_SEED,
    SELECTED_CLASSES,
    ensure_output_dirs,
)

REQUIRED_COLUMNS = [
    "image_id",
    "class_id",
    "width",
    "height",
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "relative_path",
]


def load_annotations(manifest_path: Path = MANIFEST_PATH) -> pd.DataFrame:
    """Load the GTSRB annotation CSV and verify required columns exist."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Annotation manifest not found at '{manifest_path.name}'. "
            "Run 'python src/download_data.py' first to prepare data/gtsrb/."
        )

    df = pd.read_csv(manifest_path)
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Annotation CSV is missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("Annotation CSV loaded 0 rows.")

    return df


def inspect_full_dataset(
    df: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
    selected_classes: dict[int, str] = SELECTED_CLASSES,
    check_files: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Audit full-dataset quality (types, nulls, duplicates, ROIs, paths) and class counts."""
    processed_dir, _ = ensure_output_dirs(project_root)

    duplicate_rows = int(df.duplicated().sum())
    missing_by_col = {col: int(cnt) for col, cnt in df.isna().sum().items()}
    total_missing = int(sum(missing_by_col.values()))

    # Check ROI coordinates against stated image dimensions
    bad_roi_mask = (
        df[["width", "height", "roi_x1", "roi_y1", "roi_x2", "roi_y2"]].isna().any(axis=1)
        | (df["roi_x1"] < 0)
        | (df["roi_y1"] < 0)
        | (df["roi_x2"] > df["width"])
        | (df["roi_y2"] > df["height"])
        | (df["roi_x1"] >= df["roi_x2"])
        | (df["roi_y1"] >= df["roi_y2"])
    )
    invalid_roi_count = int(bad_roi_mask.sum())

    # Check image path formatting and existence on disk
    invalid_path_count = 0
    if check_files:
        for raw_path in df["relative_path"]:
            if not isinstance(raw_path, str) or not raw_path.strip():
                invalid_path_count += 1
                continue
            rel = Path(raw_path)
            if rel.is_absolute() or not (project_root / rel).exists():
                invalid_path_count += 1

    # Full-dataset class distribution (kept separate from sampled counts)
    full_counts = (
        df.groupby("class_id", as_index=False)
        .size()
        .rename(columns={"size": "full_count"})
        .sort_values("full_count", ascending=False)
        .reset_index(drop=True)
    )
    full_counts["selected"] = full_counts["class_id"].isin(selected_classes.keys())
    full_counts["class_name"] = (
        full_counts["class_id"].map(selected_classes).fillna("Not selected")
    )
    full_counts.to_csv(processed_dir / "full_class_counts.csv", index=False)

    summary = {
        "total_rows": int(len(df)),
        "num_classes": int(df["class_id"].nunique()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "duplicate_rows": duplicate_rows,
        "total_missing_values": total_missing,
        "missing_by_col": missing_by_col,
        "invalid_roi_count": invalid_roi_count,
        "invalid_path_count": invalid_path_count,
    }
    return summary, full_counts


def create_stratified_sample(
    df: pd.DataFrame,
    selected_classes: dict[int, str] = SELECTED_CLASSES,
    max_per_class: int = MAX_SAMPLES_PER_CLASS,
    seed: int = RANDOM_SEED,
    project_root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, dict]:
    """Select target classes and draw a reproducible stratified sample (up to max_per_class)."""
    processed_dir, _ = ensure_output_dirs(project_root)

    # Drop records missing critical metadata before sampling, tracking exclusions explicitly
    critical_cols = ["image_id", "class_id", "width", "height", "relative_path"]
    valid_meta_mask = df[critical_cols].notna().all(axis=1)
    dropped_malformed = int((~valid_meta_mask).sum())
    clean_df = df[valid_meta_mask].copy()

    in_scope_mask = clean_df["class_id"].isin(selected_classes.keys())
    excluded_other_classes = int((~in_scope_mask).sum())
    filtered_df = clean_df[in_scope_mask].copy()

    sampled_parts = []
    class_sampling_log = []
    for class_id, class_name in selected_classes.items():
        cls_rows = filtered_df[filtered_df["class_id"] == class_id]
        available = int(len(cls_rows))
        taken = min(available, max_per_class)
        sampled_cls = cls_rows.sample(n=taken, random_state=seed)
        sampled_parts.append(sampled_cls)
        class_sampling_log.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "full_count": available,
                "sampled_count": taken,
                "capped": available > max_per_class,
            }
        )

    sample_df = pd.concat(sampled_parts, ignore_index=True)
    sample_df["class_name"] = sample_df["class_id"].map(selected_classes)

    out_path = processed_dir / "sampled_annotations.csv"
    sample_df.to_csv(out_path, index=False)

    sampling_report = {
        "dropped_malformed_rows": dropped_malformed,
        "excluded_other_classes": excluded_other_classes,
        "selected_total_available": int(len(filtered_df)),
        "sampled_total": int(len(sample_df)),
        "seed": seed,
        "max_per_class": max_per_class,
        "per_class": class_sampling_log,
    }
    return sample_df, sampling_report


if __name__ == "__main__":
    raw_df = load_annotations()
    audit_info, full_counts_df = inspect_full_dataset(raw_df)
    sample_df, sample_info = create_stratified_sample(raw_df)

    print("=== Full Dataset Inspection ===")
    print(f"Total rows: {audit_info['total_rows']:,} | Classes: {audit_info['num_classes']}")
    print(f"Duplicate rows: {audit_info['duplicate_rows']} | Missing values: {audit_info['total_missing_values']}")
    print(f"Invalid ROIs: {audit_info['invalid_roi_count']} | Invalid/missing paths: {audit_info['invalid_path_count']}")
    print("Column dtypes:", audit_info["dtypes"])

    print("\n=== Stratified Sample Summary ===")
    print(
        f"Excluded non-selected classes: {sample_info['excluded_other_classes']:,} | "
        f"Sampled: {sample_info['sampled_total']:,} (seed={sample_info['seed']}, cap={sample_info['max_per_class']}/class)"
    )
    print(pd.DataFrame(sample_info["per_class"]).to_string(index=False))
