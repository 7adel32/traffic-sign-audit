"""OpenCV feature extraction, ROI validation, and ROI-vs-full comparison."""

from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    PROCESSED_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    ROI_COMPARE_SAMPLE_SIZE,
    ensure_output_dirs,
)


def validate_roi(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> bool:
    """Return True if bounding-box ROI coordinates fall strictly inside image bounds."""
    return 0 <= x1 < x2 <= img_w and 0 <= y1 < y2 <= img_h


def compute_image_metrics(img_bgr: np.ndarray) -> dict[str, float]:
    """Compute brightness, contrast, and Laplacian variance on a BGR image or crop.

    Note: Variance of the Laplacian is a mathematical edge/focus proxy, not a
    direct human readability or safety score.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
    }


def extract_image_features(
    sample_df: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
    save_csv: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load each sampled image with OpenCV, validate ROI, and extract visibility metrics."""
    processed_dir, _ = ensure_output_dirs(project_root)

    records = []
    skipped = {"missing_file": 0, "unreadable_image": 0, "read_error": 0}
    skipped_examples = []
    roi_fallback_count = 0

    for _, row in sample_df.iterrows():
        rel_str = str(row["relative_path"]).replace("\\", "/")
        rel_path = Path(rel_str)
        full_path = project_root / rel_path

        if rel_path.is_absolute() or not full_path.exists():
            skipped["missing_file"] += 1
            if len(skipped_examples) < 5:
                skipped_examples.append(f"missing_file: {rel_str}")
            continue

        try:
            img_bgr = cv2.imread(str(full_path))
        except Exception as exc:
            skipped["read_error"] += 1
            if len(skipped_examples) < 5:
                skipped_examples.append(f"read_error ({exc}): {rel_str}")
            continue

        if img_bgr is None or img_bgr.size == 0:
            skipped["unreadable_image"] += 1
            if len(skipped_examples) < 5:
                skipped_examples.append(f"unreadable_image: {rel_str}")
            continue

        img_h, img_w = img_bgr.shape[:2]

        # Validate ROI coordinates before cropping; fall back to full image if invalid
        roi_used = False
        target_img = img_bgr
        try:
            x1 = int(row["roi_x1"])
            y1 = int(row["roi_y1"])
            x2 = int(row["roi_x2"])
            y2 = int(row["roi_y2"])
            if validate_roi(x1, y1, x2, y2, img_w, img_h):
                target_img = img_bgr[y1:y2, x1:x2]
                roi_used = True
            else:
                roi_fallback_count += 1
        except (TypeError, ValueError):
            roi_fallback_count += 1

        metrics = compute_image_metrics(target_img)

        records.append(
            {
                "image_id": int(row["image_id"]),
                "image_path": rel_str,
                "class_id": int(row["class_id"]),
                "class_name": str(row["class_name"]),
                "width": int(img_w),
                "height": int(img_h),
                "min_dim": int(min(img_w, img_h)),
                "image_area": int(img_w * img_h),
                "aspect_ratio": round(img_w / img_h, 4),
                "roi_used": roi_used,
                "brightness": round(metrics["brightness"], 4),
                "contrast": round(metrics["contrast"], 4),
                "sharpness": round(metrics["sharpness"], 4),
            }
        )

    features_df = pd.DataFrame(records)
    if save_csv and not features_df.empty:
        features_df.to_csv(processed_dir / "image_features.csv", index=False)

    stats = {
        "total_input": int(len(sample_df)),
        "processed": int(len(features_df)),
        "skipped": skipped,
        "skipped_total": int(sum(skipped.values())),
        "roi_fallback_count": roi_fallback_count,
        "skipped_examples": skipped_examples,
    }
    return features_df, stats


def compare_roi_vs_full(
    sample_df: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
    n_sample: int = ROI_COMPARE_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Compare brightness, contrast, and sharpness on ROI crops vs full images for a sample."""
    processed_dir, _ = ensure_output_dirs(project_root)

    # Sample evenly across classes so every selected sign type is represented
    n_classes = max(1, int(sample_df["class_id"].nunique()))
    per_class = max(1, n_sample // n_classes)
    parts = [
        grp.sample(n=min(len(grp), per_class), random_state=seed)
        for _, grp in sample_df.groupby("class_id")
    ]
    subset = pd.concat(parts, ignore_index=True)

    rows = []
    for _, row in subset.iterrows():
        rel_str = str(row["relative_path"]).replace("\\", "/")
        img_bgr = cv2.imread(str(project_root / rel_str))
        if img_bgr is None or img_bgr.size == 0:
            continue

        img_h, img_w = img_bgr.shape[:2]
        x1, y1, x2, y2 = int(row["roi_x1"]), int(row["roi_y1"]), int(row["roi_x2"]), int(row["roi_y2"])
        if not validate_roi(x1, y1, x2, y2, img_w, img_h):
            continue

        full_m = compute_image_metrics(img_bgr)
        roi_m = compute_image_metrics(img_bgr[y1:y2, x1:x2])

        rows.append(
            {
                "image_id": int(row["image_id"]),
                "class_id": int(row["class_id"]),
                "class_name": str(row["class_name"]),
                "roi_brightness": round(roi_m["brightness"], 3),
                "full_brightness": round(full_m["brightness"], 3),
                "diff_brightness": round(roi_m["brightness"] - full_m["brightness"], 3),
                "roi_contrast": round(roi_m["contrast"], 3),
                "full_contrast": round(full_m["contrast"], 3),
                "diff_contrast": round(roi_m["contrast"] - full_m["contrast"], 3),
                "roi_sharpness": round(roi_m["sharpness"], 3),
                "full_sharpness": round(full_m["sharpness"], 3),
                "diff_sharpness": round(roi_m["sharpness"] - full_m["sharpness"], 3),
            }
        )

    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(processed_dir / "roi_vs_full_sample.csv", index=False)
    return compare_df


if __name__ == "__main__":
    sample_path = PROCESSED_DIR / "sampled_annotations.csv"
    if not sample_path.exists():
        raise FileNotFoundError("Run src/data_prep.py first to create sampled_annotations.csv.")

    sampled = pd.read_csv(sample_path)
    feat_df, feat_stats = extract_image_features(sampled)
    roi_comp_df = compare_roi_vs_full(sampled)

    print("=== Feature Extraction Summary ===")
    print(f"Processed: {feat_stats['processed']}/{feat_stats['total_input']} | Skipped: {feat_stats['skipped']}")
    print(f"ROI fallbacks: {feat_stats['roi_fallback_count']}")
    print(feat_df[["width", "height", "brightness", "contrast", "sharpness"]].describe().round(2))

    print("\n=== ROI Crop vs. Full Image Mean Differences (n=" + str(len(roi_comp_df)) + ") ===")
    print(roi_comp_df[["diff_brightness", "diff_contrast", "diff_sharpness"]].mean().round(2))
