"""Vectorized NumPy triage rule and sensitivity checks."""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    BASELINE_CUTOFFS,
    PROCESSED_DIR,
    PROJECT_ROOT,
    SENSITIVITY_SCENARIOS,
    ensure_output_dirs,
)


def assign_review_priority(
    features_df: pd.DataFrame,
    sharpness_pct: float = 10.0,
    brightness_low_pct: float = 10.0,
    brightness_high_pct: float = 90.0,
    contrast_pct: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """Flag low-visibility images using vectorized NumPy percentile thresholds.

    Important: These percentile cutoffs are project-specific triage settings for
    prioritizing manual review, not official road-safety standards.
    """
    b = features_df["brightness"].to_numpy(dtype=float)
    c = features_df["contrast"].to_numpy(dtype=float)
    s = features_df["sharpness"].to_numpy(dtype=float)

    # Sample-derived percentile cutoffs
    s_thresh = float(np.percentile(s, sharpness_pct))
    b_low_thresh = float(np.percentile(b, brightness_low_pct))
    b_high_thresh = float(np.percentile(b, brightness_high_pct))
    c_thresh = float(np.percentile(c, contrast_pct))

    # Vectorized condition masks
    flag_blurry = s <= s_thresh
    flag_dark = b <= b_low_thresh
    flag_bright = b >= b_high_thresh
    flag_low_contrast = c <= c_thresh

    issue_count = (
        flag_blurry.astype(int)
        + flag_dark.astype(int)
        + flag_bright.astype(int)
        + flag_low_contrast.astype(int)
    )

    # High: 2+ simultaneous quality flags; Medium: 1 extreme flag; Normal: 0 flags
    priority_level = np.select(
        [issue_count >= 2, issue_count == 1],
        ["High", "Medium"],
        default="Normal",
    )
    needs_review = (issue_count >= 1).astype(bool)

    reasons = []
    for blur, dark, bright, low_c in zip(flag_blurry, flag_dark, flag_bright, flag_low_contrast):
        tags = []
        if blur:
            tags.append("blurry")
        if dark:
            tags.append("dark")
        if bright:
            tags.append("overbright")
        if low_c:
            tags.append("low_contrast")
        reasons.append(", ".join(tags) if tags else "none")

    result_df = features_df.copy()
    result_df["flag_blurry"] = flag_blurry
    result_df["flag_dark"] = flag_dark
    result_df["flag_bright"] = flag_bright
    result_df["flag_low_contrast"] = flag_low_contrast
    result_df["issue_count"] = issue_count
    result_df["priority_level"] = priority_level
    result_df["needs_review"] = needs_review
    result_df["flag_reasons"] = reasons

    thresholds = {
        "sharpness_pct": sharpness_pct,
        "brightness_low_pct": brightness_low_pct,
        "brightness_high_pct": brightness_high_pct,
        "contrast_pct": contrast_pct,
        "sharpness_cutoff": round(s_thresh, 4),
        "brightness_low_cutoff": round(b_low_thresh, 4),
        "brightness_high_cutoff": round(b_high_thresh, 4),
        "contrast_cutoff": round(c_thresh, 4),
    }
    return result_df, thresholds


def run_sensitivity_check(
    features_df: pd.DataFrame,
    scenarios: dict[str, dict[str, float]] = SENSITIVITY_SCENARIOS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare overall and class-level flag rates across alternative percentile settings."""
    overall_rows = []
    by_class_frames = []

    for scenario_name, params in scenarios.items():
        triaged, _ = assign_review_priority(features_df, **params)
        total = int(len(triaged))
        flagged = int(triaged["needs_review"].sum())
        high_n = int((triaged["priority_level"] == "High").sum())
        med_n = int((triaged["priority_level"] == "Medium").sum())

        overall_rows.append(
            {
                "scenario": scenario_name,
                "sharpness_pct": params["sharpness_pct"],
                "brightness_low_pct": params["brightness_low_pct"],
                "brightness_high_pct": params["brightness_high_pct"],
                "contrast_pct": params["contrast_pct"],
                "total_images": total,
                "flagged_count": flagged,
                "flagged_rate_pct": round(100.0 * flagged / total, 2),
                "high_priority_count": high_n,
                "medium_priority_count": med_n,
            }
        )

        cls_summary = (
            triaged.groupby(["class_id", "class_name"], as_index=False)
            .agg(
                total=("needs_review", "size"),
                flagged_count=("needs_review", "sum"),
                high_count=("priority_level", lambda x: int((x == "High").sum())),
                medium_count=("priority_level", lambda x: int((x == "Medium").sum())),
            )
        )
        cls_summary["flagged_rate_pct"] = (
            100.0 * cls_summary["flagged_count"] / cls_summary["total"]
        ).round(2)
        cls_summary["rank_within_scenario"] = (
            cls_summary["flagged_rate_pct"].rank(ascending=False, method="min").astype(int)
        )
        cls_summary["scenario"] = scenario_name
        by_class_frames.append(cls_summary)

    overall_df = pd.DataFrame(overall_rows)
    by_class_df = pd.concat(by_class_frames, ignore_index=True)
    return overall_df, by_class_df


def save_triage_outputs(
    triage_df: pd.DataFrame,
    overall_df: pd.DataFrame,
    by_class_df: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Write triage and sensitivity CSV outputs into outputs/processed/."""
    processed_dir, _ = ensure_output_dirs(project_root)
    triage_df.to_csv(processed_dir / "triage_results.csv", index=False)
    overall_df.to_csv(processed_dir / "sensitivity_overall.csv", index=False)
    by_class_df.to_csv(processed_dir / "sensitivity_by_class.csv", index=False)


if __name__ == "__main__":
    feat_path = PROCESSED_DIR / "image_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError("Run src/features.py first to create image_features.csv.")

    features = pd.read_csv(feat_path)
    triage_df, thresholds = assign_review_priority(features, **BASELINE_CUTOFFS)
    overall_sens, by_class_sens = run_sensitivity_check(features)
    save_triage_outputs(triage_df, overall_sens, by_class_sens)

    print("=== Baseline Vectorized Triage Thresholds ===")
    print(thresholds)
    print("\n=== Sensitivity Comparison (Overall) ===")
    print(overall_sens.to_string(index=False))
    print("\n=== Sensitivity Comparison (Flag Rate % by Class) ===")
    pivot_rates = by_class_sens.pivot(index="class_name", columns="scenario", values="flagged_rate_pct")
    print(pivot_rates.round(2))
