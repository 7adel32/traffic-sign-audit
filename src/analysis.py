"""Pandas groupby summaries answering the five project analysis questions."""

from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DIR, PROJECT_ROOT, ensure_output_dirs


def summarize_class_counts(
    full_counts_df: pd.DataFrame,
    triage_df: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, dict]:
    """Q1: Compare full-dataset counts vs sampled counts and measure class imbalance."""
    processed_dir, _ = ensure_output_dirs(project_root)

    sampled_counts = (
        triage_df.groupby(["class_id", "class_name"], as_index=False)
        .size()
        .rename(columns={"size": "sampled_count"})
    )
    selected_full = full_counts_df[full_counts_df["selected"]].copy()
    merged = selected_full[["class_id", "class_name", "full_count"]].merge(
        sampled_counts, on=["class_id", "class_name"], how="left"
    )
    merged["sampled_count"] = merged["sampled_count"].fillna(0).astype(int)
    merged = merged.sort_values("full_count", ascending=False).reset_index(drop=True)
    merged.to_csv(processed_dir / "class_counts_comparison.csv", index=False)

    full_max_row = full_counts_df.loc[full_counts_df["full_count"].idxmax()]
    full_min_row = full_counts_df.loc[full_counts_df["full_count"].idxmin()]
    sel_max_row = merged.loc[merged["full_count"].idxmax()]
    sel_min_row = merged.loc[merged["full_count"].idxmin()]

    info = {
        "full_total_images": int(full_counts_df["full_count"].sum()),
        "full_total_classes": int(len(full_counts_df)),
        "full_largest_class": f"{int(full_max_row['class_id'])} ({int(full_max_row['full_count'])})",
        "full_smallest_class": f"{int(full_min_row['class_id'])} ({int(full_min_row['full_count'])})",
        "selected_largest": f"{sel_max_row['class_name']} (full={int(sel_max_row['full_count'])}, sample={int(sel_max_row['sampled_count'])})",
        "selected_smallest": f"{sel_min_row['class_name']} (full={int(sel_min_row['full_count'])}, sample={int(sel_min_row['sampled_count'])})",
        "selected_full_imbalance_ratio": round(float(sel_max_row["full_count"] / sel_min_row["full_count"]), 2),
        "selected_sample_imbalance_ratio": round(float(merged["sampled_count"].max() / merged["sampled_count"].min()), 2),
    }
    return merged, info


def summarize_features_by_class(
    triage_df: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Q2: Compute class-wise size, brightness, contrast, and sharpness summaries."""
    processed_dir, _ = ensure_output_dirs(project_root)

    grouped = triage_df.groupby(["class_id", "class_name"], as_index=False).agg(
        count=("image_id", "size"),
        min_dim_mean=("min_dim", "mean"),
        min_dim_median=("min_dim", "median"),
        area_mean=("image_area", "mean"),
        brightness_mean=("brightness", "mean"),
        brightness_median=("brightness", "median"),
        brightness_std=("brightness", "std"),
        contrast_mean=("contrast", "mean"),
        contrast_median=("contrast", "median"),
        contrast_std=("contrast", "std"),
        sharpness_mean=("sharpness", "mean"),
        sharpness_median=("sharpness", "median"),
        sharpness_std=("sharpness", "std"),
    )
    numeric_cols = [c for c in grouped.columns if c not in ("class_id", "class_name", "count")]
    grouped[numeric_cols] = grouped[numeric_cols].round(2)
    grouped = grouped.sort_values("brightness_median").reset_index(drop=True)
    grouped.to_csv(processed_dir / "class_feature_summary.csv", index=False)
    return grouped


def summarize_flags_by_class(
    triage_df: pd.DataFrame,
    sensitivity_by_class_df: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Q3: Compare baseline flag rates by class and check rank stability across cutoffs."""
    processed_dir, _ = ensure_output_dirs(project_root)

    base_cls = triage_df.groupby(["class_id", "class_name"], as_index=False).agg(
        total=("needs_review", "size"),
        flagged_count=("needs_review", "sum"),
        high_count=("priority_level", lambda x: int((x == "High").sum())),
        medium_count=("priority_level", lambda x: int((x == "Medium").sum())),
        blurry_pct=("flag_blurry", lambda x: round(100.0 * x.mean(), 2)),
        dark_pct=("flag_dark", lambda x: round(100.0 * x.mean(), 2)),
        bright_pct=("flag_bright", lambda x: round(100.0 * x.mean(), 2)),
        low_contrast_pct=("flag_low_contrast", lambda x: round(100.0 * x.mean(), 2)),
    )
    base_cls["flagged_rate_pct"] = (100.0 * base_cls["flagged_count"] / base_cls["total"]).round(2)
    base_cls = base_cls.sort_values("flagged_rate_pct", ascending=False).reset_index(drop=True)
    base_cls.to_csv(processed_dir / "class_flag_summary.csv", index=False)

    rank_pivot = sensitivity_by_class_df.pivot(
        index="class_name", columns="scenario", values="rank_within_scenario"
    )
    rate_pivot = sensitivity_by_class_df.pivot(
        index="class_name", columns="scenario", values="flagged_rate_pct"
    )
    # Ranking changes if any class has more than one unique rank across the scenarios
    ranking_changed = bool((rank_pivot.nunique(axis=1) > 1).any())

    combined_pivot = rate_pivot.join(rank_pivot, lsuffix=" Rate(%)", rsuffix=" Rank")
    return base_cls, combined_pivot, ranking_changed


def analyze_feature_associations(
    triage_df: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
) -> pd.DataFrame:
    """Q4: Compute Pearson and Spearman correlations (association only, not causation)."""
    processed_dir, _ = ensure_output_dirs(project_root)

    cols = ["brightness", "contrast", "sharpness", "image_area"]
    pearson = triage_df[cols].corr(method="pearson").round(3)
    pearson.insert(0, "method", "pearson")
    pearson.insert(1, "feature", pearson.index)

    spearman = triage_df[cols].corr(method="spearman").round(3)
    spearman.insert(0, "method", "spearman")
    spearman.insert(1, "feature", spearman.index)

    corr_df = pd.concat([pearson, spearman], ignore_index=True)
    corr_df.to_csv(processed_dir / "feature_correlations.csv", index=False)
    return corr_df


def select_illustrative_examples(
    triage_df: pd.DataFrame,
    project_root: Path = PROJECT_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Q5: Select 3 illustrative example images plus borderline false-alarm/missed cases."""
    processed_dir, _ = ensure_output_dirs(project_root)

    chosen = []
    used_classes = set()

    # 1. High-priority multi-defect image (prefer highest issue count and blurry+dark)
    high_df = triage_df[triage_df["priority_level"] == "High"].sort_values(
        ["issue_count", "sharpness", "brightness"], ascending=[False, True, True]
    )
    if not high_df.empty:
        row1 = high_df.iloc[0].copy()
        row1["example_role"] = "1. High Priority (Multi-Defect)"
        chosen.append(row1)
        used_classes.add(int(row1["class_id"]))

    # 2. Medium-priority single-issue case from a different class (illustrates potential false alarm if sharp)
    med_df = triage_df[
        (triage_df["priority_level"] == "Medium")
        & (~triage_df["class_id"].isin(used_classes))
        & (~triage_df["flag_blurry"])
    ].sort_values("sharpness", ascending=False)
    if med_df.empty:
        med_df = triage_df[
            (triage_df["priority_level"] == "Medium") & (~triage_df["class_id"].isin(used_classes))
        ]
    if not med_df.empty:
        row2 = med_df.iloc[0].copy()
        row2["example_role"] = "2. Medium Priority (Single-Flag / Potential False Alarm)"
        chosen.append(row2)
        used_classes.add(int(row2["class_id"]))

    # 3. Unflagged normal sign from a third class (clear reference near median quality)
    norm_df = triage_df[
        (triage_df["priority_level"] == "Normal") & (~triage_df["class_id"].isin(used_classes))
    ].copy()
    if not norm_df.empty:
        med_b = norm_df["brightness"].median()
        med_s = norm_df["sharpness"].median()
        norm_df["dist_to_med"] = (norm_df["brightness"] - med_b).abs() + (
            (norm_df["sharpness"] - med_s).abs() / max(1.0, med_s) * 20.0
        )
        row3 = norm_df.sort_values("dist_to_med").iloc[0].drop(labels=["dist_to_med"]).copy()
        row3["example_role"] = "3. Normal Reference (Unflagged)"
        chosen.append(row3)

    examples_df = pd.DataFrame(chosen).reset_index(drop=True)
    examples_df.to_csv(processed_dir / "illustrative_examples.csv", index=False)

    # Borderline cases for Part 4 false-alarm / missed-case inspection
    false_alarm_candidates = triage_df[
        (triage_df["needs_review"]) & (~triage_df["flag_blurry"]) & (triage_df["sharpness"] >= triage_df["sharpness"].median())
    ].head(3)
    missed_candidates = triage_df[
        (~triage_df["needs_review"]) & (triage_df["min_dim"] <= 30)
    ].sort_values("sharpness").head(3)
    review_samples = pd.concat([false_alarm_candidates, missed_candidates], ignore_index=True)

    return examples_df, review_samples


if __name__ == "__main__":
    full_counts = pd.read_csv(PROCESSED_DIR / "full_class_counts.csv")
    triage = pd.read_csv(PROCESSED_DIR / "triage_results.csv")
    sens_cls = pd.read_csv(PROCESSED_DIR / "sensitivity_by_class.csv")

    counts_cmp, count_info = summarize_class_counts(full_counts, triage)
    feat_summary = summarize_features_by_class(triage)
    flag_summary, sens_pivot, rank_changed = summarize_flags_by_class(triage, sens_cls)
    corr_df = analyze_feature_associations(triage)
    examples_df, borderline_df = select_illustrative_examples(triage)

    print("=== Q1: Full vs. Sampled Class Counts ===")
    print(count_info)
    print(counts_cmp.to_string(index=False))

    print("\n=== Q2: Class-Wise Feature Summary ===")
    print(feat_summary.to_string(index=False))

    print("\n=== Q3: Baseline Flag Rate & Sensitivity Rank Stability ===")
    print(flag_summary.to_string(index=False))
    print(f"\nClass ranking changed across sensitivity cutoffs: {rank_changed}")
    print(sens_pivot.to_string())

    print("\n=== Q4: Feature Associations (Correlation != Causation) ===")
    print(corr_df.to_string(index=False))

    print("\n=== Q5: Three Illustrative Examples ===")
    cols = ["example_role", "image_id", "class_name", "width", "height", "brightness", "contrast", "sharpness", "priority_level", "flag_reasons"]
    print(examples_df[cols].to_string(index=False))

    print("\n=== Borderline Cases (Potential False Alarms vs. Potential Missed Small Signs) ===")
    b_cols = ["image_id", "class_name", "min_dim", "brightness", "contrast", "sharpness", "needs_review", "flag_reasons"]
    print(borderline_df[b_cols].to_string(index=False))
