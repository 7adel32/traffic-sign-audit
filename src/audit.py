"""TrafficSignAudit workflow class and automated validation checks."""

from pathlib import Path
import sys
import tempfile
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis import (
    analyze_feature_associations,
    select_illustrative_examples,
    summarize_class_counts,
    summarize_features_by_class,
    summarize_flags_by_class,
)
from src.config import (
    BASELINE_CUTOFFS,
    MAX_SAMPLES_PER_CLASS,
    PROJECT_ROOT,
    RANDOM_SEED,
    SELECTED_CLASSES,
    SENSITIVITY_SCENARIOS,
    ensure_output_dirs,
)
from src.data_prep import create_stratified_sample, inspect_full_dataset, load_annotations
from src.features import compare_roi_vs_full, extract_image_features
from src.plots import generate_all_plots
from src.triage import assign_review_priority, run_sensitivity_check, save_triage_outputs


class TrafficSignAudit:
    """End-to-end manager for loading, measuring, flagging, summarizing, and plotting GTSRB signs."""

    def __init__(
        self,
        project_root: Path | None = None,
        selected_classes: dict[int, str] = SELECTED_CLASSES,
        max_per_class: int = MAX_SAMPLES_PER_CLASS,
        seed: int = RANDOM_SEED,
        cutoffs: dict[str, float] = BASELINE_CUTOFFS,
    ) -> None:
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.selected_classes = dict(selected_classes)
        self.max_per_class = max_per_class
        self.seed = seed
        self.cutoffs = dict(cutoffs)
        ensure_output_dirs(self.project_root)

        self.raw_df: pd.DataFrame | None = None
        self.audit_info: dict = {}
        self.full_counts_df: pd.DataFrame | None = None
        self.sample_df: pd.DataFrame | None = None
        self.sampling_info: dict = {}
        self.features_df: pd.DataFrame | None = None
        self.extraction_stats: dict = {}
        self.roi_compare_df: pd.DataFrame | None = None
        self.triage_df: pd.DataFrame | None = None
        self.thresholds: dict = {}
        self.sensitivity_overall_df: pd.DataFrame | None = None
        self.sensitivity_by_class_df: pd.DataFrame | None = None
        self.examples_df: pd.DataFrame | None = None
        self.figure_paths: list[Path] = []

    def load_and_inspect(self) -> pd.DataFrame:
        """Load raw manifest, audit full-dataset quality, and build stratified sample."""
        manifest_path = self.project_root / "data" / "gtsrb" / "train_manifest.csv"
        self.raw_df = load_annotations(manifest_path)
        self.audit_info, self.full_counts_df = inspect_full_dataset(
            self.raw_df,
            project_root=self.project_root,
            selected_classes=self.selected_classes,
        )
        self.sample_df, self.sampling_info = create_stratified_sample(
            self.raw_df,
            selected_classes=self.selected_classes,
            max_per_class=self.max_per_class,
            seed=self.seed,
            project_root=self.project_root,
        )
        return self.sample_df

    def extract_features(self) -> pd.DataFrame:
        """Extract OpenCV metrics and compare ROI crops against full images."""
        if self.sample_df is None:
            raise RuntimeError("Call load_and_inspect() before extract_features().")
        self.features_df, self.extraction_stats = extract_image_features(
            self.sample_df, project_root=self.project_root
        )
        self.roi_compare_df = compare_roi_vs_full(
            self.sample_df, project_root=self.project_root, seed=self.seed
        )
        return self.features_df

    def apply_triage(self) -> pd.DataFrame:
        """Apply vectorized NumPy percentile triage and sensitivity checks."""
        if self.features_df is None:
            raise RuntimeError("Call extract_features() before apply_triage().")
        self.triage_df, self.thresholds = assign_review_priority(
            self.features_df, **self.cutoffs
        )
        self.sensitivity_overall_df, self.sensitivity_by_class_df = run_sensitivity_check(
            self.features_df, scenarios=SENSITIVITY_SCENARIOS
        )
        save_triage_outputs(
            self.triage_df,
            self.sensitivity_overall_df,
            self.sensitivity_by_class_df,
            project_root=self.project_root,
        )
        return self.triage_df

    def summarize(self) -> dict:
        """Run pandas groupby summaries answering all five case questions."""
        if self.triage_df is None or self.full_counts_df is None or self.sensitivity_by_class_df is None:
            raise RuntimeError("Call apply_triage() before summarize().")

        counts_cmp, count_info = summarize_class_counts(
            self.full_counts_df, self.triage_df, project_root=self.project_root
        )
        feat_summary = summarize_features_by_class(
            self.triage_df, project_root=self.project_root
        )
        flag_summary, sens_pivot, rank_changed = summarize_flags_by_class(
            self.triage_df, self.sensitivity_by_class_df, project_root=self.project_root
        )
        corr_df = analyze_feature_associations(
            self.triage_df, project_root=self.project_root
        )
        self.examples_df, borderline_df = select_illustrative_examples(
            self.triage_df, project_root=self.project_root
        )
        return {
            "counts_comparison": counts_cmp,
            "count_info": count_info,
            "feature_summary": feat_summary,
            "flag_summary": flag_summary,
            "sensitivity_pivot": sens_pivot,
            "ranking_changed": rank_changed,
            "correlations": corr_df,
            "examples": self.examples_df,
            "borderline_samples": borderline_df,
        }

    def plot_all(self) -> list[Path]:
        """Generate all Matplotlib figures."""
        if self.triage_df is None or self.sensitivity_by_class_df is None or self.examples_df is None:
            raise RuntimeError("Call summarize() before plot_all().")
        self.figure_paths = generate_all_plots(
            self.triage_df,
            self.sensitivity_by_class_df,
            self.examples_df,
            cutoffs=self.cutoffs,
            project_root=self.project_root,
        )
        return self.figure_paths

    def run_all(self) -> dict:
        """Execute the full pipeline end-to-end and return summary artifacts."""
        self.load_and_inspect()
        self.extract_features()
        self.apply_triage()
        summaries = self.summarize()
        self.plot_all()
        return summaries

    def run_validation_checks(self) -> None:
        """Verify error handling, schema completeness, boolean flags, and path portability."""
        # 1. Verify missing file and unreadable corrupted image are caught gracefully
        with tempfile.NamedTemporaryFile(
            dir=self.project_root / "outputs", suffix=".png", delete=False
        ) as tmp:
            tmp.write(b"not a real png image")
            corrupt_path = Path(tmp.name)

        try:
            corrupt_rel = corrupt_path.relative_to(self.project_root).as_posix()
            bad_rows = pd.DataFrame(
                [
                    {
                        "image_id": -1,
                        "class_id": 2,
                        "class_name": "Test missing",
                        "roi_x1": 0,
                        "roi_y1": 0,
                        "roi_x2": 10,
                        "roi_y2": 10,
                        "relative_path": "data/gtsrb/images/does_not_exist.png",
                    },
                    {
                        "image_id": -2,
                        "class_id": 2,
                        "class_name": "Test corrupt",
                        "roi_x1": 0,
                        "roi_y1": 0,
                        "roi_x2": 10,
                        "roi_y2": 10,
                        "relative_path": corrupt_rel,
                    },
                ]
            )
            bad_df, bad_stats = extract_image_features(
                bad_rows, project_root=self.project_root, save_csv=False
            )
            assert bad_df.empty, "Expected 0 valid feature rows from bad test inputs."
            assert bad_stats["skipped"]["missing_file"] == 1, "Missing file was not counted."
            assert bad_stats["skipped"]["unreadable_image"] == 1, "Corrupt image was not counted."
        finally:
            corrupt_path.unlink(missing_ok=True)

        # 2. Verify required output columns exist
        triage_df = (
            self.triage_df
            if self.triage_df is not None
            else pd.read_csv(self.project_root / "outputs" / "processed" / "triage_results.csv")
        )
        required_cols = [
            "image_id",
            "image_path",
            "class_id",
            "class_name",
            "width",
            "height",
            "brightness",
            "contrast",
            "sharpness",
            "priority_level",
            "needs_review",
        ]
        missing_cols = [c for c in required_cols if c not in triage_df.columns]
        assert not missing_cols, f"Missing expected columns in triage_results: {missing_cols}"

        # 3. Verify review flags are strictly True/False booleans with no NaNs
        assert triage_df["needs_review"].isna().sum() == 0, "needs_review contains NaNs."
        assert set(triage_df["needs_review"].unique()).issubset({True, False}), (
            "needs_review must contain only True/False values."
        )
        assert pd.api.types.is_bool_dtype(triage_df["needs_review"]), (
            "needs_review column must have boolean dtype."
        )

        # 4. Verify every row traces back to an existing relative path inside the repo
        sample_paths = triage_df["image_path"].astype(str)
        assert not sample_paths.str.contains(r":|^/|\\Users\\", regex=True).any(), (
            "Found non-relative or machine-specific path in triage_results."
        )
        missing_targets = [
            p for p in sample_paths if not (self.project_root / p).exists()
        ]
        assert not missing_targets, f"Found {len(missing_targets)} missing source files."

        print("All 4 validation checks passed (error catching, columns, boolean flags, relative traceability).")


if __name__ == "__main__":
    audit = TrafficSignAudit()
    summaries = audit.run_all()
    print(f"Full GTSRB dataset : {audit.audit_info['total_rows']:,} images across {audit.audit_info['num_classes']} classes")
    print(f"Analyzed sample    : {len(audit.triage_df):,} images across {audit.triage_df['class_id'].nunique()} classes")
    print(f"Flagged for review : {int(audit.triage_df['needs_review'].sum()):,} ({100 * audit.triage_df['needs_review'].mean():.2f}%)")
    print(f"Saved figures      : {len(audit.figure_paths)} PNGs in outputs/figures/")
    audit.run_validation_checks()
