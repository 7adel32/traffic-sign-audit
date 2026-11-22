"""Run the full Traffic Sign Visibility Audit pipeline and validation checks."""

from src.audit import TrafficSignAudit


def main() -> None:
    audit = TrafficSignAudit()
    summaries = audit.run_all()

    print("=== Traffic Sign Visibility Audit Complete ===")
    print(f"Full GTSRB dataset : {audit.audit_info['total_rows']:,} images across {audit.audit_info['num_classes']} classes")
    print(f"Selected sample    : {len(audit.triage_df):,} images across {audit.triage_df['class_id'].nunique()} classes (seed={audit.seed})")
    print(f"Skipped on load    : {audit.extraction_stats['skipped']} | ROI fallbacks: {audit.extraction_stats['roi_fallback_count']}")
    print(f"Baseline thresholds: {audit.thresholds}")
    print("\nFlag rate by class:")
    print(summaries["flag_summary"][["class_name", "total", "flagged_count", "flagged_rate_pct"]].to_string(index=False))
    print("\nSaved figures:")
    for fig_path in audit.figure_paths:
        print(f"  - {fig_path.relative_to(audit.project_root).as_posix()}")

    print("\nRunning validation checks...")
    audit.run_validation_checks()


if __name__ == "__main__":
    main()
