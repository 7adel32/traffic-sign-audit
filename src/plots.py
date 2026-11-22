"""Generate Matplotlib figures (no Seaborn) for the Traffic Sign Visibility Audit."""

from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import BASELINE_CUTOFFS, PROCESSED_DIR, PROJECT_ROOT, ensure_output_dirs

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def _short_label(name: str) -> str:
    return name.replace("Speed limit ", "Speed ")


def _load_rgb(project_root: Path, rel_path: str) -> np.ndarray | None:
    img_bgr = cv2.imread(str(project_root / rel_path))
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def plot_contact_sheet(triage_df: pd.DataFrame, project_root: Path, figures_dir: Path) -> Path:
    """Figure 1: 2x5 contact sheet showing flagged and unflagged examples across all 5 classes."""
    classes = sorted(triage_df["class_id"].unique())
    fig, axes = plt.subplots(2, len(classes), figsize=(14.5, 7.4))

    for col_idx, cid in enumerate(classes):
        cls_df = triage_df[triage_df["class_id"] == cid]
        flagged = cls_df[cls_df["needs_review"]].sort_values(
            ["issue_count", "sharpness"], ascending=[False, True]
        )
        clear = cls_df[~cls_df["needs_review"]].sort_values("sharpness", ascending=False)
        # Pick median-sharpness unflagged sign so it is representative
        clear_row = clear.iloc[len(clear) // 2] if not clear.empty else cls_df.iloc[0]
        flag_row = flagged.iloc[0] if not flagged.empty else cls_df.iloc[0]

        for row_idx, (sample_row, status, color) in enumerate(
            [(flag_row, "FLAGGED", "#b2182b"), (clear_row, "OK", "#1b7837")]
        ):
            ax = axes[row_idx, col_idx]
            rgb = _load_rgb(project_root, sample_row["image_path"])
            if rgb is not None:
                ax.imshow(rgb)
            cls_short = _short_label(sample_row["class_name"])
            title = (
                f"{status}: {cls_short}\n"
                f"ID {int(sample_row['image_id'])} | B:{sample_row['brightness']:.0f} "
                f"C:{sample_row['contrast']:.0f} S:{sample_row['sharpness']:.0f}"
            )
            if status == "FLAGGED":
                title += f"\n[{sample_row['flag_reasons']}]"
            ax.set_title(title, fontsize=8.5, color=color, fontweight="bold", pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle(
        "Sample Contact Sheet: Flagged (Top) vs. Unflagged (Bottom) Across Selected Classes",
        fontsize=12.5,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.02,
        "Converted from BGR to RGB. Top row shows flagged images per class; bottom row shows representative unflagged images.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.06, hspace=0.40, wspace=0.14)
    out_path = figures_dir / "fig1_contact_sheet.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_distributions_with_cutoffs(
    triage_df: pd.DataFrame, cutoffs: dict[str, float], figures_dir: Path
) -> Path:
    """Figure 2: Brightness and Sharpness distributions with percentile cutoff lines."""
    b_low = float(np.percentile(triage_df["brightness"], cutoffs["brightness_low_pct"]))
    b_high = float(np.percentile(triage_df["brightness"], cutoffs["brightness_high_pct"]))
    s_cut = float(np.percentile(triage_df["sharpness"], cutoffs["sharpness_pct"]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].hist(triage_df["brightness"], bins=35, color="#4c78a8", edgecolor="white", alpha=0.85)
    axes[0].axvline(b_low, color="#d62728", linestyle="--", linewidth=2, label=f"P{cutoffs['brightness_low_pct']:.0f} cutoff ({b_low:.1f})")
    axes[0].axvline(b_high, color="#ff7f0e", linestyle="--", linewidth=2, label=f"P{cutoffs['brightness_high_pct']:.0f} cutoff ({b_high:.1f})")
    axes[0].set_title("ROI Brightness Distribution", fontweight="bold")
    axes[0].set_xlabel("Mean Grayscale Intensity (0-255)")
    axes[0].set_ylabel("Image Count")
    axes[0].legend(frameon=True)

    log_sharp = np.log10(1.0 + triage_df["sharpness"])
    log_s_cut = float(np.log10(1.0 + s_cut))
    axes[1].hist(log_sharp, bins=35, color="#2ca02c", edgecolor="white", alpha=0.85)
    axes[1].axvline(log_s_cut, color="#d62728", linestyle="--", linewidth=2, label=f"P{cutoffs['sharpness_pct']:.0f} cutoff (var={s_cut:.1f})")
    axes[1].set_title("ROI Sharpness Proxy Distribution (log10 scale)", fontweight="bold")
    axes[1].set_xlabel("log10(1 + Laplacian Variance)")
    axes[1].set_ylabel("Image Count")
    axes[1].legend(frameon=True)

    fig.suptitle("Brightness and Sharpness Distributions with Baseline Triage Cutoffs", fontsize=12.5, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "Dashed lines show project-specific sample percentile cutoffs used for review triage (not official safety limits).",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    out_path = figures_dir / "fig2_brightness_sharpness_distributions.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_class_boxplots(triage_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Figure 3: Class-wise boxplots of Brightness, Contrast, and log10 Sharpness."""
    classes = sorted(triage_df["class_name"].unique())
    labels = [_short_label(c) for c in classes]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    specs = [
        ("brightness", "Brightness (Mean Grayscale)", False),
        ("contrast", "Contrast (Grayscale Std Dev)", False),
        ("sharpness", "log10(1 + Laplacian Variance)", True),
    ]

    for ax, (col, ylabel, use_log) in zip(axes, specs):
        series_list = [triage_df[triage_df["class_name"] == c][col].to_numpy() for c in classes]
        if use_log:
            series_list = [np.log10(1.0 + vals) for vals in series_list]

        ax.boxplot(
            series_list,
            tick_labels=labels,
            patch_artist=True,
            boxprops=dict(facecolor="#cfe2f3", color="#2b5c8f"),
            medianprops=dict(color="#c0392b", linewidth=1.8),
            flierprops=dict(marker="o", markersize=2.5, alpha=0.4),
        )
        ax.set_title(ylabel.split(" (")[0] + " by Sign Class", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=22)

    bright_medians = triage_df.groupby("class_name")["brightness"].median().sort_values(ascending=False)
    top_bright = [_short_label(c) for c in bright_medians.index[:2]]

    fig.suptitle("Visibility Feature Distributions Across Selected Sign Classes", fontsize=12.5, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        f"Boxes show IQR and median (red). Sharpness spans wide ranges within classes; {' and '.join(top_bright)} have the highest median brightness.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    out_path = figures_dir / "fig3_class_feature_boxplots.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_flag_rate_by_class(triage_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Figure 4: Bar chart of baseline review-flag rate (%) by class."""
    cls_df = (
        triage_df.groupby("class_name", as_index=False)
        .agg(total=("needs_review", "size"), flagged=("needs_review", "sum"))
    )
    cls_df["rate_pct"] = 100.0 * cls_df["flagged"] / cls_df["total"]
    cls_df = cls_df.sort_values("rate_pct", ascending=False).reset_index(drop=True)
    overall_rate = 100.0 * float(triage_df["needs_review"].mean())

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    labels = [_short_label(c) for c in cls_df["class_name"]]
    bars = ax.bar(labels, cls_df["rate_pct"], color="#4c78a8", edgecolor="#2b4c70", width=0.58)
    ax.axhline(overall_rate, color="#d62728", linestyle="--", linewidth=1.8, label=f"Sample average ({overall_rate:.1f}%)")

    for bar, (_, row) in zip(bars, cls_df.iterrows()):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.8,
            f"{h:.1f}%\n({int(row['flagged'])}/{int(row['total'])})",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    above_avg = [_short_label(c) for c in cls_df[cls_df["rate_pct"] > overall_rate]["class_name"]]
    above_str = ", ".join(above_avg) if above_avg else "none"
    low_cls = _short_label(cls_df.iloc[-1]["class_name"])
    low_rate = cls_df.iloc[-1]["rate_pct"]

    ax.set_ylim(0, max(cls_df["rate_pct"].max() * 1.22, overall_rate + 8))
    ax.set_ylabel("Flagged for Manual Review (%)")
    ax.set_xlabel("Traffic Sign Class")
    ax.set_title("Baseline Review-Flag Rate by Sign Class (Percentile Triage Rule)", fontweight="bold")
    ax.legend(frameon=True)
    fig.text(
        0.5,
        0.015,
        f"Above-average flag rates: {above_str}; lowest: {low_cls} ({low_rate:.1f}%).",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    out_path = figures_dir / "fig4_flag_rate_by_class.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_size_and_contrast_by_class(triage_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Figure 5: Comparison of image size (min dimension in px) and contrast across classes."""
    classes = sorted(triage_df["class_name"].unique())
    labels = [_short_label(c) for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    size_data = [triage_df[triage_df["class_name"] == c]["min_dim"].to_numpy() for c in classes]
    axes[0].boxplot(
        size_data,
        tick_labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor="#fde0dd", color="#ae017e"),
        medianprops=dict(color="#7a0177", linewidth=1.8),
        flierprops=dict(marker="o", markersize=2.5, alpha=0.4),
    )
    axes[0].set_title("Image Size (Shortest Edge) by Class", fontweight="bold")
    axes[0].set_ylabel("Shortest Image Dimension (pixels)")
    axes[0].tick_params(axis="x", rotation=20)

    contrast_data = [triage_df[triage_df["class_name"] == c]["contrast"].to_numpy() for c in classes]
    axes[1].boxplot(
        contrast_data,
        tick_labels=labels,
        patch_artist=True,
        boxprops=dict(facecolor="#d9f0d3", color="#1b7837"),
        medianprops=dict(color="#00441b", linewidth=1.8),
        flierprops=dict(marker="o", markersize=2.5, alpha=0.4),
    )
    axes[1].set_title("Image Contrast (Grayscale Std Dev) by Class", fontweight="bold")
    axes[1].set_ylabel("Pixel Intensity Std Dev")
    axes[1].tick_params(axis="x", rotation=20)

    min_px = int(triage_df["min_dim"].min())
    max_px = int(triage_df["min_dim"].max())
    fig.suptitle("Comparison of Resolution and Contrast Across Sign Classes", fontsize=12.5, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        f"Shortest edge spans {min_px}-{max_px} px across classes (many small captures <=35 px), while contrast spread varies by sign design.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    out_path = figures_dir / "fig5_size_and_contrast_by_class.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_sensitivity_by_class(sensitivity_by_class_df: pd.DataFrame, figures_dir: Path) -> Path:
    """Figure 6: Grouped bar chart showing sensitivity of class flag rates to cutoff choices."""
    scenarios = list(sensitivity_by_class_df["scenario"].drop_duplicates())
    pivot = sensitivity_by_class_df.pivot(index="class_name", columns="scenario", values="flagged_rate_pct")
    sort_col = scenarios[1] if len(scenarios) > 1 else scenarios[0]
    pivot = pivot.sort_values(sort_col, ascending=False)

    classes = list(pivot.index)
    labels = [_short_label(c) for c in classes]
    x = np.arange(len(classes))
    width = 0.25
    colors = ["#2b5c8f", "#e67e22", "#7f8c8d", "#1b7837"]

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    center = (len(scenarios) - 1) / 2.0
    for i, scen in enumerate(scenarios):
        vals = pivot[scen].to_numpy()
        offset = (i - center) * width
        bars = ax.bar(x + offset, vals, width=width, label=scen, color=colors[i % len(colors)], edgecolor="white")
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Flagged Rate (%)")
    ax.set_xlabel("Traffic Sign Class")
    ax.set_ylim(0, pivot.max().max() * 1.18)
    ax.set_title("Sensitivity Check: Class Flag Rates Across Alternative Cutoffs", fontweight="bold")
    ax.legend(title="Cutoff Setting", frameon=True)
    fig.text(
        0.5,
        0.015,
        "Tightening or loosening percentile bounds shifts workload linearly while showing whether top-flagged classes stay consistent.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    out_path = figures_dir / "fig6_sensitivity_by_class.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_three_examples(examples_df: pd.DataFrame, project_root: Path, figures_dir: Path) -> Path:
    """Figure 7: Three illustrative example images with measurements and class labels."""
    n = len(examples_df)
    fig, axes = plt.subplots(1, max(1, n), figsize=(4.5 * max(1, n), 4.8))
    if n == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, examples_df.iterrows()):
        rgb = _load_rgb(project_root, row["image_path"])
        if rgb is not None:
            ax.imshow(rgb)
        ax.set_title(
            f"{row['example_role']}\n{_short_label(row['class_name'])} (ID {int(row['image_id'])})",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel(
            f"Size: {int(row['width'])}x{int(row['height'])} px | Priority: {row['priority_level']}\n"
            f"B={row['brightness']:.1f}, C={row['contrast']:.1f}, S={row['sharpness']:.1f}\n"
            f"Flags: {row['flag_reasons']}",
            fontsize=8.8,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Three Illustrative Examples Demonstrating the Review-Priority Rule", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    out_path = figures_dir / "fig7_three_examples.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def generate_all_plots(
    triage_df: pd.DataFrame,
    sensitivity_by_class_df: pd.DataFrame,
    examples_df: pd.DataFrame,
    cutoffs: dict[str, float] = BASELINE_CUTOFFS,
    project_root: Path = PROJECT_ROOT,
) -> list[Path]:
    """Generate and save all Matplotlib audit figures."""
    _, figures_dir = ensure_output_dirs(project_root)
    paths = [
        plot_contact_sheet(triage_df, project_root, figures_dir),
        plot_distributions_with_cutoffs(triage_df, cutoffs, figures_dir),
        plot_class_boxplots(triage_df, figures_dir),
        plot_flag_rate_by_class(triage_df, figures_dir),
        plot_size_and_contrast_by_class(triage_df, figures_dir),
        plot_sensitivity_by_class(sensitivity_by_class_df, figures_dir),
        plot_three_examples(examples_df, project_root, figures_dir),
    ]
    return paths


if __name__ == "__main__":
    triage = pd.read_csv(PROCESSED_DIR / "triage_results.csv")
    sens_cls = pd.read_csv(PROCESSED_DIR / "sensitivity_by_class.csv")
    examples = pd.read_csv(PROCESSED_DIR / "illustrative_examples.csv")
    saved = generate_all_plots(triage, sens_cls, examples)
    print(f"Generated {len(saved)} figures:")
    for p in saved:
        print(f"  - {p.relative_to(PROJECT_ROOT).as_posix()}")
