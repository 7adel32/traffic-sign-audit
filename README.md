# Traffic Sign Visibility Audit

A reproducible Python workflow using **NumPy**, **pandas**, **Matplotlib**, and **OpenCV** (`cv2`) to inspect real traffic sign images, measure basic visual properties (brightness, contrast, and sharpness), and build a percentile-based triage rule for manual review.

**Central Question:** Which traffic sign images should be reviewed first, and what measured evidence supports that priority?

> Note: Triage cutoffs used in this project are project-specific settings for prioritizing manual inspection—not official safety standards or classifier predictions.

---

## Project Structure

```text
traffic-sign-audit/
├── README.md
├── requirements.txt
├── main.py                         # Runs the full audit pipeline and validation checks
├── traffic_sign_audit.ipynb        # Synced notebook (%%writefile + %run for each module)
├── data/
│   ├── README.md                   # Dataset source, citation, and download instructions
│   └── gtsrb/                      # Raw GTSRB images (ignored in Git) + train_manifest.csv
├── src/
│   ├── config.py                   # Paths, selected classes, seed, and cutoff settings
│   ├── data_prep.py                # Full-dataset inspection and stratified sampling
│   ├── features.py                 # ROI validation, OpenCV feature extraction, ROI vs full check
│   ├── triage.py                   # Vectorized NumPy triage rule and sensitivity scenarios
│   ├── analysis.py                 # pandas groupby summaries answering the 5 audit questions
│   ├── plots.py                    # Matplotlib figures (no Seaborn)
│   ├── audit.py                    # TrafficSignAudit class and validation checks
│   └── download_data.py            # Optional helper to fetch and unpack GTSRB locally
├── outputs/
│   ├── figures/                    # Saved PNG plots and contact sheet
│   └── processed/                  # Derived CSV tables and summaries
└── reports/
    ├── Traffic_Sign_Visibility_Audit_Report_Adel_Alatawi_481000670.docx
    └── Traffic_Sign_Visibility_Audit_Report_Adel_Alatawi_481000670.pdf
```

---

## Setup Instructions

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Obtaining the Dataset

Raw image files are not tracked in Git (`data/gtsrb/images/` is listed in `.gitignore`). If `data/gtsrb/train_manifest.csv` and `data/gtsrb/images/` are not present locally, run:

```bash
python src/download_data.py
```

See [data/README.md](data/README.md) for dataset documentation and manual setup notes.

---

## Running the Project

### Option 1: Run the Full Pipeline from Terminal

```bash
python main.py
```

This runs all stages in order (`load_and_inspect`, `extract_features`, `apply_triage`, `summarize`, `plot_all`, and `run_validation_checks`) and saves all derived CSV tables in `outputs/processed/` and figures in `outputs/figures/`.

You can also run any individual module directly (for example, `python src/triage.py`).

### Option 2: Interactive Notebook Workflow

Open `traffic_sign_audit.ipynb` in Jupyter or VS Code and run all cells from top to bottom.

- Each module has a `%%writefile "src/<module>.py"` cell so edits made in the notebook update the `.py` file directly, followed by a `%run "src/<module>.py"` cell to execute it.
- To change selected sign classes, sample size, random seed, or percentile cutoffs, edit `src/config.py` (or its cell in the notebook) and re-run.

---

## Dataset Citation

- **Official benchmark page:** https://benchmark.ini.rub.de/gtsrb_dataset.html
- **Archival dataset record:** https://zenodo.org/records/13741936
- **Dataset citation:** German Traffic Sign Recognition Benchmark (GTSRB), Real-Time Computer Vision Group, Ruhr-Universität Bochum.

```bibtex
@inproceedings{stallkamp2011german,
  title={The German Traffic Sign Recognition Benchmark: a multi-class classification competition},
  author={Stallkamp, Johannes and Schlipsing, Marc and Salmen, Jan and Igel, Christian},
  booktitle={The 2011 International Joint Conference on Neural Networks (IJCNN)},
  pages={1453--1460},
  year={2011},
  organization={IEEE}
}
```
