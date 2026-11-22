# GTSRB Dataset Notes

This project uses the labeled training split of the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

## Official Source & Archive

- **Official benchmark page:** https://benchmark.ini.rub.de/gtsrb_dataset.html
- **Archival Zenodo record:** https://zenodo.org/records/13741936

## Citation

> German Traffic Sign Recognition Benchmark (GTSRB), Real-Time Computer Vision Group, Ruhr-Universität Bochum.

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

## How to Obtain the Data Locally

Raw GTSRB images (`data/gtsrb/images/`) are excluded from Git via `.gitignore` to keep the repository lightweight.

### Option 1: Automatic Download Helper
From the project root directory, run:

```bash
python src/download_data.py
```

This downloads the training parquet archive and class metadata into `data/gtsrb/`, extracts the PNG images into `data/gtsrb/images/<class_id>/`, and generates `data/gtsrb/train_manifest.csv` using portable relative paths.

### Option 2: Manual Placement
If you download the official GTSRB training archive manually from Zenodo or the benchmark page, extract the class folders into `data/gtsrb/images/` and ensure `data/gtsrb/train_manifest.csv` contains columns:
`image_id, class_id, width, height, roi_x1, roi_y1, roi_x2, roi_y2, relative_path`.
