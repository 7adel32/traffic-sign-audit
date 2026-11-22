"""Download and unpack the GTSRB training dataset into data/gtsrb/."""

import io
from pathlib import Path
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def prepare_gtsrb(project_root: Path | None = None) -> Path:
    """Download GTSRB parquet + metadata and extract PNGs with a relative-path manifest."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    data_dir = project_root / "data" / "gtsrb"
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading GTSRB dataset files...")
    parquet_file = hf_hub_download(
        repo_id="bazyl/GTSRB",
        filename="data/train-00000-of-00001-dc762c064c221993.parquet",
        repo_type="dataset",
    )
    meta_csv = hf_hub_download(
        repo_id="bazyl/GTSRB",
        filename="Meta.csv",
        repo_type="dataset",
    )

    meta_df = pd.read_csv(meta_csv)
    meta_df.to_csv(data_dir / "meta.csv", index=False)

    table = pq.read_table(parquet_file)
    total_rows = table.num_rows
    print(f"Extracting {total_rows} training images into {images_dir.relative_to(project_root)}...")

    manifest_rows = []
    for idx in range(total_rows):
        row = {col: table[col][idx].as_py() for col in table.column_names}
        class_id = int(row["ClassId"])
        class_dir = images_dir / f"{class_id:05d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        img_path = class_dir / f"{idx:05d}.png"
        if not img_path.exists():
            img_bytes = row["Path"]["bytes"]
            with Image.open(io.BytesIO(img_bytes)) as img:
                img.save(img_path, format="PNG")

        rel_path = img_path.relative_to(project_root).as_posix()
        manifest_rows.append(
            {
                "image_id": idx,
                "class_id": class_id,
                "width": int(row["Width"]),
                "height": int(row["Height"]),
                "roi_x1": int(row["Roi.X1"]),
                "roi_y1": int(row["Roi.Y1"]),
                "roi_x2": int(row["Roi.X2"]),
                "roi_y2": int(row["Roi.Y2"]),
                "relative_path": rel_path,
            }
        )

        if (idx + 1) % 5000 == 0 or (idx + 1) == total_rows:
            print(f"  Extracted {idx + 1}/{total_rows} images")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = data_dir / "train_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Saved manifest: {manifest_path.relative_to(project_root)}")
    return manifest_path


if __name__ == "__main__":
    prepare_gtsrb()
