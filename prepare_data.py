import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

from common import DATA_DIR, GENRES, PREPARED_DIR, SEED, save_json


def read_labels(path):
    rows = pd.read_csv(path, sep=r"\s+", names=["filename", "label"], dtype={"filename": str})
    if rows.empty or not rows["label"].isin(range(len(GENRES))).all():
        raise ValueError(
            f"Labels must contain filenames and integer genre IDs from 0 to 29: {path}"
        )
    if rows["filename"].duplicated().any():
        raise ValueError(f"Repeated filenames in {path}")
    return rows


def download_labels():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        filename = f"bookcover30-labels-{split}.txt"
        path = DATA_DIR / filename
        if not path.exists():
            url = f"https://raw.githubusercontent.com/uchidalab/book-dataset/master/Task1/{filename}"
            with urlopen(url, timeout=60) as response:
                content = response.read()
            temporary = path.with_suffix(".download.txt")
            try:
                temporary.write_bytes(content)
                read_labels(temporary)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Downloaded {filename}")


def check_images(rows, images):
    paths, hashes, errors = [], [], []
    for index, filename in enumerate(rows["filename"]):
        path = images.get(filename)
        try:
            if path is None:
                raise FileNotFoundError("Image is missing")
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                pixels = np.asarray(image)
                digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            paths.append(str(path.resolve()))
            hashes.append(digest)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            paths.append("")
            hashes.append("")
            errors.append(f"{filename}: {error}")
        if (index + 1) % 5000 == 0:
            print(f"Checked {index + 1:,} images", flush=True)
    return rows.assign(path=paths, image_hash=hashes), errors


def clean_training(train, test):
    conflicts = train.groupby("image_hash")["label"].nunique()
    invalid_hashes = set(test["image_hash"]) | set(conflicts[conflicts > 1].index)
    return train[~train["image_hash"].isin(invalid_hashes)].drop_duplicates("image_hash")


def prepare(images_folder):
    images_folder = Path(images_folder)
    if not images_folder.is_dir():
        raise FileNotFoundError(
            f"Extract the cover images first. Folder not found: {images_folder}"
        )
    if any((PREPARED_DIR / f"{name}.csv").exists() for name in ("train", "validation", "test")):
        raise FileExistsError(
            "Prepared splits already exist. Rename prepared_data before preparing a new dataset."
        )
    download_labels()
    train = read_labels(DATA_DIR / "bookcover30-labels-train.txt")
    test = read_labels(DATA_DIR / "bookcover30-labels-test.txt")
    if set(train["filename"]) & set(test["filename"]):
        raise ValueError("Training and test labels contain overlapping filenames.")
    images = {}
    for path in images_folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            if path.name in images:
                raise ValueError(
                    f"Repeated image filename: {path.name}. Choose one extracted image folder."
                )
            images[path.name] = path
    train, train_errors = check_images(train, images)
    test, test_errors = check_images(test, images)
    audit = {
        "image_errors": train_errors + test_errors,
        "official_train": len(train),
        "official_test": len(test)
    }
    save_json(PREPARED_DIR / "audit.json", audit)
    if audit["image_errors"]:
        raise ValueError(
            "Missing/corrupt images found. See prepared_data/audit.json and fix them first."
        )
    cleaned = clean_training(train, test)
    audit["excluded_training_files"] = train.loc[~train.index.isin(cleaned.index),
                                                 "filename"].tolist()
    audit["duplicate_test_covers"] = int(test["image_hash"].duplicated().sum())
    save_json(PREPARED_DIR / "audit.json", audit)
    if any(cleaned["label"].value_counts().get(label, 0) < 2 for label in range(len(GENRES))):
        raise ValueError("Each genre needs at least two usable training images.")
    if set(test["label"]) != set(range(len(GENRES))):
        raise ValueError("The official test set must contain all 30 genres.")
    train, validation = train_test_split(
        cleaned, test_size=0.1, random_state=SEED, stratify=cleaned["label"]
    )
    distribution = pd.DataFrame({"genre": GENRES})
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        rows[["filename", "path", "label"]].to_csv(PREPARED_DIR / f"{name}.csv", index=False)
        distribution[name] = rows["label"].value_counts().reindex(range(len(GENRES)),
                                                                  fill_value=0).to_numpy()
        print(f"{name}: {len(rows):,} images")
    distribution.to_csv(PREPARED_DIR / "class_distribution.csv", index=False)
    save_json(PREPARED_DIR / "genres.json", GENRES)
    print(
        f"Excluded {len(audit['excluded_training_files'])} duplicate/conflicting training covers."
    )
    print("Dataset preparation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the official BookCover30 dataset")
    parser.add_argument("--images", type=Path, default=DATA_DIR / "images")
    args = parser.parse_args()
    try:
        prepare(args.images)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
