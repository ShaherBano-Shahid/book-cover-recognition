import argparse
import subprocess
import sys

from common import MODEL_DIR, PREPARED_DIR, RESULT_DIR, ROOT, dataset_id, read_json


def run_script(filename, *arguments):
    print(f"\nRunning {filename}", flush=True)
    subprocess.run([sys.executable, "-u", str(ROOT / filename), *arguments], cwd=ROOT, check=True)


def main(with_mobilenet=False):
    splits = [PREPARED_DIR / f"{name}.csv" for name in ("train", "validation", "test")]
    if not all(path.is_file() for path in splits):
        run_script("prepare_data.py")
    models = [("svm", "svm.joblib", "train_svm.py")]
    if with_mobilenet:
        models.append(("mobilenet", "mobilenet.keras", "train_mobilenet.py"))
    for name, filename, trainer in models:
        checkpoint = MODEL_DIR / filename
        metadata = MODEL_DIR / f"{name}_info.json"
        if not checkpoint.is_file() or not metadata.is_file():
            run_script(trainer)
        elif read_json(metadata)["dataset_id"] != dataset_id():
            raise ValueError(
                "Saved models use different dataset splits. Rename models and results before retraining."
            )
        else:
            print(f"Keeping trained {name} model.", flush=True)
        report = RESULT_DIR / f"{name}_metrics.json"
        if not report.is_file() or report.stat().st_mtime < checkpoint.stat().st_mtime:
            run_script("evaluate.py", "--model", name)
    print("\nTraining and evaluation finished. Run: python app.py", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the Milestone 2 SVM baseline")
    parser.add_argument(
        "--with-mobilenet", action="store_true", help="Also train and evaluate MobileNetV2"
    )
    args = parser.parse_args()
    try:
        main(**vars(args))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Project stopped: {error}")
