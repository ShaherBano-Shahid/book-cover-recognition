import argparse
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from common import (
    GENRES,
    MODEL_DIR,
    MODEL_FILENAMES,
    MODEL_LABELS,
    RESULT_DIR,
    dataset_id,
    hog_matrix,
    load_classifier,
    make_dataset,
    predict_scores,
    read_json,
    read_split,
    save_json,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate(name, split="test", scores=None):
    rows = read_split(split)
    info = read_json(MODEL_DIR / f"{name}_info.json")
    if info["dataset_id"] != dataset_id():
        raise ValueError("The prepared dataset has changed since this model was trained.")
    model = load_classifier(name)
    if scores is None:
        if name != "svm":
            scores = model.predict(make_dataset(rows), verbose=1)
        else:
            scores = model.decision_function(hog_matrix(rows))
    scores = np.asarray(scores)
    if scores.shape != (len(rows), len(GENRES)) or not np.isfinite(scores).all():
        raise ValueError("Evaluation requires one finite score for each image and genre.")
    predicted = np.argmax(scores, axis=1)
    actual = rows["label"].to_numpy()
    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(GENRES))),
        target_names=GENRES,
        output_dict=True,
        zero_division=0
    )
    matrix = confusion_matrix(actual, predicted, labels=list(range(len(GENRES))))
    top3 = np.argsort(-scores, axis=1)[:, :3]
    paths = rows["path"].sample(min(50, len(rows)), random_state=42)
    predict_scores(model, name, paths.iloc[0])
    timings = []
    for path in paths:
        started = perf_counter()
        predict_scores(model, name, path)
        timings.append((perf_counter() - started) * 1000)
    summary = {
        "model": name,
        "dataset_id": info["dataset_id"],
        "split": split,
        "images": len(rows),
        "origin": info.get("origin", "locally_trained"),
        "independent_test_verified": info.get("origin") != "imported"
        or info.get("training_split_verified", False),
        "evaluation_note": "Training/test separation is documented in the supplied report, but original split files were not supplied."
        if info.get("origin") == "imported" else
        "Locally trained with prepared splits and duplicate checks.",
        "accuracy": float(accuracy_score(actual, predicted)),
        "top3_accuracy": float(np.mean(np.any(top3 == actual[:, None], axis=1))),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "mean_inference_ms": float(np.mean(timings)),
        "latency_samples": len(timings),
        "latency_scope": "Warm model; includes image loading and preprocessing",
    }
    if split == "test":
        summary["test_images"] = len(rows)
    prefix = name if split == "test" else f"{name}_{split}"
    save_json(RESULT_DIR / f"{prefix}_metrics.json", summary)
    pd.DataFrame(report).T.to_csv(RESULT_DIR / f"{prefix}_classification_report.csv")
    rows.assign(
        predicted_label=predicted,
        actual_genre=[GENRES[label] for label in actual],
        predicted_genre=[GENRES[label] for label in predicted],
        correct=actual == predicted
    ).to_csv(RESULT_DIR / f"{prefix}_predictions.csv", index=False)
    pd.DataFrame(matrix, index=GENRES,
                 columns=GENRES).to_csv(RESULT_DIR / f"{prefix}_confusion_matrix.csv")
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0
    )
    figure, axes = plt.subplots(figsize=(14, 12))
    chart = axes.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axes.set_xticks(range(len(GENRES)), GENRES, rotation=90, fontsize=7)
    axes.set_yticks(range(len(GENRES)), GENRES, fontsize=7)
    axes.set(
        xlabel="Predicted genre",
        ylabel="Actual genre",
        title=f"{MODEL_LABELS[name]}: {split} confusion matrix"
    )
    figure.colorbar(chart, ax=axes, label="Fraction of actual genre")
    figure.tight_layout()
    figure.savefig(RESULT_DIR / f"{prefix}_confusion_matrix.png", dpi=140)
    plt.close(figure)
    for key in ("accuracy", "top3_accuracy", "macro_precision", "macro_recall", "macro_f1"):
        print(f"{key}: {summary[key]:.2%}")
    print(f"Mean inference time: {summary['mean_inference_ms']:.1f} ms")
    print("Reports saved in results/")
    if split == "test":
        save_comparison()
    return summary


def save_comparison():
    paths = [RESULT_DIR / f"{name}_metrics.json" for name in MODEL_FILENAMES]
    reports = [read_json(path) for path in paths if path.is_file()]
    if len(reports) < 2:
        return
    if len({report["dataset_id"] for report in reports}) != 1:
        return
    for report in reports:
        report.setdefault("independent_test_verified", report["model"] == "svm")
    columns = [
        "model", "accuracy", "top3_accuracy", "macro_precision", "macro_recall", "macro_f1",
        "mean_inference_ms", "independent_test_verified"
    ]
    table = pd.DataFrame(reports)[columns]
    table["model"] = table["model"].map(MODEL_LABELS)
    table.to_csv(RESULT_DIR / "model_comparison.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    if not table["independent_test_verified"].all():
        figure.suptitle(
            "Recorded test-split results; imported training overlap not independently verified",
            fontsize=10
        )
    table.set_index("model")[["accuracy", "macro_precision", "macro_recall",
                              "macro_f1"]].T.plot.bar(ax=axes[0], rot=0)
    axes[0].set(ylim=(0, 1), ylabel="Score", title="Test-set performance")
    axes[0].set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    axes[0].legend(title="Model")
    axes[1].bar(table["model"], table["mean_inference_ms"], color="#167bd4")
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].set(ylabel="Milliseconds per image", title="Inference time")
    figure.tight_layout()
    figure.savefig(RESULT_DIR / "model_comparison.png", dpi=140)
    plt.close(figure)
    print("Model comparison saved in results/model_comparison.csv and .png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the official test set"
    )
    parser.add_argument("--model", choices=(*MODEL_FILENAMES, "all"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    try:
        names = [
            name for name, filename in MODEL_FILENAMES.items() if name != "mobilenet_updated" or
            ((MODEL_DIR / filename).is_file() and (MODEL_DIR / f"{name}_info.json").is_file())
        ] if args.model == "all" else [args.model]
        for name in names:
            print(f"\nEvaluating {MODEL_LABELS[name]} on {args.split}", flush=True)
            evaluate(name, args.split)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
