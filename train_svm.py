import joblib
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import MODEL_DIR, RESULT_DIR, SEED, hog_matrix, read_split, save_json, save_model_info


def train():
    if (MODEL_DIR / "svm.joblib").exists():
        raise FileExistsError("models/svm.joblib already exists. Rename it before training again.")
    training = read_split("train")
    validation = read_split("validation")
    train_features = hog_matrix(training)
    validation_features = hog_matrix(validation)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_f1, trials = -1, []
    for c in (0.1, 1.0, 10.0):
        model = make_pipeline(
            StandardScaler(), LinearSVC(C=c, dual="auto", max_iter=5000, random_state=SEED)
        )
        model.fit(train_features, training["label"])
        score = float(
            f1_score(
                validation["label"],
                model.predict(validation_features),
                average="macro",
                zero_division=0
            )
        )
        trials.append({"C": c, "validation_macro_f1": score})
        print(f"C={c}: validation macro F1={score:.4f}", flush=True)
        if score > best_f1:
            best_f1, best_c = score, c
            best_model = model
    joblib.dump(best_model, MODEL_DIR / "svm.joblib")
    save_model_info("svm", selected_C=best_c, validation_macro_f1=best_f1)
    save_json(RESULT_DIR / "svm_validation.json", trials)
    print("Saved models/svm.joblib")


if __name__ == "__main__":
    try:
        train()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Error: {error}")
