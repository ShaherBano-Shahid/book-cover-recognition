import hashlib
from time import perf_counter

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from common import (
    GENRES,
    MODEL_DIR,
    RESULT_DIR,
    dataset_id,
    load_classifier,
    make_dataset,
    read_json,
    read_split,
    save_json,
)


def feature_model(model):
    original = model.layers[-1]
    pooling = next(
        layer for layer in original.layers
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)
    )
    embedding = tf.keras.Model(original.input, pooling.output)
    pixels = model.get_layer("normalization")(model.input)
    extractor = tf.keras.Model(model.input, embedding(pixels, training=False))
    extractor.trainable = False
    return extractor, original.layers[-1]


def features(extractor, rows, split):
    print(f"Extracting {split} features: {len(rows):,} images", flush=True)
    batches = []
    started = perf_counter()
    done = 0
    for images, _ in make_dataset(rows, batch_size=32):
        batches.append(extractor(images, training=False).numpy())
        done += len(images)
        if done % 512 == 0 or done == len(rows):
            elapsed = perf_counter() - started
            remaining = elapsed / done * (len(rows) - done)
            print(
                f"{split}: {done:,}/{len(rows):,}; elapsed {elapsed / 60:.1f} min; estimated remaining {remaining / 60:.1f} min",
                flush=True
            )
    return np.concatenate(batches)


class ValidationScores(tf.keras.callbacks.Callback):

    def __init__(self, values, labels):
        super().__init__()
        self.values = values
        self.labels = labels

    def on_epoch_end(self, epoch, logs=None):
        predicted = np.argmax(self.model(self.values, training=False).numpy(), axis=1)
        logs["val_macro_f1"] = float(
            f1_score(self.labels, predicted, average="macro", zero_division=0)
        )
        print(f"Epoch {epoch + 1}: validation macro F1={logs['val_macro_f1']:.4f}", flush=True)


def train_head(kind, initial, train_x, train_y, val_x, val_y, epochs=20):
    inputs = tf.keras.Input((train_x.shape[1], ))
    x = inputs
    if kind == "hidden":
        x = tf.keras.layers.Dense(
            128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(len(GENRES), activation="softmax")
    head = tf.keras.Model(inputs, output(x))
    if kind == "linear":
        output.set_weights(initial.get_weights())
    head.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4 if kind == "linear" else 1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    checkpoint = RESULT_DIR / f"update_{kind}.keras"
    print(f"Training {kind} classifier on cached features", flush=True)
    history = head.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        batch_size=128,
        epochs=epochs,
        verbose=2,
        callbacks=[
            ValidationScores(val_x, val_y),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint, monitor="val_macro_f1", mode="max", save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_macro_f1", mode="max", patience=4, restore_best_weights=True
            ),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
    )
    save_json(RESULT_DIR / f"update_{kind}_history.json", history.history)
    return tf.keras.models.load_model(checkpoint, compile=False)


def improve():
    tf.keras.utils.set_random_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    target = MODEL_DIR / "mobilenet_updated.keras"
    if target.exists() or (MODEL_DIR / "mobilenet_updated_info.json").exists():
        raise FileExistsError(
            "An updated candidate already exists. Keep or rename it before another experiment."
        )
    training = read_split("train").groupby("label", group_keys=False
                                           ).sample(n=200, random_state=42).sort_values("label")
    validation = read_split("validation").sort_values("label")
    source = MODEL_DIR / "mobilenet_tuned.keras"
    signature = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dataset_id": dataset_id(),
        "training_images": 6000,
        "seed": 42
    }
    model = load_classifier("mobilenet_tuned")
    extractor, initial = feature_model(model)
    cache = RESULT_DIR / "update_features.npz"
    cache_info = RESULT_DIR / "update_features.json"
    if cache.is_file() and cache_info.is_file() and read_json(cache_info) == signature:
        with np.load(cache) as stored:
            train_x, val_x = stored["train"], stored["validation"]
        print("Using cached features", flush=True)
    else:
        train_x = features(extractor, training, "train")
        val_x = features(extractor, validation, "validation")
        np.savez(cache, train=train_x, validation=val_x)
        save_json(cache_info, signature)
    train_y, val_y = training["label"].to_numpy(), validation["label"].to_numpy()
    baseline_scores = initial(val_x, training=False).numpy()
    baseline_predicted = baseline_scores.argmax(axis=1)
    baseline = {
        "accuracy": float(accuracy_score(val_y, baseline_predicted)),
        "macro_f1": float(f1_score(val_y, baseline_predicted, average="macro", zero_division=0))
    }
    print(f"Baseline validation: {baseline}", flush=True)
    trials = []
    best = None
    best_score = baseline["macro_f1"]
    for kind in ("linear", "hidden"):
        head = train_head(kind, initial, train_x, train_y, val_x, val_y)
        scores = head(val_x, training=False).numpy()
        for blend in (False, True):
            predicted = ((scores + baseline_scores) / 2 if blend else scores).argmax(axis=1)
            trial = {
                "head": kind,
                "blend_with_original": blend,
                "accuracy": float(accuracy_score(val_y, predicted)),
                "macro_f1": float(f1_score(val_y, predicted, average="macro", zero_division=0))
            }
            trials.append(trial)
            print(f"Candidate validation: {trial}", flush=True)
            if trial["macro_f1"] > best_score and trial["accuracy"] > baseline["accuracy"]:
                best_score, best = trial["macro_f1"], trial
    report = {
        "source_model": "mobilenet_tuned",
        "training_images": len(training),
        "validation_images": len(validation),
        "training_method": "Frozen CNN features, balanced training sample, retrained classifier",
        "baseline_validation": baseline,
        "trials": trials,
        "selected": best,
        "promoted": best is not None
    }
    save_json(RESULT_DIR / "improvement_experiment.json", report)
    if best is None:
        print(
            "No candidate improved both validation accuracy and macro F1. Original models retained.",
            flush=True
        )
        return
    head = tf.keras.models.load_model(RESULT_DIR / f"update_{best['head']}.keras", compile=False)
    scores = head(extractor.output, training=False)
    if best["blend_with_original"]:
        scores = tf.keras.layers.Average()([scores, initial(extractor.output, training=False)])
    updated = tf.keras.Model(extractor.input, scores, name="book_cover_updated")
    updated.save(target)
    info = read_json(MODEL_DIR / "mobilenet_tuned_info.json")
    info.update(
        {
            "source_model": "mobilenet_tuned",
            "training_method": report["training_method"],
            "training_images": len(training),
            "training_seed": 42,
            "candidate_settings": best,
            "dataset_id": dataset_id()
        }
    )
    save_json(MODEL_DIR / "mobilenet_updated_info.json", info)
    print(f"Saved {target.name}; all four original models retained.", flush=True)


if __name__ == "__main__":
    try:
        improve()
    except (OSError, ValueError) as error:
        raise SystemExit(f"Experiment stopped: {error}")
