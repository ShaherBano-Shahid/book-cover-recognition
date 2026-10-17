import argparse

import tensorflow as tf

from common import (
    GENRES,
    MODEL_DIR,
    RESULT_DIR,
    SEED,
    make_dataset,
    read_split,
    save_json,
    save_model_info,
)


def build_model(weights="imagenet"):
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=weights,
        name="mobilenet_backbone",
    )
    backbone.trainable = False
    inputs = tf.keras.Input((224, 224, 3))
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.015),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.08, value_range=(0, 255)),
        ],
        name="augmentation"
    )
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="normalization")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(GENRES), activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


def fine_tune(model):
    backbone = model.get_layer("mobilenet_backbone")
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-40:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    compile_model(model, 1e-5)


def callbacks(filename):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_DIR / filename, monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def train(head_epochs=8, finetune_epochs=20, batch_size=16, weights="imagenet"):
    if head_epochs < 1 or finetune_epochs < 0 or batch_size < 1:
        raise ValueError(
            "Head epochs and batch size must be positive; fine-tuning epochs cannot be negative."
        )
    if (MODEL_DIR / "mobilenet.keras").exists():
        raise FileExistsError(
            "models/mobilenet.keras already exists. Rename it before training again."
        )
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    training = make_dataset(read_split("train"), batch_size, training=True)
    validation = make_dataset(read_split("validation"), batch_size)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(weights)
    compile_model(model, 1e-3)
    print("Stage 1: training the classification head", flush=True)
    head = model.fit(
        training,
        validation_data=validation,
        epochs=head_epochs,
        callbacks=callbacks("mobilenet_head.keras")
    )
    model = tf.keras.models.load_model(MODEL_DIR / "mobilenet_head.keras")
    best_loss = float(model.evaluate(validation, verbose=0)[0])
    best_stage = "head"
    best_checkpoint = MODEL_DIR / "mobilenet_head.keras"
    history = {"head": head.history}
    if finetune_epochs:
        fine_tune(model)
        print("Stage 2: fine-tuning the last 40 backbone layers", flush=True)
        fine = model.fit(
            training,
            validation_data=validation,
            epochs=finetune_epochs,
            callbacks=callbacks("mobilenet_finetuned.keras")
        )
        history["finetune"] = fine.history
        model = tf.keras.models.load_model(MODEL_DIR / "mobilenet_finetuned.keras")
        fine_loss = float(model.evaluate(validation, verbose=0)[0])
        if fine_loss < best_loss:
            best_checkpoint = MODEL_DIR / "mobilenet_finetuned.keras"
            best_loss, best_stage = fine_loss, "fine-tuned"
    model = tf.keras.models.load_model(best_checkpoint)
    model.save(MODEL_DIR / "mobilenet.keras")
    save_model_info(
        "mobilenet",
        best_stage=best_stage,
        validation_loss=best_loss,
        head_epochs=head_epochs,
        finetune_epochs=finetune_epochs,
        batch_size=batch_size,
        pretrained=weights is not None
    )
    save_json(RESULT_DIR / "training_history.json", history)
    print(f"Saved models/mobilenet.keras ({best_stage}, validation loss {best_loss:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and fine-tune MobileNetV2")
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    try:
        train(**vars(args))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
