import hashlib
import json
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import tensorflow as tf

from common import (
    GENRES,
    MODEL_DIR,
    MODEL_FILENAMES,
    ROOT,
    dataset_id,
    read_image,
    read_json,
    read_split,
    save_json,
)

SOURCES = {
    "cnn": "book_cover_cnn.keras.zip",
    "mobilenet": "book_cover_mobilenet.keras.zip",
    "mobilenet_tuned": "book_cover_mobilenet_finetuned.keras.zip",
}


def compatible_config(value):
    if isinstance(value, dict):
        if value.get("class_name") == "Dense":
            config = value.get("config", {})
            if config.get("quantization_config", "absent") is None:
                config.pop("quantization_config")
        for child in value.values():
            compatible_config(child)
    elif isinstance(value, list):
        for child in value:
            compatible_config(child)


def import_model(name, source):
    source = Path(source)
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = MODEL_DIR / MODEL_FILENAMES[name]
    metadata = MODEL_DIR / f"{name}_info.json"
    if destination.exists() or metadata.exists():
        if destination.is_file() and metadata.is_file():
            info = read_json(metadata)
            if info.get("source_sha256") == checksum and info.get("adapter_version") == 1:
                print(f"Keeping imported {name} model.", flush=True)
                return
        raise FileExistsError(f"Keep or rename the existing {destination.name} and metadata first.")
    with tempfile.TemporaryDirectory(prefix=".model-check-", dir=ROOT) as folder:
        temporary = Path(folder)
        compatible = temporary / "compatible.keras"
        with ZipFile(source) as original, ZipFile(compatible, "w") as adapted:
            if original.testzip() is not None:
                raise ValueError(f"Damaged model archive: {source.name}")
            config = json.loads(original.read("config.json"))
            compatible_config(config)
            for entry in original.infolist():
                content = json.dumps(config).encode(
                ) if entry.filename == "config.json" else original.read(entry.filename)
                adapted.writestr(entry, content)
        model = tf.keras.models.load_model(compatible, compile=False, safe_mode=True)
        if tuple(model.input_shape[1:]) != (224, 224, 3) or model.output_shape[-1] != len(GENRES):
            raise ValueError(
                "Imported model must accept 224 x 224 RGB images and return 30 genres."
            )
        scale, offset = (1 / 255, 0) if name == "cnn" else (1 / 127.5, -1)
        inputs = tf.keras.Input((224, 224, 3))
        pixels = tf.keras.layers.Rescaling(scale, offset=offset, name="normalization")(inputs)
        wrapped = tf.keras.Model(inputs, model(pixels, training=False), name=f"book_cover_{name}")
        sample = read_image(read_split("validation").iloc[0]["path"])[None]
        expected = model(sample * scale + offset, training=False).numpy()
        actual = wrapped(sample, training=False).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        if not np.isfinite(actual).all() or not np.allclose(actual.sum(axis=1), 1, atol=1e-4):
            raise ValueError("Imported model returned invalid probabilities.")
        checkpoint = temporary / destination.name
        wrapped.save(checkpoint)
        info = {
            "genres": GENRES,
            "dataset_id": dataset_id(),
            "dataset_id_scope": "Evaluation splits only; original training splits were not supplied",
            "origin": "imported",
            "source_file": source.name,
            "source_sha256": checksum,
            "adapter_version": 1,
            "training_split_verified": False,
            "genre_mapping_verified": False,
            "genre_mapping_source": "Official BookCover30 order, matching the supplied project document",
            "preprocessing": "RGB [0,255]; scaling is embedded in the saved model",
            "source_pixel_range": "[0,1]" if name == "cnn" else "[-1,1]",
            "preprocessing_source": "Supplied document screenshot"
            if name == "cnn" else "Supplied document training code",
        }
        save_json(temporary / metadata.name, info)
        checkpoint.replace(destination)
        (temporary / metadata.name).replace(metadata)
    tf.keras.backend.clear_session()
    print(f"Imported {source.name} as {destination.name}; original preserved.", flush=True)


if __name__ == "__main__":
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    try:
        for name, filename in SOURCES.items():
            import_model(name, MODEL_DIR / filename)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Import stopped: {error}")
