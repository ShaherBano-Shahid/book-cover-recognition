import hashlib
import json
from pathlib import Path

import joblib
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PREPARED_DIR = ROOT / "prepared_data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
IMAGE_SIZE = 224
SEED = 42
MODEL_FILENAMES = {
    "mobilenet": "mobilenet.keras",
    "mobilenet_tuned": "mobilenet_tuned.keras",
    "cnn": "cnn.keras",
    "svm": "svm.joblib",
    "mobilenet_updated": "mobilenet_updated.keras",
}
MODEL_LABELS = {
    "mobilenet_updated": "MobileNetV2 + Dense",
    "mobilenet_tuned": "MobileNetV2 (fine-tuned)",
    "mobilenet": "MobileNetV2",
    "cnn": "Basic CNN",
    "svm": "HOG + SVM",
}
GENRES = [
    "Arts & Photography",
    "Biographies & Memoirs",
    "Business & Money",
    "Calendars",
    "Children's Books",
    "Comics & Graphic Novels",
    "Computers & Technology",
    "Cookbooks, Food & Wine",
    "Crafts, Hobbies & Home",
    "Christian Books & Bibles",
    "Engineering & Transportation",
    "Health, Fitness & Dieting",
    "History",
    "Humor & Entertainment",
    "Law",
    "Literature & Fiction",
    "Medical Books",
    "Mystery, Thriller & Suspense",
    "Parenting & Relationships",
    "Politics & Social Sciences",
    "Reference",
    "Religion & Spirituality",
    "Romance",
    "Science & Math",
    "Science Fiction & Fantasy",
    "Self-Help",
    "Sports & Outdoors",
    "Teen & Young Adult",
    "Test Preparation",
    "Travel",
]


def save_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_image(source, size=IMAGE_SIZE):
    if hasattr(source, "getvalue"):
        encoded = np.frombuffer(source.getvalue(), dtype=np.uint8)
    elif hasattr(source, "read"):
        encoded = np.frombuffer(source.read(), dtype=np.uint8)
    else:
        encoded = np.fromfile(str(source), dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError("The image file is empty.")
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not read this image.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return image.astype(np.float32)


def read_split(name):
    path = PREPARED_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError("Run prepare_data.py after extracting the dataset images.")
    rows = pd.read_csv(path)
    if rows.empty or not rows["label"].isin(range(len(GENRES))).all():
        raise ValueError(f"Invalid or empty split: {path}")
    return rows


def dataset_id():
    digest = hashlib.sha256()
    for name in ("train", "validation", "test"):
        digest.update((PREPARED_DIR / f"{name}.csv").read_bytes())
    return digest.hexdigest()


def save_model_info(name, **settings):
    save_json(
        MODEL_DIR / f"{name}_info.json", {
            "genres": GENRES,
            "dataset_id": dataset_id(),
            **settings,
        }
    )


def load_classifier(name):
    info = read_json(MODEL_DIR / f"{name}_info.json")
    if info["genres"] != GENRES:
        raise ValueError("The saved genre mapping does not match BookCover30.")
    if name in ("cnn", "mobilenet", "mobilenet_tuned", "mobilenet_updated"):
        import tensorflow as tf

        model = tf.keras.models.load_model(MODEL_DIR / MODEL_FILENAMES[name], compile=False)
        if tuple(model.input_shape[1:]) != (224, 224, 3) or model.output_shape[-1] != len(GENRES):
            raise ValueError("The model must accept 224 x 224 RGB images and return 30 genres.")
        return model
    if name == "svm":
        model = joblib.load(MODEL_DIR / "svm.joblib")
        if list(model.classes_) != list(range(len(GENRES))):
            raise ValueError("The SVM genre order is incorrect.")
        return model
    raise ValueError(f"Unknown model: {name}")


def make_dataset(rows, batch_size=16, training=False):
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices((rows["path"].tolist(), rows["label"].tolist()))
    if training:
        dataset = dataset.shuffle(len(rows), seed=SEED, reshuffle_each_iteration=True)

    def load(path, label):
        image = tf.numpy_function(
            lambda value: read_image(value.decode("utf-8")), [path], tf.float32
        )
        image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
        return image, label

    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    return dataset.map(load,
                       num_parallel_calls=2).batch(batch_size).with_options(options).prefetch(1)


def extract_hog(source):
    image = cv2.cvtColor(read_image(source, 128) / 255.0, cv2.COLOR_RGB2GRAY)
    return hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True
    ).astype(np.float32)


def hog_matrix(rows):
    features = np.empty((len(rows), 1764), dtype=np.float32)
    for index, path in enumerate(rows["path"]):
        features[index] = extract_hog(path)
        if (index + 1) % 1000 == 0:
            print(f"HOG features: {index + 1:,}/{len(rows):,}", flush=True)
    return features


def predict_scores(model, name, source):
    if name != "svm":
        return model(read_image(source)[None], training=False).numpy()[0]
    return model.decision_function(extract_hog(source)[None])[0]
