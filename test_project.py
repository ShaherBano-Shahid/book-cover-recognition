import io
import tempfile
import tkinter as tk
import unittest
import warnings
import hashlib
from contextlib import ExitStack
from pathlib import Path
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

import common
import app as desktop
import evaluate
import import_models
import improve_mobilenet
import prepare_data
import train_mobilenet
import train_svm


class ProjectTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50%",
            category=UserWarning
        )
        cls.temporary = tempfile.TemporaryDirectory(prefix=".test-", dir=common.ROOT)
        cls.root = Path(cls.temporary.name)
        cls.patches = ExitStack()
        paths = {
            "DATA_DIR": cls.root / "data",
            "PREPARED_DIR": cls.root / "prepared_data",
            "MODEL_DIR": cls.root / "models",
            "RESULT_DIR": cls.root / "results",
        }
        for module in (common, prepare_data, train_mobilenet, train_svm, evaluate, desktop):
            settings = {name: value for name, value in paths.items() if hasattr(module, name)}
            cls.patches.enter_context(patch.multiple(module, **settings))
        images = paths["DATA_DIR"] / "images"
        images.mkdir(parents=True)
        labels = {"train": [], "test": []}
        random = np.random.default_rng(42)
        for genre in range(30):
            for index in range(11):
                filename = f"cover_{genre}_{index}.jpg"
                pixels = random.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(images / filename)
                split = "test" if index == 10 else "train"
                labels[split].append(f"{filename} {genre}\n")
        for split, lines in labels.items():
            (paths["DATA_DIR"] / f"bookcover30-labels-{split}.txt").write_text("".join(lines))
        prepare_data.prepare(images)

    @classmethod
    def tearDownClass(cls):
        cls.patches.close()
        tf.keras.backend.clear_session()
        cls.temporary.cleanup()

    def test_dataset_splits(self):
        training = common.read_split("train")
        validation = common.read_split("validation")
        test = common.read_split("test")
        self.assertEqual((len(training), len(validation), len(test)), (270, 30, 30))
        self.assertFalse(set(training["filename"]) & set(validation["filename"]))
        self.assertFalse(set(training["filename"]) & set(test["filename"]))
        self.assertEqual(set(test["label"]), set(range(30)))
        self.assertEqual(common.GENRES[4], "Children's Books")

    def test_duplicate_covers(self):
        training = pd.DataFrame(
            {
                "image_hash": ["same", "same", "test", "conflict", "conflict"],
                "label": [0, 0, 1, 2, 3]
            }
        )
        test = pd.DataFrame({"image_hash": ["test"], "label": [1]})
        cleaned = prepare_data.clean_training(training, test)
        self.assertEqual(cleaned.index.tolist(), [0])
        self.assertEqual(len(test), 1)

    def test_invalid_labels(self):
        path = self.root / "invalid.txt"
        path.write_text("cover.jpg 30")
        with self.assertRaises(ValueError):
            prepare_data.read_labels(path)

    def test_missing_image(self):
        rows = pd.DataFrame({"filename": ["missing.jpg"], "label": [0]})
        checked, errors = prepare_data.check_images(rows, {})
        self.assertEqual(len(errors), 1)
        self.assertEqual(checked.iloc[0]["path"], "")

    def test_opencv_preprocessing(self):
        image = self.root / "red_cover.png"
        Image.new("RGB", (30, 40), (255, 0, 0)).save(image)
        pixels = common.read_image(image, 32)
        self.assertEqual(pixels.shape, (32, 32, 3))
        np.testing.assert_array_equal(pixels[0, 0], [255, 0, 0])
        self.assertEqual(pixels.dtype, np.float32)
        np.testing.assert_array_equal(common.read_image(io.BytesIO(image.read_bytes()), 32), pixels)

    def test_app_without_models(self):
        root = tk.Tk()
        root.withdraw()
        try:
            with patch.object(desktop, "MODEL_DIR", self.root / "empty_models"):
                gui = desktop.BookCoverApp(root)
                self.assertIn("Model unavailable", gui.status.get())
                self.assertEqual(str(gui.predict_button["state"]), "disabled")
        finally:
            root.destroy()

    def test_training_and_evaluation(self):
        original = (common.PREPARED_DIR / "train.csv").read_bytes()
        small = common.read_split("train").groupby("label").head(1)
        small.to_csv(common.PREPARED_DIR / "train.csv", index=False)
        try:
            train_mobilenet.train(head_epochs=1, finetune_epochs=1, batch_size=15, weights=None)
            train_svm.train()
            image = common.read_split("test").iloc[0]["path"]
            for name in ("mobilenet", "svm"):
                result = evaluate.evaluate(name)
                self.assertEqual(result["test_images"], 30)
                self.assertTrue(0 <= result["accuracy"] <= 1)
                self.assertTrue((common.RESULT_DIR / f"{name}_confusion_matrix.png").is_file())
                classifier = common.load_classifier(name)
                scores = common.predict_scores(classifier, name, image)
                self.assertEqual(scores.shape, (30, ))
                with open(image, "rb") as handle:
                    uploaded = io.BytesIO(handle.read())
                np.testing.assert_allclose(
                    common.predict_scores(classifier, name, uploaded), scores, atol=1e-6
                )
            self.assertTrue((common.RESULT_DIR / "model_comparison.csv").is_file())
            self.assertTrue((common.RESULT_DIR / "model_comparison.png").is_file())
            model = common.load_classifier("mobilenet")
            raw = common.read_image(image)[None]
            np.testing.assert_allclose(
                common.predict_scores(model, "mobilenet", image),
                model(raw, training=False).numpy()[0],
                atol=1e-6
            )
            fine = tf.keras.models.load_model(common.MODEL_DIR / "mobilenet_finetuned.keras")
            backbone = fine.get_layer("mobilenet_backbone")
            self.assertTrue(any(layer.trainable for layer in backbone.layers))
            self.assertTrue(
                all(
                    not layer.trainable for layer in backbone.layers
                    if isinstance(layer, tf.keras.layers.BatchNormalization)
                )
            )
            with self.assertRaises(FileExistsError):
                train_svm.train()
            root = tk.Tk()
            root.withdraw()
            try:
                gui = desktop.BookCoverApp(root)
                self.assertTrue(gui.select_image(image))
                for choice in ("MobileNetV2", "HOG + SVM"):
                    gui.model_choice.set(choice)
                    gui.change_model()
                    gui.predict_category()
                    wait_for_prediction(gui, root)
                    self.assertEqual(len(gui.results.get_children()), 3)
                    self.assertIn(gui.category.get(), common.GENRES)
                with patch.object(
                    desktop, "MODEL_NAMES", {
                        "MobileNetV2": "mobilenet",
                        "HOG + SVM": "svm"
                    }
                ):
                    gui.model_choice.set("Compare all")
                    gui.change_model()
                    gui.predict_category()
                    wait_for_prediction(gui, root)
                    self.assertEqual(len(gui.results.get_children()), 2)
            finally:
                root.destroy()
        finally:
            (common.PREPARED_DIR / "train.csv").write_bytes(original)


def wait_for_prediction(gui, root):
    deadline = monotonic() + 30
    while gui.busy and monotonic() < deadline:
        root.update()
        sleep(0.01)
    if gui.busy:
        raise AssertionError("Desktop prediction did not finish.")


class DesktopTests(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix=".test-", dir=common.ROOT)
        self.directory = Path(self.temporary.name)
        self.patches = ExitStack()
        self.patches.enter_context(patch.object(desktop, "MODEL_DIR", self.directory))
        self.patches.enter_context(patch.object(desktop, "RESULT_DIR", self.directory))
        self.root = tk.Tk()
        self.root.withdraw()
        self.gui = desktop.BookCoverApp(self.root)
        self.cover = self.directory / "cover.png"
        Image.new("RGB", (120, 180), "navy").save(self.cover)

    def tearDown(self):
        self.gui.close()
        self.patches.close()
        self.temporary.cleanup()

    def create_model_files(self):
        for name in desktop.MODEL_NAMES.values():
            checkpoint, metadata = desktop.model_files(name)
            checkpoint.write_bytes(b"test placeholder")
            common.save_json(metadata, {"dataset_id": "test"})

    def test_upload_clear_and_missing_model(self):
        self.assertTrue(self.gui.select_image(self.cover))
        self.assertEqual(self.gui.filename.get(), "cover.png")
        self.assertEqual(self.gui.original_image.size, (120, 180))
        self.assertEqual(str(self.gui.predict_button["state"]), "disabled")
        self.assertIn("Model unavailable", self.gui.status.get())
        self.gui.clear()
        self.assertIsNone(self.gui.image_path)
        self.assertEqual(self.gui.filename.get(), "")
        self.assertEqual(self.gui.category.get(), "—")
        self.assertEqual(len(self.gui.results.get_children()), 0)
        self.assertEqual(self.gui.results.winfo_manager(), "")

    def test_invalid_image_and_cancel(self):
        self.gui.select_image(self.cover)
        invalid = self.directory / "bad.png"
        invalid.write_bytes(b"not an image")
        with patch.object(desktop.messagebox, "showerror") as error:
            self.assertFalse(self.gui.select_image(invalid))
            error.assert_called_once()
        self.assertEqual(self.gui.image_path, self.cover)
        with patch.object(desktop.filedialog, "askopenfilename", return_value=""):
            self.gui.upload_image()
        self.assertEqual(self.gui.image_path, self.cover)

    def test_prediction_and_switching_models(self):
        self.create_model_files()
        self.gui.model_choice.set("MobileNetV2")
        self.gui.select_image(self.cover)
        scores = np.zeros(30)
        scores[18], scores[4], scores[1] = 0.8488, 0.1, 0.05
        with patch.object(desktop, "load_classifier", return_value=object()) as loader:
            with patch.object(desktop, "predict_scores", return_value=scores):
                self.gui.predict_category()
                self.assertTrue(self.gui.busy)
                wait_for_prediction(self.gui, self.root)
                self.assertEqual(self.gui.category.get(), "Parenting & Relationships")
                self.assertIn("84.88%", self.gui.score.get())
                self.assertNotIn("Time", self.gui.score.get())
                self.assertNotIn("ms", self.gui.score.get())
                self.assertEqual(len(self.gui.results.get_children()), 3)
                self.assertEqual(self.gui.results.winfo_manager(), "grid")
                self.gui.predict_category()
                wait_for_prediction(self.gui, self.root)
                self.assertEqual(loader.call_count, 1)
            self.gui.model_choice.set("HOG + SVM")
            self.gui.change_model()
            self.assertEqual(self.gui.category.get(), "—")
            scores[4] = 2.4
            with patch.object(desktop, "predict_scores", return_value=scores):
                self.gui.predict_category()
                wait_for_prediction(self.gui, self.root)
            self.assertEqual(self.gui.category.get(), "Children's Books")
            self.assertIn("2.400", self.gui.score.get())
            self.assertNotIn("%", self.gui.score.get())

    def test_compare_all_models(self):
        self.create_model_files()
        self.gui.select_image(self.cover)
        mobile_scores, svm_scores = np.zeros(30), np.zeros(30)
        mobile_scores[18], svm_scores[4] = 0.8, 2.4
        with patch.object(desktop, "load_classifier", return_value=object()):
            with patch.object(
                desktop,
                "predict_scores",
                side_effect=[mobile_scores] * (len(desktop.MODEL_NAMES) - 1) + [svm_scores]
            ):
                self.gui.predict_category()
                wait_for_prediction(self.gui, self.root)
        items = self.gui.results.get_children()
        self.assertEqual(len(items), len(desktop.MODEL_NAMES))
        self.assertEqual(
            self.gui.results.item(items[0], "values")[0], next(iter(desktop.MODEL_NAMES))
        )
        self.assertEqual(self.gui.results.item(items[-1], "values")[1], "Children's Books")
        self.assertEqual(self.gui.category.get(), "")
        self.assertEqual(self.gui.score.get(), "")
        self.assertEqual(self.gui.result_title.get(), "Model comparison")
        self.assertTrue(all("updated" not in label.lower() for label in desktop.MODEL_NAMES))

    def test_prediction_error_restores_controls(self):
        self.create_model_files()
        self.gui.select_image(self.cover)
        with patch.object(desktop, "predict_image", side_effect=ValueError("Broken checkpoint")):
            with patch.object(desktop.messagebox, "showerror") as error:
                self.gui.predict_category()
                wait_for_prediction(self.gui, self.root)
                error.assert_called_once_with("Prediction failed", "Broken checkpoint")
        self.assertEqual(str(self.gui.predict_button["state"]), "normal")
        self.assertIn("Prediction failed", self.gui.status.get())

    def test_measured_results(self):
        with patch.object(desktop.messagebox, "showinfo") as popup:
            self.gui.show_test_results()
            self.assertIn("No test results", popup.call_args.args[1])
        self.create_model_files()
        common.save_json(
            self.directory / "mobilenet_metrics.json", {
                "dataset_id": "test",
                "accuracy": 0.3,
                "macro_precision": 0.31,
                "macro_recall": 0.3,
                "macro_f1": 0.29,
                "mean_inference_ms": 18,
            }
        )
        with patch.object(desktop.messagebox, "showinfo") as popup:
            self.gui.show_test_results()
            self.assertIn("30.00%", popup.call_args.args[1])
            self.assertNotIn("inference", popup.call_args.args[1].lower())
            self.assertNotIn(" ms", popup.call_args.args[1])
        with patch.object(desktop, "predict_scores", return_value=np.zeros(29)):
            with patch.object(desktop, "load_classifier", return_value=object()):
                with self.assertRaises(ValueError):
                    desktop.predict_image(self.cover, "mobilenet", {})


class ImprovementTests(unittest.TestCase):

    def test_validation_checkpoint_can_be_loaded(self):
        values = np.random.default_rng(42).normal(size=(30, 4)).astype(np.float32)
        labels = np.arange(30)
        initial = tf.keras.layers.Dense(30, activation="softmax")
        initial.build((None, 4))
        with tempfile.TemporaryDirectory(prefix=".test-", dir=common.ROOT) as folder:
            with patch.object(improve_mobilenet, "RESULT_DIR", Path(folder)):
                head = improve_mobilenet.train_head(
                    "linear", initial, values, labels, values, labels, epochs=1
                )
                self.assertTrue((Path(folder) / "update_linear.keras").is_file())
                history = common.read_json(Path(folder) / "update_linear_history.json")
                self.assertTrue(np.isfinite(history["val_macro_f1"]).all())
                scores = head(values, training=False).numpy()
                self.assertEqual(scores.shape, (30, 30))
                np.testing.assert_allclose(scores.sum(axis=1), 1, atol=1e-5)
        tf.keras.backend.clear_session()

    def test_cached_features_preserve_original_predictions(self):
        inputs = tf.keras.Input((224, 224, 3))
        x = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        original = tf.keras.Model(inputs, tf.keras.layers.Dense(30, activation="softmax")(x))
        pixels = tf.keras.Input((224, 224, 3))
        scaled = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="normalization")(pixels)
        wrapped = tf.keras.Model(pixels, original(scaled, training=False))
        extractor, classifier = improve_mobilenet.feature_model(wrapped)
        sample = np.random.default_rng(42).uniform(0, 255, (2, 224, 224, 3)).astype(np.float32)
        expected = wrapped(sample, training=False).numpy()
        actual = classifier(extractor(sample, training=False), training=False).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        self.assertFalse(extractor.trainable)
        self.assertEqual(len(extractor.trainable_weights), 0)
        self.assertTrue(classifier.trainable)
        tf.keras.backend.clear_session()


class ImportTests(unittest.TestCase):

    def test_preprocessing_and_original_files(self):
        with tempfile.TemporaryDirectory(prefix=".test-", dir=common.ROOT) as folder:
            root = Path(folder)
            models = root / "models"
            prepared = root / "prepared_data"
            models.mkdir()
            prepared.mkdir()
            image = root / "cover.png"
            Image.new("RGB", (32, 32), (255, 64, 0)).save(image)
            for split in ("train", "validation", "test"):
                pd.DataFrame({
                    "path": [str(image)],
                    "label": [0]
                }).to_csv(prepared / f"{split}.csv", index=False)
            with ExitStack() as patches:
                patches.enter_context(
                    patch.multiple(common, MODEL_DIR=models, PREPARED_DIR=prepared)
                )
                patches.enter_context(patch.multiple(import_models, MODEL_DIR=models, ROOT=root))
                for name in ("cnn", "mobilenet", "mobilenet_tuned"):
                    source = tf.keras.Sequential(
                        [
                            tf.keras.Input((224, 224, 3)),
                            tf.keras.layers.GlobalAveragePooling2D(),
                            tf.keras.layers.Dense(30, activation="softmax"),
                        ]
                    )
                    archive = models / f"source_{name}.keras"
                    source.save(archive)
                    archive = archive.rename(archive.with_suffix(".keras.zip"))
                    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
                    import_models.import_model(name, archive)
                    loaded = common.load_classifier(name)
                    scale, offset = (1 / 255, 0) if name == "cnn" else (1 / 127.5, -1)
                    expected = source(
                        common.read_image(image)[None] * scale + offset, training=False
                    ).numpy()[0]
                    np.testing.assert_allclose(
                        common.predict_scores(loaded, name, image), expected, atol=1e-6
                    )
                    self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), checksum)
                    self.assertFalse(
                        common.read_json(models / f"{name}_info.json")["training_split_verified"]
                    )
                    import_models.import_model(name, archive)
                    with self.assertRaises(FileExistsError):
                        info = models / f"{name}_info.json"
                        info.unlink()
                        import_models.import_model(name, archive)
                    tf.keras.backend.clear_session()


if __name__ == "__main__":
    unittest.main(verbosity=2)
