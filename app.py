import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.font import Font

import numpy as np
from PIL import Image, ImageOps, ImageTk

from common import (
    GENRES,
    MODEL_DIR,
    MODEL_FILENAMES,
    MODEL_LABELS,
    RESULT_DIR,
    load_classifier,
    predict_scores,
    read_json,
)

MODEL_NAMES = {
    label: name
    for name, label in MODEL_LABELS.items() if name != "mobilenet_updated" or
    ((MODEL_DIR / MODEL_FILENAMES[name]).is_file() and (MODEL_DIR / f"{name}_info.json").is_file())
}
BACKGROUND = "#f3f6fa"
TEXT = "#20334d"
MUTED = "#697b91"
BLUE = "#167bd4"
GREEN = "#237d5b"


def model_files(name):
    filename = MODEL_FILENAMES[name]
    return MODEL_DIR / filename, MODEL_DIR / f"{name}_info.json"


def predict_image(path, name, models):
    checkpoint, metadata = model_files(name)
    if not checkpoint.is_file() or not metadata.is_file():
        raise FileNotFoundError("The selected model is unavailable.")
    if name not in models:
        models[name] = load_classifier(name)
    scores = np.asarray(predict_scores(models[name], name, path))
    if scores.shape != (len(GENRES), ) or not np.isfinite(scores).all():
        raise ValueError("The model returned invalid genre scores.")
    ranked = [(int(index), float(scores[index])) for index in np.argsort(-scores)[:3]]
    return {"name": name, "ranked": ranked}


class BookCoverApp:

    def __init__(self, root):
        self.root = root
        self.image_path = None
        self.original_image = None
        self.preview_image = None
        self.models = {}
        self.messages = queue.Queue()
        self.busy = False
        self.closed = False
        self.filename = tk.StringVar()
        self.model_choice = tk.StringVar(value="Compare all")
        self.category = tk.StringVar(value="—")
        self.result_title = tk.StringVar(value="Model comparison")
        self.score = tk.StringVar()
        self.status = tk.StringVar()
        root.title("Book Cover Recognition")
        root.configure(bg=BACKGROUND)
        self.table_font = Font(root=root, family="Segoe UI", size=10)
        self.model_width = max(self.table_font.measure(label) for label in MODEL_NAMES) + 20
        self.score_width = self.table_font.measure("Model probability") + 20
        genre_width = max(self.table_font.measure(genre) for genre in GENRES) + 20
        width = min(
            max(760, self.model_width + self.score_width + genre_width + 60),
            root.winfo_screenwidth() - 60
        )
        root.geometry(f"{width}x{min(840, max(600, root.winfo_screenheight() - 115))}")
        root.minsize(680, 600)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.build_interface()
        self.update_controls()
        root.bind("<Control-o>", lambda event: self.upload_image())
        root.bind("<Return>", lambda event: self.predict_category())
        root.bind("<FocusIn>", self.on_focus)

    def build_interface(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "Treeview",
            font=self.table_font,
            rowheight=self.table_font.metrics("linespace") + 6,
            background="white",
            fieldbackground="white",
            foreground=TEXT
        )
        style.configure(
            "Treeview.Heading",
            font=("Segoe UI", 10, "bold"),
            background="#e8eef5",
            foreground=TEXT
        )
        style.configure("TCombobox", padding=5, font=("Segoe UI", 10))
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        body = tk.Frame(self.root, bg=BACKGROUND, padx=28, pady=12)
        body.grid(sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1, minsize=120)
        header = tk.Frame(body, bg=BACKGROUND)
        header.grid(row=0, pady=(0, 10))
        tk.Label(
            header,
            text="BOOK COVER RECOGNITION",
            bg=BACKGROUND,
            fg=TEXT,
            font=("Segoe UI", 22, "bold")
        ).pack()
        tk.Label(
            header,
            text="Upload a book cover and predict its category",
            bg=BACKGROUND,
            fg=MUTED,
            font=("Segoe UI", 11)
        ).pack(pady=(3, 0))
        self.preview = tk.Canvas(
            body,
            width=360,
            height=300,
            bg="white",
            highlightbackground="#cdd8e5",
            highlightthickness=1
        )
        self.preview.grid(row=1, sticky="ns", pady=(0, 8))
        self.preview.bind("<Configure>", lambda event: self.draw_preview())
        self.file_entry = tk.Entry(
            body,
            textvariable=self.filename,
            state="readonly",
            justify="center",
            font=("Segoe UI", 10),
            readonlybackground=BACKGROUND,
            fg=MUTED,
            relief="flat",
            bd=0
        )
        self.file_entry.grid(row=2, sticky="ew", pady=(0, 6))
        model_row = tk.Frame(body, bg=BACKGROUND)
        model_row.grid(row=3, pady=(0, 8))
        tk.Label(model_row, text="Prediction model", bg=BACKGROUND, fg=TEXT,
                 font=("Segoe UI", 10)).pack(side="left", padx=(0, 10))
        self.model_selector = ttk.Combobox(
            model_row,
            textvariable=self.model_choice,
            values=["Compare all", *MODEL_NAMES],
            state="readonly",
            width=27
        )
        self.model_selector.pack(side="left")
        self.model_selector.bind("<<ComboboxSelected>>", self.change_model)
        ttk.Button(model_row, text="Test results",
                   command=self.show_test_results).pack(side="left", padx=(12, 0))
        actions = tk.Frame(body, bg=BACKGROUND)
        actions.grid(row=4, pady=(0, 10))
        self.upload_button = self.make_button(actions, "Upload Image", BLUE, self.upload_image)
        self.predict_button = self.make_button(
            actions, "Predict Category", GREEN, self.predict_category
        )
        self.clear_button = self.make_button(actions, "Clear", "#365f92", self.clear)
        ttk.Separator(body).grid(row=5, sticky="ew", pady=(0, 6))
        result = tk.Frame(body, bg=BACKGROUND)
        result.grid(row=6, sticky="ew", pady=(0, 6))
        tk.Label(
            result,
            textvariable=self.result_title,
            bg=BACKGROUND,
            fg=GREEN,
            font=("Segoe UI", 11, "bold")
        ).pack()
        self.category_label = tk.Label(
            result,
            textvariable=self.category,
            bg=BACKGROUND,
            fg=GREEN,
            font=("Segoe UI", 16, "bold"),
            wraplength=620
        )
        self.category_label.pack(pady=(2, 3))
        self.score_label = tk.Label(
            result, textvariable=self.score, bg=BACKGROUND, fg=MUTED, font=("Segoe UI", 10)
        )
        self.score_label.pack()
        self.results = ttk.Treeview(
            body,
            columns=("model", "genre", "score"),
            show="headings",
            height=4,
            selectmode="none",
            takefocus=False
        )
        self.results.heading("model", text="Model")
        self.results.heading("genre", text="Top 3 categories")
        self.results.heading("score", text="Model probability")
        self.results.column("model", width=self.model_width, anchor="w", stretch=False)
        self.results.column("genre", width=280, anchor="w")
        self.results.column("score", width=self.score_width, anchor="center", stretch=False)
        self.results.grid(row=7, sticky="ew", pady=(0, 6))
        self.results.grid_remove()
        footer = tk.Frame(body, bg=BACKGROUND)
        footer.grid(row=8, sticky="ew")
        footer.columnconfigure(0, weight=1)
        tk.Label(
            footer,
            textvariable=self.status,
            bg=BACKGROUND,
            fg=MUTED,
            font=("Segoe UI", 9),
            anchor="w",
            wraplength=520
        ).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(footer, mode="indeterminate", length=100)
        self.progress.grid(row=0, column=1, padx=(10, 0))
        self.progress.grid_remove()
        self.draw_preview()

    def make_button(self, parent, text, color, command):
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg="white",
            activebackground=color,
            activeforeground="white",
            disabledforeground="#d5e2ed",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            bd=0,
            padx=18,
            pady=8,
            cursor="hand2"
        )
        button.pack(side="left", padx=5)
        button.bind(
            "<Enter>", lambda event: button.configure(relief="raised")
            if button["state"] == "normal" else None
        )
        button.bind("<Leave>", lambda event: button.configure(relief="flat"))
        return button

    def draw_preview(self):
        width, height = self.preview.winfo_width(), self.preview.winfo_height()
        self.preview.delete("all")
        if self.original_image is None:
            self.preview.create_text(
                width / 2, height / 2, text="No image selected", fill=MUTED, font=("Segoe UI", 15)
            )
        else:
            image = self.original_image.copy()
            image.thumbnail((max(1, width - 24), max(1, height - 24)), Image.Resampling.LANCZOS)
            self.preview_image = ImageTk.PhotoImage(image, master=self.root)
            self.preview.create_image(width / 2, height / 2, image=self.preview_image)

    def upload_image(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(
            title="Choose a book cover",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")]
        )
        if path:
            self.select_image(path)

    def select_image(self, path):
        try:
            with Image.open(path) as image:
                preview = ImageOps.exif_transpose(image).convert("RGB").copy()
        except (OSError, ValueError, Image.DecompressionBombError):
            messagebox.showerror(
                "Invalid image", "Please choose a readable JPG, PNG, BMP, or WEBP image."
            )
            return False
        self.image_path = Path(path)
        self.original_image = preview
        self.filename.set(self.image_path.name)
        self.reset_results()
        self.draw_preview()
        self.update_controls()
        return True

    def reset_results(self):
        self.result_title.set(
            "Model comparison" if self.model_choice.get() ==
            "Compare all" else "Predicted Category:"
        )
        self.category.set("—")
        self.score.set("")
        self.results.delete(*self.results.get_children())
        self.results.grid_remove()

    def change_model(self, event=None):
        self.reset_results()
        self.update_controls()

    def on_focus(self, event):
        if event.widget == self.root and not self.busy:
            self.update_controls()

    def selected_models(self):
        if self.model_choice.get() == "Compare all":
            return list(MODEL_NAMES.values())
        return [MODEL_NAMES[self.model_choice.get()]]

    def update_controls(self):
        names = self.selected_models()
        if len(names) > 1:
            self.category_label.pack_forget()
            self.score_label.pack_forget()
        else:
            self.category_label.pack(pady=(2, 3))
            self.score_label.pack()
        self.results.configure(height=max(3, len(names)))
        ready = all(path.is_file() for name in names for path in model_files(name))
        self.upload_button.configure(state="disabled" if self.busy else "normal")
        self.clear_button.configure(
            state="normal" if self.image_path and not self.busy else "disabled"
        )
        self.predict_button.configure(
            state="normal" if ready and self.image_path and not self.busy else "disabled",
            text="Predicting..." if self.busy else "Predict Category"
        )
        self.model_selector.configure(state="disabled" if self.busy else "readonly")
        self.results.heading(
            "genre", text="Predicted genre" if len(names) > 1 else "Top 3 categories"
        )
        self.results.heading(
            "score",
            text="Score"
            if len(names) > 1 else "Model probability" if names[0] != "svm" else "Decision score"
        )
        if not self.busy:
            if not ready:
                self.status.set("Model unavailable")
            else:
                self.status.set("Ready")

    def predict_category(self):
        if self.busy or self.image_path is None:
            return
        names = self.selected_models()
        if not all(path.is_file() for name in names for path in model_files(name)):
            self.update_controls()
            return
        self.reset_results()
        self.busy = True
        self.update_controls()
        self.status.set("Predicting...")
        self.progress.grid()
        self.progress.start(12)
        threading.Thread(
            target=self.predict_in_background, args=(self.image_path, names), daemon=True
        ).start()
        self.root.after(100, self.check_prediction)

    def predict_in_background(self, path, names):
        try:
            results = [predict_image(path, name, self.models) for name in names]
            self.messages.put((True, results))
        except Exception as error:
            self.messages.put((False, str(error)))

    def check_prediction(self):
        if self.closed:
            return
        try:
            success, result = self.messages.get_nowait()
        except queue.Empty:
            self.root.after(100, self.check_prediction)
            return
        self.busy = False
        self.progress.stop()
        self.progress.grid_remove()
        self.update_controls()
        if success:
            self.display_result(result)
        else:
            self.status.set("Prediction failed")
            messagebox.showerror("Prediction failed", result)

    def display_result(self, results):
        if len(results) == 1:
            result = results[0]
            name = result["name"]
            best_label, best_score = result["ranked"][0]
            self.result_title.set("Predicted Category:")
            self.category.set(GENRES[best_label])
            heading = "Model probability" if name != "svm" else "Decision score"
            value = f"{best_score:.2%}" if name != "svm" else f"{best_score:.3f}"
            self.score.set(f"{heading}: {value}")
        else:
            self.result_title.set("Model comparison")
            self.category.set("")
            self.score.set("")
        self.results.delete(*self.results.get_children())
        for result in results:
            name = result["name"]
            model_label = MODEL_LABELS[name]
            ranked = result["ranked"][:1] if len(results) > 1 else result["ranked"]
            for label, score in ranked:
                value = f"{score:.2%}" if name != "svm" else f"{score:.3f}"
                self.results.insert("", "end", values=(model_label, GENRES[label], value))
        self.results.grid()
        self.status.set("Done")

    def clear(self):
        if self.busy:
            return
        self.image_path = None
        self.original_image = None
        self.preview_image = None
        self.filename.set("")
        self.reset_results()
        self.draw_preview()
        self.update_controls()

    def show_test_results(self):
        lines = []
        try:
            for label, name in MODEL_NAMES.items():
                report_path = RESULT_DIR / f"{name}_metrics.json"
                metadata = MODEL_DIR / f"{name}_info.json"
                if report_path.is_file() and metadata.is_file():
                    report, info = read_json(report_path), read_json(metadata)
                    if report["dataset_id"] == info["dataset_id"]:
                        lines.append(
                            f"{label}\n"
                            f"Accuracy: {report['accuracy']:.2%}\n"
                            f"Precision: {report['macro_precision']:.3f}\n"
                            f"Recall: {report['macro_recall']:.3f}\n"
                            f"Macro F1: {report['macro_f1']:.3f}"
                        )
        except (OSError, ValueError, KeyError) as error:
            messagebox.showerror("Test results", f"Could not read the evaluation report: {error}")
            return
        messagebox.showinfo(
            "Test results", "\n\n".join(lines) if lines else "No test results available."
        )

    def close(self):
        self.closed = True
        self.root.destroy()


if __name__ == "__main__":
    window = tk.Tk()
    BookCoverApp(window)
    window.mainloop()
