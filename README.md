# Book Cover Recognition

Semester 5 Artificial Intelligence Project — Milestone 2

This project predicts the genre of a book from its cover image. It has a Python desktop interface where a user can upload a cover, select a model, and see the predicted categories. The models can also be compared on the same image.

## Tools used

- Python 3.10
- Tkinter for the interface
- TensorFlow and Keras for CNN models
- Scikit-learn and Scikit-image for HOG and SVM
- OpenCV, Pillow, NumPy, Pandas, and Matplotlib

## Dataset

The project uses [BookCover30](https://github.com/uchidalab/book-dataset), which has 30 book genres. The images are resized to 224 × 224 pixels.

After checking for missing images and duplicate or conflicting covers, the prepared dataset has:

| Split | Images |
| --- | ---: |
| Training | 46,150 |
| Validation | 5,128 |
| Test | 5,700 |

The official test set is kept separate. Ten percent of the remaining training data is used for validation. Class counts and the data checks are included in `prepared_data`.

## Run the project

Open the project folder in VS Code. In its PowerShell terminal, run:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

If the virtual environment already exists and the packages are installed, only the last command is needed.

The trained model files are kept locally and are not included in this repository. Copy the `models` folder from the trained project to use all saved models. Each model needs its matching `_info.json` file. Without these files, the app opens and previews images, but prediction is unavailable for the missing model.

1. Click **Upload Image** and choose a book cover.
2. Select a model.
3. Click **Predict Category** to see its top three categories.
4. Select **Compare all** to see one prediction from each available model. This option needs all listed models.
5. Use **Clear** to reset the image or **Test results** to view the saved evaluation scores.

## Prepare data and train the baseline

Download the BookCover30 images from the dataset page and extract them into `data/images/224x224`. The label files are downloaded by the preparation script when needed.

```powershell
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe train_svm.py
.\.venv\Scripts\python.exe evaluate.py --model svm
.\.venv\Scripts\python.exe app.py
```

After training, select **HOG + SVM** in the interface. This is enough to run the baseline without the imported CNN models. Training may take several hours on a CPU.

The preparation and training scripts stop if their output already exists. Keep a backup before preparing another dataset or retraining a model.

## Models and results

| Model | Test accuracy | Macro F1 |
| --- | ---: | ---: |
| HOG + SVM | 11.54% | 9.72% |
| Basic CNN | 9.77% | 7.37% |
| MobileNetV2 | 25.40% | 23.85% |
| MobileNetV2 (fine-tuned) | 26.86% | 25.37% |
| MobileNetV2 + Dense | 27.11% | 26.06% |

These results are from the recorded 5,700-image test split. HOG + SVM uses edge features and a linear SVM. MobileNetV2 + Dense uses frozen features from the fine-tuned model with a classifier retrained on 6,000 balanced training images.

The Basic CNN and the two original MobileNetV2 models were imported from an existing project. Their original training split files were not provided, so overlap with the evaluation data could not be independently checked. The SVM was trained in this project with checked split separation.

The `results` folder includes evaluation metrics, classification reports, confusion matrices, and comparison charts. To evaluate all saved models again:

```powershell
.\.venv\Scripts\python.exe evaluate.py --model all --split validation
.\.venv\Scripts\python.exe evaluate.py --model all
```

Evaluation needs the images, prepared split lists, and saved models from the same run. The split lists contain local image paths, so they are not uploaded to GitHub.

## Main files

| File | Purpose |
| --- | --- |
| `app.py` | Desktop interface and image upload |
| `common.py` | Image loading, HOG features, and model loading |
| `prepare_data.py` | Dataset checks and train/validation/test splits |
| `train_svm.py` | HOG + SVM training |
| `train_mobilenet.py` | MobileNetV2 training and fine-tuning |
| `import_models.py` | Convert the original CNN model archives for this project |
| `improve_mobilenet.py` | Train and compare MobileNetV2 classifier heads |
| `evaluate.py` | Calculate metrics and save reports |
| `run_project.py` | Run preparation, SVM training, and evaluation |
| `test_project.py` | Check the pipeline and interface using temporary images |

## Tests

```powershell
.\.venv\Scripts\python.exe -B test_project.py
```

The tests use small synthetic images and do not overwrite the real dataset or saved models. Their scores are only for checking that the code works.

## Current limitations

The models use visual features. There is no separate OCR step to read the title. Similar covers can belong to different genres, so incorrect predictions are still common. A displayed probability is a model score for one cover, not the overall accuracy. SVM scores are not percentages.

The dataset, trained models, temporary files, and virtual environment are excluded from GitHub. The source code and small evaluation reports are included.
