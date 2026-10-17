# Book Cover Recognition

Semester 5 Artificial Intelligence Project - Milestone 2

This project predicts the genre of a book from its cover image. It has a Python desktop interface where a user can upload a cover, select a model, and see the predicted categories. The models can also be compared on the same image.

## Tools used

- Python 3.10
- Tkinter for the interface
- TensorFlow and Keras for CNN models
- Scikit-learn and Scikit-image for HOG and SVM
- OpenCV, Pillow, NumPy, Pandas, and Matplotlib

## Dataset

The project uses [BookCover30](https://github.com/uchidalab/book-dataset), which contains 57,000 cover images across 30 genres. After image checks and the exclusion of 22 duplicate or conflicting training covers, the prepared dataset has:

| Split | Images |
| --- | ---: |
| Training | 46,150 |
| Validation | 5,128 |
| Test | 5,700 |

The official test list is kept separate and unchanged. Ten percent of the remaining official training data is used for validation, with a stratified split and random seed 42. Every genre has 190 test images. The checks found no missing or unreadable images and recorded one duplicate cover pair within the official test set. Class counts and the audit are included in `prepared_data`.

## Preprocessing

- Images are converted to RGB and resized to 224 x 224 pixels.
- Genre labels are stored as integers from 0 to 29.
- The Basic CNN scales pixels to 0-1. MobileNetV2 scales them to -1 to 1. These steps are included inside the saved CNN models.
- HOG uses 128 x 128 grayscale covers and produces 1,764 features per image. StandardScaler is fitted on training features only.
- The MobileNetV2 training code includes small rotations, zooms, and changes in contrast and brightness. HOG + SVM does not use image augmentation.

## Run the project

Open the project folder in VS Code. In its PowerShell terminal, run:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

If the virtual environment already exists and the packages are installed, only the last command is needed.

The trained model files are kept locally and are not included in this repository. Copy the `models` folder from the trained project into the project folder to use the saved models. The model filenames are listed in the results table below. Each model also needs its matching metadata file, such as `svm_info.json` or `mobilenet_updated_info.json`. Without a model and its metadata, the app can preview images but cannot predict with that model.

1. Click **Upload Image** and choose a book cover.
2. Select a model.
3. Click **Predict Category** to see its top three categories.
4. Select **Compare all** to see one prediction from each listed model. This option needs all listed models and shows separate predictions rather than combining their scores.
5. Use **Clear** to reset the image or **Test results** to view the saved evaluation scores.

To use the project on another laptop, copy the saved models and their metadata, create a new virtual environment, and install `requirements.txt`. The full dataset is needed for training and evaluation, but not for predicting the genre of an uploaded cover. The saved reports in `results` are also needed to display the recorded test scores.

## Prepare data and train the baseline

Download the BookCover30 images from the dataset page and extract them into `data/images/224x224`. The label files are downloaded by the preparation script when needed.

```powershell
.\.venv\Scripts\python.exe prepare_data.py --images .\data\images\224x224
.\.venv\Scripts\python.exe train_svm.py
.\.venv\Scripts\python.exe evaluate.py --model svm
.\.venv\Scripts\python.exe app.py
```

After training, select **HOG + SVM** in the interface. This baseline can run on its own. Training may take several hours on a CPU. The SVM compares C values of 0.1, 1 and 10 using validation macro F1; the current saved classifier uses C = 0.1.

The preparation and training scripts stop if their output already exists. Keep a backup before preparing another dataset or retraining a model.

`train_mobilenet.py` provides two-stage training: first the final classifier, then fine-tuning of later backbone layers at a lower learning rate. `improve_mobilenet.py` trains and compares additional classifiers using frozen MobileNetV2 features.

## Models and results

| Model | Saved file in `models` | Test accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| HOG + SVM | `svm.joblib` | 11.54% | 9.72% |
| Basic CNN | `cnn.keras` | 9.77% | 7.37% |
| MobileNetV2 | `mobilenet.keras` | 25.40% | 23.85% |
| MobileNetV2 (fine-tuned) | `mobilenet_tuned.keras` | 26.86% | 25.37% |
| MobileNetV2 + Dense | `mobilenet_updated.keras` | 27.11% | 26.06% |

These results are from the recorded 5,700-image test list. HOG + SVM uses edge features and a linear SVM. Basic CNN provides a small deep-learning baseline. The MobileNetV2 variants provide a transfer-learning comparison. MobileNetV2 + Dense uses frozen features from the fine-tuned model with a classifier trained on 6,000 balanced training images.

MobileNetV2 + Dense has the highest recorded first-choice accuracy of 27.11%. Its top-three accuracy is 47.16%, meaning that the correct genre appears among its top three choices for about 47 out of 100 test covers. Its first-choice accuracy is about 0.25 percentage points above the fine-tuned MobileNetV2 model, so the improvement is small.

The SVM used the checked training split. Complete training/test separation has not yet been verified against the full training history of every CNN variant, so the CNN scores are preliminary comparisons rather than confirmed independent test estimates.

### Image score and test accuracy

A score above 50% or 90% on one uploaded cover is different from the test accuracy listed in the table.

- **Image score:** The probability a CNN model assigns to its predicted genre for that one cover. For example, it may give a cookbook cover a score of 95% for Cookbooks, Food & Wine.
- **Test accuracy:** The percentage of correct predictions across the whole test set. An accuracy of 27.11% means about 27 out of every 100 test covers were classified correctly.

A 95% score on one image does not mean the model has 95% overall accuracy. CNN probabilities have not been calibrated, and a high score can still be wrong. HOG + SVM uses a decision score, which can be negative and is not a percentage.

The `results` folder includes evaluation metrics, classification reports, confusion matrices, and comparison charts. To evaluate all saved models again:

```powershell
.\.venv\Scripts\python.exe evaluate.py --model all --split validation
.\.venv\Scripts\python.exe evaluate.py --model all
```

Evaluation needs the images, prepared split lists, and saved models from the same run. The split lists contain local image paths and are not uploaded to GitHub. After moving the dataset, those paths and the matching split metadata must be handled before re-running evaluation. Copying the models and their metadata is enough for image prediction.

## Milestone 2 work

The prototype covers dataset exploration, class distribution, preprocessing, a trained baseline, initial evaluation, and a working prediction pipeline.

Beyond the required baseline, it includes:

- Comparison of five models and an additional Dense classifier experiment.
- A Tkinter interface with cover upload, image preview, top-three predictions, and model comparison.
- Duplicate-cover checks, genre-wise evaluation reports, and confusion matrices.
- A 16-test software suite for the data pipeline, model handling, and interface.

## Main files

| File | Purpose |
| --- | --- |
| `app.py` | Desktop interface and image upload |
| `common.py` | Image loading, HOG features, and model loading |
| `prepare_data.py` | Dataset checks and train/validation/test splits |
| `train_svm.py` | HOG + SVM training |
| `train_mobilenet.py` | MobileNetV2 training and fine-tuning |
| `import_models.py` | Prepare compatible Keras models from existing model archives |
| `improve_mobilenet.py` | Train and compare MobileNetV2 classifier heads |
| `evaluate.py` | Calculate metrics and save reports |
| `run_project.py` | Run preparation, SVM training, and evaluation |
| `test_project.py` | Check the pipeline and interface using temporary images |

## Tests

```powershell
.\.venv\Scripts\python.exe -B test_project.py
```

The 16 software tests passed in the recorded project check. They use small synthetic images and do not overwrite the real dataset or saved models. Their sample training scores only check that the code works and are separate from the BookCover30 accuracy results.

## Current limitations

The models use visual features. There is no separate OCR step to read the title. Similar covers can belong to different genres, so incorrect predictions are still common. Further work should focus on controlled CNN training with fully documented splits, validation-based model selection, and a separate untouched final test set.

The dataset, trained models, temporary files, and virtual environment are excluded from GitHub. The source code and small evaluation reports are included.
