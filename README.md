# Sleep Position Classification — Pose Landmarks + Deep Learning

This notebook builds and compares several pipelines for classifying **sleep position from images** into five classes:

- **Foetus**
- **Log**
- **Starfish**
- **Unknown**
- **Yearner**

The dataset (`SleepStressDataset_2`) contains 455 images organized into one folder per class. It is provided as a zip file and is expected to be extracted into a `dataset/` directory with one subfolder per class label.

## Overview

The notebook explores three broad approaches to the same classification problem, in roughly the order they appear:

1. **Pose-landmark + classical ML pipeline**
   - Extracts 12 body keypoints (shoulders, elbows, wrists, hips, knees, ankles) from each image using **YOLOv11-pose**.
   - Imputes missing keypoints via nearest-neighbor matching (Euclidean distance) followed by mean imputation.
   - Engineers ~50 geometric features per image: pairwise distances, joint angles (knee/hip/elbow), horizontal/vertical spread, left-right symmetry, and sin/cos of each angle.
   - Selects the most useful features using a **custom Q-learning agent** that searches feature subsets to maximize cross-validated accuracy (evaluated with an `ExtraTreesClassifier`).
   - Trains classical models (Random Forest, Extra Trees, SVM, KNN, MLP, XGBoost) on the engineered/selected features.

2. **End-to-end CNN image classifiers (transfer learning)**
   - Fine-tunes pretrained backbones directly on the raw images: **ResNet50**, **EfficientNetB3**, and **ConvNeXtTiny**.
   - Uses `image_dataset_from_directory` for loading, data augmentation (flip/rotate/zoom/contrast/translation/noise), class-weighting for imbalance, and callbacks (`EarlyStopping`, `ReduceLROnPlateau`).
   - Offline augmentation is also explored by writing augmented copies of the dataset to disk before training.

3. **Hybrid (fused) feature pipeline**
   - Extracts deep CNN embeddings (ResNet50, pooled) and compresses them with **PCA**.
   - Concatenates these embeddings with the geometric pose features into a single fused feature vector.
   - Applies `SelectKBest` (mutual information) to reduce dimensionality, then trains an **XGBoost** classifier with stratified 5-fold cross-validation.

## Pipeline Summary (in notebook order)

| Step | What it does |
|---|---|
| Setup & config | Imports; defines class names, paths, and global constants |
| Dataset extraction | Unzips the dataset and verifies per-class image counts |
| Landmark extraction | Runs YOLOv11-pose on every image, normalizes keypoints to a 0–1000 box-relative scale |
| Missing-value imputation | Nearest-neighbor + mean imputation for undetected keypoints |
| Feature engineering | Distances, joint angles, spread, symmetry, trig features (~50 features/image) |
| Q-learning feature selection | RL agent searches feature subsets to maximize CV accuracy |
| Classical ML models | Random Forest, Extra Trees, SVM, KNN, MLP, XGBoost on selected features |
| CNN transfer learning | ResNet50 → EfficientNetB3 → ConvNeXtTiny, with augmentation & fine-tuning |
| Deep + pose feature fusion | ResNet embeddings (PCA) + pose features → XGBoost |
| Final model & evaluation | ConvNeXtTiny on augmented data; classification report + confusion matrix |

## Results (as recorded in the notebook's saved outputs)

- **Dataset:** 455 images total (Foetus: 96, Log: 126, Starfish: 71, Unknown: 86, Yearner: 76). YOLO landmark extraction succeeded on all 455 images.
- **Geometric features + Extra Trees (Q-learning-selected subset):** mean 5-fold CV accuracy ≈ **0.61**
- **Fused (deep + pose) features + XGBoost, top-100 features:** mean 5-fold CV accuracy ≈ **0.57**
- **Final CNN (ConvNeXtTiny, fine-tuned on augmented data):** **0.81** validation accuracy overall (macro F1 ≈ 0.80), the best-performing approach in the notebook. Per-class F1 ranged from 0.75 (Unknown) to 0.86 (Yearner).

The CNN-based approach substantially outperformed both the pure pose-geometry pipeline and the hybrid fused-feature pipeline on this dataset.

## Requirements

```
numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, lightgbm, catboost,
tensorflow (keras), ultralytics (YOLO), opencv-python, tqdm
```
Optional: `sklearn_crfsuite`, `vit-keras` (imported but not central to the final pipeline).

GPU is strongly recommended for the CNN/transfer-learning and YOLO sections.

## How to Run

1. Place the dataset zip at the path referenced by `ZIP_FILE` (default: `/content/SleepStressDataset_2 (1).zip`), or update that variable to point to your copy.
2. Run cells top to bottom. Note: the notebook was developed iteratively in Google Colab and cells are **not all in clean linear order** — some later cells (e.g., the hybrid fusion section) depend on variables defined in earlier exploratory cells (`landmark_df`, `feature_df_yolo`, `y`, etc.). If you hit a `NameError`, re-run the corresponding earlier section first.
3. Key configurable constants live near the top: `CLASS_NAMES`, `IMG_SIZE`, `BATCH_SIZE`, `TEST_SIZE`, `RL_EPISODES`.
4. Trained model weights are saved to `convnext_best.keras`; intermediate features are cached to `landmarks.csv` and `fused_features.npy`.

## Notes / Caveats

- This notebook was authored interactively (cell execution order is non-sequential), so treat it as an experimental/research log rather than a polished script — sections were rerun and modified multiple times while iterating on the approach.
- The "Unknown" class likely represents ambiguous/unclassifiable sleep postures and is harder to separate from the others (lowest F1 in the final model).
- If `ultralytics` is not installed, the landmark-extraction step falls back to randomly generated dummy landmarks purely so the rest of the pipeline can still run end-to-end for testing — real results require YOLO installed and working.
