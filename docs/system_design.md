# System design

## Overview

The pipeline takes a raw CT scan (DICOM) and produces a list of candidate nodules
with malignancy probabilities, plus a human-readable report.

## Stages

1. **Load** (`src/preprocessing/load_dicom.py`) — read pixel data and convert to
   Hounsfield units using the DICOM rescale slope/intercept.
2. **Preprocess** (`src/preprocessing/normalize.py`, `lung_segmentation.py`) —
   clip to the lung HU window, segment the lung region, normalize to [0, 1].
3. **Detect** (`src/detection/nodule_detector.py`) — Laplacian-of-Gaussian blob
   detection generates candidate nodule locations and radii; filtered by
   plausible size.
4. **Extract features** (`src/detection/feature_extraction.py`) — shape,
   intensity, and GLCM texture features per candidate, for classical ML or
   interpretability alongside the CNN.
5. **Classify** (`src/models/cnn_model.py`) — a 2D CNN predicts
   benign/malignant probability for each candidate patch.
6. **Segment (optional)** (`src/models/unet_model.py`) — U-Net can refine
   nodule boundaries when precise volume/shape is needed.
7. **Evaluate** (`src/evaluation/`) — sensitivity, specificity, ROC-AUC, and
   FROC curves on held-out patients (patient-level split, never slice-level,
   to avoid data leakage).
8. **Report** (`src/inference/generate_report.py`) — writes a per-patient
   text report and appends to a running predictions CSV.

## Design decisions

- **Patient-level splitting**: all slices from a patient belong to exactly
  one of train/validation/test, to prevent leakage.
- **Sensitivity is prioritized over specificity** in threshold tuning: a
  missed malignant nodule is more costly than a false positive.
- **Modularity**: preprocessing, detection, models, evaluation, and inference
  are separate packages so any stage (e.g. swapping in a 3D CNN detector)
  can be replaced independently.

## Known limitations

This scaffold is for research/educational use. It has not been validated
against a regulatory-grade dataset or clinical workflow.