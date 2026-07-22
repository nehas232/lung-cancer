# User guide

## Installation

```bash
git clone <your-repo-url>
cd lung-cancer-detection
pip install -r requirements.txt
```

## Running inference on a scan

```bash
python main.py --input data/raw/patient_001.dcm
```

This writes:
- `outputs/report_patient_001.txt` — human-readable findings
- `outputs/predictions.csv` — appended row(s) with per-nodule probabilities

## Training a new model

1. Populate `data/raw/`, `data/annotations/labels.csv`, and `data/splits/*.csv`.
2. Preprocess scans into `data/processed/` (see `src/preprocessing/`).
3. Run training:

```bash
python -m src.models.train_model
```

Best weights (by validation loss) are saved to `models_saved/lung_cancer_cnn.pth`.

## Evaluating a trained model

```bash
python -m src.evaluation.evaluate
```

Writes `outputs/evaluation_report.txt` with accuracy, sensitivity,
specificity, precision, and AUC-ROC.

## Running tests

```bash
pytest tests/
```