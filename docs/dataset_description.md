# Dataset description

## Expected data layout

- `data/raw/` — original DICOM (`.dcm`) CT scan slices, one file per slice
  (or organized in per-patient subfolders for multi-slice volumes).
- `data/processed/` — preprocessed, normalized arrays saved as `.npy` for
  fast loading during training.
- `data/annotations/labels.csv` — one row per patient/scan with a
  benign/malignant label.
- `data/annotations/nodule_coordinates.csv` — nodule center coordinates
  (x, y, z) and diameter in mm, used for supervised detection/segmentation.
- `data/splits/{train,validation,test}.csv` — patient ID lists defining the
  split. Splitting is done at the patient level to avoid leakage.

## Suggested public sources

- **LIDC-IDRI** — CT scans with multi-radiologist nodule annotations.
- **LUNA16** — a curated subset of LIDC-IDRI standardized for nodule
  detection benchmarks.
- **NLST** (National Lung Screening Trial) — large screening-trial cohort.

## Column reference

**labels.csv**
| column | description |
|---|---|
| patient_id | unique patient identifier |
| scan_path | relative path to the raw scan |
| label | `benign` or `malignant` |

**nodule_coordinates.csv**
| column | description |
|---|---|
| patient_id | unique patient identifier |
| x, y, z | voxel coordinates of nodule center |
| diameter_mm | nodule diameter in millimeters |