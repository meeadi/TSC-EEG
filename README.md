# Baseline Model for EEG Classification using MiniROCKET + Ridge Classifier

This project implements a **baseline model for EEG signal classification** using:
- **MiniROCKET** (for time series feature extraction)
- **RidgeClassifierCV** (for classification)

## Dataset:
- **BCI Competition IV Dataset 2a (BNCI2014_001)** via MOABB.
- Using only **Subject 1**, **Session '0train'**, **Run '0'** for this baseline.

## Methodology:
1. Load raw EEG signals.
2. Extract event markers (motor imagery tasks: feet, left hand, right hand, tongue).
3. Epoch signals into trials (0–2 seconds windows).
4. Transform using **MiniROCKET**.
5. Train **RidgeClassifierCV**.
6. Evaluate accuracy & classification metrics.

## Results (Baseline):
| Metric     | Value |
|------------|--------|
| Accuracy   | ~50%  |
| Macro F1   | ~0.47 |
| Support    | 10 samples (small test set) |

> Note: Accuracy is low because only a **small subset (run '0' of subject 1)** was used. This is intentional to establish a reproducible **baseline**.

## Future Improvements:
- Combine multiple runs per subject (e.g., runs '1', '2', '3', etc.).
- Include more subjects (1–9) for better generalization.
- Experiment with **S-Rocket, HDC-Rocket** as enhancements.
- Evaluate on larger test splits & cross-validation.

## Usage:
```bash
pip install -r requirements.txt
python src/data_preprocessing.py
