# X-NIDS — Intelligent Network Defense

**An AI-Powered Network Intrusion Detection System with Explainable Alerting**

X-NIDS combines a Random Forest classifier, an F1-optimized severity engine, and SHAP-based explainability into a single Streamlit dashboard. It detects malicious traffic on the UNSW-NB15 benchmark at **95.74% accuracy** and produces analyst-facing alerts that explain *why* each prediction was made.

> Capstone project — IST 584: Applied AI Security, Penn State University.
> Author: **Aldiyar Utebekov**

## Performance

| Indicator | Result |
|---|---|
| Detection Accuracy | **95.74%** |
| AUC-ROC | **0.9943** |
| F1 Score | **0.9673** |
| Optimal Threshold | **0.42** (auto-selected via F1 maximization) |
| Inference Time | **< 5 seconds** on 175,341 records |

## Quick Start

```bash
git clone https://github.com/aldiyar-sec/x-nids.git
cd x-nids
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run xnids.py
```

Then upload `demo_all_attacks.csv` from the dashboard.

To retrain the model from scratch (download UNSW-NB15 CSVs first from https://research.unsw.edu.au/projects/unsw-nb15-dataset and place in project root):

```bash
python train_model.py
```

## Project Files

| File | Purpose |
|---|---|
| `xnids.py` | Streamlit dashboard application |
| `train_model.py` | Training pipeline with stratified split and F1 threshold optimization |
| `x_nids_model.joblib` | Trained Random Forest classifier |
| `preprocessor.joblib` | Saved label encoders |
| `model_metrics.json` | Dynamic metrics file |
| `demo_all_attacks.csv` | Demonstration dataset |

## Architecture

Three-layer pipeline:

- **Detection Layer** — Random Forest + saved encoders
- **Security Layer** — F1-optimized threshold (0.42) + HIGH/MEDIUM/LOW tiering
- **Analyst Layer** — Streamlit dashboard + SHAP attribution

## Tech Stack

Python 3.10+ · scikit-learn · SHAP · Streamlit · pandas · joblib

## Dataset

[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) — 257,673 labeled network records spanning nine attack categories. Developed at the University of New South Wales.

## License

MIT

## Contact

**Aldiyar Utebekov** · IST 584 — Applied AI Security · Penn State University
