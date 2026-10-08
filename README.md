# SMS Security Classifier (Flask Version)

A Flask-based SMS classification project for the CS190 final assessment.

## What this project does

It classifies incoming SMS messages into these classes:
- `ads`
- `gov`
- `notifs`
- `otp`
- `spam`

The project includes **two separate models**:
1. **Ensemble Classification Model**
2. **Neural Network Model**

The web app now asks for both:
- **Sender**
- **Message body**

and shows the **two model results separately**, including confidence information.

## Key upgrades made

- Reworked around **Flask-only deployment**
- Added **sender-aware feature engineering**
- Improved the **ensemble pipeline** using richer TF-IDF feature unions
- Refactored inference so web and CLI use the same prediction helpers
- Added separate pages for:
  - Analyzer
  - Project/About
  - Metrics
- Removed the old Streamlit run instruction

## Project structure

```text
sms-classifier-main/
├── app.py
├── data/
├── models/
├── outputs/
├── src/
│   ├── model_service.py
│   ├── preprocessing.py
│   ├── train_ensemble.py
│   ├── train_nn.py
│   ├── predict.py
│   └── evaluate_confidence.py
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── about.html
│   └── metrics.html
└── requirements.txt
```

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` has only what the web app needs to run (this is also what Vercel installs).
To retrain models or run EDA, install `requirements-train.txt` instead.
The saved models require `scikit-learn==1.8.0` exactly.

## Run the Flask app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Train the models

### Ensemble

```bash
python src/train_ensemble.py
```

### Neural Network

```bash
python src/train_nn.py
python src/export_nn_numpy.py
```

Training needs `requirements-train.txt` (TensorFlow). The second command exports the trained
network to `models/nn_numpy_weights.npz` and `models/nn_numpy_meta.json`. The web app and CLI
run the network from those files with plain numpy, so TensorFlow is **not** needed to serve
predictions (this keeps the Vercel deployment small). Re-run the export after every retrain.

## Run prediction from the command line

```bash
python src/predict.py --sender "GCash" "Your OTP is 123456"
```

Or force one model only:

```bash
python src/predict.py --model ensemble --sender "BDO Deals" "Promo alert..."
python src/predict.py --model nn --sender "Maya" "Your OTP is 654321"
```

## Metrics and reports

Generated reports are stored in:

```text
outputs/reports/
```

Useful files include:
- `best_model_test_metrics.json`
- `best_model_classification_report.txt`
- `ensemble_comparison.csv`
- `nn_metrics_summary.json`
- `nn_best_model_report.txt`

## Course requirement alignment

This version is designed to align with the brief by providing:
- a supervised classification solution
- an ensemble model
- a neural network model
- Flask-only web integration
- descriptive pages about the project and models
- user-entered inputs for prediction
- separate display of both model outputs
- confidence reporting

## Notes

- If the exported NN files are missing, the Flask app still runs the ensemble model and clearly reports that the neural network is unavailable (run `python src/export_nn_numpy.py`).
- English stopwords are bundled in `src/english_stopwords.txt`, so no NLTK download is needed at runtime.
