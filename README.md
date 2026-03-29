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
```

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

- If TensorFlow is not installed, the Flask app will still run the ensemble model and will clearly report that the neural network is unavailable.
- For full requirement compliance during presentation/demo, install TensorFlow and retrain the neural network.
