# SMS Classifier — Tagalog Multi-Class NLP Pipeline

A clean, production-ready Python project for classifying Tagalog SMS messages
into five categories: **spam · gov · notif · otp · ads**.

---

## Project Structure

```
sms-classifier/
│
├── data/
│   └── raw/
│       └── tagalog-sms.xlsx        ← place your dataset here
│
├── src/
│   ├── utils.py                    ← shared paths, logger, helpers
│   ├── eda.py                      ← exploratory data analysis
│   └── preprocessing.py            ← text cleaning & feature prep
│
├── outputs/
│   ├── figures/                    ← generated plots (PNG)
│   └── reports/                    ← generated text reports
│
├── models/                         ← saved model artefacts (future)
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup

### 1. Clone & enter the repository

```bash
git clone https://github.com/<your-username>/sms-classifier.git
cd sms-classifier
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place `tagalog-sms.xlsx` inside `data/raw/`.

---

## Running the Scripts

Both scripts are run from the **project root** so that relative paths resolve correctly.

### EDA

```bash
python src/eda.py
```

What it does:

| Step | Output |
|------|--------|
| `basic_info` | Shape, dtypes, first 5 rows printed to console |
| `check_missing_values` | Missing-value audit printed to console |
| `class_distribution` | Console table + `outputs/figures/class_distribution.png` |
| `message_length_analysis` | Console stats + `outputs/figures/message_length_analysis.png` |
| `top_words_per_class` | Console word lists + `outputs/figures/top_words_per_class.png` |

### Preprocessing

```bash
python src/preprocessing.py
```

What it does:

- Loads the raw dataset
- Runs the full clean → tokenise → remove-stopwords pipeline
- Prints a sample of raw vs. cleaned messages
- Performs a stratified 80/20 split and prints split shapes

---

## Labels

| Label | Description |
|-------|-------------|
| `spam` | Unsolicited commercial or scam messages |
| `gov`  | Government announcements / alerts |
| `notif` | Service notifications (banks, apps, etc.) |
| `otp` | One-time passwords / verification codes |
| `ads` | Advertisements from legitimate senders |

---

## Roadmap

- [ ] Feature engineering (TF-IDF, n-grams)
- [ ] Baseline models (Naive Bayes, Logistic Regression)
- [ ] Model evaluation & comparison
- [ ] Hyperparameter tuning
- [ ] Export trained model to `models/`

---

## Contributing

1. Fork the repo.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m "Add your feature"`.
4. Push and open a pull request.

---

## License

MIT
