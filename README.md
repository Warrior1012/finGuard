# FinGuard

FinGuard is an educational digital-transaction fraud-detection prototype built with Flask, SQLite, and a scikit-learn model. It calculates a risk score, produces a Safe / Suspicious / Fraud verdict, records transaction results, and provides a dashboard with explanations and risk flags.

> **Prototype disclaimer:** This project is not a bank-grade fraud prevention system. Its outputs require independent validation and should not be used as the sole basis for real financial decisions.

## Features

- Random Forest-based transaction risk classification
- Composite risk scoring and rule-based risk flags
- SQLite transaction history and statistics
- Plain-language explanations for results
- Email OTP flow for selected suspicious transactions
- Browser UI served by the Flask application

## Stack

- Python, Flask, Flask-CORS
- scikit-learn, joblib, NumPy, pandas
- SQLite
- HTML, CSS, and JavaScript
- SMTP for email OTP

## Requirements

- Python 3.10+ recommended
- Dependencies from `requirements.txt`
- Model artifacts expected under the `model/` directory (including `fraud_model.pkl` and the scaler used by the application)

## Setup

```bash
git clone https://github.com/Warrior1012/finGuard.git
cd finGuard
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`. If the repository's platform-specific startup scripts are present, they can also be used to launch the application.

## Configuration and security

**Before deploying, remove all credentials from source code.** The current application source contains hard-coded SMTP credential values. Treat those credentials as exposed: revoke/rotate them immediately, move replacements into environment variables or a secret manager, and never reuse committed credentials. This README intentionally does not reproduce them.

Configure mail settings through environment variables (for example, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, and `SMTP_FROM`) after updating the application to read them. Do not commit `.env` files, database files, or private model/data artifacts unless there is a deliberate reason to publish them.

## Model and evaluation

The repository contains model artifacts and describes training on a credit-card transaction dataset. Dataset size and performance figures should only be published when reproducible training/evaluation scripts and a documented test split support them. The application combines model predictions with hand-written rules; the final score is not necessarily the model's raw probability.

## Known limitations

- Model artifact paths and the expected feature schema must match the committed files.
- Email delivery depends on valid SMTP configuration and provider policies.
- Add input validation, rate limiting, authentication, tests, and production-grade secret handling before deployment.
- Fraud scores can generate false positives and false negatives; provide human review for consequential decisions.

## License

No license is specified in this repository.