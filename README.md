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

Copy `.env.example` to `.env` and configure `SMTP_USER` and `SMTP_PASSWORD` if email OTP is required. The application now reads SMTP settings from the environment; without them, email OTP is not sent. The repository previously contained hard-coded SMTP credentials. **Revoke/rotate them immediately**; this source change cannot erase old commits from Git history.

The local SQLite database and Python bytecode cache are excluded from version control. Each installation creates its own runtime database.

The old SMTP credentials were removed from the current source, but remain potentially recoverable from Git history. Rotate/revoke them with the email provider now; deleting them from a new commit does not invalidate them. Keep replacement credentials only in local `.env` files or your deployment secret store. Do not commit `.env` files, local database files, or private data artifacts.

## Model and evaluation

The repository contains model artifacts and describes training on a credit-card transaction dataset. Dataset size and performance figures should only be published when reproducible training/evaluation scripts and a documented test split support them. The application combines model predictions with hand-written rules; the final score is not necessarily the model's raw probability.

## Known limitations

- Model artifact paths and the expected feature schema must match the committed files.
- Email delivery depends on valid SMTP configuration and provider policies.
- Add input validation, rate limiting, authentication, tests, and production-grade secret handling before deployment.
- Fraud scores can generate false positives and false negatives; provide human review for consequential decisions.

## License

No license is specified in this repository.