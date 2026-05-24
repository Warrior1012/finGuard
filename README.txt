# FinGuard — Invisible Fraud Detector System
> AI Solution Expo | JSS University | Team: Tanishq, Raunak,Shraddha,Yashica

## What is this?
FinGuard is a real-time, behavior-based fraud detection system using a Random Forest ML model.
It analyzes UPI/digital transactions and classifies them as Safe / Suspicious / Fraud — with
a plain-language explanation for every decision.

## Quick Start

### Windows
Double-click `START_SERVER.bat`

### Mac / Linux
```bash
chmod +x start_server.sh
./start_server.sh
```

Then open your browser: **http://localhost:5000**

## Features
- 🛡️ Random Forest ML model (trained on 284,807 transactions)
- ⚡ Real-time risk scoring (0–100 composite score)
- 🔐 OTP email verification for suspicious transactions
- 📊 Transaction history with SQLite database
- 📈 Statistics dashboard
- 🤖 Explainable AI — plain language verdict for every transaction
- 🚩 Up to 6 risk flags with severity levels (High / Medium / Low)

## Risk Score Thresholds
| Score | Verdict     | Action              |
|-------|-------------|---------------------|
| 0–29  | ✅ Safe      | Transaction Allowed |
| 30–59 | ⚠️ Suspicious | OTP Required       |
| 60+   | 🚨 Fraud     | Transaction Blocked |

## Tech Stack
- **Backend:** Python, Flask, SQLite
- **ML Model:** Random Forest (scikit-learn), trained on Kaggle credit card fraud dataset
- **Frontend:** Vanilla HTML/CSS/JS (no frameworks needed)
- **Email OTP:** SMTP via Gmail

## Email OTP Setup (Optional)
Edit `app.py` and set your Gmail credentials:
```python
sender_email = "your_email@gmail.com"
app_password = "your_16_char_app_password"
```
Generate an App Password at: https://myaccount.google.com/apppasswords

## Model Performance
- Trained on: 227,845 transactions
- False Positive Rate: ~1.7%
- Features: amount ratio, hour risk, location, device, merchant, recipient, account age, prior fraud