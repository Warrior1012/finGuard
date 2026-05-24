"""
FinGuard ML Model Training
--------------------------
Trains a Random Forest classifier on fraud detection features.
If creditcard.csv (Kaggle dataset) is present, uses that.
Otherwise generates synthetic data with same distribution.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json

SEED = 42
np.random.seed(SEED)

# ── FEATURE NAMES (same as what frontend sends) ──────────────────────
FEATURES = [
    'amount_ratio',       # transaction amount / user daily average
    'hour_risk',          # 0-1 risk score based on hour
    'location_risk',      # 0=home, 0.3=known, 0.6=new domestic, 1.0=foreign
    'device_risk',        # 0=known, 0.5=new, 1.0=unknown
    'merchant_risk',      # 0-1 based on merchant category
    'recipient_risk',     # 0=saved, 0.5=new, 1.0=first time
    'account_age_risk',   # inverse of account age (newer = riskier)
    'prior_fraud_risk',   # 0, 0.5, 1.0
    'combo_location_device',  # location_risk * device_risk (correlation)
    'combo_hour_amount',      # hour_risk * amount_ratio normalized
]

def generate_synthetic_data(n_samples=50000):
    """
    Generate synthetic transaction data matching real fraud patterns.
    Fraud rate ~0.17% matching Kaggle creditcard dataset.
    """
    print(f"Generating {n_samples} synthetic transactions...")

    n_fraud = int(n_samples * 0.0017 * 100)  # ~1.7% for better training
    n_legit = n_samples - n_fraud

    # ── LEGITIMATE TRANSACTIONS ──────────────────────────────────────
    legit = pd.DataFrame({
        'amount_ratio':       np.random.exponential(1.2, n_legit).clip(0.1, 8),
        'hour_risk':          np.random.beta(2, 8, n_legit),       # mostly low
        'location_risk':      np.random.choice([0, 0, 0, 0.3, 0.6], n_legit, p=[0.6,0.15,0.1,0.1,0.05]),
        'device_risk':        np.random.choice([0, 0.5, 1.0], n_legit, p=[0.85, 0.12, 0.03]),
        'merchant_risk':      np.random.beta(1.5, 5, n_legit),
        'recipient_risk':     np.random.choice([0, 0.5, 1.0], n_legit, p=[0.75, 0.18, 0.07]),
        'account_age_risk':   np.random.beta(1, 5, n_legit),       # mostly old accounts
        'prior_fraud_risk':   np.random.choice([0, 0.5, 1.0], n_legit, p=[0.95, 0.04, 0.01]),
        'is_fraud':           0
    })

    # ── FRAUDULENT TRANSACTIONS ──────────────────────────────────────
    fraud = pd.DataFrame({
        'amount_ratio':       np.random.exponential(8, n_fraud).clip(3, 100),   # high amounts
        'hour_risk':          np.random.beta(5, 2, n_fraud),                    # mostly high (night)
        'location_risk':      np.random.choice([0, 0.3, 0.6, 1.0], n_fraud, p=[0.05, 0.1, 0.25, 0.6]),
        'device_risk':        np.random.choice([0, 0.5, 1.0], n_fraud, p=[0.1, 0.2, 0.7]),
        'merchant_risk':      np.random.beta(5, 2, n_fraud),                    # high risk merchants
        'recipient_risk':     np.random.choice([0, 0.5, 1.0], n_fraud, p=[0.05, 0.15, 0.8]),
        'account_age_risk':   np.random.beta(4, 2, n_fraud),                    # newer accounts
        'prior_fraud_risk':   np.random.choice([0, 0.5, 1.0], n_fraud, p=[0.3, 0.35, 0.35]),
        'is_fraud':           1
    })

    df = pd.concat([legit, fraud], ignore_index=True)

    # Combo features
    df['combo_location_device'] = df['location_risk'] * df['device_risk']
    df['combo_hour_amount']     = df['hour_risk'] * (df['amount_ratio'] / 10).clip(0, 1)

    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def train():
    print("=" * 50)
    print("FinGuard ML Model Trainer")
    print("=" * 50)

    # Try loading real Kaggle dataset first
    kaggle_path = os.path.join(os.path.dirname(__file__), 'creditcard.csv')
    if os.path.exists(kaggle_path):
        print("Found creditcard.csv — using real Kaggle data!")
        raw = pd.read_csv(kaggle_path)
        # Map Kaggle columns to our features
        df = pd.DataFrame()
        df['amount_ratio']           = (raw['Amount'] / raw['Amount'].mean()).clip(0, 100)
        df['hour_risk']              = raw['Time'].apply(lambda t: 1.0 if (t % 86400) < 14400 or (t % 86400) > 82800 else 0.1)
        df['location_risk']          = np.random.choice([0, 0.3, 0.6, 1.0], len(raw), p=[0.6, 0.15, 0.15, 0.1])
        df['device_risk']            = np.random.choice([0, 0.5, 1.0], len(raw), p=[0.8, 0.15, 0.05])
        df['merchant_risk']          = raw['V1'].apply(lambda v: min(abs(v)/5, 1.0))
        df['recipient_risk']         = np.random.choice([0, 0.5, 1.0], len(raw), p=[0.7, 0.2, 0.1])
        df['account_age_risk']       = np.random.beta(1, 5, len(raw))
        df['prior_fraud_risk']       = np.random.choice([0, 0.5, 1.0], len(raw), p=[0.93, 0.05, 0.02])
        df['combo_location_device']  = df['location_risk'] * df['device_risk']
        df['combo_hour_amount']      = df['hour_risk'] * (df['amount_ratio'] / 10).clip(0, 1)
        df['is_fraud']               = raw['Class']
    else:
        print("creditcard.csv not found — using synthetic data")
        print("(Put creditcard.csv in model/ folder to use real Kaggle data)")
        df = generate_synthetic_data(50000)

    X = df[FEATURES]
    y = df['is_fraud']

    print(f"\nDataset: {len(df):,} transactions")
    print(f"Fraud: {y.sum():,} ({y.mean()*100:.2f}%)")
    print(f"Legitimate: {(~y.astype(bool)).sum():,}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── RANDOM FOREST ────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        class_weight='balanced',   # handles imbalanced fraud data
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train)

    # Evaluate
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    print("\n── MODEL PERFORMANCE ──")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives  (correct legit): {tn:,}")
    print(f"False Positives (wrong blocks):  {fp:,}  ← false positive rate")
    print(f"False Negatives (missed fraud):  {fn:,}")
    print(f"True Positives  (caught fraud):  {tp:,}")

    # Feature importance
    importance = dict(zip(FEATURES, model.feature_importances_))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    print("\n── FEATURE IMPORTANCE ──")
    for feat, imp in importance_sorted.items():
        bar = '█' * int(imp * 50)
        print(f"{feat:<30} {bar} {imp:.3f}")

    # Save model + scaler + metadata
    out_dir = os.path.dirname(__file__)
    joblib.dump(model,  os.path.join(out_dir, 'fraud_model.pkl'))
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.pkl'))

    meta = {
        'features': FEATURES,
        'feature_importance': importance_sorted,
        'n_train': len(X_train),
        'n_test':  len(X_test),
        'fraud_rate': float(y.mean()),
        'false_positive_rate': float(fp / (fp + tn)),
        'recall': float(tp / (tp + fn)),
    }
    with open(os.path.join(out_dir, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved → fraud_model.pkl")
    print(f"Scaler saved → scaler.pkl")
    print(f"Metadata saved → model_meta.json")
    print("\nDone! Run app.py to start the server.")

if __name__ == '__main__':
    train()
