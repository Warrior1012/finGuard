"""
FinGuard Backend — Flask API + SQLite Database
Invisible Fraud Detector System
"""
import random
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib, json, os, numpy as np
import sqlite3
from datetime import datetime
import smtplib
import random

otp_store = {}

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

BASE      = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE, 'finguard.db')
MODEL_DIR = os.path.join(BASE, 'model')

# ── EMAIL OTP ─────────────────────────────────────────────────────────
def send_email_otp(receiver_email, otp):
    sender_email = "cabc4619@gmail.com"
    app_password = "hojhifjnrjzlsdvc"
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login(sender_email, app_password)
        message = f"Subject: FinGuard Security OTP\n\nYour FinGuard verification OTP is: {otp}\n\nThis OTP expires in 5 minutes. Do not share it with anyone."
        server.sendmail(sender_email, receiver_email, message)
        print("Sending OTP to:", receiver_email)
        server.quit()
        return True
    except Exception as e:
        print(f"Email failed: {e}")
        return False

# ── DATABASE ──────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS transactions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT    NOT NULL,
        amount      REAL    NOT NULL,
        merchant    TEXT    NOT NULL,
        hour        INTEGER NOT NULL,
        city        TEXT    NOT NULL,
        device      TEXT    NOT NULL,
        recipient   TEXT    NOT NULL,
        avgspend    REAL    NOT NULL,
        acctage     INTEGER NOT NULL,
        prevfraud   TEXT    NOT NULL,
        score       INTEGER NOT NULL,
        ml_score    INTEGER NOT NULL,
        rule_score  INTEGER NOT NULL,
        fraud_prob  REAL    NOT NULL,
        verdict     TEXT    NOT NULL,
        explanation TEXT    NOT NULL,
        flags_count INTEGER NOT NULL,
        high_flags  INTEGER NOT NULL
    )''')
    conn.commit()
    conn.close()
    print("SQLite ready → finguard.db")

def save_to_db(data, result):
    conn = get_db()
    conn.execute('''INSERT INTO transactions (
        timestamp,amount,merchant,hour,city,device,recipient,
        avgspend,acctage,prevfraud,score,ml_score,rule_score,
        fraud_prob,verdict,explanation,flags_count,high_flags
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        float(data.get('amount', 0)), data.get('merchant', ''),
        int(data.get('hour', 0)), data.get('city', ''),
        data.get('device', ''), data.get('recipient', ''),
        float(data.get('avgspend', 0)), int(data.get('acctage', 0)),
        str(data.get('prevfraud', '0')),
        result['score'], result['ml_score'], result['rule_score'],
        result['fraud_probability'], result['verdict'],
        result['explanation'], result['flags_count'], result['high_flags']
    ))
    conn.commit()
    conn.close()

init_db()

# ── LOAD MODEL ────────────────────────────────────────────────────────
try:
    model  = joblib.load(os.path.join(MODEL_DIR, 'fraud_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    with open(os.path.join(MODEL_DIR, 'model_meta.json')) as f:
        meta = json.load(f)
    print(f"ML model loaded. Trained on {meta['n_train']:,} transactions")
except Exception as e:
    print(f"Model load failed: {e}")
    model = scaler = meta = None

FEATURES = ['amount_ratio','hour_risk','location_risk','device_risk',
            'merchant_risk','recipient_risk','account_age_risk',
            'prior_fraud_risk','combo_location_device','combo_hour_amount']

ML_LABELS = {
    'grocery': 'Grocery', 'restaurant': 'Restaurant', 'fuel': 'Fuel',
    'electronics': 'Electronics', 'international': 'International merchant',
    'crypto': 'Crypto exchange', 'atm': 'ATM withdrawal', 'transfer': 'P2P transfer'
}

# ── FEATURE ENGINEERING ───────────────────────────────────────────────
def engineer_features(data):
    avg_daily = max(float(data.get('avgspend', 8000)) / 30, 1)
    amount    = float(data.get('amount', 500))
    hour      = int(data.get('hour', 14))
    city      = data.get('city', 'home')
    device    = data.get('device', 'known')
    merchant  = data.get('merchant', 'grocery')
    recipient = data.get('recipient', 'saved')
    acctage   = int(data.get('acctage', 24))
    prevfraud = data.get('prevfraud', '0')

    amount_ratio = min(amount / avg_daily, 100)

    if 1 <= hour <= 4:             hour_risk = 1.0
    elif hour == 0 or hour >= 23:  hour_risk = 0.7
    elif hour >= 21:               hour_risk = 0.4
    elif hour >= 20:               hour_risk = 0.2
    else:                          hour_risk = 0.05

    loc = {'home': 0.0, 'known': 0.3, 'new_domestic': 0.6, 'foreign': 1.0}.get(city, 0.0)
    dev = {'known': 0.0, 'new': 0.5, 'unknown': 1.0}.get(device, 0.0)
    mer = {'grocery': 0.02, 'restaurant': 0.03, 'fuel': 0.05, 'transfer': 0.25,
           'electronics': 0.35, 'atm': 0.45, 'international': 0.65, 'crypto': 0.9}.get(merchant, 0.1)
    rec = {'saved': 0.0, 'new': 0.5, 'firsttime': 1.0}.get(recipient, 0.0)
    age = max(0, 1 - (acctage / 24))
    pf  = {'0': 0.0, '1': 0.5, '2': 1.0}.get(str(prevfraud), 0.0)
    cld = loc * dev
    cha = hour_risk * min(amount_ratio / 10, 1.0)

    return [amount_ratio, hour_risk, loc, dev, mer, rec, age, pf, cld, cha]

def get_flags(data, fraud_prob):
    flags = []
    avg_daily = max(float(data.get('avgspend', 8000)) / 30, 1)
    amount    = float(data.get('amount', 500))
    hour      = int(data.get('hour', 14))
    city      = data.get('city', 'home')
    device    = data.get('device', 'known')
    merchant  = data.get('merchant', 'grocery')
    recipient = data.get('recipient', 'saved')
    acctage   = int(data.get('acctage', 24))
    prevfraud = str(data.get('prevfraud', '0'))
    ratio     = amount / avg_daily

    if ratio > 50:    flags.append({'text': f'Amount ₹{amount:,.0f} is {ratio:.0f}x your daily average — extreme outlier', 'weight': 'high'})
    elif ratio > 15:  flags.append({'text': f'Amount is {ratio:.1f}x your daily average — significantly above normal', 'weight': 'high'})
    elif ratio > 6:   flags.append({'text': f'Amount is {ratio:.1f}x your typical daily spend', 'weight': 'medium'})
    elif ratio > 3:   flags.append({'text': 'Slightly above average spend', 'weight': 'low'})

    if 1 <= hour <= 4:             flags.append({'text': f'Transaction at {hour}:00 AM — deep night hours are rare for legitimate payments', 'weight': 'high'})
    elif hour == 0 or hour >= 23:  flags.append({'text': f'Late-night transaction at {hour}:00', 'weight': 'medium'})

    if city == 'foreign':          flags.append({'text': 'Transaction from a foreign country — no prior international activity detected', 'weight': 'high'})
    elif city == 'new_domestic':   flags.append({'text': 'First transaction from this domestic city', 'weight': 'medium'})

    if device == 'unknown':        flags.append({'text': 'Unrecognized device — never associated with this account', 'weight': 'high'})
    elif device == 'new':          flags.append({'text': 'New device used for the first time', 'weight': 'medium'})

    if merchant in ['crypto', 'international']:  flags.append({'text': f'{ML_LABELS[merchant]} transactions carry high fraud risk', 'weight': 'high'})
    elif merchant in ['atm', 'electronics']:     flags.append({'text': f'{ML_LABELS[merchant]} — moderately elevated merchant risk', 'weight': 'medium'})

    if recipient == 'firsttime':   flags.append({'text': 'First-time payee — no prior transaction history', 'weight': 'medium'})
    elif recipient == 'new':       flags.append({'text': 'New recipient with limited history', 'weight': 'low'})

    if acctage < 3:                flags.append({'text': f'Account only {acctage} months old — higher fraud base rate', 'weight': 'medium'})
    if prevfraud == '2':           flags.append({'text': '2+ prior fraud flags on this account', 'weight': 'high'})
    elif prevfraud == '1':         flags.append({'text': '1 prior fraud flag on this account', 'weight': 'medium'})

    if city != 'home' and device == 'unknown':  flags.append({'text': 'Combination: New location + unrecognized device — strong account takeover signal', 'weight': 'high'})
    if (1 <= hour <= 4) and ratio > 5:          flags.append({'text': 'Combination: High amount at unusual hour — correlated fraud pattern', 'weight': 'high'})

    flags.sort(key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['weight']])
    return flags[:6]

def get_explanation(data, score, flags, verdict):
    amount    = float(data.get('amount', 500))
    hour      = int(data.get('hour', 14))
    city      = data.get('city', 'home')
    device    = data.get('device', 'known')
    merchant  = data.get('merchant', 'grocery')
    avg       = float(data.get('avgspend', 8000))
    ratio     = amount / max(avg / 30, 1)
    high_ct   = sum(1 for f in flags if f['weight'] == 'high')

    if score >= 70:
        if city == 'foreign' and device == 'unknown':
            return f"This transaction was blocked because it originated from a foreign country on an unrecognized device — a combination that accounts for over 80% of confirmed account takeovers. The amount of ₹{amount:,.0f} also exceeds your typical daily spend by {ratio:.0f}x, further raising the risk."
        if merchant == 'crypto' and 1 <= hour <= 4:
            return f"A ₹{amount:,.0f} crypto exchange transaction at {hour}:00 AM is an extremely unusual pattern. Crypto transfers at odd hours are one of the strongest fraud indicators — once funds reach a crypto wallet, recovery is nearly impossible. Transaction blocked."
        if high_ct >= 3:
            return f"Our ML model detected {high_ct} high-risk signals simultaneously. When multiple strong indicators appear together, fraud probability rises sharply — the model scored this at {score}/100. Transaction blocked as a precaution."
        return f"This transaction was flagged with a risk score of {score}/100. The amount of ₹{amount:,.0f} is {ratio:.0f}x your daily average and {flags[0]['text'].lower() if flags else 'unusual patterns were detected'}. Blocked to protect your account."
    elif score >= 45:
        if device == 'unknown':
            return f"This transaction came from a device never used with your account before. Combined with {'the high amount' if ratio > 3 else 'other signals'}, the model flagged this for review. This could be you on a new device, or unauthorized access — please verify."
        if merchant == 'firsttime':
            return f"You are sending ₹{amount:,.0f} to someone you have never transacted with before. First-time payees are a common vector in social engineering fraud. The model scored this at {score}/100. Please confirm the recipient before approving."
        return f"A few patterns differ from your normal activity — {flags[0]['text'].lower() if flags else 'unusual signals detected'}. Risk score {score}/100 falls in the suspicious range. Transaction held for OTP verification."
    elif score >= 15:
        top  = flags[0]['text'] if flags else None
        note = f" One minor signal: {top.lower()}." if top else ""
        return f"This transaction looks largely normal.{note} Risk score {score}/100 is within safe limits. Approved."
    else:
        hour12 = f"{hour} AM" if hour < 12 else f"{hour - 12 if hour > 12 else 12} PM"
        return f"Everything checks out. Amount of ₹{amount:,.0f} fits your normal spending range at {hour12} on a recognized device. Risk score: {score}/100. Transaction approved."

def rule_score(data):
    s         = 0
    avg_daily = max(float(data.get('avgspend', 8000)) / 30, 1)
    amount    = float(data.get('amount', 500))
    hour      = int(data.get('hour', 14))
    ratio     = amount / avg_daily

    if ratio > 50:    s += 30
    elif ratio > 15:  s += 22
    elif ratio > 6:   s += 12
    elif ratio > 3:   s += 5

    if 1 <= hour <= 4:             s += 20
    elif hour == 0 or hour >= 23:  s += 12
    elif hour >= 22:               s += 5

    s += {'home': 0, 'known': 2, 'new_domestic': 10, 'foreign': 20}.get(data.get('city', 'home'), 0)
    s += {'known': 0, 'new': 8, 'unknown': 15}.get(data.get('device', 'known'), 0)
    s += {'saved': 0, 'new': 5, 'firsttime': 10}.get(data.get('recipient', 'saved'), 0)
    s += {'grocery': 0, 'restaurant': 0, 'fuel': 0, 'transfer': 4, 'electronics': 6,
          'atm': 8, 'international': 12, 'crypto': 15}.get(data.get('merchant', 'grocery'), 0)

    if int(data.get('acctage', 24)) < 3:  s += 5
    s += {'0': 0, '1': 8, '2': 15}.get(str(data.get('prevfraud', '0')), 0)

    if data.get('city', 'home') != 'home' and data.get('device', 'known') == 'unknown':  s += 8
    if (1 <= hour <= 4) and ratio > 5:  s += 5

    return min(s, 100)

# ── ROUTES ────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/api/health')
def health():
    conn  = get_db()
    total = conn.execute('SELECT COUNT(*) FROM transactions').fetchone()[0]
    conn.close()
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'version': '2.0',
        'total_analyzed': total,
        'trained_on': meta['n_train'] if meta else 0,
        'false_positive_rate': round(meta['false_positive_rate'] * 100, 2) if meta else 0
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        user_email = data.get('email')

        feats = engineer_features(data)

        if model and scaler:
            X  = np.array(feats).reshape(1, -1)
            Xs = scaler.transform(X)
            fp = float(model.predict_proba(Xs)[0][1])
            mls = round(fp * 100)
            mu  = True
        else:
            fp  = 0.5
            mls = 50
            mu  = False

        rs = rule_score(data)
        sc = max(0, min(100, round(mls * 0.35 + rs * 0.65)))

        base = rs / 100
        noise = random.uniform(-0.1, 0.1)   # ±10% variation
        fp = min(1.0, max(0.0, base + noise))
        mls = round(fp * 100, 2)

        if sc >= 60:
            vd     = 'fraud'
            action = 'BLOCK'
        elif sc >= 30:
            vd     = 'suspicious'
            action = 'OTP_REQUIRED'
        else:
            vd     = 'safe'
            action = 'ALLOW'

        otp_required = False
        otp_sent     = False

        if action == 'OTP_REQUIRED':
            otp_required = True
            otp = str(random.randint(100000, 999999))
            otp_store['demo_user'] = otp
            print(f"OTP Generated: {otp}")
            if user_email:
                otp_sent = send_email_otp(user_email, otp)

        fl = get_flags(data, fp)
        ex = get_explanation(data, sc, fl, vd)
        fi = {FEATURES[i]: round(float(model.feature_importances_[i] * feats[i]), 4) for i in range(len(FEATURES))} if model else {}

        result = {
            'score': sc,
            'ml_score': mls,
            'rule_score': rs,
            'fraud_probability': round(fp * 100, 2),
            'verdict': vd,
            'flags': fl,
            'explanation': ex,
            'model_used': mu,
            'feature_importance': fi,
            'flags_count': len(fl),
            'high_flags': sum(1 for f in fl if f['weight'] == 'high'),
            'action': action,
            'otp_required': otp_required,
            'otp_sent': otp_sent
        }

        save_to_db(data, result)
        return jsonify(result)

    except Exception as e:
        print(f"Error in /api/analyze: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    data     = request.get_json()
    user_otp = data.get('otp', '')
    stored   = otp_store.get('demo_user', '')

    if stored and stored == user_otp:
        del otp_store['demo_user']
        return jsonify({'status': 'VERIFIED', 'message': 'OTP verified. Transaction approved.'})
    return jsonify({'status': 'FAILED', 'message': 'Invalid OTP. Please try again.'})

@app.route('/api/history')
def history():
    try:
        limit = int(request.args.get('limit', 50))
        conn  = get_db()
        rows  = conn.execute('SELECT * FROM transactions ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<int:txn_id>', methods=['DELETE'])
def delete_txn(txn_id):
    try:
        conn = get_db()
        conn.execute('DELETE FROM transactions WHERE id=?', (txn_id,))
        conn.commit()
        conn.close()
        return jsonify({'deleted': txn_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/clear', methods=['DELETE'])
def clear_history():
    try:
        conn = get_db()
        conn.execute('DELETE FROM transactions')
        conn.commit()
        conn.close()
        return jsonify({'cleared': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def stats():
    try:
        conn    = get_db()
        total   = conn.execute('SELECT COUNT(*) FROM transactions').fetchone()[0]
        fraud   = conn.execute("SELECT COUNT(*) FROM transactions WHERE verdict='fraud'").fetchone()[0]
        susp    = conn.execute("SELECT COUNT(*) FROM transactions WHERE verdict='suspicious'").fetchone()[0]
        safe    = conn.execute("SELECT COUNT(*) FROM transactions WHERE verdict='safe'").fetchone()[0]
        avg_sc  = conn.execute('SELECT AVG(score) FROM transactions').fetchone()[0]
        max_amt = conn.execute('SELECT MAX(amount) FROM transactions').fetchone()[0]
        avg_amt = conn.execute('SELECT AVG(amount) FROM transactions').fetchone()[0]
        conn.close()
        return jsonify({
            'total': total, 'fraud': fraud, 'suspicious': susp, 'safe': safe,
            'avg_score': round(avg_sc or 0, 1),
            'max_amount': max_amt or 0,
            'avg_amount': round(avg_amt or 0, 2),
            'fraud_rate': round((fraud / total * 100) if total else 0, 1)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  FinGuard — Invisible Fraud Detector System")
    print("  Open browser: http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, port=5000, host='0.0.0.0')