from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import random
import json
import os

# ============================================
# 1. DATABASE
# ============================================

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///decisions.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Decision(Base):
    __tablename__ = "decisions"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), nullable=False)
    decision_value = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text)
    anchor = Column(Float)
    signal = Column(Float)
    difficulty = Column(Float)
    true_value = Column(Float)
    adjusted_estimate = Column(Float)
    anchor_warning = Column(Boolean, default=False)
    outcome_value = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Intervention(Base):
    __tablename__ = "interventions"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50))
    action = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)


class BanditState(Base):
    __tablename__ = "bandit_states"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True)
    values_json = Column(Text)
    counts_json = Column(Text)


Base.metadata.create_all(engine)

# ============================================
# 2. TASK GENERATOR
# ============================================

def generate_task():
    # Anchor = last month's users (30k-70k), presented to user
    anchor = float(np.random.choice([30, 40, 45, 50, 55, 60, 70]))
    # True value = actual this month's users, ±15% of anchor
    true_value = float(anchor * np.random.uniform(0.85, 1.15))
    # Signal = analyst estimate, ±20% of anchor (may be wrong)
    signal = float(anchor * np.random.uniform(0.80, 1.20))
    # Difficulty = how far signal is from true value
    difficulty = float(abs(signal - true_value))
    return {
        "true_value": round(true_value, 2),
        "signal": round(signal, 2),
        "anchor": round(anchor, 2),
        "difficulty": round(difficulty, 2)
    }

# ============================================
# 3. ANALYTICS
# ============================================

def get_user_df(user_id):
    with Session() as session:
        decisions = session.query(Decision).filter(
            Decision.user_id == user_id,
            Decision.outcome_value.isnot(None)
        ).all()
        data = [{
            "decision": d.decision_value,
            "confidence": d.confidence,
            "actual_outcome": d.outcome_value,
            "anchor": d.anchor or 50.0,
        } for d in decisions]
    return pd.DataFrame(data)


def compute_calibration(df):
    if len(df) < 10:
        return {"status": "insufficient_data"}
    df = df.copy()
    df["error"] = df["decision"] - df["actual_outcome"]
    tolerance = 5.0
    df["correct"] = (abs(df["error"]) < tolerance).astype(int)
    brier = float(np.mean((df["confidence"] / 100.0 - df["correct"]) ** 2))
    bins = np.linspace(0, 100, 6)
    df["bin"] = pd.cut(df["confidence"], bins)
    calib = df.groupby("bin", observed=True).agg(
        confidence=("confidence", "mean"),
        correct=("correct", "mean")
    ).dropna()
    gap = float((calib["confidence"] / 100.0 - calib["correct"]).mean()) if len(calib) > 0 else 0.0
    return {"brier": round(brier, 4), "calibration_gap": round(gap, 4), "n": len(df)}


def infer_state(df):
    if len(df) < 20:
        return "unknown"
    df = df.copy()
    error = (df["decision"] - df["actual_outcome"]).values
    confidence = df["confidence"].values
    X = np.column_stack([error, confidence])
    try:
        model = GaussianHMM(n_components=3, n_iter=100, random_state=42)
        model.fit(X)
        states = model.predict(X)
        state_errors = {}
        for s in range(3):
            mask = states == s
            if mask.sum() > 0:
                state_errors[s] = np.mean(np.abs(error[mask]))
        sorted_states = sorted(state_errors, key=state_errors.get)
        label_map = {sorted_states[0]: "stable", sorted_states[1]: "noisy", sorted_states[2]: "biased"}
        return label_map[states[-1]]
    except Exception:
        return "unknown"


def compute_trajectory_metrics(df):
    if len(df) < 5:
        return {}
    df = df.copy()
    df["bias"] = df["decision"] - df["anchor"]
    abai = float(np.mean(np.abs(df["bias"])))
    be = float(np.corrcoef(df["anchor"], df["bias"])[0, 1]) if df["anchor"].std() > 0 else 0.0
    bias_changes = np.diff(df["bias"].values)
    volatility = float(np.std(bias_changes)) if len(bias_changes) > 0 else 0.0
    return {"abai": round(abai, 3), "bias_elasticity": round(be, 3), "volatility": round(volatility, 3)}

# ============================================
# 4. PER-USER BANDIT
# ============================================

ACTIONS = ["reduce_conf", "increase_conf", "debias", "slow"]


def load_bandit(user_id):
    with Session() as session:
        state = session.query(BanditState).filter_by(user_id=user_id).first()
        if state:
            return {"values": json.loads(state.values_json), "counts": json.loads(state.counts_json)}
    return {"values": {a: 0.0 for a in ACTIONS}, "counts": {a: 1 for a in ACTIONS}}


def save_bandit(user_id, bandit):
    with Session() as session:
        state = session.query(BanditState).filter_by(user_id=user_id).first()
        if state:
            state.values_json = json.dumps(bandit["values"])
            state.counts_json = json.dumps(bandit["counts"])
        else:
            session.add(BanditState(
                user_id=user_id,
                values_json=json.dumps(bandit["values"]),
                counts_json=json.dumps(bandit["counts"])
            ))
        session.commit()


def select_action(bandit):
    if random.random() < 0.3:
        return random.choice(ACTIONS)
    return max(bandit["values"], key=bandit["values"].get)


def update_bandit(bandit, action, reward):
    bandit["counts"][action] += 1
    bandit["values"][action] += (reward - bandit["values"][action]) / bandit["counts"][action]
    return bandit


FEEDBACK_TEXT = {
    "reduce_conf": "Your confidence is running ahead of your accuracy. Dial it back.",
    "increase_conf": "You are underestimating yourself. Trust your estimates more.",
    "debias": "Your estimates are tracking the anchor too closely. Re-evaluate independently.",
    "slow": "You are deciding too fast. Take more time before committing."
}

# ============================================
# 5. FLASK APP
# ============================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "ddvm_dev_secret")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    try:
        user_id = request.form.get("user_id", "").strip()
        decision_value = float(request.form.get("decision_value"))
        confidence = float(request.form.get("confidence", 50))
        reasoning = request.form.get("reasoning", "")
        anchor = float(request.form.get("anchor", 50))
        signal = float(request.form.get("signal", 50))
        difficulty = float(request.form.get("difficulty", 10))
        true_value = float(request.form.get("true_value", 50))

        anchor_warning = abs(decision_value - anchor) < (abs(anchor) * 0.05 + 1)
        adjusted_estimate = round(decision_value * 0.7 + true_value * 0.3, 2)
        error = round(abs(decision_value - true_value), 2)

        with Session() as session:
            d = Decision(
                user_id=user_id,
                decision_value=decision_value,
                confidence=confidence,
                reasoning=reasoning,
                anchor=anchor,
                signal=signal,
                difficulty=difficulty,
                true_value=true_value,
                adjusted_estimate=adjusted_estimate,
                anchor_warning=anchor_warning,
                outcome_value=true_value
            )
            session.add(d)
            session.commit()
            decision_id = d.id

        df = get_user_df(user_id)
        intervention = None
        if len(df) >= 5:
            bandit = load_bandit(user_id)
            action = select_action(bandit)
            errors = np.abs(df["decision"] - df["actual_outcome"]).values
            reward = float(np.mean(errors[:-3]) - np.mean(errors[-3:])) if len(errors) > 3 else 0.0
            bandit = update_bandit(bandit, action, reward)
            save_bandit(user_id, bandit)
            with Session() as session:
                session.add(Intervention(user_id=user_id, action=action))
                session.commit()
            intervention = FEEDBACK_TEXT[action]

        return jsonify({
            "status": "ok",
            "decision_id": decision_id,
            "true_value": round(true_value, 1),
            "your_estimate": round(decision_value, 1),
            "error": error,
            "anchor_warning": anchor_warning,
            "intervention": intervention
        })

    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 400


@app.route("/dashboard")
def dashboard():
    with Session() as session:
        decisions = session.query(Decision).order_by(Decision.timestamp.desc()).all()
        session.expunge_all()
    return render_template("dashboard.html", decisions=decisions)


@app.route("/log_outcome/<int:decision_id>", methods=["GET", "POST"])
def log_outcome(decision_id):
    with Session() as session:
        decision = session.query(Decision).get(decision_id)
        if not decision:
            flash("Decision not found.", "error")
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            try:
                outcome = float(request.form.get("outcome_value"))
                decision.outcome_value = outcome
                session.commit()
                flash("Outcome recorded.")
                return redirect(url_for("dashboard"))
            except Exception as e:
                flash(f"Error: {str(e)}", "error")

        session.expunge(decision)

    return render_template("log_outcome.html", decision=decision)


@app.route("/task", methods=["GET"])
def task():
    return jsonify(generate_task())


@app.route("/api/feedback/<user_id>", methods=["GET"])
def api_feedback(user_id):
    df = get_user_df(user_id)
    if len(df) < 10:
        return jsonify({"msg": f"Need more data. Have {len(df)} decisions with outcomes."})

    calib = compute_calibration(df)
    state = infer_state(df)
    trajectory = compute_trajectory_metrics(df)
    bandit = load_bandit(user_id)
    action = select_action(bandit)

    error = np.abs(df["decision"] - df["actual_outcome"]).values
    reward = float(np.mean(error[:-5]) - np.mean(error[-5:])) if len(error) > 5 else 0.0
    bandit = update_bandit(bandit, action, reward)
    save_bandit(user_id, bandit)

    with Session() as session:
        session.add(Intervention(user_id=user_id, action=action))
        session.commit()

    return jsonify({
        "state": state,
        "calibration": calib,
        "trajectory": trajectory,
        "action": action,
        "feedback": FEEDBACK_TEXT[action]
    })


@app.route("/api/summary/<user_id>", methods=["GET"])
def api_summary(user_id):
    df = get_user_df(user_id)
    if df.empty:
        return jsonify({"msg": "No data for this user"})
    return jsonify({
        "n_decisions": len(df),
        "mean_error": round(float(np.mean(np.abs(df["decision"] - df["actual_outcome"]))), 3),
        "mean_confidence": round(float(df["confidence"].mean()), 1),
        "trajectory": compute_trajectory_metrics(df)
    })


# ============================================
# 6. RUN
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)