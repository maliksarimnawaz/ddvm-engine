from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, Text
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
    decision_id = Column(Integer, primary_key=True)
    user_id = Column(String)
    decision = Column(Float)
    confidence = Column(Float)
    anchor = Column(Float)
    signal = Column(Float)
    difficulty = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Outcome(Base):
    __tablename__ = "outcomes"
    outcome_id = Column(Integer, primary_key=True)
    decision_id = Column(Integer, ForeignKey("decisions.decision_id"))
    actual_outcome = Column(Float)


class Intervention(Base):
    __tablename__ = "interventions"
    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    action = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)


# Per-user bandit state persisted in DB
class BanditState(Base):
    __tablename__ = "bandit_states"
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True)
    values_json = Column(Text)  # JSON serialized action values
    counts_json = Column(Text)  # JSON serialized action counts


Base.metadata.create_all(engine)

# ============================================
# 2. TASK GENERATOR
# ============================================

def generate_task():
    true_value = float(np.random.normal(50, 10))
    difficulty = float(np.random.choice([5, 10, 15]))
    signal = float(true_value + np.random.normal(0, difficulty))
    # Anchor is independent of true_value to avoid circular bias measurement
    anchor = float(np.random.choice([30, 40, 50, 60, 70]))

    return {
        "true_value": true_value,
        "signal": signal,
        "anchor": anchor,
        "difficulty": difficulty
    }

# ============================================
# 3. DATA ACCESS
# ============================================

def get_user_df(user_id):
    with Session() as session:
        decisions = session.query(Decision).filter_by(user_id=user_id).all()
        outcomes = session.query(Outcome).all()
        outcome_map = {o.decision_id: o.actual_outcome for o in outcomes}

        data = []
        for d in decisions:
            if d.decision_id in outcome_map:
                data.append({
                    "decision": d.decision,
                    "confidence": d.confidence,
                    "actual_outcome": outcome_map[d.decision_id],
                    "anchor": d.anchor,
                    "timestamp": d.timestamp
                })

    return pd.DataFrame(data)

# ============================================
# 4. CALIBRATION
# ============================================

def compute_calibration(df):
    if len(df) < 10:
        return {"status": "insufficient_data"}

    df = df.copy()
    df["error"] = df["decision"] - df["actual_outcome"]

    # Fixed tolerance — not outcome-variance dependent
    tolerance = 5.0
    df["correct"] = (abs(df["error"]) < tolerance).astype(int)

    brier = float(np.mean((df["confidence"] / 100.0 - df["correct"]) ** 2))

    bins = np.linspace(0, 100, 6)
    df["bin"] = pd.cut(df["confidence"], bins)

    calib = df.groupby("bin", observed=True).agg(
        confidence=("confidence", "mean"),
        correct=("correct", "mean")
    ).dropna()

    gap = float((calib["confidence"] / 100.0 - calib["correct"]).mean())

    return {
        "brier": brier,
        "calibration_gap": gap,
        "n": len(df)
    }

# ============================================
# 5. HMM STATE — labels assigned by emission means
# ============================================

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

        # Assign labels by mean absolute error per state — not by index
        state_errors = {}
        for s in range(3):
            mask = states == s
            if mask.sum() > 0:
                state_errors[s] = np.mean(np.abs(error[mask]))

        sorted_states = sorted(state_errors, key=state_errors.get)
        label_map = {
            sorted_states[0]: "stable",
            sorted_states[1]: "noisy",
            sorted_states[2]: "biased"
        }

        return label_map[states[-1]]

    except Exception:
        return "unknown"

# ============================================
# 6. TRAJECTORY METRICS (ABAI, BE)
# ============================================

def compute_trajectory_metrics(df):
    if len(df) < 5:
        return {}

    df = df.copy()
    df["bias"] = df["decision"] - df["anchor"]

    # ABAI — Absolute Bias Across Anchors: mean absolute deviation from anchor
    abai = float(np.mean(np.abs(df["bias"])))

    # BE — Bias Elasticity: how much bias changes per unit anchor change
    if df["anchor"].std() > 0:
        be = float(np.corrcoef(df["anchor"], df["bias"])[0, 1])
    else:
        be = 0.0

    # Volatility — std of sequential bias changes
    bias_changes = np.diff(df["bias"].values)
    volatility = float(np.std(bias_changes)) if len(bias_changes) > 0 else 0.0

    return {
        "abai": abai,
        "bias_elasticity": be,
        "volatility": volatility
    }

# ============================================
# 7. PER-USER BANDIT
# ============================================

ACTIONS = ["reduce_conf", "increase_conf", "debias", "slow"]


def load_bandit(user_id):
    with Session() as session:
        state = session.query(BanditState).filter_by(user_id=user_id).first()
        if state:
            return {
                "values": json.loads(state.values_json),
                "counts": json.loads(state.counts_json)
            }
    return {
        "values": {a: 0.0 for a in ACTIONS},
        "counts": {a: 1 for a in ACTIONS}
    }


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
    bandit["values"][action] += (
        (reward - bandit["values"][action]) / bandit["counts"][action]
    )
    return bandit

# ============================================
# 8. FEEDBACK TEXT
# ============================================

def generate_feedback(action):
    return {
        "reduce_conf": "Your confidence is running ahead of your accuracy. Dial it back.",
        "increase_conf": "You are underestimating yourself. Trust your estimates more.",
        "debias": "Your estimates are tracking the anchor too closely. Re-evaluate independently.",
        "slow": "You are deciding too fast. Take more time before committing."
    }[action]

# ============================================
# 9. FLASK API
# ============================================

app = Flask(__name__)


@app.route("/")
def home():
    return jsonify({"status": "Decision Engine Running", "version": "2.0"})


@app.route("/task", methods=["GET"])
def task():
    return jsonify(generate_task())


@app.route("/submit_decision", methods=["POST"])
def submit_decision():
    data = request.json
    required = ["user_id", "decision", "confidence", "anchor", "signal", "difficulty"]
    if not all(k in data for k in required):
        return jsonify({"error": "Missing fields"}), 400

    with Session() as session:
        d = Decision(
            user_id=data["user_id"],
            decision=float(data["decision"]),
            confidence=float(data["confidence"]),
            anchor=float(data["anchor"]),
            signal=float(data["signal"]),
            difficulty=float(data["difficulty"])
        )
        session.add(d)
        session.commit()
        decision_id = d.decision_id

    return jsonify({"decision_id": decision_id})


@app.route("/submit_outcome", methods=["POST"])
def submit_outcome():
    data = request.json
    if "decision_id" not in data or "actual_outcome" not in data:
        return jsonify({"error": "Missing fields"}), 400

    with Session() as session:
        session.add(Outcome(
            decision_id=int(data["decision_id"]),
            actual_outcome=float(data["actual_outcome"])
        ))
        session.commit()

    return jsonify({"status": "ok"})


@app.route("/feedback/<user_id>", methods=["GET"])
def feedback(user_id):
    df = get_user_df(user_id)

    if len(df) < 10:
        return jsonify({"msg": f"Need more data. Have {len(df)} decisions with outcomes."})

    calib = compute_calibration(df)
    state = infer_state(df)
    trajectory = compute_trajectory_metrics(df)

    # Per-user bandit
    bandit = load_bandit(user_id)
    action = select_action(bandit)

    # Reward: recent error reduction
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
        "feedback": generate_feedback(action)
    })


@app.route("/user_summary/<user_id>", methods=["GET"])
def user_summary(user_id):
    df = get_user_df(user_id)
    if df.empty:
        return jsonify({"msg": "No data for this user"})

    return jsonify({
        "n_decisions": len(df),
        "mean_error": float(np.mean(np.abs(df["decision"] - df["actual_outcome"]))),
        "mean_confidence": float(df["confidence"].mean()),
        "trajectory": compute_trajectory_metrics(df)
    })


# ============================================
# 10. RUN
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
