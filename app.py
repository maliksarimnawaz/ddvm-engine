from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats
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
DBSession = sessionmaker(bind=engine)
Base = declarative_base()

TOTAL_ROUNDS = 20


class UserSession(Base):
    """Backend-authoritative session state. Frontend has no state."""
    __tablename__ = "user_sessions"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True, nullable=False)
    condition = Column(String(20), default="control")
    current_round = Column(Integer, default=1)
    completed = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # current task stored server-side — no sensitive data sent to frontend
    task_json = Column(Text)


class Decision(Base):
    __tablename__ = "decisions"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), nullable=False)
    condition = Column(String(20), default="control")
    round_number = Column(Integer)
    decision_value = Column(Float, nullable=False)
    reasoning = Column(Text)
    anchor = Column(Float)
    anchor_displacement = Column(Float)
    signal = Column(Float)
    signal_noise = Column(Float)
    true_value = Column(Float)
    outcome_value = Column(Float)
    abi = Column(Float)
    relative_error = Column(Float)
    submitted_at = Column(DateTime, default=datetime.utcnow)


class Intervention(Base):
    __tablename__ = "interventions"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50))
    round_number = Column(Integer)
    action = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class BanditState(Base):
    __tablename__ = "bandit_states"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True)
    values_json = Column(Text)
    counts_json = Column(Text)


class EventLog(Base):
    """Observability layer."""
    __tablename__ = "event_log"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50))
    round_number = Column(Integer)
    event_type = Column(String(50))
    metadata_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)


def migrate_schema():
    """Safe column additions for existing Postgres tables."""
    alterations = [
        ("decisions", "condition",           "VARCHAR(20) DEFAULT 'control'"),
        ("decisions", "round_number",        "INTEGER"),
        ("decisions", "anchor_displacement", "FLOAT"),
        ("decisions", "signal_noise",        "FLOAT"),
        ("decisions", "abi",                 "FLOAT"),
        ("decisions", "relative_error",      "FLOAT"),
        ("decisions", "submitted_at",        "TIMESTAMP"),
    ]
    import sqlalchemy as sa
    with engine.connect() as conn:
        for table, col, typ in alterations:
            try:
                conn.execute(sa.text(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {typ}"
                ))
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass


try:
    migrate_schema()
except Exception:
    pass

# ============================================
# 2. TASK GENERATOR — fully independent
# ============================================

ANCHOR_DISPLACEMENTS = [-20, -12, -6, 6, 12, 20]
SIGNAL_NOISE_TIERS = [3, 6, 10]


def generate_task():
    true_value = float(max(15.0, np.random.normal(50, 8)))
    displacement = float(np.random.choice(ANCHOR_DISPLACEMENTS))
    anchor = true_value + displacement
    noise_tier = float(np.random.choice(SIGNAL_NOISE_TIERS))
    signal = float(true_value + np.random.normal(0, noise_tier))
    return {
        "true_value": round(true_value, 3),
        "anchor": round(anchor, 2),
        "anchor_displacement": round(displacement, 2),
        "signal": round(signal, 2),
        "signal_noise": round(noise_tier, 2)
    }

# ============================================
# 3. METRICS
# ============================================

def compute_abi(estimate, true_value, anchor):
    denom = anchor - true_value
    if abs(denom) < 0.01:
        return 0.0
    return float((estimate - true_value) / denom)


def compute_relative_error(estimate, true_value):
    if abs(true_value) < 0.01:
        return 0.0
    return float(abs(estimate - true_value) / true_value)


def get_user_df(user_id):
    with DBSession() as session:
        decisions = session.query(Decision).filter(
            Decision.user_id == user_id,
            Decision.outcome_value.isnot(None)
        ).order_by(Decision.round_number).all()
        data = [{
            "round": d.round_number or 0,
            "estimate": d.decision_value,
            "true_value": d.true_value or 0,
            "anchor": d.anchor or 50,
            "anchor_displacement": d.anchor_displacement or 0,
            "signal": d.signal or 50,
            "signal_noise": d.signal_noise or 6,
            "abi": d.abi or 0,
            "relative_error": d.relative_error or 0,
            "condition": d.condition or "control"
        } for d in decisions]
    return pd.DataFrame(data)


def compute_full_metrics(df):
    if len(df) < 5:
        return {"status": "insufficient_data", "n": len(df)}

    df = df.copy().sort_values("round").reset_index(drop=True)
    n = len(df)
    results = {"n": n}

    abi_vals = df["abi"].fillna(0).values

    # Mean ABI
    results["mean_abi"] = round(float(np.mean(abi_vals)), 4)
    results["std_abi"] = round(float(np.std(abi_vals)), 4)

    # TLI
    if n >= 10:
        half = n // 2
        early_std = float(np.std(abi_vals[:half]))
        late_std = float(np.std(abi_vals[half:]))
        results["tli"] = round(float(1 - (late_std / early_std)) if early_std > 0.01 else 0.0, 4)
    else:
        results["tli"] = None

    # BE
    displacements = np.abs(df["anchor_displacement"].fillna(0).values)
    if displacements.std() > 0.01 and n >= 8:
        slope, _, r, p, _ = stats.linregress(displacements, abi_vals)
        results["be_slope"] = round(float(slope), 4)
        results["be_r"] = round(float(r), 4)
        results["be_p"] = round(float(p), 4)
    else:
        results["be_slope"] = None
        results["be_r"] = None
        results["be_p"] = None

    # SII
    sig_err = (df["signal"] - df["true_value"]).fillna(0).values
    est_err = (df["estimate"] - df["true_value"]).fillna(0).values
    if np.std(sig_err) > 0.01 and np.std(est_err) > 0.01:
        r_sig = float(np.corrcoef(est_err, sig_err)[0, 1])
        results["sii"] = round(float(1 - abs(r_sig)), 4)
        results["signal_correlation"] = round(r_sig, 4)
    else:
        results["sii"] = 1.0
        results["signal_correlation"] = 0.0

    # CD
    rounds = df["round"].values.astype(float)
    rel_errors = df["relative_error"].fillna(0).values
    if n >= 8 and np.std(rounds) > 0:
        slope, _, r, p, _ = stats.linregress(rounds, rel_errors)
        results["cd_slope"] = round(float(slope), 6)
        results["cd_r"] = round(float(r), 4)
        results["cd_p"] = round(float(p), 4)
    else:
        results["cd_slope"] = None

    results["mean_relative_error"] = round(float(np.mean(rel_errors)), 4)

    # ADS
    adjustments = np.abs(df["estimate"].values - df["anchor"].fillna(50).values)
    if np.std(displacements) > 0.01 and np.std(adjustments) > 0.01:
        results["ads"] = round(float(np.corrcoef(adjustments, displacements)[0, 1]), 4)
    else:
        results["ads"] = None

    # CSS via HMM
    if n >= 15:
        try:
            X = abi_vals.reshape(-1, 1)
            model = GaussianHMM(n_components=3, n_iter=200, random_state=42, covariance_type="full")
            model.fit(X)
            states = model.predict(X)
            means = {s: float(model.means_[s][0]) for s in range(3)}
            sorted_states = sorted(means, key=means.get)
            label_map = {sorted_states[0]: "under_anchored", sorted_states[1]: "moderate", sorted_states[2]: "over_anchored"}
            state_seq = [label_map[s] for s in states]
            results["css_terminal_state"] = state_seq[-1]
            results["css_transition_count"] = sum(1 for i in range(1, len(state_seq)) if state_seq[i] != state_seq[i-1])
            results["css_state_sequence"] = state_seq
        except Exception:
            results["css_terminal_state"] = "unknown"
            results["css_transition_count"] = None
            results["css_state_sequence"] = []
    else:
        results["css_terminal_state"] = "insufficient_data"
        results["css_transition_count"] = None
        results["css_state_sequence"] = []

    results["round_abi"] = [round(float(v), 3) for v in abi_vals]
    results["round_relative_error"] = [round(float(v), 3) for v in rel_errors]
    results["rounds"] = [int(r) for r in df["round"].values]

    return results

# ============================================
# 4. BANDIT
# ============================================

ACTIONS = ["debias", "slow", "reanchor", "ignore_signal"]

FEEDBACK_TEXT = {
    "debias": "Your estimates are tracking the reference figure closely. Try forming your prediction before looking at the reference.",
    "slow": "Consider whether your first instinct is being influenced by the figures shown. Take a moment before committing.",
    "reanchor": "The reference figure may be misleading. Focus on what you know independently of the value shown.",
    "ignore_signal": "The market forecast has been inaccurate in recent rounds. Weight it accordingly."
}


def load_bandit(user_id):
    with DBSession() as session:
        state = session.query(BanditState).filter_by(user_id=user_id).first()
        if state:
            return {"values": json.loads(state.values_json), "counts": json.loads(state.counts_json)}
    return {"values": {a: 0.0 for a in ACTIONS}, "counts": {a: 1 for a in ACTIONS}}


def save_bandit(user_id, bandit):
    with DBSession() as session:
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


def select_action(bandit, df):
    if len(df) >= 5:
        recent_abi = df["abi"].iloc[-5:].mean()
        if recent_abi > 0.5 and random.random() < 0.6:
            return random.choice(["debias", "reanchor"])
        if recent_abi < 0.2 and random.random() < 0.4:
            return "ignore_signal"
    if random.random() < 0.25:
        return random.choice(ACTIONS)
    return max(bandit["values"], key=bandit["values"].get)


def update_bandit(bandit, action, reward):
    bandit["counts"][action] += 1
    bandit["values"][action] += (reward - bandit["values"][action]) / bandit["counts"][action]
    return bandit

# ============================================
# 5. SESSION MANAGEMENT (backend authority)
# ============================================

def get_or_create_session(user_id, condition):
    with DBSession() as session:
        us = session.query(UserSession).filter_by(user_id=user_id).first()
        if not us:
            task = generate_task()
            us = UserSession(
                user_id=user_id,
                condition=condition,
                current_round=1,
                completed=0,
                task_json=json.dumps(task)
            )
            session.add(us)
            session.commit()
            return {
                "user_id": user_id,
                "condition": condition,
                "current_round": 1,
                "completed": 0,
                "task": task
            }
        return {
            "user_id": us.user_id,
            "condition": us.condition,
            "current_round": us.current_round,
            "completed": us.completed,
            "task": json.loads(us.task_json) if us.task_json else generate_task()
        }


def log_event(user_id, round_number, event_type, metadata=None):
    try:
        with DBSession() as session:
            session.add(EventLog(
                user_id=user_id,
                round_number=round_number,
                event_type=event_type,
                metadata_json=json.dumps(metadata or {})
            ))
            session.commit()
    except Exception:
        pass

# ============================================
# 6. FLASK APP
# ============================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "ddvm_dev_secret")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    """
    Initialize or resume a session.
    Returns current round number and task.
    Frontend calls this once on study begin.
    """
    try:
        user_id = request.form.get("user_id", "").strip()
        condition = request.form.get("condition", "control")

        if not user_id:
            return jsonify({"status": "error", "msg": "Participant ID is required"}), 400

        session_data = get_or_create_session(user_id, condition)
        log_event(user_id, session_data["current_round"], "session_start")

        return jsonify({
            "status": "ok",
            "current_round": session_data["current_round"],
            "total_rounds": TOTAL_ROUNDS,
            "condition": session_data["condition"],
            "task": session_data["task"],
            "completed": session_data["completed"]
        })
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500


@app.route("/submit", methods=["POST"])
def submit():
    """
    Atomic submission endpoint.
    Backend validates round, commits decision,
    increments session, returns next task in single response.
    Frontend never manages round state.
    """
    user_id = request.form.get("user_id", "").strip()

    if not user_id:
        return jsonify({"status": "error", "msg": "Missing user_id"}), 400

    # normalize input — handle locale formats
    raw_val = request.form.get("decision_value", "").strip().replace(",", ".")
    try:
        decision_value = float(raw_val)
        if decision_value < 0 or not np.isfinite(decision_value):
            raise ValueError("out of range")
    except (ValueError, TypeError):
        log_event(user_id, None, "validation_error", {"raw": raw_val})
        return jsonify({"status": "error", "error_type": "invalid_input",
                        "msg": "Please enter a valid positive number"}), 400

    reasoning = request.form.get("reasoning", "")

    try:
        with DBSession() as session:
            # ── Load backend session state (authoritative)
            us = session.query(UserSession).filter_by(user_id=user_id).first()
            if not us:
                return jsonify({"status": "error", "error_type": "no_session",
                                "msg": "Session not found. Please restart."}), 400

            submitted_round = int(request.form.get("round_number", us.current_round))
            expected_round = us.current_round

            # ── Round validation
            if submitted_round != expected_round:
                log_event(user_id, submitted_round, "round_mismatch",
                          {"submitted": submitted_round, "expected": expected_round})
                # return current task so frontend can resync
                task = json.loads(us.task_json) if us.task_json else generate_task()
                return jsonify({
                    "status": "resync",
                    "current_round": expected_round,
                    "task": task,
                    "msg": "Round mismatch corrected"
                })

            # ── Load stored task from backend (not from frontend)
            task = json.loads(us.task_json) if us.task_json else generate_task()
            true_value = task["true_value"]
            anchor = task["anchor"]
            anchor_displacement = task["anchor_displacement"]
            signal = task["signal"]
            signal_noise = task["signal_noise"]

            # ── Compute metrics
            abi = compute_abi(decision_value, true_value, anchor)
            rel_error = compute_relative_error(decision_value, true_value)

            # ── Commit decision atomically
            d = Decision(
                user_id=user_id,
                condition=us.condition,
                round_number=expected_round,
                decision_value=decision_value,
                reasoning=reasoning,
                anchor=anchor,
                anchor_displacement=anchor_displacement,
                signal=signal,
                signal_noise=signal_noise,
                true_value=true_value,
                outcome_value=true_value,
                abi=abi,
                relative_error=rel_error
            )
            session.add(d)

            # ── Determine intervention (treatment only, round >= 5)
            intervention = None
            if us.condition == "treatment" and expected_round >= 5:
                df = get_user_df(user_id)
                if len(df) >= 4:
                    bandit = load_bandit(user_id)
                    action = select_action(bandit, df)
                    recent_errors = df["relative_error"].values
                    reward = float(np.mean(recent_errors[:-3]) - np.mean(recent_errors[-3:])) \
                        if len(recent_errors) > 3 else 0.0
                    bandit = update_bandit(bandit, action, reward)
                    save_bandit(user_id, bandit)
                    session.add(Intervention(
                        user_id=user_id,
                        round_number=expected_round,
                        action=action
                    ))
                    intervention = FEEDBACK_TEXT[action]

            # ── Determine if session complete
            is_last = (expected_round >= TOTAL_ROUNDS)

            if is_last:
                us.current_round = TOTAL_ROUNDS
                us.completed = 1
                us.task_json = None
                next_task = None
            else:
                next_task = generate_task()
                us.current_round = expected_round + 1
                us.task_json = json.dumps(next_task)
                us.updated_at = datetime.utcnow()

            session.commit()
            decision_id = d.id

        log_event(user_id, expected_round, "submission_success")

        # ── Build response with mechanism-aware feedback
        error_magnitude = abs(decision_value - true_value)
        abi_interpretation = (
            "Your estimate was close to the reference figure." if abi > 0.6 else
            "Your estimate moved away from the reference figure." if abi < 0.1 else
            "Your estimate showed moderate reference influence."
        )

        response = {
            "status": "ok",
            "decision_id": decision_id,
            "round_completed": expected_round,
            "true_value": round(true_value, 2),
            "your_estimate": round(decision_value, 2),
            "error": round(error_magnitude, 2),
            "abi": round(abi, 3),
            "relative_error": round(rel_error, 3),
            "abi_interpretation": abi_interpretation,
            "intervention": intervention,
            "session_complete": is_last,
            "next_round": None if is_last else expected_round + 1,
            "next_task": next_task
        }

        return jsonify(response)

    except Exception as e:
        log_event(user_id, None, "submission_failure", {"error": str(e)})
        return jsonify({"status": "error", "error_type": "server_error",
                        "msg": "Something went wrong. Please try again."}), 500


@app.route("/session_metrics/<user_id>", methods=["GET"])
def session_metrics(user_id):
    df = get_user_df(user_id)
    if df.empty:
        return jsonify({"status": "no_data"})
    return jsonify(compute_full_metrics(df))


@app.route("/dashboard")
def dashboard():
    with DBSession() as session:
        rows = session.query(Decision).order_by(Decision.submitted_at.desc()).all()
        decisions = [{
            "id": d.id,
            "user_id": d.user_id,
            "condition": d.condition or "control",
            "round_number": d.round_number,
            "decision_value": d.decision_value,
            "anchor": d.anchor,
            "signal": d.signal,
            "true_value": d.true_value,
            "outcome_value": d.outcome_value,
            "abi": round(d.abi, 3) if d.abi is not None else None,
            "relative_error": round(d.relative_error, 3) if d.relative_error is not None else None,
            "timestamp": d.submitted_at.strftime("%Y-%m-%d %H:%M") if d.submitted_at else ""
        } for d in rows]
        sessions = session.query(UserSession).order_by(UserSession.created_at.desc()).all()
        session_data = [{
            "user_id": s.user_id,
            "condition": s.condition,
            "current_round": s.current_round,
            "completed": s.completed
        } for s in sessions]
    return render_template("dashboard.html", decisions=decisions, sessions=session_data)


@app.route("/api/population_metrics", methods=["GET"])
def api_population_metrics():
    with DBSession() as session:
        decisions = session.query(Decision).filter(
            Decision.outcome_value.isnot(None),
            Decision.abi.isnot(None)
        ).all()
        if not decisions:
            return jsonify({"status": "no_data"})
        all_abi = [d.abi for d in decisions]
        all_rel_err = [d.relative_error for d in decisions if d.relative_error is not None]
        user_ids = list(set(d.user_id for d in decisions))

    pop = {
        "n_users": len(user_ids),
        "n_decisions": len(all_abi),
        "population_mean_abi": round(float(np.mean(all_abi)), 4),
        "population_std_abi": round(float(np.std(all_abi)), 4),
        "population_mean_rel_error": round(float(np.mean(all_rel_err)), 4) if all_rel_err else None
    }
    user_mean_abis = []
    for uid in user_ids:
        df = get_user_df(uid)
        if len(df) >= 5:
            user_mean_abis.append(float(df["abi"].mean()))
    if user_mean_abis:
        pop["individual_abi_variance"] = round(float(np.var(user_mean_abis)), 4)
        pop["individual_abi_range"] = [round(min(user_mean_abis), 4), round(max(user_mean_abis), 4)]
    return jsonify(pop)


# ============================================
# 7. RUN
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)