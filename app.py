from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
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
Session = sessionmaker(bind=engine)
Base = declarative_base()


class Decision(Base):
    __tablename__ = "decisions"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), nullable=False)
    condition = Column(String(20), default="control")
    round_number = Column(Integer)
    decision_value = Column(Float, nullable=False)
    reasoning = Column(Text)
    anchor = Column(Float)
    anchor_displacement = Column(Float)   # anchor - true_value (signed)
    signal = Column(Float)
    signal_noise = Column(Float)          # difficulty tier
    true_value = Column(Float)
    outcome_value = Column(Float)
    # derived metrics stored per round
    abi = Column(Float)                   # Anchoring Bias Index
    relative_error = Column(Float)        # |estimate - true| / true
    timestamp = Column(DateTime, default=datetime.utcnow)


class Intervention(Base):
    __tablename__ = "interventions"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50))
    round_number = Column(Integer)
    action = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)


class BanditState(Base):
    __tablename__ = "bandit_states"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True)
    values_json = Column(Text)
    counts_json = Column(Text)


Base.metadata.create_all(engine)

# ── Schema migration: add new columns if they don't exist (Postgres safe)
def migrate_schema():
    new_columns = [
        ("condition",           "VARCHAR(20) DEFAULT 'control'"),
        ("round_number",        "INTEGER"),
        ("anchor_displacement", "FLOAT"),
        ("signal_noise",        "FLOAT"),
        ("abi",                 "FLOAT"),
        ("relative_error",      "FLOAT"),
    ]
    with engine.connect() as conn:
        for col_name, col_type in new_columns:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE decisions ADD COLUMN IF NOT EXISTS {col_name} {col_type}"
                    )
                )
                conn.commit()
            except Exception:
                conn.rollback()

try:
    migrate_schema()
except Exception:
    pass

# ============================================
# 2. TASK GENERATOR — fully independent
# ============================================

ANCHOR_DISPLACEMENTS = [-20, -12, -6, 6, 12, 20]   # signed offset from truth
SIGNAL_NOISE_TIERS = [3, 6, 10]                      # difficulty


def generate_task():
    # Ground truth: independent of anchor
    true_value = float(np.random.normal(50, 8))
    true_value = max(15.0, true_value)               # floor at 15k

    # Anchor: truth + known displacement
    displacement = float(np.random.choice(ANCHOR_DISPLACEMENTS))
    anchor = true_value + displacement

    # Signal: noisy estimate of truth, independent of anchor
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
# 3. ROUND-LEVEL METRIC COMPUTATION
# ============================================

def compute_abi(estimate, true_value, anchor):
    """
    Anchoring Bias Index.
    0 = estimate at true value (no anchoring)
    1 = estimate at anchor (full anchoring)
    >1 = overshot anchor
    <0 = estimate moved opposite to anchor
    """
    denom = anchor - true_value
    if abs(denom) < 0.01:
        return 0.0
    return float((estimate - true_value) / denom)


def compute_relative_error(estimate, true_value):
    """Scale-invariant error."""
    if abs(true_value) < 0.01:
        return 0.0
    return float(abs(estimate - true_value) / true_value)

# ============================================
# 4. SESSION-LEVEL METRIC SUITE
# ============================================

def get_user_df(user_id):
    with Session() as session:
        decisions = session.query(Decision).filter(
            Decision.user_id == user_id,
            Decision.outcome_value.isnot(None)
        ).order_by(Decision.round_number).all()

        data = [{
            "round": d.round_number,
            "estimate": d.decision_value,
            "true_value": d.true_value,
            "anchor": d.anchor,
            "anchor_displacement": d.anchor_displacement,
            "signal": d.signal,
            "signal_noise": d.signal_noise,
            "abi": d.abi,
            "relative_error": d.relative_error,
            "condition": d.condition
        } for d in decisions]

    return pd.DataFrame(data)


def compute_full_metrics(df):
    """
    Compute all 7 session-level metrics.
    Requires df with columns: estimate, true_value, anchor,
    anchor_displacement, signal, abi, relative_error, round
    """
    if len(df) < 5:
        return {"status": "insufficient_data", "n": len(df)}

    df = df.copy().sort_values("round").reset_index(drop=True)
    n = len(df)

    results = {"n": n}

    # ── METRIC 1: Mean ABI ──────────────────────────────────────
    # Primary anchoring measure. Scale-invariant, sign-preserving.
    abi_vals = df["abi"].values
    results["mean_abi"] = round(float(np.mean(abi_vals)), 4)
    results["std_abi"] = round(float(np.std(abi_vals)), 4)

    # ── METRIC 2: Trajectory Lock-In (TLI) ─────────────────────
    # Whether bias pattern stabilizes across rounds.
    # Requires at least 10 rounds.
    if n >= 10:
        half = n // 2
        early_std = float(np.std(abi_vals[:half]))
        late_std = float(np.std(abi_vals[half:]))
        if early_std > 0.01:
            tli = float(1 - (late_std / early_std))
        else:
            tli = 0.0
        results["tli"] = round(tli, 4)
        results["early_abi_std"] = round(early_std, 4)
        results["late_abi_std"] = round(late_std, 4)
    else:
        results["tli"] = None

    # ── METRIC 3: Bias Elasticity (BE) ─────────────────────────
    # Regression slope of ABI on |anchor_displacement|.
    # High positive = bias gets stronger as anchor gets more extreme.
    displacements = np.abs(df["anchor_displacement"].values)
    if displacements.std() > 0.01 and n >= 8:
        slope, intercept, r, p, se = stats.linregress(displacements, abi_vals)
        results["be_slope"] = round(float(slope), 4)
        results["be_r"] = round(float(r), 4)
        results["be_p"] = round(float(p), 4)
    else:
        results["be_slope"] = None
        results["be_r"] = None
        results["be_p"] = None

    # ── METRIC 4: Signal Independence Index (SII) ──────────────
    # How much participant ignores signal vs follows it.
    # 1 = fully independent, 0 = follows signal exactly.
    signal_errors = (df["signal"] - df["true_value"]).values
    estimate_errors = (df["estimate"] - df["true_value"]).values
    if np.std(signal_errors) > 0.01 and np.std(estimate_errors) > 0.01:
        r_signal = float(np.corrcoef(estimate_errors, signal_errors)[0, 1])
        sii = float(1 - abs(r_signal))
    else:
        sii = 1.0
        r_signal = 0.0
    results["sii"] = round(sii, 4)
    results["signal_correlation"] = round(r_signal, 4)

    # ── METRIC 5: Calibration Drift (CD) ───────────────────────
    # Regression slope of relative_error on round number.
    # Negative = improving, positive = deteriorating.
    rounds = df["round"].values.astype(float)
    rel_errors = df["relative_error"].values
    if n >= 8 and np.std(rounds) > 0:
        slope, _, r, p, _ = stats.linregress(rounds, rel_errors)
        results["cd_slope"] = round(float(slope), 6)
        results["cd_r"] = round(float(r), 4)
        results["cd_p"] = round(float(p), 4)
        results["mean_relative_error"] = round(float(np.mean(rel_errors)), 4)
    else:
        results["cd_slope"] = None
        results["cd_r"] = None
        results["cd_p"] = None
        results["mean_relative_error"] = round(float(np.mean(rel_errors)), 4)

    # ── METRIC 6: Anchor Displacement Sensitivity (ADS) ────────
    # Do participants push back harder against extreme anchors?
    # High ADS = partial correction for displacement magnitude.
    adjustment = np.abs(df["estimate"].values - df["anchor"].values)
    displacements_signed = np.abs(df["anchor_displacement"].values)
    if np.std(displacements_signed) > 0.01 and n >= 8:
        r_ads = float(np.corrcoef(adjustment, displacements_signed)[0, 1])
        results["ads"] = round(r_ads, 4)
    else:
        results["ads"] = None

    # ── METRIC 7: Cognitive State Sequence (CSS) ───────────────
    # HMM fit on full ABI sequence. States assigned by mean ABI.
    # under-anchored (<0.2), moderate (0.2-0.6), over-anchored (>0.6)
    if n >= 15:
        try:
            X = abi_vals.reshape(-1, 1)
            model = GaussianHMM(n_components=3, n_iter=200,
                                random_state=42, covariance_type="full")
            model.fit(X)
            states = model.predict(X)

            # assign labels by emission mean
            means = {s: float(model.means_[s][0]) for s in range(3)}
            sorted_states = sorted(means, key=means.get)
            label_map = {
                sorted_states[0]: "under_anchored",
                sorted_states[1]: "moderate",
                sorted_states[2]: "over_anchored"
            }
            state_seq = [label_map[s] for s in states]

            # TLI via HMM: does terminal state match initial dominant state?
            from collections import Counter
            initial_dominant = Counter(state_seq[:5]).most_common(1)[0][0]
            terminal_dominant = Counter(state_seq[-5:]).most_common(1)[0][0]
            tli_hmm = 1.0 if initial_dominant == terminal_dominant else 0.0

            results["css_terminal_state"] = state_seq[-1]
            results["css_tli_hmm"] = tli_hmm
            results["css_state_sequence"] = state_seq
            results["css_transition_count"] = sum(
                1 for i in range(1, len(state_seq))
                if state_seq[i] != state_seq[i-1]
            )
        except Exception:
            results["css_terminal_state"] = "unknown"
            results["css_tli_hmm"] = None
            results["css_state_sequence"] = []
            results["css_transition_count"] = None
    else:
        results["css_terminal_state"] = "insufficient_data"
        results["css_tli_hmm"] = None
        results["css_state_sequence"] = []
        results["css_transition_count"] = None

    # ── Per-round ABI for frontend visualization ────────────────
    results["round_abi"] = [round(float(v), 3) for v in abi_vals]
    results["round_relative_error"] = [round(float(v), 3)
                                        for v in rel_errors]
    results["rounds"] = [int(r) for r in df["round"].values]

    return results

# ============================================
# 5. PER-USER BANDIT
# ============================================

ACTIONS = ["debias", "slow", "reanchor", "ignore_signal"]

FEEDBACK_TEXT = {
    "debias": "Your estimates are tracking the reference figure closely. Try forming your prediction before looking at the reference.",
    "slow": "Consider whether your first instinct is being influenced by the figures shown. Take a moment before committing.",
    "reanchor": "The reference figure may be misleading. Focus on what you know independently of the value shown.",
    "ignore_signal": "The market forecast has been inaccurate in recent rounds. Weight it accordingly."
}


def load_bandit(user_id):
    with Session() as session:
        state = session.query(BanditState).filter_by(user_id=user_id).first()
        if state:
            return {
                "values": json.loads(state.values_json),
                "counts": json.loads(state.counts_json)
            }
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


def select_action(bandit, df):
    """
    Context-aware action selection.
    Uses recent ABI and SII to bias exploration toward relevant actions.
    Falls back to epsilon-greedy.
    """
    if len(df) >= 5:
        recent_abi = df["abi"].iloc[-5:].mean()
        recent_sii = None

        # prefer debias/reanchor for high ABI
        if recent_abi > 0.5 and random.random() < 0.6:
            return random.choice(["debias", "reanchor"])
        # prefer ignore_signal if signal is being followed too closely
        # (approximated by low SII estimate from recent data)
        if recent_abi < 0.2 and random.random() < 0.4:
            return "ignore_signal"

    # epsilon-greedy fallback
    if random.random() < 0.25:
        return random.choice(ACTIONS)
    return max(bandit["values"], key=bandit["values"].get)


def update_bandit(bandit, action, reward):
    bandit["counts"][action] += 1
    bandit["values"][action] += (
        (reward - bandit["values"][action]) / bandit["counts"][action]
    )
    return bandit

# ============================================
# 6. FLASK APP
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
        condition = request.form.get("condition", "control")
        round_number = int(request.form.get("round_number", 1))
        decision_value = float(request.form.get("decision_value"))
        reasoning = request.form.get("reasoning", "")
        anchor = float(request.form.get("anchor", 50))
        anchor_displacement = float(request.form.get("anchor_displacement", 0))
        signal = float(request.form.get("signal", 50))
        signal_noise = float(request.form.get("signal_noise", 6))
        true_value = float(request.form.get("true_value", 50))

        # compute round-level metrics
        abi = compute_abi(decision_value, true_value, anchor)
        rel_error = compute_relative_error(decision_value, true_value)

        with Session() as session:
            d = Decision(
                user_id=user_id,
                condition=condition,
                round_number=round_number,
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
            session.commit()
            decision_id = d.id

        # intervention — treatment condition only, after round 5
        intervention = None
        if condition == "treatment" and round_number >= 5:
            df = get_user_df(user_id)
            if len(df) >= 5:
                bandit = load_bandit(user_id)
                action = select_action(bandit, df)

                # reward = reduction in relative error over last 3 rounds
                recent_errors = df["relative_error"].values
                reward = (float(np.mean(recent_errors[:-3])) -
                          float(np.mean(recent_errors[-3:]))) \
                    if len(recent_errors) > 3 else 0.0

                bandit = update_bandit(bandit, action, reward)
                save_bandit(user_id, bandit)

                with Session() as session:
                    session.add(Intervention(
                        user_id=user_id,
                        round_number=round_number,
                        action=action
                    ))
                    session.commit()

                intervention = FEEDBACK_TEXT[action]

        return jsonify({
            "status": "ok",
            "decision_id": decision_id,
            "true_value": round(true_value, 2),
            "your_estimate": round(decision_value, 2),
            "abi": round(abi, 3),
            "relative_error": round(rel_error, 3),
            "intervention": intervention
        })

    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 400


@app.route("/session_metrics/<user_id>", methods=["GET"])
def session_metrics(user_id):
    """Full metric suite for a completed session."""
    df = get_user_df(user_id)
    if df.empty:
        return jsonify({"status": "no_data"})
    metrics = compute_full_metrics(df)
    return jsonify(metrics)


@app.route("/dashboard")
def dashboard():
    with Session() as session:
        rows = session.query(Decision).order_by(Decision.timestamp.desc()).all()
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
            "timestamp": d.timestamp.strftime("%Y-%m-%d %H:%M") if d.timestamp else ""
        } for d in rows]
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


@app.route("/api/population_metrics", methods=["GET"])
def api_population_metrics():
    """
    Population-level metric aggregates across all users.
    Used for claim 3 validation — stable aggregates vs individual variance.
    """
    with Session() as session:
        decisions = session.query(Decision).filter(
            Decision.outcome_value.isnot(None)
        ).all()

        if not decisions:
            return jsonify({"status": "no_data"})

        all_abi = [d.abi for d in decisions if d.abi is not None]
        all_rel_err = [d.relative_error for d in decisions
                       if d.relative_error is not None]
        user_ids = list(set(d.user_id for d in decisions))

    pop_metrics = {
        "n_users": len(user_ids),
        "n_decisions": len(all_abi),
        "population_mean_abi": round(float(np.mean(all_abi)), 4),
        "population_std_abi": round(float(np.std(all_abi)), 4),
        "population_mean_rel_error": round(float(np.mean(all_rel_err)), 4),
        "population_std_rel_error": round(float(np.std(all_rel_err)), 4)
    }

    # per-user mean ABI to show individual variance
    user_mean_abis = []
    for uid in user_ids:
        df = get_user_df(uid)
        if len(df) >= 5:
            user_mean_abis.append(float(df["abi"].mean()))

    if user_mean_abis:
        pop_metrics["individual_abi_variance"] = round(
            float(np.var(user_mean_abis)), 4)
        pop_metrics["individual_abi_range"] = [
            round(min(user_mean_abis), 4),
            round(max(user_mean_abis), 4)
        ]

    return jsonify(pop_metrics)


# ============================================
# 7. RUN
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)