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

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
DBSession = sessionmaker(bind=engine)
Base = declarative_base()

TOTAL_ROUNDS = 20


class UserSession(Base):
    __tablename__ = "user_sessions"
    id           = Column(Integer, primary_key=True)
    user_id      = Column(String(50), unique=True, nullable=False)
    condition    = Column(String(20), default="control")
    current_round = Column(Integer, default=1)
    completed    = Column(Integer, default=0)
    task_json    = Column(Text)
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow)


class Decision(Base):
    __tablename__ = "decisions"
    id                 = Column(Integer, primary_key=True)
    user_id            = Column(String(50), nullable=False)
    condition          = Column(String(20), default="control")
    round_number       = Column(Integer)
    decision_value     = Column(Float, nullable=False)
    reasoning          = Column(Text)
    anchor             = Column(Float)
    anchor_displacement = Column(Float)
    signal             = Column(Float)
    signal_noise       = Column(Float)
    true_value         = Column(Float)
    outcome_value      = Column(Float)
    abi                = Column(Float)
    relative_error     = Column(Float)
    submitted_at       = Column(DateTime, default=datetime.utcnow)


class Intervention(Base):
    __tablename__ = "interventions"
    id           = Column(Integer, primary_key=True)
    user_id      = Column(String(50))
    round_number = Column(Integer)
    action       = Column(String(50))
    created_at   = Column(DateTime, default=datetime.utcnow)


class BanditState(Base):
    __tablename__ = "bandit_states"
    id          = Column(Integer, primary_key=True)
    user_id     = Column(String(50), unique=True)
    values_json = Column(Text)
    counts_json = Column(Text)


class EventLog(Base):
    __tablename__ = "event_log"
    id            = Column(Integer, primary_key=True)
    user_id       = Column(String(50))
    round_number  = Column(Integer)
    event_type    = Column(String(50))
    metadata_json = Column(Text)
    created_at    = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)


def migrate_schema():
    import sqlalchemy as sa
    cols = [
        ("decisions",     "condition",            "VARCHAR(20) DEFAULT 'control'"),
        ("decisions",     "round_number",         "INTEGER"),
        ("decisions",     "anchor_displacement",  "FLOAT"),
        ("decisions",     "signal_noise",         "FLOAT"),
        ("decisions",     "abi",                  "FLOAT"),
        ("decisions",     "relative_error",       "FLOAT"),
        ("decisions",     "submitted_at",         "TIMESTAMP"),
    ]
    with engine.connect() as conn:
        for table, col, typ in cols:
            try:
                conn.execute(sa.text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {typ}"))
                conn.commit()
            except Exception:
                try: conn.rollback()
                except Exception: pass

try:
    migrate_schema()
except Exception:
    pass

# ============================================
# 2. TASK GENERATOR
# ============================================

ANCHOR_DISPLACEMENTS = [-20, -12, -6, 6, 12, 20]
SIGNAL_NOISE_TIERS   = [3, 6, 10]


def generate_task():
    true_value   = float(max(15.0, np.random.normal(50, 8)))
    displacement = float(np.random.choice(ANCHOR_DISPLACEMENTS))
    anchor       = true_value + displacement
    noise_tier   = float(np.random.choice(SIGNAL_NOISE_TIERS))
    signal       = float(true_value + np.random.normal(0, noise_tier))
    return {
        "true_value":          round(true_value, 3),
        "anchor":              round(anchor, 2),
        "anchor_displacement": round(displacement, 2),
        "signal":              round(signal, 2),
        "signal_noise":        round(noise_tier, 2)
    }

# ============================================
# 3. CORE METRICS  (DO NOT MODIFY)
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
    """Load completed decisions for a user into a DataFrame."""
    with DBSession() as s:
        rows = s.query(Decision).filter(
            Decision.user_id == user_id,
            Decision.outcome_value.isnot(None)
        ).order_by(Decision.round_number).all()
        data = [{
            "round":              r.round_number or 0,
            "estimate":          r.decision_value,
            "true_value":        r.true_value or 0,
            "anchor":            r.anchor or 50,
            "anchor_displacement": r.anchor_displacement or 0,
            "signal":            r.signal or 50,
            "signal_noise":      r.signal_noise or 6,
            "abi":               r.abi or 0,
            "relative_error":    r.relative_error or 0,
            "condition":         r.condition or "control"
        } for r in rows]
    return pd.DataFrame(data)


def compute_full_metrics(df):
    if len(df) < 5:
        return {"status": "insufficient_data", "n": len(df)}

    df  = df.copy().sort_values("round").reset_index(drop=True)
    n   = len(df)
    res = {"n": n}

    abi_vals = df["abi"].fillna(0).values

    res["mean_abi"] = round(float(np.mean(abi_vals)), 4)
    res["std_abi"]  = round(float(np.std(abi_vals)),  4)

    if n >= 10:
        half      = n // 2
        early_std = float(np.std(abi_vals[:half]))
        late_std  = float(np.std(abi_vals[half:]))
        res["tli"] = round(float(1 - (late_std / early_std)) if early_std > 0.01 else 0.0, 4)
    else:
        res["tli"] = None

    disps = np.abs(df["anchor_displacement"].fillna(0).values)
    if disps.std() > 0.01 and n >= 8:
        slope, _, r, p, _ = stats.linregress(disps, abi_vals)
        res["be_slope"] = round(float(slope), 4)
        res["be_r"]     = round(float(r), 4)
        res["be_p"]     = round(float(p), 4)
    else:
        res["be_slope"] = res["be_r"] = res["be_p"] = None

    sig_err = (df["signal"]   - df["true_value"]).fillna(0).values
    est_err = (df["estimate"] - df["true_value"]).fillna(0).values
    if np.std(sig_err) > 0.01 and np.std(est_err) > 0.01:
        r_sig       = float(np.corrcoef(est_err, sig_err)[0, 1])
        res["sii"]  = round(float(1 - abs(r_sig)), 4)
        res["signal_correlation"] = round(r_sig, 4)
    else:
        res["sii"] = 1.0
        res["signal_correlation"] = 0.0

    rounds     = df["round"].values.astype(float)
    rel_errors = df["relative_error"].fillna(0).values
    if n >= 8 and np.std(rounds) > 0:
        slope, _, r, p, _ = stats.linregress(rounds, rel_errors)
        res["cd_slope"] = round(float(slope), 6)
        res["cd_r"]     = round(float(r), 4)
        res["cd_p"]     = round(float(p), 4)
    else:
        res["cd_slope"] = None

    res["mean_relative_error"] = round(float(np.mean(rel_errors)), 4)

    adjustments = np.abs(df["estimate"].values - df["anchor"].fillna(50).values)
    if np.std(disps) > 0.01 and np.std(adjustments) > 0.01:
        res["ads"] = round(float(np.corrcoef(adjustments, disps)[0, 1]), 4)
    else:
        res["ads"] = None

    if n >= 15:
        try:
            X     = abi_vals.reshape(-1, 1)
            model = GaussianHMM(n_components=3, n_iter=200, random_state=42, covariance_type="full")
            model.fit(X)
            states       = model.predict(X)
            means        = {s: float(model.means_[s][0]) for s in range(3)}
            sorted_s     = sorted(means, key=means.get)
            lmap         = {sorted_s[0]: "under_anchored", sorted_s[1]: "moderate", sorted_s[2]: "over_anchored"}
            state_seq    = [lmap[s] for s in states]
            res["css_terminal_state"]  = state_seq[-1]
            res["css_transition_count"] = sum(1 for i in range(1, len(state_seq)) if state_seq[i] != state_seq[i-1])
            res["css_state_sequence"]  = state_seq
        except Exception:
            res["css_terminal_state"]  = "unknown"
            res["css_transition_count"] = None
            res["css_state_sequence"]  = []
    else:
        res["css_terminal_state"]  = "insufficient_data"
        res["css_transition_count"] = None
        res["css_state_sequence"]  = []

    res["round_abi"]            = [round(float(v), 3) for v in abi_vals]
    res["round_relative_error"] = [round(float(v), 3) for v in rel_errors]
    res["rounds"]               = [int(r) for r in df["round"].values]

    return res

# ============================================
# 4. ROBUSTNESS SUITE  (separate layer — never touches core metrics)
# ============================================

def perturb_and_recompute(df, noise_scale=0.5, n_runs=50):
    """
    Sensitivity analysis via perturbation testing.
    Adds Gaussian noise to anchor and signal, recomputes metrics n_runs times.
    Returns std of key metrics across runs.
    """
    if len(df) < 5:
        return {"status": "insufficient_data"}

    abi_samples  = []
    sii_samples  = []
    tli_samples  = []

    for _ in range(n_runs):
        df_p = df.copy()
        df_p["anchor"] = df_p["anchor"] + np.random.normal(0, noise_scale, len(df_p))
        df_p["signal"] = df_p["signal"] + np.random.normal(0, noise_scale, len(df_p))

        # Recompute ABI per row with perturbed anchor
        df_p["abi"] = [
            compute_abi(row["estimate"], row["true_value"], row["anchor"])
            for _, row in df_p.iterrows()
        ]
        df_p["anchor_displacement"] = df_p["anchor"] - df_p["true_value"]

        m = compute_full_metrics(df_p)
        if m.get("status") == "insufficient_data":
            continue
        abi_samples.append(m.get("mean_abi", 0))
        sii_samples.append(m.get("sii", 1))
        if m.get("tli") is not None:
            tli_samples.append(m["tli"])

    if not abi_samples:
        return {"status": "insufficient_data"}

    abi_std = float(np.std(abi_samples))
    result = {
        "abi_stability_std": round(abi_std, 4),
        "sii_stability_std": round(float(np.std(sii_samples)), 4) if sii_samples else None,
        "tli_stability_std": round(float(np.std(tli_samples)), 4) if tli_samples else None,
        "n_runs":            n_runs,
        "noise_scale":       noise_scale,
        # Stability label: std < 0.05 = high, < 0.15 = moderate, else low
        "abi_stability_label": "high" if abi_std < 0.05 else ("moderate" if abi_std < 0.15 else "low")
    }
    return result


def compute_split_half(df):
    """
    Split-half reliability.
    Correlates ABI sequences from first and second halves of session.
    """
    if len(df) < 10:
        return {"status": "insufficient_data"}

    df   = df.copy().sort_values("round").reset_index(drop=True)
    half = len(df) // 2
    first_abi  = df["abi"].values[:half]
    second_abi = df["abi"].values[half: half * 2]

    min_len = min(len(first_abi), len(second_abi))
    if min_len < 3:
        return {"status": "insufficient_data"}

    first_abi  = first_abi[:min_len]
    second_abi = second_abi[:min_len]

    if np.std(first_abi) < 0.01 or np.std(second_abi) < 0.01:
        corr = 0.0
    else:
        corr = float(np.corrcoef(first_abi, second_abi)[0, 1])

    # Spearman-Brown corrected reliability
    sb_corr = (2 * corr) / (1 + corr) if (1 + corr) > 0.01 else 0.0

    return {
        "abi_split_half_corr":    round(corr, 4),
        "spearman_brown_corr":    round(sb_corr, 4),
        # Reliability label
        "reliability_label": "high" if abs(sb_corr) > 0.7 else ("moderate" if abs(sb_corr) > 0.4 else "low")
    }


def compute_internal_consistency(df):
    """
    Internal consistency checks.
    Tests expected directional relationships between metrics.
    """
    if len(df) < 8:
        return {"status": "insufficient_data"}

    df = df.copy()
    abi_vals     = df["abi"].fillna(0).values
    rel_err_vals = df["relative_error"].fillna(0).values
    adjustments  = np.abs(df["estimate"].values - df["anchor"].fillna(50).values)

    sig_err = (df["signal"]   - df["true_value"]).fillna(0).values
    est_err = (df["estimate"] - df["true_value"]).fillna(0).values

    def safe_corr(a, b):
        if np.std(a) < 0.01 or np.std(b) < 0.01:
            return None
        return round(float(np.corrcoef(a, b)[0, 1]), 4)

    sii_vals = np.array([1 - abs(float(np.corrcoef(
        est_err[:i+1], sig_err[:i+1])[0, 1]))
        if i >= 2 and np.std(est_err[:i+1]) > 0.01 and np.std(sig_err[:i+1]) > 0.01
        else np.nan for i in range(len(df))])
    sii_vals = sii_vals[~np.isnan(sii_vals)]

    result = {
        # ABI vs relative_error: expect positive (more anchoring → more error)
        "abi_relerr_corr": safe_corr(abi_vals, rel_err_vals),
        # ABI vs adjustments: expect negative (more anchoring → less adjustment away)
        "abi_adj_corr":    safe_corr(abi_vals, adjustments),
    }

    if len(sii_vals) >= 5:
        result["abi_sii_corr"] = safe_corr(abi_vals[:len(sii_vals)], sii_vals)
    else:
        result["abi_sii_corr"] = None

    # Coherence flags
    result["coherent_abi_relerr"] = (
        result["abi_relerr_corr"] is not None and result["abi_relerr_corr"] > 0
    )
    result["coherent_abi_adj"] = (
        result["abi_adj_corr"] is not None and result["abi_adj_corr"] < 0
    )
    result["n_coherent"] = sum([result["coherent_abi_relerr"], result["coherent_abi_adj"]])
    result["consistency_label"] = (
        "high" if result["n_coherent"] == 2 else
        "moderate" if result["n_coherent"] == 1 else "low"
    )

    return result


def compare_to_random(df):
    """
    Null model benchmarking.
    Compares real user metrics to a random agent.
    Random agent generates estimates uniformly around the mean anchor.
    """
    if len(df) < 5:
        return {"status": "insufficient_data"}

    df = df.copy()

    # Build random agent dataframe
    df_random = df.copy()
    anchor_mean = df["anchor"].mean()
    anchor_std  = df["anchor"].std() if df["anchor"].std() > 0 else 10
    df_random["estimate"] = np.random.uniform(
        anchor_mean - 2 * anchor_std,
        anchor_mean + 2 * anchor_std,
        len(df)
    )
    df_random["abi"] = [
        compute_abi(row["estimate"], row["true_value"], row["anchor"])
        for _, row in df_random.iterrows()
    ]
    df_random["relative_error"] = [
        compute_relative_error(row["estimate"], row["true_value"])
        for _, row in df_random.iterrows()
    ]

    real_metrics   = compute_full_metrics(df)
    random_metrics = compute_full_metrics(df_random)

    if real_metrics.get("status") == "insufficient_data":
        return {"status": "insufficient_data"}

    real_abi   = real_metrics.get("mean_abi", 0)
    random_abi = random_metrics.get("mean_abi", 0)
    separation = abs(real_abi - random_abi)

    return {
        "real_mean_abi":   round(real_abi, 4),
        "random_mean_abi": round(random_abi, 4),
        "separation":      round(separation, 4),
        "real_mean_rel_error":   round(real_metrics.get("mean_relative_error", 0), 4),
        "random_mean_rel_error": round(random_metrics.get("mean_relative_error", 0), 4),
        # Meaningful if separation > 0.1
        "meaningful_label": "yes" if separation > 0.1 else ("borderline" if separation > 0.05 else "no")
    }


def compute_robustness_suite(df):
    """
    Master robustness function.
    Runs all four validation components.
    Returns structured output with confidence signals for each metric.
    """
    sensitivity  = perturb_and_recompute(df, noise_scale=0.5, n_runs=50)
    split_half   = compute_split_half(df)
    consistency  = compute_internal_consistency(df)
    null_model   = compare_to_random(df)

    # Derive confidence labels for frontend display
    abi_stability   = sensitivity.get("abi_stability_label", "unknown")
    reliability     = split_half.get("reliability_label", "unknown")
    consistency_lbl = consistency.get("consistency_label", "unknown")
    meaningful      = null_model.get("meaningful_label", "unknown")

    # Overall confidence: all four must be high/yes for high confidence
    high_count = sum([
        abi_stability   == "high",
        reliability     == "high",
        consistency_lbl == "high",
        meaningful      == "yes"
    ])
    overall = "high" if high_count >= 3 else ("moderate" if high_count >= 1 else "low")

    return {
        "sensitivity":         sensitivity,
        "split_half":          split_half,
        "internal_consistency": consistency,
        "null_model":          null_model,
        "confidence_summary": {
            "abi_stability":    abi_stability,
            "reliability":      reliability,
            "consistency":      consistency_lbl,
            "meaningful":       meaningful,
            "overall":          overall
        }
    }

# ============================================
# 5. BANDIT
# ============================================

ACTIONS = ["debias", "slow", "reanchor", "ignore_signal"]

FEEDBACK_TEXT = {
    "debias":        "Your estimates are tracking the reference figure closely. Try forming your prediction before looking at the reference.",
    "slow":          "Consider whether your first instinct is being influenced by the figures shown. Take a moment before committing.",
    "reanchor":      "The reference figure may be misleading. Focus on what you know independently of the value shown.",
    "ignore_signal": "The market forecast has been inaccurate in recent rounds. Weight it accordingly."
}


def load_bandit(user_id):
    with DBSession() as s:
        state = s.query(BanditState).filter_by(user_id=user_id).first()
        if state:
            return {"values": json.loads(state.values_json), "counts": json.loads(state.counts_json)}
    return {"values": {a: 0.0 for a in ACTIONS}, "counts": {a: 1 for a in ACTIONS}}


def save_bandit(user_id, bandit):
    with DBSession() as s:
        state = s.query(BanditState).filter_by(user_id=user_id).first()
        if state:
            state.values_json = json.dumps(bandit["values"])
            state.counts_json = json.dumps(bandit["counts"])
        else:
            s.add(BanditState(user_id=user_id,
                              values_json=json.dumps(bandit["values"]),
                              counts_json=json.dumps(bandit["counts"])))
        s.commit()


def select_action(bandit, recent_abi_mean):
    if recent_abi_mean > 0.5 and random.random() < 0.6:
        return random.choice(["debias", "reanchor"])
    if recent_abi_mean < 0.2 and random.random() < 0.4:
        return "ignore_signal"
    if random.random() < 0.25:
        return random.choice(ACTIONS)
    return max(bandit["values"], key=bandit["values"].get)


def update_bandit(bandit, action, reward):
    bandit["counts"][action] += 1
    bandit["values"][action] += (reward - bandit["values"][action]) / bandit["counts"][action]
    return bandit

# ============================================
# 6. SESSION MANAGEMENT
# ============================================

def get_or_create_session(user_id, condition):
    with DBSession() as s:
        us = s.query(UserSession).filter_by(user_id=user_id).first()
        if not us:
            task = generate_task()
            us = UserSession(user_id=user_id, condition=condition,
                             current_round=1, completed=0,
                             task_json=json.dumps(task))
            s.add(us)
            s.commit()
            return {"user_id": user_id, "condition": condition,
                    "current_round": 1, "completed": 0, "task": task}
        return {"user_id": us.user_id, "condition": us.condition,
                "current_round": us.current_round, "completed": us.completed,
                "task": json.loads(us.task_json) if us.task_json else generate_task()}


def log_event(user_id, round_number, event_type, metadata=None):
    try:
        with DBSession() as s:
            s.add(EventLog(user_id=user_id, round_number=round_number,
                           event_type=event_type,
                           metadata_json=json.dumps(metadata or {})))
            s.commit()
    except Exception:
        pass

# ============================================
# 7. FLASK APP
# ============================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "changeme-set-in-env")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start():
    try:
        user_id   = request.form.get("user_id", "").strip()
        condition = request.form.get("condition", "control")
        if not user_id:
            return jsonify({"status": "error", "msg": "Participant ID required"}), 400
        sd = get_or_create_session(user_id, condition)
        log_event(user_id, sd["current_round"], "session_start")
        return jsonify({"status": "ok", "current_round": sd["current_round"],
                        "total_rounds": TOTAL_ROUNDS, "condition": sd["condition"],
                        "task": sd["task"], "completed": sd["completed"]})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500


@app.route("/submit", methods=["POST"])
def submit():
    user_id = request.form.get("user_id", "").strip()
    if not user_id:
        return jsonify({"status": "error", "msg": "Missing user_id"}), 400

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
        # ── Single session block — no nested sessions
        with DBSession() as s:
            us = s.query(UserSession).filter_by(user_id=user_id).first()
            if not us:
                return jsonify({"status": "error", "error_type": "no_session",
                                "msg": "Session not found. Please restart."}), 400

            submitted_round = int(request.form.get("round_number", us.current_round))
            expected_round  = us.current_round

            if submitted_round != expected_round:
                log_event(user_id, submitted_round, "round_mismatch",
                          {"submitted": submitted_round, "expected": expected_round})
                task = json.loads(us.task_json) if us.task_json else generate_task()
                return jsonify({"status": "resync", "current_round": expected_round, "task": task})

            # Load task from backend — never trust frontend values
            task             = json.loads(us.task_json) if us.task_json else generate_task()
            true_value       = task["true_value"]
            anchor           = task["anchor"]
            anchor_disp      = task["anchor_displacement"]
            signal           = task["signal"]
            signal_noise     = task["signal_noise"]

            abi       = compute_abi(decision_value, true_value, anchor)
            rel_error = compute_relative_error(decision_value, true_value)

            d = Decision(
                user_id=user_id, condition=us.condition,
                round_number=expected_round, decision_value=decision_value,
                reasoning=reasoning, anchor=anchor,
                anchor_displacement=anchor_disp, signal=signal,
                signal_noise=signal_noise, true_value=true_value,
                outcome_value=true_value, abi=abi, relative_error=rel_error
            )
            s.add(d)

            # ── Bandit intervention (uses already-committed rows only)
            intervention    = None
            recent_abi_mean = 0.0
            if us.condition == "treatment" and expected_round >= 5:
                # Query previous decisions inside same session — avoids nested DBSession
                prev = s.query(Decision).filter(
                    Decision.user_id == user_id,
                    Decision.abi.isnot(None)
                ).order_by(Decision.round_number.desc()).limit(5).all()
                if len(prev) >= 4:
                    recent_abi_mean = float(np.mean([p.abi for p in prev]))
                    recent_errors   = [p.relative_error for p in prev if p.relative_error is not None]
                    bandit          = load_bandit(user_id)
                    action          = select_action(bandit, recent_abi_mean)
                    reward          = float(np.mean(recent_errors[:-3]) - np.mean(recent_errors[-3:])) \
                        if len(recent_errors) > 3 else 0.0
                    bandit          = update_bandit(bandit, action, reward)
                    save_bandit(user_id, bandit)
                    s.add(Intervention(user_id=user_id, round_number=expected_round, action=action))
                    intervention = FEEDBACK_TEXT[action]

            is_last  = (expected_round >= TOTAL_ROUNDS)
            next_task = None
            if is_last:
                us.current_round = TOTAL_ROUNDS
                us.completed     = 1
                us.task_json     = None
            else:
                next_task            = generate_task()
                us.current_round     = expected_round + 1
                us.task_json         = json.dumps(next_task)
                us.updated_at        = datetime.utcnow()

            s.commit()
            decision_id = d.id

        log_event(user_id, expected_round, "submission_success")

        abi_interp = (
            "Your estimate was close to the reference figure — possible anchoring effect." if abi > 0.6 else
            "Your estimate moved away from the reference figure." if abi < 0.1 else
            "Your estimate showed moderate reference influence."
        )

        return jsonify({
            "status":           "ok",
            "decision_id":      decision_id,
            "round_completed":  expected_round,
            "true_value":       round(true_value, 2),
            "your_estimate":    round(decision_value, 2),
            "error":            round(abs(decision_value - true_value), 2),
            "abi":              round(abi, 3),
            "relative_error":   round(rel_error, 3),
            "abi_interpretation": abi_interp,
            "intervention":     intervention,
            "session_complete": is_last,
            "next_round":       None if is_last else expected_round + 1,
            "next_task":        next_task
        })

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


@app.route("/robustness/<user_id>", methods=["GET"])
def robustness(user_id):
    """On-demand robustness suite — never called during task flow."""
    df = get_user_df(user_id)
    if df.empty or len(df) < 5:
        return jsonify({"status": "insufficient_data", "n": len(df)})
    result = compute_robustness_suite(df)
    log_event(user_id, None, "robustness_computed", {"n": len(df)})
    return jsonify(result)


@app.route("/dashboard")
def dashboard():
    with DBSession() as s:
        rows = s.query(Decision).order_by(Decision.submitted_at.desc()).all()
        decisions = [{
            "id":             d.id,
            "user_id":        d.user_id,
            "condition":      d.condition or "control",
            "round_number":   d.round_number,
            "decision_value": d.decision_value,
            "anchor":         d.anchor,
            "signal":         d.signal,
            "true_value":     d.true_value,
            "outcome_value":  d.outcome_value,
            "abi":            round(d.abi, 3) if d.abi is not None else None,
            "relative_error": round(d.relative_error, 3) if d.relative_error is not None else None,
            "timestamp":      d.submitted_at.strftime("%Y-%m-%d %H:%M") if d.submitted_at else ""
        } for d in rows]
        sessions = s.query(UserSession).order_by(UserSession.created_at.desc()).all()
        session_data = [{
            "user_id":       ss.user_id,
            "condition":     ss.condition,
            "current_round": ss.current_round,
            "completed":     ss.completed
        } for ss in sessions]
    return render_template("dashboard.html", decisions=decisions, sessions=session_data)


@app.route("/api/population_metrics", methods=["GET"])
def api_population_metrics():
    with DBSession() as s:
        rows = s.query(Decision).filter(
            Decision.outcome_value.isnot(None),
            Decision.abi.isnot(None)
        ).all()
        if not rows:
            return jsonify({"status": "no_data"})
        all_abi     = [r.abi for r in rows]
        all_rel_err = [r.relative_error for r in rows if r.relative_error is not None]
        user_ids    = list(set(r.user_id for r in rows))

    pop = {
        "n_users":                len(user_ids),
        "n_decisions":            len(all_abi),
        "population_mean_abi":    round(float(np.mean(all_abi)), 4),
        "population_std_abi":     round(float(np.std(all_abi)),  4),
        "population_mean_rel_error": round(float(np.mean(all_rel_err)), 4) if all_rel_err else None
    }
    user_abis = []
    for uid in user_ids:
        df = get_user_df(uid)
        if len(df) >= 5:
            user_abis.append(float(df["abi"].mean()))
    if user_abis:
        pop["individual_abi_variance"] = round(float(np.var(user_abis)), 4)
        pop["individual_abi_range"]    = [round(min(user_abis), 4), round(max(user_abis), 4)]
    return jsonify(pop)


# ============================================
# 8. RUN
# ============================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)