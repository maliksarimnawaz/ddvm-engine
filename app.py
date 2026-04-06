"""
DDVM Decision Engine — Production Flask Application
All sessions controlled by backend. Frontend is display-only.
"""
import logging
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from hmmlearn.hmm import GaussianHMM
from scipy import stats
from sqlalchemy import (Column, DateTime, Float, Integer, String, Text,
                        create_engine, text)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

# ── Logging (visible in Railway)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ============================================================
# 1. DATABASE SETUP
# ============================================================

_raw_db = os.environ.get("DATABASE_URL", "sqlite:///decisions.db")
DATABASE_URL = (
    _raw_db.replace("postgres://", "postgresql://", 1)
    if _raw_db.startswith("postgres://")
    else _raw_db
)

# NullPool avoids connection-pool exhaustion on Postgres/Railway
_engine_kwargs = dict(echo=False, pool_pre_ping=True)
if DATABASE_URL.startswith("postgresql"):
    _engine_kwargs["poolclass"] = NullPool

engine = create_engine(DATABASE_URL, **_engine_kwargs)
DBSession = sessionmaker(bind=engine)
Base = declarative_base()

TOTAL_ROUNDS = 20


# ── Models ───────────────────────────────────────────────────

class UserSession(Base):
    __tablename__ = "user_sessions"
    id            = Column(Integer, primary_key=True)
    user_id       = Column(String(50), unique=True, nullable=False)
    condition     = Column(String(20), default="control")
    current_round = Column(Integer, default=1)
    completed     = Column(Integer, default=0)
    task_json     = Column(Text)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow)


class Decision(Base):
    __tablename__ = "decisions"
    id                  = Column(Integer, primary_key=True)
    user_id             = Column(String(50), nullable=False)
    condition           = Column(String(20), default="control")
    round_number        = Column(Integer)
    decision_value      = Column(Float, nullable=False)
    reasoning           = Column(Text)
    anchor              = Column(Float)
    anchor_displacement = Column(Float)
    signal              = Column(Float)
    signal_noise        = Column(Float)
    true_value          = Column(Float)
    outcome_value       = Column(Float)
    abi                 = Column(Float)
    relative_error      = Column(Float)
    submitted_at        = Column(DateTime, default=datetime.utcnow)


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


# ── Schema creation & migration ──────────────────────────────

def _create_tables():
    try:
        Base.metadata.create_all(engine)
        log.info("Tables verified/created.")
    except Exception as exc:
        log.error("Table creation failed: %s", exc)


def _migrate():
    """Add columns that may be missing in existing Postgres tables."""
    if not DATABASE_URL.startswith("postgresql"):
        return
    additions = [
        ("decisions", "condition",            "VARCHAR(20) DEFAULT 'control'"),
        ("decisions", "round_number",         "INTEGER"),
        ("decisions", "anchor_displacement",  "DOUBLE PRECISION"),
        ("decisions", "signal_noise",         "DOUBLE PRECISION"),
        ("decisions", "abi",                  "DOUBLE PRECISION"),
        ("decisions", "relative_error",       "DOUBLE PRECISION"),
        ("decisions", "submitted_at",         "TIMESTAMP"),
    ]
    with engine.connect() as conn:
        for tbl, col, typ in additions:
            try:
                conn.execute(text(
                    f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} {typ}"
                ))
                conn.commit()
            except Exception as exc:
                log.warning("Migration skip %s.%s: %s", tbl, col, exc)
                try:
                    conn.rollback()
                except Exception:
                    pass


_create_tables()
_migrate()

# ============================================================
# 2. TASK GENERATOR
# ============================================================

ANCHOR_DISPLACEMENTS = [-20, -12, -6, 6, 12, 20]
SIGNAL_NOISE_TIERS   = [3, 6, 10]


def generate_task() -> dict:
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
        "signal_noise":        round(noise_tier, 2),
    }

# ============================================================
# 3. CORE METRICS  — DO NOT MODIFY
# ============================================================

def compute_abi(estimate: float, true_value: float, anchor: float) -> float:
    denom = anchor - true_value
    if abs(denom) < 0.01:
        return 0.0
    return float((estimate - true_value) / denom)


def compute_relative_error(estimate: float, true_value: float) -> float:
    if abs(true_value) < 0.01:
        return 0.0
    return float(abs(estimate - true_value) / true_value)


def get_user_df(user_id: str) -> pd.DataFrame:
    """Load all completed decisions for a user. Uses its own session scope."""
    with DBSession() as s:
        rows = (
            s.query(Decision)
            .filter(
                Decision.user_id == user_id,
                Decision.outcome_value.isnot(None),
            )
            .order_by(Decision.round_number)
            .all()
        )
        data = [
            {
                "round":               r.round_number or 0,
                "estimate":            r.decision_value or 0.0,
                "true_value":          r.true_value or 0.0,
                "anchor":              r.anchor or 50.0,
                "anchor_displacement": r.anchor_displacement or 0.0,
                "signal":              r.signal or 50.0,
                "signal_noise":        r.signal_noise or 6.0,
                "abi":                 r.abi or 0.0,
                "relative_error":      r.relative_error or 0.0,
                "condition":           r.condition or "control",
            }
            for r in rows
        ]
    return pd.DataFrame(data)


def compute_full_metrics(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return {"status": "insufficient_data", "n": len(df)}

    df  = df.copy().sort_values("round").reset_index(drop=True)
    n   = len(df)
    res = {"n": n}

    abi_vals = df["abi"].fillna(0.0).values

    res["mean_abi"] = round(float(np.mean(abi_vals)), 4)
    res["std_abi"]  = round(float(np.std(abi_vals)), 4)

    if n >= 10:
        half      = n // 2
        early_std = float(np.std(abi_vals[:half]))
        late_std  = float(np.std(abi_vals[half:]))
        res["tli"] = round(
            float(1 - (late_std / early_std)) if early_std > 0.01 else 0.0, 4
        )
    else:
        res["tli"] = None

    disps = np.abs(df["anchor_displacement"].fillna(0.0).values)
    if disps.std() > 0.01 and n >= 8:
        slope, _, r, p, _ = stats.linregress(disps, abi_vals)
        res["be_slope"] = round(float(slope), 4)
        res["be_r"]     = round(float(r), 4)
        res["be_p"]     = round(float(p), 4)
    else:
        res["be_slope"] = res["be_r"] = res["be_p"] = None

    sig_err = (df["signal"] - df["true_value"]).fillna(0.0).values
    est_err = (df["estimate"] - df["true_value"]).fillna(0.0).values
    if np.std(sig_err) > 0.01 and np.std(est_err) > 0.01:
        r_sig       = float(np.corrcoef(est_err, sig_err)[0, 1])
        res["sii"]  = round(float(1 - abs(r_sig)), 4)
        res["signal_correlation"] = round(r_sig, 4)
    else:
        res["sii"] = 1.0
        res["signal_correlation"] = 0.0

    rounds     = df["round"].values.astype(float)
    rel_errors = df["relative_error"].fillna(0.0).values
    if n >= 8 and np.std(rounds) > 0:
        slope, _, r, p, _ = stats.linregress(rounds, rel_errors)
        res["cd_slope"] = round(float(slope), 6)
        res["cd_r"]     = round(float(r), 4)
        res["cd_p"]     = round(float(p), 4)
    else:
        res["cd_slope"] = None

    res["mean_relative_error"] = round(float(np.mean(rel_errors)), 4)

    adjustments = np.abs(df["estimate"].values - df["anchor"].fillna(50.0).values)
    if np.std(disps) > 0.01 and np.std(adjustments) > 0.01:
        res["ads"] = round(float(np.corrcoef(adjustments, disps)[0, 1]), 4)
    else:
        res["ads"] = None

    if n >= 15:
        try:
            X     = abi_vals.reshape(-1, 1)
            model = GaussianHMM(
                n_components=3, n_iter=200,
                random_state=42, covariance_type="full"
            )
            model.fit(X)
            states    = model.predict(X)
            means     = {s: float(model.means_[s][0]) for s in range(3)}
            sorted_s  = sorted(means, key=means.get)
            lmap      = {
                sorted_s[0]: "under_anchored",
                sorted_s[1]: "moderate",
                sorted_s[2]: "over_anchored",
            }
            state_seq = [lmap[s] for s in states]
            res["css_terminal_state"]   = state_seq[-1]
            res["css_transition_count"] = sum(
                1 for i in range(1, len(state_seq))
                if state_seq[i] != state_seq[i - 1]
            )
            res["css_state_sequence"] = state_seq
        except Exception as exc:
            log.warning("HMM failed: %s", exc)
            res["css_terminal_state"]   = "unknown"
            res["css_transition_count"] = None
            res["css_state_sequence"]   = []
    else:
        res["css_terminal_state"]   = "insufficient_data"
        res["css_transition_count"] = None
        res["css_state_sequence"]   = []

    res["round_abi"]            = [round(float(v), 3) for v in abi_vals]
    res["round_relative_error"] = [round(float(v), 3) for v in rel_errors]
    res["rounds"]               = [int(r) for r in df["round"].values]

    return res

# ============================================================
# 4. ROBUSTNESS SUITE  — separate layer, never modifies metrics
# ============================================================

def perturb_and_recompute(df: pd.DataFrame, noise_scale=0.5, n_runs=50) -> dict:
    if len(df) < 5:
        return {"status": "insufficient_data"}
    abi_samples = []
    for _ in range(n_runs):
        df_p = df.copy()
        df_p["anchor"] += np.random.normal(0, noise_scale, len(df_p))
        df_p["signal"] += np.random.normal(0, noise_scale, len(df_p))
        df_p["abi"]    = [
            compute_abi(row["estimate"], row["true_value"], row["anchor"])
            for _, row in df_p.iterrows()
        ]
        df_p["anchor_displacement"] = df_p["anchor"] - df_p["true_value"]
        m = compute_full_metrics(df_p)
        if m.get("status") != "insufficient_data":
            abi_samples.append(m.get("mean_abi", 0.0))
    if not abi_samples:
        return {"status": "insufficient_data"}
    std = float(np.std(abi_samples))
    return {
        "abi_stability_std":   round(std, 4),
        "n_runs":              n_runs,
        "abi_stability_label": "high" if std < 0.05 else ("moderate" if std < 0.15 else "low"),
    }


def compute_split_half(df: pd.DataFrame) -> dict:
    if len(df) < 10:
        return {"status": "insufficient_data"}
    df   = df.copy().sort_values("round").reset_index(drop=True)
    half = len(df) // 2
    a    = df["abi"].values[:half]
    b    = df["abi"].values[half: half * 2]
    n    = min(len(a), len(b))
    if n < 3 or np.std(a[:n]) < 0.01 or np.std(b[:n]) < 0.01:
        return {"abi_split_half_corr": 0.0, "reliability_label": "low"}
    corr = float(np.corrcoef(a[:n], b[:n])[0, 1])
    sb   = (2 * corr) / (1 + corr) if (1 + corr) > 0.01 else 0.0
    return {
        "abi_split_half_corr":  round(corr, 4),
        "spearman_brown_corr":  round(sb, 4),
        "reliability_label":    "high" if abs(sb) > 0.7 else ("moderate" if abs(sb) > 0.4 else "low"),
    }


def compute_internal_consistency(df: pd.DataFrame) -> dict:
    if len(df) < 8:
        return {"status": "insufficient_data"}
    abi  = df["abi"].fillna(0.0).values
    rerr = df["relative_error"].fillna(0.0).values
    adj  = np.abs(df["estimate"].values - df["anchor"].fillna(50.0).values)

    def sc(a, b):
        if np.std(a) < 0.01 or np.std(b) < 0.01:
            return None
        return round(float(np.corrcoef(a, b)[0, 1]), 4)

    r_re  = sc(abi, rerr)
    r_adj = sc(abi, adj)
    coherent = sum([
        r_re  is not None and r_re  > 0,
        r_adj is not None and r_adj < 0,
    ])
    return {
        "abi_relerr_corr":     r_re,
        "abi_adj_corr":        r_adj,
        "n_coherent":          coherent,
        "consistency_label":   "high" if coherent == 2 else ("moderate" if coherent == 1 else "low"),
    }


def compare_to_random(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return {"status": "insufficient_data"}
    mu  = df["anchor"].mean()
    sig = max(df["anchor"].std(), 10.0)
    dr  = df.copy()
    dr["estimate"]       = np.random.uniform(mu - 2 * sig, mu + 2 * sig, len(df))
    dr["abi"]            = [compute_abi(r["estimate"], r["true_value"], r["anchor"]) for _, r in dr.iterrows()]
    dr["relative_error"] = [compute_relative_error(r["estimate"], r["true_value"]) for _, r in dr.iterrows()]
    rm = compute_full_metrics(df)
    rnd = compute_full_metrics(dr)
    if rm.get("status") == "insufficient_data":
        return {"status": "insufficient_data"}
    real_abi = rm.get("mean_abi", 0.0)
    rand_abi = rnd.get("mean_abi", 0.0)
    sep = abs(real_abi - rand_abi)
    return {
        "real_mean_abi":    round(real_abi, 4),
        "random_mean_abi":  round(rand_abi, 4),
        "separation":       round(sep, 4),
        "meaningful_label": "yes" if sep > 0.1 else ("borderline" if sep > 0.05 else "no"),
    }


def compute_robustness_suite(df: pd.DataFrame) -> dict:
    sensitivity = perturb_and_recompute(df)
    split_half  = compute_split_half(df)
    consistency = compute_internal_consistency(df)
    null_model  = compare_to_random(df)
    high = sum([
        sensitivity.get("abi_stability_label") == "high",
        split_half.get("reliability_label")    == "high",
        consistency.get("consistency_label")   == "high",
        null_model.get("meaningful_label")     == "yes",
    ])
    overall = "high" if high >= 3 else ("moderate" if high >= 1 else "low")
    return {
        "sensitivity":          sensitivity,
        "split_half":           split_half,
        "internal_consistency": consistency,
        "null_model":           null_model,
        "confidence_summary": {
            "abi_stability": sensitivity.get("abi_stability_label", "unknown"),
            "reliability":   split_half.get("reliability_label", "unknown"),
            "consistency":   consistency.get("consistency_label", "unknown"),
            "meaningful":    null_model.get("meaningful_label", "unknown"),
            "overall":       overall,
        },
    }

# ============================================================
# 5. BANDIT
# ============================================================

ACTIONS = ["debias", "slow", "reanchor", "ignore_signal"]

FEEDBACK_TEXT = {
    "debias":        "Your estimates are tracking the reference figure closely. Try forming your prediction before looking at the reference.",
    "slow":          "Consider whether your first instinct is being influenced by the figures shown. Take a moment before committing.",
    "reanchor":      "The reference figure may be misleading. Focus on what you know independently of the value shown.",
    "ignore_signal": "The market forecast has been inaccurate in recent rounds. Weight it accordingly.",
}


def _load_bandit_inline(s, user_id: str) -> dict:
    """Load bandit state using an already-open session."""
    state = s.query(BanditState).filter_by(user_id=user_id).first()
    if state:
        return {
            "values": json.loads(state.values_json),
            "counts": json.loads(state.counts_json),
        }
    return {"values": {a: 0.0 for a in ACTIONS}, "counts": {a: 1 for a in ACTIONS}}


def _save_bandit_inline(s, user_id: str, bandit: dict) -> None:
    """Save bandit state using an already-open session."""
    state = s.query(BanditState).filter_by(user_id=user_id).first()
    if state:
        state.values_json = json.dumps(bandit["values"])
        state.counts_json = json.dumps(bandit["counts"])
    else:
        s.add(BanditState(
            user_id=user_id,
            values_json=json.dumps(bandit["values"]),
            counts_json=json.dumps(bandit["counts"]),
        ))


def _select_action(bandit: dict, recent_abi: float) -> str:
    if recent_abi > 0.5 and random.random() < 0.6:
        return random.choice(["debias", "reanchor"])
    if recent_abi < 0.2 and random.random() < 0.4:
        return "ignore_signal"
    if random.random() < 0.25:
        return random.choice(ACTIONS)
    return max(bandit["values"], key=bandit["values"].get)


def _update_bandit(bandit: dict, action: str, reward: float) -> dict:
    bandit["counts"][action] += 1
    bandit["values"][action] += (
        (reward - bandit["values"][action]) / bandit["counts"][action]
    )
    return bandit

# ============================================================
# 6. SESSION MANAGEMENT
# ============================================================

def _get_or_create_session(s, user_id: str, condition: str) -> UserSession:
    """Get or create a UserSession. Requires an open session `s`."""
    us = s.query(UserSession).filter_by(user_id=user_id).first()
    if not us:
        task = generate_task()
        us = UserSession(
            user_id=user_id,
            condition=condition,
            current_round=1,
            completed=0,
            task_json=json.dumps(task),
        )
        s.add(us)
        s.flush()  # assign id without full commit
        log.info("New session: user=%s condition=%s", user_id, condition)
    return us


def _log_event(user_id: str, round_number, event_type: str, meta: dict = None) -> None:
    """Fire-and-forget event logging — never raises."""
    try:
        with DBSession() as s:
            s.add(EventLog(
                user_id=user_id,
                round_number=round_number,
                event_type=event_type,
                metadata_json=json.dumps(meta or {}),
            ))
            s.commit()
    except Exception as exc:
        log.warning("EventLog write failed: %s", exc)

# ============================================================
# 7. FLASK APPLICATION
# ============================================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-in-production")


def _err(msg: str, code: int = 400, error_type: str = "error") -> tuple:
    log.warning("API error [%d] %s: %s", code, error_type, msg)
    return jsonify({"status": "error", "error_type": error_type, "msg": msg}), code


# ── Health check ─────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "ts": datetime.utcnow().isoformat()})


# ── Main page ─────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Start / Resume session ────────────────────────────────────

@app.route("/start", methods=["POST"])
def start():
    user_id   = (request.form.get("user_id") or "").strip()
    condition = (request.form.get("condition") or "control").strip()

    if not user_id:
        return _err("Participant ID is required.")

    try:
        with DBSession() as s:
            us   = _get_or_create_session(s, user_id, condition)
            task = json.loads(us.task_json) if us.task_json else generate_task()
            if not us.task_json:
                us.task_json = json.dumps(task)
            s.commit()
            return jsonify({
                "status":        "ok",
                "current_round": us.current_round,
                "total_rounds":  TOTAL_ROUNDS,
                "condition":     us.condition,
                "task":          task,
                "completed":     us.completed,
            })
    except Exception as exc:
        log.exception("start failed for user=%s", user_id)
        return _err(str(exc), 500, "server_error")


# ── Atomic submit ─────────────────────────────────────────────

@app.route("/submit", methods=["POST"])
def submit():
    user_id = (request.form.get("user_id") or "").strip()
    if not user_id:
        return _err("Missing user_id.")

    # Input normalisation — handle comma-decimal locales
    raw_val = (request.form.get("decision_value") or "").strip().replace(",", ".")
    try:
        decision_value = float(raw_val)
        if not np.isfinite(decision_value) or decision_value < 0:
            raise ValueError("out of range")
    except (ValueError, TypeError):
        _log_event(user_id, None, "validation_error", {"raw": raw_val})
        return _err("Please enter a valid positive number.", 400, "invalid_input")

    reasoning = (request.form.get("reasoning") or "").strip()

    try:
        with DBSession() as s:
            us = s.query(UserSession).filter_by(user_id=user_id).first()
            if not us:
                return _err("Session not found. Please restart.", 400, "no_session")

            submitted_round = int(request.form.get("round_number") or us.current_round)
            expected_round  = us.current_round

            # ── Round guard: resync if mismatch ──────────────
            if submitted_round != expected_round:
                _log_event(user_id, submitted_round, "round_mismatch",
                           {"submitted": submitted_round, "expected": expected_round})
                task = json.loads(us.task_json) if us.task_json else generate_task()
                return jsonify({
                    "status":        "resync",
                    "current_round": expected_round,
                    "task":          task,
                })

            # ── Load task from backend (authoritative) ────────
            if us.task_json:
                task = json.loads(us.task_json)
            else:
                task = generate_task()
                us.task_json = json.dumps(task)

            true_value  = task["true_value"]
            anchor      = task["anchor"]
            anchor_disp = task["anchor_displacement"]
            signal      = task["signal"]
            signal_noise = task["signal_noise"]

            abi       = compute_abi(decision_value, true_value, anchor)
            rel_error = compute_relative_error(decision_value, true_value)

            # ── Commit decision ───────────────────────────────
            d = Decision(
                user_id=user_id, condition=us.condition,
                round_number=expected_round,
                decision_value=decision_value, reasoning=reasoning,
                anchor=anchor, anchor_displacement=anchor_disp,
                signal=signal, signal_noise=signal_noise,
                true_value=true_value, outcome_value=true_value,
                abi=abi, relative_error=rel_error,
            )
            s.add(d)

            # ── Bandit (inline — no nested sessions) ──────────
            intervention = None
            if us.condition == "treatment" and expected_round >= 5:
                prev = (
                    s.query(Decision)
                    .filter(
                        Decision.user_id == user_id,
                        Decision.abi.isnot(None),
                    )
                    .order_by(Decision.round_number.desc())
                    .limit(5)
                    .all()
                )
                if len(prev) >= 4:
                    recent_abi   = float(np.mean([p.abi for p in prev]))
                    recent_err   = [p.relative_error for p in prev if p.relative_error is not None]
                    bandit       = _load_bandit_inline(s, user_id)
                    action       = _select_action(bandit, recent_abi)
                    reward       = (
                        float(np.mean(recent_err[:-3]) - np.mean(recent_err[-3:]))
                        if len(recent_err) > 3 else 0.0
                    )
                    bandit       = _update_bandit(bandit, action, reward)
                    _save_bandit_inline(s, user_id, bandit)
                    s.add(Intervention(
                        user_id=user_id,
                        round_number=expected_round,
                        action=action,
                    ))
                    intervention = FEEDBACK_TEXT[action]

            # ── Advance session ───────────────────────────────
            is_last   = (expected_round >= TOTAL_ROUNDS)
            next_task = None
            if is_last:
                us.current_round = TOTAL_ROUNDS
                us.completed     = 1
                us.task_json     = None
            else:
                next_task        = generate_task()
                us.current_round = expected_round + 1
                us.task_json     = json.dumps(next_task)
                us.updated_at    = datetime.utcnow()

            s.commit()
            decision_id = d.id

        _log_event(user_id, expected_round, "submission_success")
        log.info("submit ok user=%s round=%d abi=%.3f", user_id, expected_round, abi)

        abi_interp = (
            "Your estimate was close to the reference figure — possible anchoring effect."
            if abi > 0.6 else
            "Your estimate moved away from the reference figure."
            if abi < 0.1 else
            "Your estimate showed moderate reference influence."
        )

        return jsonify({
            "status":             "ok",
            "decision_id":        decision_id,
            "round_completed":    expected_round,
            "true_value":         round(true_value, 2),
            "your_estimate":      round(decision_value, 2),
            "error":              round(abs(decision_value - true_value), 2),
            "abi":                round(abi, 3),
            "relative_error":     round(rel_error, 3),
            "abi_interpretation": abi_interp,
            "intervention":       intervention,
            "session_complete":   is_last,
            "next_round":         None if is_last else expected_round + 1,
            "next_task":          next_task,
        })

    except Exception as exc:
        log.exception("submit failed user=%s", user_id)
        _log_event(user_id, None, "submission_failure", {"error": str(exc)})
        return _err("Something went wrong. Please try again.", 500, "server_error")


# ── Analytics ─────────────────────────────────────────────────

@app.route("/session_metrics/<path:user_id>")
def session_metrics(user_id):
    df = get_user_df(user_id)
    if df.empty:
        return jsonify({"status": "no_data"})
    return jsonify(compute_full_metrics(df))


@app.route("/robustness/<path:user_id>")
def robustness(user_id):
    df = get_user_df(user_id)
    if len(df) < 5:
        return jsonify({"status": "insufficient_data", "n": len(df)})
    result = compute_robustness_suite(df)
    _log_event(user_id, None, "robustness_computed", {"n": len(df)})
    return jsonify(result)


@app.route("/api/population_metrics")
def api_population_metrics():
    with DBSession() as s:
        rows = s.query(Decision).filter(
            Decision.outcome_value.isnot(None),
            Decision.abi.isnot(None),
        ).all()
        if not rows:
            return jsonify({"status": "no_data"})
        all_abi     = [r.abi for r in rows]
        all_rel_err = [r.relative_error for r in rows if r.relative_error is not None]
        user_ids    = list({r.user_id for r in rows})

    pop = {
        "n_users":                   len(user_ids),
        "n_decisions":               len(all_abi),
        "population_mean_abi":       round(float(np.mean(all_abi)), 4),
        "population_std_abi":        round(float(np.std(all_abi)), 4),
        "population_mean_rel_error": round(float(np.mean(all_rel_err)), 4) if all_rel_err else None,
    }
    user_abis = []
    for uid in user_ids:
        udf = get_user_df(uid)
        if len(udf) >= 5:
            user_abis.append(float(udf["abi"].mean()))
    if user_abis:
        pop["individual_abi_variance"] = round(float(np.var(user_abis)), 4)
        pop["individual_abi_range"]    = [round(min(user_abis), 4), round(max(user_abis), 4)]
    return jsonify(pop)


# ── Dashboard ─────────────────────────────────────────────────

@app.route("/dashboard")
def dashboard():
    try:
        with DBSession() as s:
            d_rows = s.query(Decision).order_by(Decision.submitted_at.desc()).limit(500).all()
            decisions = [
                {
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
                    "timestamp":      d.submitted_at.strftime("%Y-%m-%d %H:%M") if d.submitted_at else "",
                }
                for d in d_rows
            ]
            s_rows = s.query(UserSession).order_by(UserSession.created_at.desc()).all()
            sessions = [
                {
                    "user_id":       ss.user_id,
                    "condition":     ss.condition,
                    "current_round": ss.current_round,
                    "completed":     ss.completed,
                }
                for ss in s_rows
            ]
        return render_template("dashboard.html", decisions=decisions, sessions=sessions)
    except Exception as exc:
        log.exception("dashboard error")
        return f"<pre>Dashboard error: {exc}</pre>", 500


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)