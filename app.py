"""
DDVM Decision Engine — Production Flask Application
SQLAlchemy 2.0 compatible. NullPool. All sessions backend-authoritative.
"""
import logging
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from hmmlearn.hmm import GaussianHMM
from scipy import stats
from sqlalchemy import (Column, DateTime, Float, Integer, String, Text,
                        UniqueConstraint, create_engine, event, text)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.pool import NullPool

# ── Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("ddvm")

# ============================================================
# 1. DATABASE
# ============================================================

_raw_db = os.environ.get("DATABASE_URL", "sqlite:///decisions.db")
DATABASE_URL = (
    _raw_db.replace("postgres://", "postgresql://", 1)
    if _raw_db.startswith("postgres://")
    else _raw_db
)
IS_POSTGRES = DATABASE_URL.startswith("postgresql")

log.info("DATABASE_URL prefix: %s", DATABASE_URL[:30])

_engine_kwargs: dict = {"echo": False, "pool_pre_ping": True}
if IS_POSTGRES:
    # NullPool: no connection reuse between requests.
    # Eliminates connection-state bugs across Gunicorn workers.
    _engine_kwargs["poolclass"] = NullPool

engine = create_engine(DATABASE_URL, **_engine_kwargs)
# SQLAlchemy 2.0 — do NOT use bind= kwarg on sessionmaker
DBSession = sessionmaker(engine, expire_on_commit=False)
Base = declarative_base()

TOTAL_ROUNDS = 20

# ── Models ───────────────────────────────────────────────────

class UserSession(Base):
    __tablename__ = "user_sessions"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    user_id       = Column(String(50), unique=True, nullable=False, index=True)
    condition     = Column(String(20), nullable=False, default="control")
    current_round = Column(Integer, nullable=False, default=1)
    completed     = Column(Integer, nullable=False, default=0)
    task_json     = Column(Text)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow)


class Decision(Base):
    __tablename__ = "decisions"
    id                  = Column(Integer, primary_key=True, autoincrement=True)
    user_id             = Column(String(50), nullable=False, index=True)
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
    id           = Column(Integer, primary_key=True, autoincrement=True)
    user_id      = Column(String(50))
    round_number = Column(Integer)
    action       = Column(String(50))
    created_at   = Column(DateTime, default=datetime.utcnow)


class BanditState(Base):
    __tablename__ = "bandit_states"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    user_id     = Column(String(50), unique=True)
    values_json = Column(Text)
    counts_json = Column(Text)


class EventLog(Base):
    __tablename__ = "event_log"
    id            = Column(Integer, primary_key=True, autoincrement=True)
    user_id       = Column(String(50))
    round_number  = Column(Integer)
    event_type    = Column(String(50))
    metadata_json = Column(Text)
    created_at    = Column(DateTime, default=datetime.utcnow)


# ── Schema setup ─────────────────────────────────────────────

def _setup_schema() -> None:
    try:
        Base.metadata.create_all(engine)
        log.info("Schema OK.")
    except Exception as exc:
        log.error("Schema creation failed: %s", exc)

    if not IS_POSTGRES:
        return

    migrations = [
        ("decisions", "condition",            "VARCHAR(20) DEFAULT 'control'"),
        ("decisions", "round_number",         "INTEGER"),
        ("decisions", "anchor_displacement",  "DOUBLE PRECISION"),
        ("decisions", "signal_noise",         "DOUBLE PRECISION"),
        ("decisions", "abi",                  "DOUBLE PRECISION"),
        ("decisions", "relative_error",       "DOUBLE PRECISION"),
        ("decisions", "submitted_at",         "TIMESTAMP"),
    ]
    # Drop NOT NULL on legacy columns that no longer exist in the model
    legacy_nullable = [
        ("decisions", "confidence"),
        ("decisions", "adjusted_estimate"),
        ("decisions", "anchor_warning"),
        ("decisions", "outcome_value"),
    ]
    with engine.connect() as conn:
        for tbl, col in legacy_nullable:
            try:
                conn.execute(text(
                    f"ALTER TABLE {tbl} ALTER COLUMN {col} DROP NOT NULL"
                ))
                conn.commit()
                log.info("Dropped NOT NULL on %s.%s", tbl, col)
            except Exception as exc:
                log.debug("legacy_nullable skip %s.%s: %s", tbl, col, exc)
                try: conn.rollback()
                except Exception: pass
    with engine.connect() as conn:
        for tbl, col, typ in migrations:
            try:
                conn.execute(text(
                    f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS {col} {typ}"
                ))
                conn.commit()
            except Exception as exc:
                log.debug("Migration skip %s.%s: %s", tbl, col, exc)
                try:
                    conn.rollback()
                except Exception:
                    pass


_setup_schema()

# ============================================================
# 2. TASK GENERATOR
# ============================================================

_DISPLACEMENTS = [-20, -12, -6, 6, 12, 20]
_NOISE_TIERS   = [3, 6, 10]


def generate_task() -> dict:
    tv  = float(max(15.0, np.random.normal(50, 8)))
    dis = float(np.random.choice(_DISPLACEMENTS))
    nt  = float(np.random.choice(_NOISE_TIERS))
    return {
        "true_value":          round(tv, 3),
        "anchor":              round(tv + dis, 2),
        "anchor_displacement": round(dis, 2),
        "signal":              round(float(tv + np.random.normal(0, nt)), 2),
        "signal_noise":        round(nt, 2),
    }

# ============================================================
# 3. CORE METRICS — DO NOT MODIFY
# ============================================================

def compute_abi(estimate: float, true_value: float, anchor: float) -> float:
    d = anchor - true_value
    if abs(d) < 0.01:
        return 0.0
    return float((estimate - true_value) / d)


def compute_relative_error(estimate: float, true_value: float) -> float:
    if abs(true_value) < 0.01:
        return 0.0
    return float(abs(estimate - true_value) / true_value)


def get_user_df(user_id: str) -> pd.DataFrame:
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
    res: dict = {"n": n}

    abi_vals = df["abi"].fillna(0.0).values

    res["mean_abi"] = round(float(np.mean(abi_vals)), 4)
    res["std_abi"]  = round(float(np.std(abi_vals)), 4)

    if n >= 10:
        half      = n // 2
        es        = float(np.std(abi_vals[:half]))
        ls        = float(np.std(abi_vals[half:]))
        res["tli"] = round(float(1 - (ls / es)) if es > 0.01 else 0.0, 4)
    else:
        res["tli"] = None

    disps = np.abs(df["anchor_displacement"].fillna(0.0).values)
    if disps.std() > 0.01 and n >= 8:
        sl, _, r, p, _ = stats.linregress(disps, abi_vals)
        res["be_slope"] = round(float(sl), 4)
        res["be_r"]     = round(float(r), 4)
        res["be_p"]     = round(float(p), 4)
    else:
        res["be_slope"] = res["be_r"] = res["be_p"] = None

    se = (df["signal"] - df["true_value"]).fillna(0.0).values
    ee = (df["estimate"] - df["true_value"]).fillna(0.0).values
    if np.std(se) > 0.01 and np.std(ee) > 0.01:
        rs = float(np.corrcoef(ee, se)[0, 1])
        res["sii"] = round(float(1 - abs(rs)), 4)
        res["signal_correlation"] = round(rs, 4)
    else:
        res["sii"] = 1.0
        res["signal_correlation"] = 0.0

    rnds = df["round"].values.astype(float)
    re   = df["relative_error"].fillna(0.0).values
    if n >= 8 and np.std(rnds) > 0:
        sl, _, r, p, _ = stats.linregress(rnds, re)
        res["cd_slope"] = round(float(sl), 6)
        res["cd_r"]     = round(float(r), 4)
        res["cd_p"]     = round(float(p), 4)
    else:
        res["cd_slope"] = None

    res["mean_relative_error"] = round(float(np.mean(re)), 4)

    adj = np.abs(df["estimate"].values - df["anchor"].fillna(50.0).values)
    if np.std(disps) > 0.01 and np.std(adj) > 0.01:
        res["ads"] = round(float(np.corrcoef(adj, disps)[0, 1]), 4)
    else:
        res["ads"] = None

    if n >= 15:
        try:
            X     = abi_vals.reshape(-1, 1)
            model = GaussianHMM(n_components=3, n_iter=200, random_state=42,
                                covariance_type="full")
            model.fit(X)
            states   = model.predict(X)
            means    = {s: float(model.means_[s][0]) for s in range(3)}
            ss       = sorted(means, key=means.get)
            lmap     = {ss[0]: "under_anchored", ss[1]: "moderate", ss[2]: "over_anchored"}
            seq      = [lmap[s] for s in states]
            res["css_terminal_state"]   = seq[-1]
            res["css_transition_count"] = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
            res["css_state_sequence"]   = seq
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
    res["round_relative_error"] = [round(float(v), 3) for v in re]
    res["rounds"]               = [int(r) for r in df["round"].values]
    return res

# ============================================================
# 4. ROBUSTNESS SUITE
# ============================================================

def _perturb_and_recompute(df: pd.DataFrame, noise_scale=0.5, n_runs=50) -> dict:
    if len(df) < 5:
        return {"status": "insufficient_data"}
    samples = []
    for _ in range(n_runs):
        p = df.copy()
        p["anchor"] += np.random.normal(0, noise_scale, len(p))
        p["signal"] += np.random.normal(0, noise_scale, len(p))
        p["abi"]    = [compute_abi(r["estimate"], r["true_value"], r["anchor"])
                       for _, r in p.iterrows()]
        p["anchor_displacement"] = p["anchor"] - p["true_value"]
        m = compute_full_metrics(p)
        if m.get("status") != "insufficient_data":
            samples.append(m.get("mean_abi", 0.0))
    if not samples:
        return {"status": "insufficient_data"}
    s = float(np.std(samples))
    return {
        "abi_stability_std":   round(s, 4),
        "n_runs":              n_runs,
        "abi_stability_label": "high" if s < 0.05 else ("moderate" if s < 0.15 else "low"),
    }


def _split_half(df: pd.DataFrame) -> dict:
    if len(df) < 10:
        return {"status": "insufficient_data"}
    df   = df.copy().sort_values("round").reset_index(drop=True)
    h    = len(df) // 2
    a, b = df["abi"].values[:h], df["abi"].values[h: h * 2]
    n    = min(len(a), len(b))
    if n < 3 or np.std(a[:n]) < 0.01 or np.std(b[:n]) < 0.01:
        return {"abi_split_half_corr": 0.0, "reliability_label": "low"}
    c  = float(np.corrcoef(a[:n], b[:n])[0, 1])
    sb = (2 * c) / (1 + c) if abs(1 + c) > 0.01 else 0.0
    return {
        "abi_split_half_corr": round(c, 4),
        "spearman_brown_corr": round(sb, 4),
        "reliability_label":   "high" if abs(sb) > 0.7 else ("moderate" if abs(sb) > 0.4 else "low"),
    }


def _internal_consistency(df: pd.DataFrame) -> dict:
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
    coh   = sum([r_re is not None and r_re > 0, r_adj is not None and r_adj < 0])
    return {
        "abi_relerr_corr":   r_re,
        "abi_adj_corr":      r_adj,
        "n_coherent":        coh,
        "consistency_label": "high" if coh == 2 else ("moderate" if coh == 1 else "low"),
    }


def _null_model(df: pd.DataFrame) -> dict:
    if len(df) < 5:
        return {"status": "insufficient_data"}
    mu  = df["anchor"].mean()
    sig = max(df["anchor"].std(), 10.0)
    dr  = df.copy()
    dr["estimate"]       = np.random.uniform(mu - 2 * sig, mu + 2 * sig, len(df))
    dr["abi"]            = [compute_abi(r["estimate"], r["true_value"], r["anchor"]) for _, r in dr.iterrows()]
    dr["relative_error"] = [compute_relative_error(r["estimate"], r["true_value"]) for _, r in dr.iterrows()]
    rm, rnd = compute_full_metrics(df), compute_full_metrics(dr)
    if rm.get("status") == "insufficient_data":
        return {"status": "insufficient_data"}
    ra, rna = rm.get("mean_abi", 0.0), rnd.get("mean_abi", 0.0)
    sep = abs(ra - rna)
    return {
        "real_mean_abi":    round(ra, 4),
        "random_mean_abi":  round(rna, 4),
        "separation":       round(sep, 4),
        "meaningful_label": "yes" if sep > 0.1 else ("borderline" if sep > 0.05 else "no"),
    }


def compute_robustness_suite(df: pd.DataFrame) -> dict:
    sens = _perturb_and_recompute(df)
    sh   = _split_half(df)
    cons = _internal_consistency(df)
    nm   = _null_model(df)
    high = sum([
        sens.get("abi_stability_label") == "high",
        sh.get("reliability_label")     == "high",
        cons.get("consistency_label")   == "high",
        nm.get("meaningful_label")      == "yes",
    ])
    overall = "high" if high >= 3 else ("moderate" if high >= 1 else "low")
    return {
        "sensitivity":          sens,
        "split_half":           sh,
        "internal_consistency": cons,
        "null_model":           nm,
        "confidence_summary": {
            "abi_stability": sens.get("abi_stability_label", "unknown"),
            "reliability":   sh.get("reliability_label", "unknown"),
            "consistency":   cons.get("consistency_label", "unknown"),
            "meaningful":    nm.get("meaningful_label", "unknown"),
            "overall":       overall,
        },
    }

# ============================================================
# 5. BANDIT
# ============================================================

_ACTIONS = ["debias", "slow", "reanchor", "ignore_signal"]

_FEEDBACK = {
    "debias":        "Your estimates are tracking the reference figure closely. Try forming your prediction before looking at the reference.",
    "slow":          "Consider whether your first instinct is being influenced by the figures shown. Take a moment before committing.",
    "reanchor":      "The reference figure may be misleading. Focus on what you know independently of the value shown.",
    "ignore_signal": "The market forecast has been inaccurate in recent rounds. Weight it accordingly.",
}


def _load_bandit(s: Session, user_id: str) -> dict:
    state = s.query(BanditState).filter_by(user_id=user_id).first()
    if state:
        return {"values": json.loads(state.values_json),
                "counts": json.loads(state.counts_json)}
    return {"values": {a: 0.0 for a in _ACTIONS}, "counts": {a: 1 for a in _ACTIONS}}


def _save_bandit(s: Session, user_id: str, bandit: dict) -> None:
    state = s.query(BanditState).filter_by(user_id=user_id).first()
    if state:
        state.values_json = json.dumps(bandit["values"])
        state.counts_json = json.dumps(bandit["counts"])
    else:
        s.add(BanditState(user_id=user_id,
                          values_json=json.dumps(bandit["values"]),
                          counts_json=json.dumps(bandit["counts"])))


def _select_action(bandit: dict, recent_abi: float) -> str:
    if recent_abi > 0.5 and random.random() < 0.6:
        return random.choice(["debias", "reanchor"])
    if recent_abi < 0.2 and random.random() < 0.4:
        return "ignore_signal"
    if random.random() < 0.25:
        return random.choice(_ACTIONS)
    return max(bandit["values"], key=bandit["values"].get)


def _update_bandit(bandit: dict, action: str, reward: float) -> dict:
    bandit["counts"][action] += 1
    bandit["values"][action] += (reward - bandit["values"][action]) / bandit["counts"][action]
    return bandit

# ============================================================
# 6. SESSION HELPERS
# ============================================================

def _upsert_session(s: Session, user_id: str, condition: str) -> "UserSession":
    """
    Get-or-create with race-condition safety.
    Uses INSERT ... ON CONFLICT (Postgres) or plain get-or-create (SQLite).
    Returns the committed UserSession row.
    """
    us = s.query(UserSession).filter_by(user_id=user_id).first()
    if us:
        return us

    task = generate_task()
    us   = UserSession(
        user_id=user_id,
        condition=condition,
        current_round=1,
        completed=0,
        task_json=json.dumps(task),
    )
    s.add(us)
    try:
        s.flush()   # write to DB, detect constraint violations immediately
    except IntegrityError:
        s.rollback()
        # Another request created the row between our SELECT and INSERT — fetch it
        us = s.query(UserSession).filter_by(user_id=user_id).first()
        if not us:
            raise RuntimeError("session upsert failed — row missing after conflict")
    return us


def _log_event(user_id: str, round_number, event_type: str, meta: dict = None) -> None:
    """Fire-and-forget. Never raises, never blocks main request."""
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
# 7. FLASK APP
# ============================================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-in-production")


# ── Observability hooks ───────────────────────────────────────

@app.before_request
def _before():
    request._start_ts = time.monotonic()
    # Use request.form carefully — Werkzeug caches the parse, safe to read here
    form_preview = {k: v for k, v in (request.form or {}).items()
                    if k not in ("decision_value",)}  # don't log numeric answer
    log.info("REQ  %s %s form=%s", request.method, request.path, form_preview)


@app.after_request
def _after(resp):
    elapsed = (time.monotonic() - getattr(request, "_start_ts", time.monotonic())) * 1000
    log.info("RESP %s %s %d (%.0fms)", request.method, request.path,
             resp.status_code, elapsed)
    return resp


@app.errorhandler(Exception)
def _unhandled(exc):
    log.exception("UNHANDLED on %s %s", request.method, request.path)
    return jsonify({"status": "error", "error_type": "unhandled", "msg": str(exc)}), 500


def _err(msg: str, code: int = 400, error_type: str = "error"):
    log.warning("ERR [%d/%s] %s", code, error_type, msg)
    return jsonify({"status": "error", "error_type": error_type, "msg": msg}), code


# ── Health ────────────────────────────────────────────────────

@app.route("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception as exc:
        db_ok = False
        log.error("health DB check failed: %s", exc)
    return jsonify({"status": "ok", "db": db_ok, "ts": datetime.utcnow().isoformat()})


# ── Index ─────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── Start / resume ────────────────────────────────────────────

@app.route("/start", methods=["POST"])
def start():
    user_id   = (request.form.get("user_id") or "").strip()
    condition = (request.form.get("condition") or "control").strip()

    if not user_id:
        return _err("Participant ID is required.", 400, "missing_user_id")

    try:
        with DBSession() as s:
            us   = _upsert_session(s, user_id, condition)
            task = json.loads(us.task_json) if us.task_json else generate_task()
            if not us.task_json:
                us.task_json = json.dumps(task)
            s.commit()
            log.info("start ok user=%s round=%d completed=%d",
                     user_id, us.current_round, us.completed)
            return jsonify({
                "status":        "ok",
                "current_round": us.current_round,
                "total_rounds":  TOTAL_ROUNDS,
                "condition":     us.condition,
                "task":          task,
                "completed":     us.completed,
            })
    except Exception as exc:
        log.exception("start FAILED user=%s", user_id)
        return _err(f"Could not start session: {exc}", 500, "server_error")


# ── Submit ────────────────────────────────────────────────────

@app.route("/submit", methods=["POST"])
def submit():
    # 1. Extract and validate user_id
    user_id = (request.form.get("user_id") or "").strip()
    if not user_id:
        return _err("Missing user_id.", 400, "missing_user_id")

    # 2. Normalise and validate decision value
    raw = (request.form.get("decision_value") or "").strip().replace(",", ".")
    try:
        decision_value = float(raw)
        if not np.isfinite(decision_value) or decision_value < 0:
            raise ValueError
    except (ValueError, TypeError):
        _log_event(user_id, None, "validation_error", {"raw": raw})
        return _err("Please enter a valid positive number.", 400, "invalid_input")

    reasoning = (request.form.get("reasoning") or "").strip()

    try:
        with DBSession() as s:
            # 3. Load session — do NOT create here; must exist from /start
            us = s.query(UserSession).filter_by(user_id=user_id).first()
            if not us:
                # Session missing: auto-create so user doesn't get stuck
                log.warning("submit: no session for user=%s, auto-creating", user_id)
                condition = (request.form.get("condition") or "control").strip()
                us = _upsert_session(s, user_id, condition)
                s.flush()

            submitted_round = int(request.form.get("round_number") or us.current_round)
            expected_round  = us.current_round

            # 4. Round guard — resync if mismatch
            if submitted_round != expected_round:
                log.warning("round mismatch user=%s submitted=%d expected=%d",
                            user_id, submitted_round, expected_round)
                task = json.loads(us.task_json) if us.task_json else generate_task()
                if not us.task_json:
                    us.task_json = json.dumps(task)
                s.commit()   # commit any auto-created session
                return jsonify({
                    "status":        "resync",
                    "current_round": expected_round,
                    "task":          task,
                })

            # 5. Load authoritative task from backend
            if us.task_json:
                task = json.loads(us.task_json)
            else:
                task = generate_task()
                us.task_json = json.dumps(task)

            tv          = task["true_value"]
            anchor      = task["anchor"]
            a_disp      = task["anchor_displacement"]
            signal      = task["signal"]
            s_noise     = task["signal_noise"]

            abi       = compute_abi(decision_value, tv, anchor)
            rel_error = compute_relative_error(decision_value, tv)

            # 6. Insert decision row
            d = Decision(
                user_id=user_id, condition=us.condition,
                round_number=expected_round,
                decision_value=decision_value, reasoning=reasoning,
                anchor=anchor, anchor_displacement=a_disp,
                signal=signal, signal_noise=s_noise,
                true_value=tv, outcome_value=tv,
                abi=abi, relative_error=rel_error,
            )
            s.add(d)

            # 7. Bandit intervention (treatment only, round >= 5, no nested sessions)
            intervention = None
            if us.condition == "treatment" and expected_round >= 5:
                prev = (
                    s.query(Decision)
                    .filter(Decision.user_id == user_id, Decision.abi.isnot(None))
                    .order_by(Decision.round_number.desc())
                    .limit(5).all()
                )
                if len(prev) >= 4:
                    r_abi   = float(np.mean([p.abi for p in prev]))
                    r_err   = [p.relative_error for p in prev if p.relative_error is not None]
                    bandit  = _load_bandit(s, user_id)
                    action  = _select_action(bandit, r_abi)
                    reward  = (float(np.mean(r_err[:-3]) - np.mean(r_err[-3:]))
                               if len(r_err) > 3 else 0.0)
                    bandit  = _update_bandit(bandit, action, reward)
                    _save_bandit(s, user_id, bandit)
                    s.add(Intervention(user_id=user_id,
                                       round_number=expected_round, action=action))
                    intervention = _FEEDBACK[action]

            # 8. Advance session state
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

            # 9. Single atomic commit
            s.commit()
            decision_id = d.id

        # 10. Async event log (outside transaction)
        _log_event(user_id, expected_round, "submission_success")
        log.info("submit OK user=%s round=%d abi=%.3f is_last=%s",
                 user_id, expected_round, abi, is_last)

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
            "true_value":         round(tv, 2),
            "your_estimate":      round(decision_value, 2),
            "error":              round(abs(decision_value - tv), 2),
            "abi":                round(abi, 3),
            "relative_error":     round(rel_error, 3),
            "abi_interpretation": abi_interp,
            "intervention":       intervention,
            "session_complete":   is_last,
            "next_round":         None if is_last else expected_round + 1,
            "next_task":          next_task,
        })

    except Exception as exc:
        log.exception("submit FAILED user=%s", user_id)
        _log_event(user_id, None, "submission_failure", {"error": str(exc)})
        return _err(f"Submission failed: {exc}", 500, "server_error")


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
    try:
        with DBSession() as s:
            rows     = s.query(Decision).filter(
                Decision.outcome_value.isnot(None),
                Decision.abi.isnot(None),
            ).all()
            if not rows:
                return jsonify({"status": "no_data"})
            all_abi     = [r.abi for r in rows]
            all_rel_err = [r.relative_error for r in rows if r.relative_error is not None]
            user_ids    = list({r.user_id for r in rows})

        pop = {
            "n_users":    len(user_ids),
            "n_decisions": len(all_abi),
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
    except Exception as exc:
        log.exception("population_metrics failed")
        return _err(str(exc), 500, "server_error")


@app.route("/dashboard")
def dashboard():
    try:
        with DBSession() as s:
            d_rows = s.query(Decision).order_by(Decision.submitted_at.desc()).limit(500).all()
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
                "timestamp":      d.submitted_at.strftime("%Y-%m-%d %H:%M") if d.submitted_at else "",
            } for d in d_rows]
            s_rows = s.query(UserSession).order_by(UserSession.created_at.desc()).all()
            sessions = [{
                "user_id":       ss.user_id,
                "condition":     ss.condition,
                "current_round": ss.current_round,
                "completed":     ss.completed,
            } for ss in s_rows]
        return render_template("dashboard.html", decisions=decisions, sessions=sessions)
    except Exception as exc:
        log.exception("dashboard error")
        return jsonify({"status": "error", "msg": str(exc)}), 500


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)