# DDVM Engine

**Decision Drift Validation Model — Behavioral Experiment Platform**

A production-grade web application for measuring cognitive bias in sequential decision-making. Built as part of an independent research program testing whether trajectory-aware metrics outperform static aggregate measures in detecting and predicting anchoring bias.

---

## What it does

Participants complete a 20-round prediction task in which they estimate startup user growth figures. Each round presents two reference values — a historical figure and a market forecast. The system silently tracks how much each estimate is influenced by these references, computes behavioral metrics in real time, and generates a full decision profile at the end.

The platform is designed as a closed-loop experimental system:

- **Data layer** — captures sequential decisions, reference values, and outcomes
- **Metric layer** — computes ABI, TLI, BE, SII, CD, ADS, RABI, and IES per participant
- **Intervention layer** — adaptive bandit algorithm selects and logs behavioral nudges
- **Analysis layer** — robustness suite validates metrics against perturbation, split-half reliability, and null model benchmarks

---

## Metrics

| Metric | Name | What it measures |
|---|---|---|
| ABI | Anchoring Bias Index | How closely estimates track the reference value |
| TLI | Trajectory Lock-In | Whether the bias pattern stabilises across rounds |
| BE | Bias Elasticity | Sensitivity of anchoring to reference magnitude |
| SII | Signal Independence Index | Independence from the market forecast |
| CD | Calibration Drift | Whether accuracy improves or worsens across rounds |
| ADS | Anchor Displacement Sensitivity | Self-correction against extreme references |
| RABI | Resource-Rational Agent Bias Index | Individual-level anchoring susceptibility (composite) |
| IES | Intervention Effectiveness Score | Whether behavioral nudges changed decision patterns |

---

## Architecture

```
Flask (backend)
├── /start          — initialize or resume session
├── /submit         — atomic round submission (validates, commits, returns next task)
├── /full_analysis  — RABI + IES + all metrics for a completed session
├── /robustness     — perturbation stability, split-half, null model
└── /dashboard      — research console

PostgreSQL (Railway)
├── user_sessions   — backend-authoritative session state
├── decisions       — per-round behavioral data
├── interventions   — logged with display flag (control vs treatment)
├── bandit_states   — per-user epsilon-greedy bandit
└── event_log       — observability layer

Gunicorn + Docker (Railway)
```

**Key design decisions:**
- Backend is the state machine. Frontend is display-only — it never holds authoritative round state.
- `NullPool` on Postgres eliminates connection-pool exhaustion across Gunicorn workers.
- Tasks are stored server-side in `task_json` — frontend cannot tamper with true values.
- Interventions fire for all participants from round 5; `displayed=1` only for treatment condition.
- Race-condition-safe session creation via optimistic concurrency (flush → catch IntegrityError → re-query).

---

## Stack

- **Backend:** Python 3.11, Flask 3.0, SQLAlchemy 2.0
- **Database:** PostgreSQL (psycopg2-binary)
- **Analytics:** NumPy, Pandas, SciPy, hmmlearn
- **Deployment:** Docker (python:3.11-slim), Railway, Gunicorn
- **Frontend:** Vanilla JS, HTML/CSS — no framework dependencies

---

## Setup

**Local (SQLite):**
```bash
pip install -r requirements.txt
python app.py
```

**Production (Railway):**
1. Fork or push to GitHub
2. Connect repo to Railway
3. Add a PostgreSQL service — Railway auto-sets `DATABASE_URL`
4. Set `SECRET_KEY` environment variable
5. Deploy — schema is created and migrated automatically on startup

**Experiment conditions:**
- Control: `your-domain.railway.app/` (interventions logged, not shown)
- Treatment: `your-domain.railway.app/?condition=treatment` (interventions shown)

---

## Research context

This platform is the empirical validation layer for the Decision Drift Validation Model (DDVM), which tests three claims from prior published work:

1. Measurement operationalization determines what anchoring effects are detected
2. Trajectory-aware metrics (ABI, TLI, BE) predict outcomes better than static aggregates
3. Stable population-level statistics can mask highly variable individual trajectories

Related publications:
- *Reframing Anchoring Bias Measurement: A Logistic Classification Approach* — under review, Undergraduate Journal of Cognitive Sciences (N=94)
- *RABI: Resource-Rational Agent Bias Index* — preprint, DOI: [10.5281/zenodo.18737769](https://doi.org/10.5281/zenodo.18737769)
- *Co-Adaptive Human–AI Bias Dynamics* — preprint, DOI: [10.5281/zenodo.18737891](https://doi.org/10.5281/zenodo.18737891)

---

## Author

**Muhammad Sarim Nawaz**  
Independent Researcher  
[GitHub](https://github.com/maliksarimnawaz) · [LinkedIn](https://www.linkedin.com/in/sarim-nawaz-810a42397/)