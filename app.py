from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import os
import datetime
from threading import Thread
import time

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_secret")

# DATABASE CONFIG
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///decisions.db")

# Fix for Heroku postgres URL
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ------------------------------
# DATABASE MODEL
# ------------------------------
class Decision(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), nullable=False)
    decision_value = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    reasoning = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    outcome_value = db.Column(db.Float)
    adjusted_estimate = db.Column(db.Float)
    anchor_warning = db.Column(db.Boolean, default=False)

# ------------------------------
# ROUTES
# ------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            user_id = request.form.get("user_id")
            decision_value = float(request.form.get("decision_value"))
            confidence = float(request.form.get("confidence"))
            reasoning = request.form.get("reasoning")

            # SIMPLE LOGIC (placeholder)
            anchor_warning = decision_value > 90 or decision_value < 10
            adjusted_estimate = decision_value * 0.7 + 50 * 0.3

            decision = Decision(
                user_id=user_id,
                decision_value=decision_value,
                confidence=confidence,
                reasoning=reasoning,
                adjusted_estimate=adjusted_estimate,
                anchor_warning=anchor_warning
            )

            db.session.add(decision)
            db.session.commit()

            flash(f"Saved! Adjusted: {round(adjusted_estimate,2)}" +
                  (" ⚠️ Anchoring suspected" if anchor_warning else ""))

        except Exception as e:
            flash(f"Error: {str(e)}")

        return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    decisions = Decision.query.order_by(Decision.timestamp.desc()).all()
    return render_template("dashboard.html", decisions=decisions)


@app.route("/log_outcome/<int:decision_id>", methods=["GET", "POST"])
def log_outcome(decision_id):
    decision = Decision.query.get_or_404(decision_id)

    if request.method == "POST":
        try:
            outcome = float(request.form.get("outcome_value"))
            decision.outcome_value = outcome
            db.session.commit()
            flash("Outcome saved!")
        except Exception as e:
            flash(f"Error: {str(e)}")

        return redirect(url_for("dashboard"))

    return render_template("log_outcome.html", decision=decision)


# ------------------------------
# BACKGROUND METRICS WORKER (SAFE VERSION)
# ------------------------------
def metrics_worker():
    while True:
        with app.app_context():
            try:
                decisions = Decision.query.filter(Decision.outcome_value.isnot(None)).all()

                for d in decisions:
                    # placeholder metric
                    pass

                db.session.commit()

            except Exception as e:
                print("Metrics error:", e)

        time.sleep(60)


# Only run background worker locally (not in Heroku dynos)
if os.environ.get("RUN_WORKER") == "true":
    thread = Thread(target=metrics_worker, daemon=True)
    thread.start()


# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)