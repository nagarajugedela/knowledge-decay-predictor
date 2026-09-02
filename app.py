import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
import resend

from dotenv import load_dotenv

from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    request,
    flash
)

from flask_sqlalchemy import SQLAlchemy

from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

app.config["SECRET_KEY"] = os.getenv(
    "SECRET_KEY",
    "knowledge_decay_secret_key"
)

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    "sqlite:///knowledge.db"
)

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


# =========================================================
# RESEND EMAIL CONFIGURATION
# =========================================================

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")

# Retention below this value is considered high risk.
FORGETTING_THRESHOLD = 40

resend.api_key = RESEND_API_KEY

print(
    "Resend API key loaded:",
    bool(RESEND_API_KEY)
)


# =========================================================
# DATABASE
# =========================================================

db = SQLAlchemy(app)


# =========================================================
# LOGIN MANAGER
# =========================================================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

model = joblib.load(
    "model/retention_model.pkl"
)


# =========================================================
# USER MODEL
# =========================================================

class User(UserMixin, db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    name = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )

    # True for admin, False for normal student.
    is_admin = db.Column(
        db.Boolean,
        default=False
    )


# =========================================================
# TOPIC MODEL
# =========================================================

class Topic(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    topic_name = db.Column(
        db.String(100),
        nullable=False
    )

    study_date = db.Column(
        db.String(20),
        nullable=False
    )

    study_duration = db.Column(
        db.Float,
        nullable=False
    )

    difficulty = db.Column(
        db.String(20),
        nullable=False
    )

    confidence_score = db.Column(
        db.Integer,
        nullable=False,
        default=5
    )

    revision_count = db.Column(
        db.Integer,
        nullable=False,
        default=0
    )


# =========================================================
# RETENTION MODEL
# =========================================================

class Retention(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    topic_id = db.Column(
        db.Integer,
        db.ForeignKey("topic.id"),
        nullable=False
    )

    retention_percentage = db.Column(
        db.Float,
        nullable=False
    )

    risk_level = db.Column(
        db.String(20),
        nullable=False
    )

    next_revision_date = db.Column(
        db.String(20),
        nullable=False
    )


# =========================================================
# REMINDER DELIVERY MODEL
# =========================================================

class ReminderDelivery(db.Model):
    """
    Stores one reminder record per topic per day.

    This prevents the same automatic reminder
    from being sent repeatedly on the same day.
    """

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    topic_id = db.Column(
        db.Integer,
        db.ForeignKey("topic.id"),
        nullable=False
    )

    reminder_date = db.Column(
        db.Date,
        nullable=False
    )

    __table_args__ = (
        db.UniqueConstraint(
            "topic_id",
            "reminder_date",
            name="unique_daily_topic_reminder"
        ),
    )


# =========================================================
# USER LOADER
# =========================================================

@login_manager.user_loader
def load_user(user_id):

    return User.query.get(int(user_id))


# =========================================================
# RETENTION CALCULATION
# =========================================================

def calculate_retention(days_passed):

    retention = max(
        100 - (days_passed * 3),
        20
    )

    return retention


# =========================================================
# DIFFICULTY CONVERSION
# =========================================================

def difficulty_to_number(difficulty):

    if difficulty == "Easy":
        return 1

    if difficulty == "Medium":
        return 2

    return 3


# =========================================================
# ML RETENTION PREDICTION
# =========================================================

def predict_retention_for_days(topic, days_passed):

    difficulty_num = difficulty_to_number(
        topic.difficulty
    )

    prediction_input = pd.DataFrame([
        {
            "study_duration": topic.study_duration,
            "difficulty": difficulty_num,
            "confidence_score": topic.confidence_score,
            "revision_count": topic.revision_count,
            "days_passed": max(days_passed, 0)
        }
    ])

    prediction = model.predict(
        prediction_input
    )

    return round(
        float(prediction[0]),
        2
    )


def predict_retention(topic):

    study_date = datetime.strptime(
        topic.study_date,
        "%Y-%m-%d"
    )

    days_passed = max(
        (datetime.now() - study_date).days,
        0
    )

    return predict_retention_for_days(
        topic,
        days_passed
    )


# =========================================================
# ADAPTIVE REVISION SCHEDULER
# =========================================================

REVISION_TARGET_RETENTION = 70


def calculate_predicted_revision_date(topic):

    study_date = datetime.strptime(
        topic.study_date,
        "%Y-%m-%d"
    )

    today = datetime.now().date()

    days_since_study = max(
        (today - study_date.date()).days,
        0
    )

    # Search future days for the point where the ML model
    # predicts retention at or below the revision target.
    for future_days in range(
        max(days_since_study, 1),
        days_since_study + 91
    ):

        predicted_retention = (
            predict_retention_for_days(
                topic,
                future_days
            )
        )

        if predicted_retention <= REVISION_TARGET_RETENTION:

            revision_date = (
                study_date.date()
                + timedelta(days=future_days)
            )

            return (
                revision_date,
                predicted_retention
            )

    # Safe fallback if the model does not cross the target
    # within the 90-day prediction window.
    fallback_date = (
        today + timedelta(days=7)
    )

    fallback_retention = (
        predict_retention_for_days(
            topic,
            days_since_study + 7
        )
    )

    return (
        fallback_date,
        fallback_retention
    )


def save_revision_schedule(topic):

    revision_date, predicted_retention = (
        calculate_predicted_revision_date(topic)
    )

    if predicted_retention < 40:

        risk_level = "High"

    elif predicted_retention < 70:

        risk_level = "Medium"

    else:

        risk_level = "Low"

    retention_record = (
        Retention.query
        .filter_by(
            topic_id=topic.id
        )
        .first()
    )

    if retention_record is None:

        retention_record = Retention(
            topic_id=topic.id,
            retention_percentage=predicted_retention,
            risk_level=risk_level,
            next_revision_date=(
                revision_date.strftime(
                    "%Y-%m-%d"
                )
            )
        )

        db.session.add(
            retention_record
        )

    else:

        retention_record.retention_percentage = (
            predicted_retention
        )

        retention_record.risk_level = (
            risk_level
        )

        retention_record.next_revision_date = (
            revision_date.strftime(
                "%Y-%m-%d"
            )
        )

    db.session.commit()

    return (
        revision_date,
        predicted_retention,
        risk_level
    )


# =========================================================
# EXPLANATION GENERATOR
# =========================================================

def generate_explanation(
    topic,
    retention
):

    reasons = []

    if topic.confidence_score <= 5:

        reasons.append(
            f"Low confidence score "
            f"({topic.confidence_score}/10)"
        )

    if topic.revision_count == 0:

        reasons.append(
            "No revisions completed"
        )

    if topic.revision_count >= 2:

        reasons.append(
            "Multiple revisions completed"
        )

    if retention < 50:

        recommendation = (
            "Revise immediately"
        )

    elif retention < 75:

        recommendation = (
            "Revise within a few days"
        )

    else:

        recommendation = (
            "Topic is well retained"
        )

    return reasons, recommendation


# =========================================================
# RESEND EMAIL FUNCTION
# =========================================================

def send_revision_email(
    recipient_email,
    topic_name,
    retention=None,
    user_name=None,
    monitored_user_email=None,
    is_admin_copy=False
):

    if not RESEND_API_KEY:

        raise RuntimeError(
            "RESEND_API_KEY is not configured."
        )

    greeting = user_name or "there"

    retention_message = ""

    if retention is not None:

        retention_message = (
            f"Current estimated retention: "
            f"{retention}%\n\n"
        )

    admin_message = ""

    if is_admin_copy:

        admin_message = (
            "This is an admin monitoring copy.\n"
            f"Student: {user_name}\n"
            f"Student email: {monitored_user_email}\n\n"
        )

    subject_prefix = ""

    if is_admin_copy:

        subject_prefix = "[Admin Copy] "

    params = {

        "from": "onboarding@resend.dev",

        "to": [
            recipient_email
        ],

        "subject": (
            subject_prefix
            + "Knowledge Decay Predictor - "
              "Revision Reminder"
        ),

        "text": f"""
Hello {greeting},

This is your revision reminder from
Knowledge Decay Predictor.

Topic: {topic_name}

{retention_message}{admin_message}
This topic is due for revision today.

Please revise this topic to strengthen
your memory and reduce knowledge decay.

Keep learning!

Knowledge Decay Predictor
"""
    }

    return resend.Emails.send(
        params
    )


# =========================================================
# AUTOMATIC REVISION REMINDER
# =========================================================

def send_due_revision_reminders():
    """
    Send revision reminders when the student's
    personalized predicted revision date arrives.
    """

    if not RESEND_API_KEY:

        raise RuntimeError(
            "RESEND_API_KEY is not configured."
        )

    today = datetime.now().date()

    sent_count = 0

    topics = Topic.query.all()

    for topic in topics:

        # Get the saved personalized revision schedule.
        retention_record = (
            Retention.query
            .filter_by(
                topic_id=topic.id
            )
            .first()
        )

        # Create a schedule automatically for older topics
        # that do not have one yet.
        if retention_record is None:

            save_revision_schedule(
                topic
            )

            retention_record = (
                Retention.query
                .filter_by(
                    topic_id=topic.id
                )
                .first()
            )

        if retention_record is None:
            continue

        # Check whether the predicted revision date is today.
        try:

            revision_date = datetime.strptime(
                retention_record.next_revision_date,
                "%Y-%m-%d"
            ).date()

        except (ValueError, TypeError):

            continue

        if revision_date != today:
            continue

        # Prevent duplicate reminder emails on the same day.
        existing_reminder = (
            ReminderDelivery.query
            .filter_by(
                topic_id=topic.id,
                reminder_date=today
            )
            .first()
        )

        if existing_reminder:
            continue

        # Find the student who owns the topic.
        user = db.session.get(
            User,
            topic.user_id
        )

        if user is None:
            continue

        # Send email to the student.
        send_revision_email(
            user.email,
            topic.topic_name,
            retention=(
                retention_record
                .retention_percentage
            ),
            user_name=user.name
        )

        # Send monitoring copy to admin if configured.
        if ADMIN_EMAIL:

            send_revision_email(
                ADMIN_EMAIL,
                topic.topic_name,
                retention=(
                    retention_record
                    .retention_percentage
                ),
                user_name=user.name,
                monitored_user_email=user.email,
                is_admin_copy=True
            )

        # Record successful delivery.
        reminder = ReminderDelivery(
            topic_id=topic.id,
            reminder_date=today
        )

        db.session.add(
            reminder
        )

        db.session.commit()

        sent_count += 1

    return sent_count


# =========================================================
# SEND REMINDERS COMMAND
# =========================================================

@app.cli.command("send-reminders")
def send_reminders_command():
    """
    Run the revision reminder job.

    This command can be scheduled to run once per day
    on the production server.
    """

    try:

        sent_count = (
            send_due_revision_reminders()
        )

    except Exception as error:

        raise SystemExit(
            f"Reminder job failed: {error}"
        ) from error

    print(
        f"Sent {sent_count} "
        f"revision reminder(s)."
    )


# =========================================================
# CREATE ADMIN COMMAND
# =========================================================

@app.cli.command("create-admin")
def create_admin():
    """
    Create a new admin user or promote
    an existing user to admin.
    """

    name = input(
        "Enter admin name: "
    ).strip()

    email = input(
        "Enter admin email: "
    ).strip()

    password = input(
        "Enter admin password: "
    )

    if not name or not email or not password:

        print(
            "Name, email and password "
            "are required."
        )

        return

    existing_user = User.query.filter_by(
        email=email
    ).first()

    # If the email already exists,
    # promote that account to admin.
    if existing_user:

        existing_user.is_admin = True

        db.session.commit()

        print(
            "Existing user is now an admin."
        )

        return

    hashed_password = (
        generate_password_hash(password)
    )

    admin = User(
        name=name,
        email=email,
        password=hashed_password,
        is_admin=True
    )

    db.session.add(
        admin
    )

    db.session.commit()

    print(
        "Admin account created successfully."
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return redirect(
        url_for("login")
    )


# =========================================================
# REGISTER
# =========================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        name = request.form[
            "name"
        ].strip()

        email = request.form[
            "email"
        ].strip().lower()

        password = request.form[
            "password"
        ]

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:

            flash(
                "Email already registered"
            )

            return redirect(
                url_for("register")
            )

        hashed_password = (
            generate_password_hash(password)
        )

        new_user = User(
            name=name,
            email=email,
            password=hashed_password,
            is_admin=False
        )

        db.session.add(
            new_user
        )

        db.session.commit()

        flash(
            "Registration Successful"
        )

        return redirect(
            url_for("login")
        )

    return render_template(
        "register.html"
    )


# =========================================================
# LOGIN
# =========================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form[
            "email"
        ].strip().lower()

        password = request.form[
            "password"
        ]

        user = User.query.filter_by(
            email=email
        ).first()

        if user and check_password_hash(
            user.password,
            password
        ):

            login_user(user)

            # Admin users go to admin area later.
            if user.is_admin:

                return redirect(
                    url_for("admin_dashboard")
                )

            return redirect(
                url_for("dashboard")
            )

        flash(
            "Invalid Email or Password"
        )

    return render_template(
        "login.html"
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
@login_required
def dashboard():

    topics = Topic.query.filter_by(
        user_id=current_user.id
    ).all()

    total_topics = len(topics)

    today = datetime.now()

    retention_values = []

    high_risk = 0

    need_revision = 0

    for topic in topics:

        study_date = datetime.strptime(
            topic.study_date,
            "%Y-%m-%d"
        )

        days_passed = max(
            (today - study_date).days,
            0
        )

        retention = calculate_retention(
            days_passed
        )

        retention_values.append(
            retention
        )

        if retention < 40:

            high_risk += 1

        if retention < 70:

            need_revision += 1

    if retention_values:

        average_retention = round(
            sum(retention_values)
            / len(retention_values),
            2
        )

    else:

        average_retention = 0

    return render_template(
        "dashboard.html",
        name=current_user.name,
        total_topics=total_topics,
        average_retention=average_retention,
        high_risk=high_risk,
        need_revision=need_revision
    )


# =========================================================
# ADD TOPIC
# =========================================================

@app.route(
    "/add_topic",
    methods=["GET", "POST"]
)
@login_required
def add_topic():

    if request.method == "POST":

        topic_name = request.form[
            "topic_name"
        ].strip()

        study_date = request.form[
            "study_date"
        ]

        study_duration = float(
            request.form[
                "study_duration"
            ]
        )

        difficulty = request.form[
            "difficulty"
        ]

        confidence_score = int(
            request.form[
                "confidence_score"
            ]
        )

        revision_count = int(
            request.form[
                "revision_count"
            ]
        )

        topic = Topic(
            user_id=current_user.id,
            topic_name=topic_name,
            study_date=study_date,
            study_duration=study_duration,
            difficulty=difficulty,
            confidence_score=confidence_score,
            revision_count=revision_count
        )

        db.session.add(
            topic
        )

        db.session.commit()

        # Calculate and save the personalized revision date.
        save_revision_schedule(
            topic
        )

        flash(
            "Topic Added Successfully"
        )

        return redirect(
            url_for("dashboard")
        )

    return render_template(
        "add_topic.html"
    )


# =========================================================
# TOPICS
# =========================================================

@app.route("/topics")
@login_required
def topics():

    user_topics = Topic.query.filter_by(
        user_id=current_user.id
    ).all()

    return render_template(
        "topics.html",
        topics=user_topics
    )


# =========================================================
# RETENTION
# =========================================================

@app.route("/retention")
@login_required
def retention():

    topics = Topic.query.filter_by(
        user_id=current_user.id
    ).all()

    report = []

    today = datetime.now()

    for topic in topics:

        study_date = datetime.strptime(
            topic.study_date,
            "%Y-%m-%d"
        )

        days_passed = max(
            (today - study_date).days,
            0
        )

        retention_value = calculate_retention(
            days_passed
        )

        report.append(
            {
                "topic": topic.topic_name,
                "days": days_passed,
                "retention": retention_value
            }
        )

    return render_template(
        "retention.html",
        report=report
    )


# =========================================================
# RECOMMENDATIONS
# =========================================================

@app.route("/recommendations")
@login_required
def recommendations():

    topics = Topic.query.filter_by(
        user_id=current_user.id
    ).all()

    recommendations_data = []

    today = datetime.now()

    for topic in topics:

        study_date = datetime.strptime(
            topic.study_date,
            "%Y-%m-%d"
        )

        days_passed = max(
            (today - study_date).days,
            0
        )

        retention_value = calculate_retention(
            days_passed
        )

        if retention_value < 40:

            priority_value = (
                "Revise Immediately"
            )

        elif retention_value < 70:

            priority_value = "Revise Soon"

        else:

            priority_value = (
                "No Revision Needed"
            )

        recommendations_data.append(
            {
                "topic": topic.topic_name,
                "retention": retention_value,
                "priority": priority_value
            }
        )

    return render_template(
        "recommendations.html",
        recommendations=recommendations_data
    )


# =========================================================
# DELETE TOPIC
# =========================================================

@app.route(
    "/delete_topic/<int:id>"
)
@login_required
def delete_topic(id):

    topic = Topic.query.get_or_404(
        id
    )

    if topic.user_id != current_user.id:

        flash(
            "Unauthorized Access"
        )

        return redirect(
            url_for("topics")
        )

    db.session.delete(
        topic
    )

    db.session.commit()

    flash(
        "Topic Deleted Successfully"
    )

    return redirect(
        url_for("topics")
    )


# =========================================================
# EDIT TOPIC
# =========================================================

@app.route(
    "/edit_topic/<int:id>",
    methods=["GET", "POST"]
)
@login_required
def edit_topic(id):

    topic = Topic.query.get_or_404(
        id
    )

    if topic.user_id != current_user.id:

        flash(
            "Unauthorized Access"
        )

        return redirect(
            url_for("topics")
        )

    if request.method == "POST":

        topic.topic_name = request.form[
            "topic_name"
        ].strip()

        topic.study_date = request.form[
            "study_date"
        ]

        topic.study_duration = float(
            request.form[
                "study_duration"
            ]
        )

        topic.difficulty = request.form[
            "difficulty"
        ]

        topic.confidence_score = int(
            request.form[
                "confidence_score"
            ]
        )

        topic.revision_count = int(
            request.form[
                "revision_count"
            ]
        )

        db.session.commit()

        # Recalculate the personalized revision schedule
        # because the learning information changed.
        save_revision_schedule(
            topic
        )

        flash(
            "Topic Updated Successfully"
        )

        return redirect(
            url_for("topics")
        )

    return render_template(
        "edit_topic.html",
        topic=topic
    )


# =========================================================
# ANALYTICS
# =========================================================

@app.route("/analytics")
@login_required
def analytics():

    topics = Topic.query.filter_by(
        user_id=current_user.id
    ).all()

    labels = []

    retentions = []

    easy = 0
    medium = 0
    hard = 0

    today = datetime.now()

    for topic in topics:

        # Difficulty count
        if topic.difficulty == "Easy":

            easy += 1

        elif topic.difficulty == "Medium":

            medium += 1

        else:

            hard += 1

        study_date = datetime.strptime(
            topic.study_date,
            "%Y-%m-%d"
        )

        days_passed = max(
            (today - study_date).days,
            0
        )

        retention_value = calculate_retention(
            days_passed
        )

        labels.append(
            topic.topic_name
        )

        retentions.append(
            retention_value
        )

    difficulty_labels = [
        "Easy",
        "Medium",
        "Hard"
    ]

    difficulty_data = [
        easy,
        medium,
        hard
    ]

    total_topics = len(
        topics
    )

    if retentions:

        average_retention = round(
            sum(retentions)
            / len(retentions),
            2
        )

    else:

        average_retention = 0

    high_risk = len(
        [
            retention_value
            for retention_value in retentions
            if retention_value < 50
        ]
    )

    return render_template(
        "analytics.html",
        labels=labels,
        retentions=retentions,
        difficulty_labels=difficulty_labels,
        difficulty_data=difficulty_data,
        total_topics=total_topics,
        average_retention=average_retention,
        high_risk=high_risk
    )


# =========================================================
# AI RECOMMENDATIONS
# =========================================================

@app.route("/ai_recommendations")
@login_required
def ai_recommendations():

    topics = Topic.query.filter_by(
        user_id=current_user.id
    ).all()

    predictions = []

    for topic in topics:

        prediction = predict_retention(
            topic
        )

        predictions.append(
            {
                "topic": topic.topic_name,
                "prediction": prediction
            }
        )

    return render_template(
        "ai_recommendations.html",
        predictions=predictions
    )


# =========================================================
# PRIORITY
# =========================================================

@app.route("/priority")
@login_required
def priority():

    topics = Topic.query.filter_by(
        user_id=current_user.id
    ).all()

    ranked_topics = []

    today = datetime.now()

    for topic in topics:

        study_date = datetime.strptime(
            topic.study_date,
            "%Y-%m-%d"
        )

        days_passed = max(
            (today - study_date).days,
            0
        )

        retention_value = calculate_retention(
            days_passed
        )

        ranked_topics.append(
            {
                "topic_name": topic.topic_name,
                "retention": retention_value
            }
        )

    ranked_topics.sort(
        key=lambda item: item["retention"]
    )

    return render_template(
        "priority.html",
        ranked_topics=ranked_topics
    )


# =========================================================
# EXPLANATIONS
# =========================================================

@app.route("/explanations")
@login_required
def explanations():

    topics = Topic.query.filter_by(
        user_id=current_user.id
    ).all()

    insights = []

    today = datetime.now()

    for topic in topics:

        study_date = datetime.strptime(
            topic.study_date,
            "%Y-%m-%d"
        )

        days_passed = max(
            (today - study_date).days,
            0
        )

        retention_value = calculate_retention(
            days_passed
        )

        if retention_value < 40:

            message = (
                "Retention is very low. "
                "Immediate revision is "
                "strongly recommended."
            )

        elif retention_value < 70:

            message = (
                "Knowledge is beginning to fade. "
                "Schedule a revision session soon."
            )

        else:

            message = (
                "Retention remains strong. "
                "No urgent revision is required."
            )

        insights.append(
            {
                "topic": topic.topic_name,
                "message": message
            }
        )

    return render_template(
        "explanations.html",
        insights=insights
    )


# =========================================================
# REVISION SCHEDULE
# =========================================================

@app.route("/revision_schedule")
@login_required
def revision_schedule():

    topics = Topic.query.filter_by(
        user_id=current_user.id
    ).all()

    schedule = []

    for topic in topics:

        retention_record = (
            Retention.query
            .filter_by(
                topic_id=topic.id
            )
            .first()
        )

        # Create a schedule if this topic does not
        # already have one.
        if retention_record is None:

            save_revision_schedule(
                topic
            )

            retention_record = (
                Retention.query
                .filter_by(
                    topic_id=topic.id
                )
                .first()
            )

        if retention_record is None:
            continue

        schedule.append(
            {
                "topic": topic.topic_name,
                "retention": retention_record.retention_percentage,
                "revision_date": retention_record.next_revision_date,
                "priority": retention_record.risk_level
            }
        )

    return render_template(
        "revision_schedule.html",
        schedule=schedule
    )


# =========================================================
# TEST EMAIL
# =========================================================

@app.route("/test_email")
@login_required
def test_email():

    try:

        send_revision_email(
            current_user.email,
            "DBMS",
            user_name=current_user.name
        )

        if ADMIN_EMAIL:

            send_revision_email(
                ADMIN_EMAIL,
                "DBMS",
                user_name=current_user.name,
                monitored_user_email=current_user.email,
                is_admin_copy=True
            )

        flash(
            "Test revision email sent successfully!"
        )

    except Exception as error:

        print(
            "Email error:",
            error
        )

        flash(
            "Email could not be sent. "
            "Check the terminal."
        )

    return redirect(
        url_for("dashboard")
    )


# =========================================================
# ADMIN ACCESS HELPER
# =========================================================

def admin_required():

    if not current_user.is_authenticated:

        return False

    return current_user.is_admin


# =========================================================
# ADMIN DASHBOARD
# =========================================================

@app.route("/admin")
@login_required
def admin_dashboard():

    if not admin_required():

        flash("Admin access required.")

        return redirect(
            url_for("dashboard")
        )

    # -----------------------------------------
    # USERS
    # -----------------------------------------

    users = User.query.all()

    total_users = len(users)


    # -----------------------------------------
    # TOPICS
    # -----------------------------------------

    all_topics = Topic.query.all()

    total_topics = len(all_topics)

    topic_data = []

    retention_values = []

    high_risk_topics = 0

    today = datetime.now()


    for topic in all_topics:

        user = db.session.get(
            User,
            topic.user_id
        )

        study_date = datetime.strptime(
            topic.study_date,
            "%Y-%m-%d"
        )

        days_passed = max(
            (today - study_date).days,
            0
        )

        retention_value = calculate_retention(
            days_passed
        )

        retention_values.append(
            retention_value
        )

        if retention_value < 40:

            high_risk_topics += 1


        topic_data.append(
            {
                "topic": topic,
                "user": user,
                "retention": retention_value
            }
        )


    # -----------------------------------------
    # AVERAGE RETENTION
    # -----------------------------------------

    if retention_values:

        average_retention = round(
            sum(retention_values)
            / len(retention_values),
            2
        )

    else:

        average_retention = 0


    # -----------------------------------------
    # ADMIN DASHBOARD
    # -----------------------------------------

    return render_template(
        "admin_dashboard.html",

        total_users=total_users,

        total_topics=total_topics,

        average_retention=average_retention,

        high_risk_topics=high_risk_topics,

        users=users,

        topic_data=topic_data
    )
# =========================================================
# ADMIN USER DETAILS
# =========================================================




# =========================================================
# ADMIN USERS
# =========================================================

@app.route("/admin/user/<int:user_id>")
@login_required
def admin_user_details(user_id):

    # Check admin access
    if not admin_required():
        flash("Admin access required.")
        return redirect(url_for("dashboard"))

    # Get selected user
    user = User.query.get_or_404(user_id)

    # Get this user's topics
    user_topics = Topic.query.filter_by(
        user_id=user.id
    ).all()

    topic_data = []

    total_retention = 0
    high_risk_topics = 0

    today = datetime.now()

    for topic in user_topics:

        try:

            study_date = datetime.strptime(
                str(topic.study_date),
                "%Y-%m-%d"
            )

            days_passed = max(
                (today - study_date).days,
                0
            )

            retention = calculate_retention(
                days_passed
            )

        except (ValueError, TypeError):

            retention = 0

        if retention < 40:
            high_risk_topics += 1

        total_retention += retention

        topic_data.append({
            "topic": topic,
            "retention": round(retention, 2)
        })

    # Average retention
    if user_topics:
        average_retention = round(
            total_retention / len(user_topics),
            2
        )
    else:
        average_retention = 0

    return render_template(
        "admin_user_details.html",
        user=user,
        topic_data=topic_data,
        total_topics=len(user_topics),
        average_retention=average_retention,
        high_risk_topics=high_risk_topics
    )
@app.route("/admin/users")
@login_required
def admin_users():

    if not admin_required():
        flash("Admin access required.")
        return redirect(url_for("dashboard"))

    users = User.query.order_by(
        User.id.asc()
    ).all()

    return render_template(
        "admin_users.html",
        users=users
    )

# =========================================================
# ADMIN TOPICS
# =========================================================

@app.route("/admin/topics")
@login_required
def admin_topics():

    # Only admins can access this page
    if not admin_required():
        flash("Admin access required.")
        return redirect(url_for("dashboard"))

    topics = Topic.query.order_by(
        Topic.id.desc()
    ).all()

    topic_data = []

    today = datetime.now()

    for topic in topics:

        user = db.session.get(
            User,
            topic.user_id
        )

        try:

            study_date = datetime.strptime(
                topic.study_date,
                "%Y-%m-%d"
            )

            days_passed = max(
                (today - study_date).days,
                0
            )

            retention = calculate_retention(
                days_passed
            )

        except (ValueError, TypeError):

            retention = 0

        topic_data.append(
            {
                "topic": topic,
                "user": user,
                "retention": retention
            }
        )

    return render_template(
        "admin_topics.html",
        topic_data=topic_data
    )

# =========================================================
# LOGOUT
# =========================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(
        url_for("login")
    )


# =========================================================
# CREATE DATABASE TABLES
# =========================================================

with app.app_context():

    db.create_all()


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )