import joblib
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_mail import Mail, Message
from datetime import datetime, timedelta
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    logout_user,
    login_required,
    current_user
)
from werkzeug.security import generate_password_hash, check_password_hash




app = Flask(__name__)
# Email Configuration

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True

app.config['MAIL_USERNAME'] = 'your_email@gmail.com'
app.config['MAIL_PASSWORD'] = 'your_app_password'

mail = Mail(app)

# Configuration
app.config['SECRET_KEY'] = 'knowledge_decay_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///knowledge.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Database
db = SQLAlchemy(app)
model = joblib.load(
    "model/retention_model.pkl"
)

# Login Manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
class Topic(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey('user.id'),
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
class Retention(db.Model):

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    topic_id = db.Column(
        db.Integer,
        db.ForeignKey('topic.id'),
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

# User Loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
def calculate_retention(days_passed):

    retention = max(
        100 - (days_passed * 3),
        20
    )

    return retention
def difficulty_to_number(difficulty):

    if difficulty == "Easy":
        return 1

    elif difficulty == "Medium":
        return 2

    else:
        return 3
def predict_retention(topic):

    study_date = datetime.strptime(
        topic.study_date,
        "%Y-%m-%d"
    )

    days_passed = (
        datetime.now() - study_date
    ).days

    difficulty_num = difficulty_to_number(
        topic.difficulty
    )

    prediction = model.predict([
        [
            topic.study_duration,
            difficulty_num,
            topic.confidence_score,
            topic.revision_count,
            days_passed
        ]
    ])

    return round(prediction[0], 2)
def generate_explanation(topic, retention):

    reasons = []

    if topic.confidence_score <= 5:
        reasons.append(
            f"Low confidence score ({topic.confidence_score}/10)"
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


# Home Route
@app.route("/")
def home():
    return redirect(url_for("login"))


# Register Route
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            flash("Email already registered")
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        new_user = User(
            name=name,
            email=email,
            password=hashed_password
        )

        db.session.add(new_user)
        db.session.commit()

        flash("Registration Successful")
        return redirect(url_for("login"))

    return render_template("register.html")


# Login Route
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):

            login_user(user)

            return redirect(url_for("dashboard"))

        flash("Invalid Email or Password")

    return render_template("login.html")


# Dashboard Route
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

        days_passed = (
            today - study_date
        ).days

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
            sum(retention_values) /
            len(retention_values),
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
@app.route("/add_topic", methods=["GET", "POST"])
@login_required
def add_topic():

    if request.method == "POST":

        topic_name = request.form["topic_name"]
        study_date = request.form["study_date"]

        study_duration = float(
            request.form["study_duration"]
        )

        difficulty = request.form["difficulty"]

        confidence_score = int(
            request.form["confidence_score"]
        )

        revision_count = int(
            request.form["revision_count"]
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

        db.session.add(topic)
        db.session.commit()

        flash("Topic Added Successfully")

        return redirect(url_for("dashboard"))

    return render_template("add_topic.html")
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

        days_passed = (
            today - study_date
        ).days

        retention = calculate_retention(
            days_passed
        )

        report.append({

            "topic": topic.topic_name,

            "days": days_passed,

            "retention": retention

        })

    return render_template(
        "retention.html",
        report=report
    )
@app.route("/recommendations")
@login_required
def recommendations():

    topics = Topic.query.filter_by(
        user_id=current_user.id
    ).all()

    recommendations = []

    today = datetime.now()

    for topic in topics:

        study_date = datetime.strptime(
            topic.study_date,
            "%Y-%m-%d"
        )

        days_passed = (
            today - study_date
        ).days

        retention = calculate_retention(
            days_passed
        )

        if retention < 40:

            priority = "Revise Immediately"

        elif retention < 70:

            priority = "Revise Soon"

        else:

            priority = "No Revision Needed"

        recommendations.append({

            "topic": topic.topic_name,

            "retention": retention,

            "priority": priority

        })

    return render_template(
        "recommendations.html",
        recommendations=recommendations
    )
@app.route("/delete_topic/<int:id>")
@login_required
def delete_topic(id):

    topic = Topic.query.get_or_404(id)

    if topic.user_id != current_user.id:
        flash("Unauthorized Access")
        return redirect(url_for("topics"))

    db.session.delete(topic)
    db.session.commit()

    flash("Topic Deleted Successfully")

    return redirect(url_for("topics"))
@app.route("/edit_topic/<int:id>", methods=["GET", "POST"])
@login_required
def edit_topic(id):

    topic = Topic.query.get_or_404(id)

    if topic.user_id != current_user.id:
        flash("Unauthorized Access")
        return redirect(url_for("topics"))
        topic.confidence_score = int(
        request.form["confidence_score"]
        )

        topic.revision_count = int(
        request.form["revision_count"]
        )

    if request.method == "POST":

        topic.topic_name = request.form["topic_name"]
        topic.study_date = request.form["study_date"]
        topic.study_duration = float(
            request.form["study_duration"]
        )
        topic.difficulty = request.form["difficulty"]

        db.session.commit()

        flash("Topic Updated Successfully")

        return redirect(url_for("topics"))

        return render_template(
        "edit_topic.html",
        topic=topic
    )
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

        # Difficulty Count

        if topic.difficulty == "Easy":
            easy += 1

        elif topic.difficulty == "Medium":
            medium += 1

        else:
            hard += 1

        # Retention Calculation

        study_date = datetime.strptime(
            topic.study_date,
            "%Y-%m-%d"
        )

        days_passed = (
            today - study_date
        ).days

        retention = calculate_retention(
            days_passed
        )

        labels.append(
            topic.topic_name
        )

        retentions.append(
            retention
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

    total_topics = len(topics)

    average_retention = (
        round(sum(retentions) / len(retentions), 2)
        if retentions else 0
    )

    high_risk = len([
        r for r in retentions
        if r < 50
    ])

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
   
@app.route("/ai_recommendations")
@login_required
def ai_recommendations():

    topics = Topic.query.filter_by(
        user_id=current_user.id
    ).all()

    predictions = []

    for topic in topics:

        retention = predict_retention(
            topic
        )

        predictions.append({
            "topic": topic.topic_name,
            "retention": retention
        })

    return render_template(
        "ai_recommendations.html",
        predictions=predictions
    )
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

        days_passed = (
            today - study_date
        ).days

        retention = calculate_retention(
            days_passed
        )

        ranked_topics.append({

            "topic_name": topic.topic_name,

            "retention": retention

        })

    ranked_topics.sort(
        key=lambda x: x["retention"]
    )

    return render_template(
        "priority.html",
        ranked_topics=ranked_topics
    )
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

        days_passed = (
            today - study_date
        ).days

        retention = calculate_retention(
            days_passed
        )

        if retention < 40:

            message = (
                "Retention is very low. "
                "Immediate revision is strongly recommended."
            )

        elif retention < 70:

            message = (
                "Knowledge is beginning to fade. "
                "Schedule a revision session soon."
            )

        else:

            message = (
                "Retention remains strong. "
                "No urgent revision is required."
            )

        insights.append({

            "topic": topic.topic_name,

            "message": message

        })

    return render_template(
        "explanations.html",
        insights=insights
    )
@app.route("/revision_schedule")
@login_required
def revision_schedule():

    topics = Topic.query.filter_by(
        user_id=current_user.id
    ).all()

    schedule = []

    today = datetime.now()

    for topic in topics:

        study_date = datetime.strptime(
            topic.study_date,
            "%Y-%m-%d"
        )

        days_passed = (
            today - study_date
        ).days

        retention = calculate_retention(
            days_passed
        )

        if retention < 40:

            next_revision = today.strftime(
                "%Y-%m-%d"
            )

            priority = "High"

        elif retention < 70:

            next_revision = (
                today + timedelta(days=2)
            ).strftime("%Y-%m-%d")

            priority = "Medium"

        else:

            next_revision = (
                today + timedelta(days=7)
            ).strftime("%Y-%m-%d")

            priority = "Low"

        schedule.append({

            "topic": topic.topic_name,

            "retention": retention,

            "revision_date": next_revision,

            "priority": priority

        })

    return render_template(
        "revision_schedule.html",
        schedule=schedule
    )
# Logout Route
@app.route("/logout")
@login_required
def logout():

    logout_user()

    return redirect(url_for("login"))


# Create Database Tables
with app.app_context():
    db.create_all()


# Run App
if __name__ == "__main__":
    app.run(debug=True)