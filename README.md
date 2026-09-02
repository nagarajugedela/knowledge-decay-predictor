# Knowledge Decay Predictor

An AI-powered web application that helps students track learning retention, predict forgetting patterns, and generate smart revision schedules.

## Email reminders

The daily reminder job sends an email for every topic whose estimated retention has fallen below 40%. Each alert goes to the affected user and to the monitoring address. A topic is only sent once on a given day, even if the job is run more than once.

1. Copy `.env.example` to `.env` and provide your Resend credentials and `ADMIN_EMAIL`.
2. In Resend, replace `onboarding@resend.dev` in `app.py` with a sender address from your verified domain before sending to real users.
3. Schedule this command to run once a day (for example, with Windows Task Scheduler):

```powershell
flask --app app send-reminders
```

The command prints the number of reminders sent and exits with an error if either email setting is missing.

## Features

* User Registration & Login
* Add Study Topics
* Retention Prediction
* Analytics Dashboard
* AI Study Recommendations
* Priority Ranking System
* Revision Scheduler
* Learning Insights
* Automated revision reminder emails to users and the admin monitor
* Interactive Charts using Chart.js

## Installation

```bash
pip install -r requirements.txt
python app.py
```

## Author

NagaRaju Gedela
B.Tech CSE (AI & ML)