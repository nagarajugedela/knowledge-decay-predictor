import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Sample Training Data

data = pd.DataFrame({

    "study_duration": [
        1, 2, 3, 4, 2,
        5, 1, 3, 4, 2
    ],

    "difficulty": [
        1, 2, 3, 2, 1,
        3, 2, 1, 3, 2
    ],

    "confidence_score": [
        4, 7, 8, 9, 5,
        10, 3, 6, 8, 7
    ],

    "revision_count": [
        0, 1, 2, 3, 0,
        4, 0, 1, 2, 2
    ],

    "days_passed": [
        15, 10, 5, 2, 20,
        1, 25, 12, 4, 8
    ],

    "retention": [
        35, 55, 80, 92, 30,
        98, 20, 60, 85, 75
    ]
})

X = data[
    [
        "study_duration",
        "difficulty",
        "confidence_score",
        "revision_count",
        "days_passed"
    ]
]

y = data["retention"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

joblib.dump(
    model,
    "model/retention_model.pkl"
)

print("Model Trained Successfully")