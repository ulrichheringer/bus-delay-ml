"""Bus-delay regression with a chronological holdout over synthetic service days."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
NUMERIC = ["hour", "weekday", "rain_mm", "traffic_index", "passenger_load", "scheduled_headway_min", "route_distance_km"]
FEATURES = ["route"] + NUMERIC


def _dataset(seed=SEED, rows=2400):
    rng = np.random.default_rng(seed)
    day = np.arange(rows)
    hour = rng.integers(5, 24, rows)
    weekday = day % 7
    rush = np.isin(hour, [7, 8, 9, 16, 17, 18]).astype(float)
    traffic = np.clip(rng.normal(0.42 + 0.48 * rush, 0.18, rows), 0, 1.4)
    route = rng.choice(["R1", "R2", "R3", "R4", "R5", "R6"], rows)
    rain = np.clip(rng.gamma(1.2, 3.0, rows), 0, 35)
    passengers = rng.integers(5, 91, rows)
    headway = rng.integers(6, 31, rows)
    distance = rng.uniform(4, 38, rows)
    X = pd.DataFrame({"route": route, "hour": hour, "weekday": weekday, "rain_mm": rain,
                      "traffic_index": traffic, "passenger_load": passengers,
                      "scheduled_headway_min": headway, "route_distance_km": distance})
    route_effect = pd.Series(route).map({"R1": 0, "R2": 1.5, "R3": 3, "R4": -0.5, "R5": 2, "R6": 4}).to_numpy()
    delay = (-2 + 5.2 * traffic + 0.32 * rain + 0.065 * passengers + 0.18 * headway
             + 0.12 * distance + 5.5 * rush + route_effect + 0.45 * (weekday == 4)
             + rng.normal(0, 2.4, rows))
    return X, np.maximum(0, delay)


def _model(seed=SEED):
    prep = ColumnTransformer([
        ("route", OneHotEncoder(handle_unknown="ignore"), ["route"]),
        ("numeric", StandardScaler(), NUMERIC),
    ])
    return make_pipeline(prep, RandomForestRegressor(n_estimators=240, min_samples_leaf=3, max_features=0.9,
                                                     random_state=seed, n_jobs=1))


def sample_input():
    return {"route": "R3", "hour": 17, "weekday": 2, "rain_mm": 4.5, "traffic_index": 0.93,
            "passenger_load": 72, "scheduled_headway_min": 15, "route_distance_km": 19.2}


def train_model(seed=SEED):
    X, y = _dataset(seed)
    return _model(seed).fit(X, y)


def predict(model, payload):
    row = pd.DataFrame([payload], columns=FEATURES)
    return {"predicted_delay_minutes": round(max(0.0, float(model.predict(row)[0])), 2)}


def evaluate(seed=SEED):
    X, y = _dataset(seed)
    split = int(0.8 * len(y))
    model = _model(seed).fit(X.iloc[:split], y[:split])
    pred = model.predict(X.iloc[split:])
    return {"mae_minutes": round(float(mean_absolute_error(y[split:], pred)), 4),
            "rmse_minutes": round(float(np.sqrt(mean_squared_error(y[split:], pred))), 4),
            "r2": round(float(r2_score(y[split:], pred)), 4),
            "train_rows": split, "test_rows": int(len(y) - split), "holdout": "last 20% of generated service days"}
