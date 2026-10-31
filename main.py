"""Delivery time API.  Run:  uvicorn main:app --reload"""
from fastapi.staticfiles import StaticFiles
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI

from schemas import DeliveryRequestWrapper, PredictionDeliveryTime

app = FastAPI(title="Delivery Time Predictor")

scaler = joblib.load("standarized_scalar.pkl")
model_pipeline = joblib.load("xgb_model.pkl")

# Test-set MAE of the tuned XGBoost (from the notebook):
#   mean_absolute_error(y_test, xgb_best.predict(x_test_scaled))
MAE = 3.19

# Must be the SAME mapping the model was trained with (matches Delivery_on_the_go_1.ipynb).
TRAFFIC_MAP = {"Low": 0, "Medium": 1, "High": 2, "Jam": 3}

# Rider details a customer doesn't know. Use typical values (df[col].median()) from your data.
DEFAULTS = {"Delivery_person_Age": 30, "Delivery_person_Ratings": 4.6,
            "Vehicle_condition": 1, "multiple_deliveries": 0, "Type_of_vehicle": "motorcycle"}

DROP_KEYS = ("role", "ID", "Delivery_person_ID",
             "Delivery_person_id")  # labels, not features
# adjust to the hours that have enough data in your notebook
BEST_TIME_HOURS = range(8, 23)

# ---- columns the model expects: read from the scaler, never typed by hand ----
EXPECTED = list(scaler.feature_names_in_)
_model_cols = getattr(model_pipeline, "feature_names_in_", None)
if _model_cols is not None and list(_model_cols) != EXPECTED:
    raise RuntimeError("Scaler and model were trained on different columns. "
                       "Re-save both from the same notebook run.")


# ---- derived columns: only built if the model was trained with them ----
def bucket_ratings(r):  # copied from your notebook
    if r < 3.0:
        return "Low (< 3.0)"
    elif r < 4.0:
        return "Medium (3.0 - 3.9)"
    elif r < 4.5:
        return "Good (4.0 - 4.4)"
    return "Excellent (4.5 - 5.0)"


DERIVED = {
    "Rating_Group": lambda row: bucket_ratings(row["Delivery_person_Ratings"]),
    # "Day_time": lambda row: ...paste your notebook's Day_time logic here...
}
for _name in ("Rating_Group", "Day_time"):
    if any(c.startswith(_name) for c in EXPECTED) and _name not in DERIVED:
        raise RuntimeError(f"The model expects '{_name}' but main.py can't build it. "
                           f"Paste the notebook logic into DERIVED, or retrain without it.")
ACTIVE_DERIVED = {n: f for n, f in DERIVED.items() if any(
    c.startswith(n) for c in EXPECTED)}


# ---- the one prediction function every endpoint uses ----
def predictors(inputs: dict) -> float:
    row = {**DEFAULTS, "hour": datetime.now().hour,
           **{k: v for k, v in inputs.items() if v is not None and k not in DROP_KEYS}}
    row["Road_traffic_density"] = TRAFFIC_MAP[row["Road_traffic_density"]]
    for name, fn in ACTIVE_DERIVED.items():
        row[name] = fn(row)

    encoded = pd.get_dummies(pd.DataFrame([row]), dtype=int)
    features = encoded.reindex(columns=EXPECTED, fill_value=0)
    scaled = pd.DataFrame(scaler.transform(features), columns=EXPECTED)
    return float(model_pipeline.predict(scaled)[0])


def result(pred: float, role: str = "customer") -> PredictionDeliveryTime:
    pred = max(pred, 0.0)
    est, lo, hi = round(pred), round(max(pred - MAE, 0.0)), round(pred + MAE)
    who = "This delivery would take you about" if role == "rider" else "Your order should arrive in about"
    return PredictionDeliveryTime(estimated_time=est, min_time=lo, max_time=hi,
                                  display_text=f"{who} {est} min ({lo} to {hi})")


# ---- endpoints (all take {"user_perspective": {"role": "customer" | "rider", ...}}) ----
@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionDeliveryTime)
def predict(body: DeliveryRequestWrapper):
    p = body.user_perspective
    return result(predictors(p.model_dump()), p.role)


@app.post("/predict/best-time")
def best_time(body: DeliveryRequestWrapper):
    base = body.user_perspective.model_dump()
    by_hour = {f"{h:02d}:00": round(max(predictors({**base, "hour": h}), 0.0), 1)
               for h in BEST_TIME_HOURS}
    return {"best_time": min(by_hour, key=by_hour.get), "by_hour": by_hour}


@app.post("/predict/vehicles", response_model=dict[str, PredictionDeliveryTime])
def vehicles(body: DeliveryRequestWrapper):
    p = body.user_perspective
    base = p.model_dump()
    return {v: result(predictors({**base, "Type_of_vehicle": v}), p.role)
            for v in ("motorcycle", "scooter", "electric_scooter", "bicycle")}


# ---- startup self-check: changing a category must change the prediction ----
def _sanity_check():
    base = dict(Distance=5.0, Weatherconditions="Sunny", Road_traffic_density="Medium",
                Type_of_order="Meal", Festival="No", City="Metropolitian",
                Type_of_vehicle="motorcycle")
    variants = {"Weatherconditions": ("Sunny", "Fog", "Stormy", "Cloudy"),
                "Type_of_vehicle": ("motorcycle", "bicycle", "scooter"),
                "Road_traffic_density": ("Low", "Jam"),
                "City": ("Urban", "Metropolitian")}
    for field, values in variants.items():
        outs = {round(predictors({**base, field: v}), 4) for v in values}
        if len(outs) == 1:
            print(f"WARNING: changing {field} never changes the prediction. "
                  f"Check that the column names match the model's.")


_sanity_check()

app.mount("/app", StaticFiles(directory="static", html=True), name="app")
