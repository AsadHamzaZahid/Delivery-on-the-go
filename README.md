# 🍕 Delivery on the Go

A food-delivery time predictor with a cinematic, story-style interface. Pick who you are (a hungry customer or a delivery rider), set the conditions, press **Predict**, and watch the delivery play out on a map before the pizza box reveals your estimated time.

## 🔗 Live app

**[ADD YOUR LIVE LINK HERE](#)**

## What it does

- Predicts the delivery time in minutes from distance, weather, traffic, order type, festival, city type, hour of day and (for riders) age, rating, vehicle and vehicle condition.
- Shows a typical error range around the estimate (the tuned model's test MAE is about 3.2 minutes).
- Suggests the **best hour to order** (the fastest hour between 08:00 and 22:00).
- Shows how the time would change under **Low / Medium / High / Jam** traffic.
- Two views: a **Customer** view on a spacious iPad, and a **Rider** view on a phone with extra rider details and Accept / Decline buttons.

## The experience

1. A scooter rides through a night skyline, then you choose **Customer** or **Rider**.
2. A first-person scene plays (waking up hungry, or grabbing your helmet) and the device slides up.
3. Tap the map to place the restaurant and the delivery spot, or type a distance.
4. Press **Predict delivery time**:
   - clouds cover the screen and part to reveal an illustrated town map,
   - the restaurant and home pins drop in,
   - a red dotted line draws between them,
   - a red delivery vehicle (scooter, motorcycle, electric scooter or bicycle, depending on the rider's vehicle) drives the route,
   - a pizza box opens with your delivery time (with confetti for fast deliveries).
5. Press **More options in the app** to see the best-time chart and the traffic comparison.

Extras: a weather sky on the reveal map (sunny, cloudy, fog, stormy, sandstorm, windy), three imaginary towns you can switch between (Sakura Harbor, Cloud Meadow, Crayon Hills), synthesized background music and sound effects, and a mute button.

## How it works

The model was trained in a Jupyter notebook and exported as two files. The app loads them and runs predictions itself, so no notebook or separate API is needed at runtime.

| File | What it is |
| --- | --- |
| `standarized_scalar.pkl` | The fitted feature scaler (also holds the expected feature names) |
| `xgb_model.pkl` | The tuned XGBoost regressor |
| `streamlit_app.py` | The whole app: prediction logic plus the full interface, embedded as a Streamlit custom component |

For each request the app encodes the inputs the same way as the notebook (traffic: Low=0, Medium=1, High=2, Jam=3; the rest one-hot encoded and aligned to the scaler's columns), scales them, and predicts. The best-time and traffic comparisons re-run the same model with a different hour or traffic level.

## Run it locally

1. Put `streamlit_app.py`, `standarized_scalar.pkl` and `xgb_model.pkl` in the same folder.
2. Install the requirements:

   ```bash
   pip install streamlit pandas joblib scikit-learn xgboost
   ```

   Use the same `scikit-learn` and `xgboost` versions you trained with, otherwise the `.pkl` files may fail to load.
3. Start the app:

   ```bash
   streamlit run streamlit_app.py
   ```

4. Open http://localhost:8501.

An internet connection is needed for the map library and the real map in the phone/iPad view (Leaflet and OpenStreetMap).

## Notes and limits

- The model only learned deliveries up to about 21 km, so distances are limited to 0.5 to 25 km.
- The map animation is a visual simulation, not live tracking.
- Estimates are a guide with a typical error of about 3 minutes.

## Tech

Python, Streamlit, XGBoost, scikit-learn, pandas, joblib. Front end in plain HTML/CSS/JavaScript with Leaflet, drawn and animated in code, and sounds synthesized with the Web Audio API.
