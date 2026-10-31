# 🍕 Delivery on the Go

A food-delivery time predictor with a cinematic, story-style interface. Pick who you are (a hungry customer or a delivery rider), set the conditions, press **Predict**, and watch the delivery play out on a map before the pizza box reveals your estimated time.

## 🌐 Live demo

**[Open the live app](https://delivery1232344324on12the12go123234432401.streamlit.app/)**

---

## 🤖 What it does

- Predicts the delivery time in minutes from distance, weather, traffic, order type, festival, city type, hour of day and (for riders) age, rating, vehicle and vehicle condition.
- Shows a typical error range around the estimate (the tuned model's test MAE is about 3.2 minutes).
- Suggests the **best hour to order** (the fastest hour between 08:00 and 22:00).
- Shows how the time would change under **Low / Medium / High / Jam** traffic.
- Two views: a **Customer** view on a spacious iPad, and a **Rider** view on a phone with extra rider details and Accept / Decline buttons.

---

## How it works

The model was trained in a Jupyter notebook and exported as two files. The app loads them and runs predictions itself, so no notebook or separate API is needed at runtime.

| File | What it is |
| --- | --- |
| `standarized_scalar.pkl` | The fitted feature scaler (also holds the expected feature names) |
| `xgb_model.pkl` | The tuned XGBoost regressor |
| `streamlit_app.py` | The whole app: prediction logic plus the full interface, embedded as a Streamlit custom component |

For each request the app encodes the inputs the same way as the notebook (traffic: Low=0, Medium=1, High=2, Jam=3; the rest one-hot encoded and aligned to the scaler's columns), scales them, and predicts. The best-time and traffic comparisons re-run the same model with a different hour or traffic level.

---

## 🚀 Getting started (local setup)

1. Put `streamlit_app.py`, `standarized_scalar.pkl` and `xgb_model.pkl` in the same folder.
2. Install the requirements:

   ```bash
   pip install streamlit pandas joblib scikit-learn xgboost
   ```

   *Use the same `scikit-learn` and `xgboost` versions you trained with, otherwise the `.pkl` files may fail to load.*
3. Start the app:

   ```bash
   streamlit run streamlit_app.py
   ```

4. Open http://localhost:8501.

An internet connection is needed for the map library and the real map in the phone/iPad view (Leaflet and OpenStreetMap).

---

## 📌 Notes and limits

- The model only learned deliveries up to about 21 km, so distances are limited to 0.5 to 25 km.
- The map animation is a visual simulation, not live tracking.
- Estimates are a guide with a typical error of about 3 minutes.

---

## 🖥️ Tech stack

- **Streamlit** — frontend / UI
- **XGBoost & scikit-learn** — prediction model
- **Python** — data pipeline and scaling (`streamlit_app.py`, `standarized_scalar.pkl`, `xgb_model.pkl`)
