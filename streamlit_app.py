"""Delivery on the Go: Streamlit app. The full cinematic UI is embedded as a custom component.
Run:  streamlit run streamlit_app.py     (needs standarized_scalar.pkl and xgb_model.pkl in the same folder)"""
import tempfile
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Delivery on the Go", page_icon="🍕",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
header[data-testid=stHeader],[data-testid=stToolbar],[data-testid=stDecoration],[data-testid=stStatusWidget],[data-testid=stSpinner],footer{display:none!important}
.stApp{background:#000}
.block-container{padding:0!important;max-width:100%!important}
iframe{position:fixed!important;top:0;left:0;width:100vw!important;height:100vh!important;border:0}
</style>""", unsafe_allow_html=True)

# test MAE of the tuned XGBoost (minutes)
MAE = 3.19
TRAFFIC_MAP = {"Low": 0, "Medium": 1, "High": 2,
               "Jam": 3}    # same as the notebook
DEFAULTS = {"Delivery_person_Age": 30, "Delivery_person_Ratings": 4.6, "Vehicle_condition": 1,
            "multiple_deliveries": 0, "Type_of_vehicle": "motorcycle"}
BEST_HOURS = range(8, 23)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("standarized_scalar.pkl"), joblib.load("xgb_model.pkl")


try:
    scaler, model = load_model()
except FileNotFoundError:
    st.error("Put standarized_scalar.pkl and xgb_model.pkl next to streamlit_app.py.")
    st.stop()
EXPECTED = list(scaler.feature_names_in_)


def predict(inputs: dict) -> float:
    row = {**DEFAULTS, **inputs}
    row["Road_traffic_density"] = TRAFFIC_MAP[row["Road_traffic_density"]]
    enc = pd.get_dummies(pd.DataFrame([row]), dtype=int).reindex(
        columns=EXPECTED, fill_value=0)
    scaled = pd.DataFrame(scaler.transform(enc), columns=EXPECTED)
    return max(float(model.predict(scaled)[0]), 0.0)


def run_request(req: dict) -> dict:
    """Same shapes the FastAPI endpoints returned, so the page code stays unchanged."""
    p = {k: v for k, v in req.items() if k not in ("role", "id")}
    who = "This delivery would take you about" if req[
        "role"] == "rider" else "Your order should arrive in about"

    def pack(v):
        est, lo, hi = round(v), max(round(v - MAE), 0), round(v + MAE)
        return dict(estimated_time=est, min_time=lo, max_time=hi, display_text=f"{who} {est} min ({lo} to {hi})")

    by_hour = {f"{h:02d}:00": round(
        predict({**p, "hour": h}), 1) for h in BEST_HOURS}
    return dict(main=pack(predict(p)),
                best=dict(best_time=min(by_hour, key=by_hour.get),
                          by_hour=by_hour),
                traffic=[pack(predict({**p, "Road_traffic_density": t})) for t in TRAFFIC_MAP])


# ------------------------------------------------------------------ the page (HTML, CSS, JS)
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Delivery on the Go</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.css">
<style>
*{box-sizing:border-box;margin:0}
html,body{height:100%;overflow:hidden;background:#000;font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
[hidden]{display:none!important}
.full{position:fixed;inset:0}
@keyframes slide{to{transform:translateX(-50%)}}
@keyframes dash{to{transform:translateX(-140px)}}
@keyframes bob{from{transform:scaleX(-1) translateY(0)}to{transform:scaleX(-1) translateY(-6px)}}
@keyframes fade{from{opacity:0}to{opacity:1}}

/* ---------- intro: scooter riding down the road ---------- */
#intro{background:linear-gradient(#1b1038,#6a2c5a 55%,#f08a4b);overflow:hidden;z-index:1}
.sun{position:absolute;left:50%;bottom:30%;width:240px;height:240px;margin-left:-120px;border-radius:50%;background:#ffd27a;box-shadow:0 0 130px #ff9a4d}
#skyline{position:absolute;left:0;bottom:30%;display:flex;align-items:flex-end;width:max-content;animation:slide 50s linear infinite}
#skyline i{display:block;margin-right:4px;background-color:#1a1033;background-image:radial-gradient(#ffd27a 1.5px,transparent 2px);background-size:14px 18px}
.road{position:absolute;left:0;right:0;bottom:0;height:30%;background:linear-gradient(#26202e,#120f18);overflow:hidden}
.dash{position:absolute;top:50%;left:0;width:calc(100% + 140px);height:6px;background:repeating-linear-gradient(90deg,#f5d76e 0 70px,transparent 70px 140px);animation:dash .5s linear infinite}
.scooter{position:absolute;left:50%;bottom:26%;margin-left:-50px;font-size:100px;animation:bob .4s ease-in-out infinite alternate}
.scooter span{position:absolute;left:6px;top:-26px;font-size:42px}
#intro h1{position:absolute;top:9%;width:100%;text-align:center;color:#fff;font:800 clamp(28px,6vw,60px) Georgia,serif;letter-spacing:2px;text-shadow:0 4px 20px #000}
#choice{z-index:5;background:rgba(10,5,25,.55);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:26px;animation:fade 1s}
#choice h2{color:#fff;font:700 clamp(22px,4vw,36px) Georgia,serif}
#choice div{display:flex;gap:20px;flex-wrap:wrap;justify-content:center}
#choice button{width:230px;padding:22px 10px;border:0;border-radius:18px;background:#fff;font-size:24px;font-weight:700;cursor:pointer;transition:.2s}
#choice button:hover{transform:translateY(-6px) scale(1.04);background:#ffd27a}
#choice small{display:block;font-size:13px;font-weight:500;color:#666;margin-top:4px}

/* ---------- first-person scenes ---------- */
#pov{z-index:2;background:#000;overflow:hidden}
#room{position:absolute;inset:0;opacity:0;transition:opacity 1.2s}
.open #room{opacity:1}
.customer #room{filter:blur(18px)}
.customer.open #room{filter:blur(0);transition:filter 3s,opacity 1s}
.customer #gar,.rider #bed{display:none}
#bed,#gar{position:absolute;inset:0}
#bed{background:linear-gradient(#2b2547,#1a1630)}
.win{position:absolute;left:12%;top:12%;width:30%;height:42%;background:linear-gradient(#ffd9a0,#ff9f6b);border:10px solid #120f22;box-shadow:0 0 90px 30px rgba(255,170,100,.45)}
.stand{position:absolute;right:10%;bottom:28%;width:14%;height:16%;background:#3a2a22;font-size:48px;text-align:center;border-radius:6px}
.blanket{position:absolute;left:-5%;right:-5%;bottom:-8%;height:38%;background:linear-gradient(#4a5aa8,#2f3a78);border-radius:50% 50% 0 0/30% 30% 0 0}
.lid{position:absolute;left:0;right:0;height:51%;background:#000;z-index:3}
.lid.t{top:0;border-radius:0 0 50% 50%/0 0 14% 14%}
.lid.b{bottom:0;border-radius:50% 50% 0 0/14% 14% 0 0}
.rider .lid{display:none}
@keyframes lidT{0%{transform:translateY(0)}25%{transform:translateY(-55%)}40%{transform:translateY(-12%)}100%{transform:translateY(-105%)}}
@keyframes lidB{0%{transform:translateY(0)}25%{transform:translateY(55%)}40%{transform:translateY(12%)}100%{transform:translateY(105%)}}
.customer.open .lid.t{animation:lidT 3s ease-out forwards}
.customer.open .lid.b{animation:lidB 3s ease-out forwards}
#gar{background:linear-gradient(90deg,#4a4d58,#2a2c34 60%,#1c1d23)}
.shelf{position:absolute;left:25%;width:50%;top:58%;height:14px;background:#7a5230;box-shadow:0 12px 24px rgba(0,0,0,.6)}
#helmet{position:absolute;left:50%;top:58%;width:160px;margin:-112px 0 0 -80px;transition:transform 1.3s ease-in,opacity .8s}
#hand{position:absolute;left:50%;margin-left:-45px;bottom:-52vh;width:90px;height:46vh;border-radius:45px 45px 0 0;background:linear-gradient(#e0ac88,#c98f6b);box-shadow:0 0 0 18px #1d2230;transition:transform 1.3s ease-in-out,opacity .8s}
.reach #hand{transform:translateY(-54vh)}
.grab #helmet{transform:translateY(34vh) scale(3.6);opacity:0}
.grab #hand{transform:translateY(-26vh) scale(1.5);opacity:0}
#visor{position:absolute;inset:0;pointer-events:none;opacity:0;transition:opacity 1s;background:radial-gradient(ellipse at center,rgba(40,60,90,.1) 45%,rgba(0,0,0,.9) 100%);box-shadow:inset 0 0 0 16px #000;z-index:2}
.wear #visor{opacity:1}
#caption{position:absolute;bottom:14%;left:0;right:0;text-align:center;color:#fff;font:600 clamp(18px,3vw,26px) Georgia,serif;text-shadow:0 2px 10px #000;opacity:0;transition:opacity .9s;z-index:4}
.cap #caption{opacity:1}


/* ---------- the phone ---------- */
#phoneWrap{z-index:10;display:grid;place-items:center;pointer-events:none}
.rig{position:relative;transform:translateY(125%) rotate(6deg);transition:transform 1.2s cubic-bezier(.2,.8,.2,1)}
.up .rig{transform:none}
.hand{position:absolute;bottom:50px;width:74px;height:210px;border-radius:38px;background:linear-gradient(#e0ac88,#c98f6b)}
.hand.l{left:-36px;transform:rotate(8deg)}
.hand.r{right:-36px;transform:rotate(-8deg)}
#phone{position:relative;z-index:1;pointer-events:auto;width:min(490px,94vw);height:min(90vh,840px);border:11px solid #0b0b10;border-radius:40px;background:#fbf6ee;overflow:hidden;box-shadow:0 30px 80px rgba(0,0,0,.7)}
#screen{height:100%;overflow-y:auto}
#top{position:sticky;top:0;z-index:500;background:#fbf6ee}
#bar{display:flex;justify-content:space-between;align-items:center;padding:14px 20px;font-size:16px;font-weight:800;color:#c2410c}
#bar button{border:0;background:none;color:#888;cursor:pointer;font-size:13px}
#map{height:230px}
.pad{padding:16px 22px 30px;color:#2b2118}
#hint{font-size:14px;color:#7a6a5a;margin-bottom:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
label{display:block;font-size:13px;font-weight:700;color:#7a6a5a}
select,input{width:100%;margin-top:5px;padding:11px;border:1px solid #e3d6c4;border-radius:12px;background:#fff;font-size:15px}
h4{margin:20px 0 8px;font-size:14px;color:#c2410c;text-transform:uppercase;letter-spacing:1px}
#go{width:100%;margin-top:22px;padding:17px;border:0;border-radius:14px;background:#e8501a;color:#fff;font-size:18px;font-weight:800;cursor:pointer}
#go:disabled{opacity:.6}
#err{color:#b91c1c;font-size:13px;margin-top:8px}
.big{text-align:center;margin:18px 0 4px;font-size:64px;font-weight:900;color:#e8501a;line-height:1}
.big small{font-size:22px;color:#2b2118}
.sub{text-align:center;color:#555;font-size:14px}
.acts{display:flex;gap:10px;margin-top:12px}
.acts button{flex:1;padding:12px;border:0;border-radius:12px;font-weight:700;cursor:pointer}
.ok{background:#16a34a;color:#fff}.no{background:#eee}
.bars{display:flex;align-items:flex-end;gap:3px;height:110px;margin-top:6px}
.bar{flex:1;height:100%;display:flex;flex-direction:column;justify-content:flex-end;align-items:center;font-size:9px;color:#7a6a5a}
.bar i{display:block;width:100%;background:#f3b48f;border-radius:4px 4px 0 0}
.bar.best i{background:#16a34a}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
.cards div{background:#fff;border:1px solid #e3d6c4;border-radius:10px;padding:8px 2px;text-align:center;font-size:11px;color:#7a6a5a}
.cards b{display:block;font-size:18px;color:#2b2118}
.cards .on{border-color:#e8501a;background:#fff1e8}
.note{font-size:11px;color:#9a8a7a;margin-top:6px}

/* ---------- reveal: map comes down, pins drop, line draws, pizza box opens ---------- */
#reveal{position:fixed;inset:0;z-index:30;background:#dfe6e0;overflow:hidden;transform:translateY(-102%);transition:transform 1s cubic-bezier(.2,.8,.2,1)}
#reveal.down{transform:none}
#map2{position:absolute;inset:0}
.back{position:absolute;top:14px;right:14px;z-index:1100;border:0;border-radius:20px;padding:8px 16px;background:#fff;font-weight:700;cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,.3)}
@keyframes drop{from{transform:translateY(-140px);opacity:0}to{transform:none;opacity:1}}
.pw{animation:drop .7s cubic-bezier(.3,1.5,.5,1) both}
.rpin{width:44px;height:44px;border-radius:50% 50% 50% 0;transform:rotate(-45deg);background:#e11d2e;border:3px solid #fff;box-shadow:0 4px 10px rgba(0,0,0,.45);display:grid;place-items:center}
.rpin span{transform:rotate(45deg);font-size:20px;line-height:1}
#box{position:absolute;left:0;right:0;bottom:4%;z-index:1100;display:flex;flex-direction:column;align-items:center;perspective:900px}
.pbox{position:relative;width:min(300px,84vw);height:180px;opacity:0;transform:translateY(120px) scale(.3);transition:transform .8s cubic-bezier(.2,1.4,.4,1),opacity .4s}
#box.in .pbox{opacity:1;transform:none}
.pbase{position:absolute;inset:0;background:#b9783b;border-radius:10px;box-shadow:0 22px 40px rgba(0,0,0,.45);display:grid;place-items:center}
.pz{width:150px;height:150px;border-radius:50%;display:grid;place-items:center;background:radial-gradient(circle at 30% 35%,#c0392b 0 9px,transparent 10px),radial-gradient(circle at 70% 28%,#c0392b 0 9px,transparent 10px),radial-gradient(circle at 58% 72%,#c0392b 0 9px,transparent 10px),radial-gradient(circle at 24% 68%,#c0392b 0 9px,transparent 10px),radial-gradient(circle,#ffd25e 0 62%,#e0a030 63% 100%)}
.tm{width:106px;height:106px;border-radius:50%;background:rgba(255,248,230,.94);display:grid;place-content:center;text-align:center;color:#7a1f0c}
.tm b{font-size:42px;line-height:1}.tm small{font-size:13px;font-weight:700}.tm em{font-size:11px;font-style:normal;color:#8a5a3a}
.plid{position:absolute;inset:0;border-radius:10px;background:#e3a863;box-shadow:inset 0 0 0 6px #d09550;transform-origin:top;transform-style:preserve-3d;transition:transform 1.1s ease-in-out}
.plid .logo{position:absolute;inset:0;display:grid;place-items:center;text-align:center;font:800 22px Georgia,serif;color:#8a3a12;backface-visibility:hidden}
#box.open .plid{transform:rotateX(-168deg)}
#boxtxt{margin-top:16px;padding:6px 14px;border-radius:14px;background:rgba(255,255,255,.93);font-weight:700;color:#2b2118;text-align:center;opacity:0;transition:opacity .6s .9s}
#more{margin-top:8px;border:0;border-radius:14px;padding:11px 18px;background:#e8501a;color:#fff;font-weight:800;cursor:pointer;opacity:0;transition:opacity .6s 1.1s}
#box.open #boxtxt,#box.open #more{opacity:1}

#confetti{position:fixed;inset:0;width:100%;height:100%;z-index:1200;pointer-events:none}

/* ---------- customer's iPad: wide, spacious, bold ---------- */
#phone.ipad{width:min(980px,96vw);height:min(92vh,860px);border:16px solid #15151b;border-radius:34px;box-shadow:0 30px 90px rgba(0,0,0,.75),inset 0 0 0 2px #3a3a46}
.ipad #bar{padding:18px 32px;font-size:21px}
.ipad #bar button{font-size:15px}
.ipad #map{height:320px}
.ipad .pad{padding:26px 38px 44px}
.ipad #hint{font-size:17px;margin-bottom:18px}
.ipad .grid{grid-template-columns:repeat(3,1fr);gap:22px}
.ipad label{font-size:15px;font-weight:800;color:#5a4a3a}
.ipad select,.ipad input{padding:15px;font-size:18px;border-radius:14px;margin-top:7px;border:2px solid #e3d6c4}
.ipad h4{font-size:16px;margin:28px 0 12px}
.ipad #go{padding:21px;font-size:23px;border-radius:18px;margin-top:30px}
.ipad .big{font-size:96px}.ipad .big small{font-size:32px}.ipad .sub{font-size:19px}
.ipad .bars{height:170px;gap:5px}.ipad .bar{font-size:12px}
.ipad .cards{gap:14px}.ipad .cards div{font-size:15px;padding:16px 4px}.ipad .cards b{font-size:30px}
.ipad .note{font-size:13px}.ipad .acts button{padding:16px;font-size:16px}
@media(max-width:760px){.ipad .grid{grid-template-columns:1fr 1fr}}

/* ---------- reveal sky (weather) + vehicle bounce ---------- */
#sky{position:absolute;left:0;right:0;top:0;height:62%}
#map2.leaflet-container{background:transparent;-webkit-mask-image:linear-gradient(to bottom,rgba(0,0,0,.5),#000 26%);mask-image:linear-gradient(to bottom,rgba(0,0,0,.5),#000 26%)}
#wxfx{position:absolute;left:0;top:0;width:100%;height:50%;z-index:500;pointer-events:none}
#wxtag{position:absolute;top:14px;left:50%;transform:translateX(-50%);z-index:1000;padding:6px 16px;border-radius:16px;background:rgba(255,255,255,.92);font:700 14px Georgia,serif;color:#2b2118;pointer-events:none}
#cloudwipe{position:fixed;inset:0;z-index:40;pointer-events:none;overflow:hidden;background:linear-gradient(#a9d2f5,#e6f2fc);opacity:0;transition:opacity .9s .3s}
#cloudwipe.in{opacity:1;transition:opacity .5s}
#cloudwipe .cl{position:absolute;width:46vw;height:44vh;border-radius:50%;background:radial-gradient(closest-side,#fff 55%,rgba(255,255,255,0));transition:transform .9s cubic-bezier(.3,.7,.3,1)}
#cloudwipe .cl.l{transform:translateX(-150vw)}
#cloudwipe .cl.r{transform:translateX(150vw)}
#cloudwipe.in .cl{transform:none}
@keyframes bobv{from{transform:translateY(0)}to{transform:translateY(-2px)}}
.bobv{animation:bobv .22s ease-in-out infinite alternate}
/* ---------- sound toggle ---------- */
#mute{position:fixed;left:12px;bottom:12px;z-index:2000;width:42px;height:42px;border:0;border-radius:50%;background:rgba(255,255,255,.92);font-size:20px;cursor:pointer;box-shadow:0 2px 10px rgba(0,0,0,.4)}

/* ---------- imaginary towns (original designs): tinted map + scattered decorations ---------- */
.town-sakura .leaflet-tile-pane{filter:sepia(.55) hue-rotate(290deg) saturate(1.7) brightness(1.04)}
.town-cloud .leaflet-tile-pane{filter:hue-rotate(165deg) saturate(1.5) brightness(1.1)}
.town-crayon .leaflet-tile-pane{filter:sepia(.7) hue-rotate(35deg) saturate(2) brightness(1.06)}
.deco{font-size:22px;line-height:26px;text-align:center;pointer-events:none;filter:drop-shadow(0 2px 2px rgba(0,0,0,.35))}
.deco.big{font-size:30px}
.townname{position:absolute;left:10px;bottom:10px;z-index:1000;padding:4px 10px;border-radius:12px;background:rgba(255,255,255,.92);font:700 12px Georgia,serif;color:#2b2118;pointer-events:none}
#reveal .townname{top:14px;bottom:auto;font-size:15px;padding:7px 14px}
</style>
</head>
<body>

<!-- 1. pizza scooter riding down the road -->
<div id="intro" class="full">
  <div class="sun"></div>
  <div id="skyline"></div>
  <div class="road"><div class="dash"></div></div>
  <div class="scooter">🛵<span>🍕</span></div>
  <h1>Delivery on the Go</h1>
</div>

<!-- 2. choose a side after 2 seconds -->
<div id="choice" class="full" hidden>
  <h2>Who are you tonight?</h2>
  <div>
    <button data-r="customer">🛋️ I am hungry<small>Customer</small></button>
    <button data-r="rider">🏍️ I deliver<small>Rider</small></button>
  </div>
</div>

<!-- 3. first-person scenes -->
<div id="pov" class="full" hidden>
  <div id="room">
    <div id="bed"><div class="win"></div><div class="stand">🍕</div><div class="blanket"></div></div>
    <div id="gar">
      <div class="shelf"></div>
      <svg id="helmet" viewBox="0 0 120 100"><path d="M8 64C8 26 34 8 64 8s48 20 48 56v20H68l-6-12H8z" fill="#d6371d"/><rect x="50" y="36" width="64" height="26" rx="10" fill="#10151f"/><rect x="8" y="26" width="104" height="7" fill="#fff" opacity=".85" transform="rotate(-8 60 30)"/></svg>
      <div id="hand"></div>
    </div>
  </div>
  <div id="visor"></div>
  <div class="lid t"></div><div class="lid b"></div>
  <div id="caption"></div>
</div>


<!-- 4. the phone -->
<div id="phoneWrap" class="full" hidden>
 <div class="rig">
  <div class="hand l"></div><div class="hand r"></div>
  <div id="phone"><div id="screen">
    <div id="top">
      <div id="bar"><span>🍕 Delivery on the Go · <span id="who"></span></span><button onclick="location.reload()">↺ switch</button></div>
      <div id="map"></div>
    </div>
    <div class="pad">
      <p id="hint">Tap the map: first the restaurant 🍕, then the delivery spot 🏠.</p>
      <label>Distance (km, 0.5 to 25)<input id="dist" type="number" min="0.5" max="25" step="0.1" value="5"></label>
      <h4>Conditions</h4>
      <div class="grid">
        <label>Weather<select id="weather"><option>Sunny</option><option>Cloudy</option><option>Fog</option><option>Stormy</option><option>Sandstorms</option><option>Windy</option></select></label>
        <label>Traffic<select id="traffic"><option>Low</option><option selected>Medium</option><option>High</option><option>Jam</option></select></label>
        <label>Order<select id="otype"><option>Meal</option><option>Snack</option><option>Drinks</option><option>Buffet</option></select></label>
        <label>Festival<select id="fest"><option>No</option><option>Yes</option></select></label>
        <label>City<select id="city"><option value="Metropolitian">Metropolitan</option><option>Urban</option><option>Semi-Urban</option></select></label>
        <label>Order time<select id="hour"><option value="">Now</option></select></label>
        <label>Town<select id="townSel"><option value="0">🌸 Sakura Harbor</option><option value="1">☁️ Cloud Meadow</option><option value="2">🖍️ Crayon Hills</option></select></label>
      </div>
      <div id="riderOnly" hidden>
        <h4>Your details</h4>
        <div class="grid">
          <label>Age<input id="age" type="number" min="17" max="60" value="29"></label>
          <label>Rating (1 to 5)<input id="rating" type="number" min="1" max="5" step="0.1" value="4.6"></label>
          <label>Vehicle condition<select id="vcond"><option>0</option><option selected>1</option><option>2</option><option>3</option></select></label>
          <label>Orders carrying<select id="multi"><option>0</option><option>1</option><option>2</option><option>3</option></select></label>
          <label>Vehicle<select id="veh"><option>motorcycle</option><option>scooter</option><option>electric_scooter</option><option>bicycle</option></select></label>
        </div>
      </div>
      <button id="go">Predict delivery time</button>
      <p id="err"></p>
      <div id="result" hidden></div>
    </div>
  </div></div>
 </div>
</div>

<!-- 5. map slides down after Predict -->
<div id="reveal" hidden>
  <div id="sky"></div>
  <div id="map2"></div>
  <canvas id="wxfx"></canvas>
  <div id="wxtag"></div>
  <button class="back" id="back">↑ Back</button>
  <div id="box" hidden>
    <div class="pbox">
      <div class="pbase"><div class="pz"><div class="tm"><b id="tm"></b><small>min</small><em id="rg"></em></div></div></div>
      <div class="plid"><div class="logo">🍕<br>Delivery<br>on the Go</div></div>
    </div>
    <p id="boxtxt"></p>
    <button id="more">More options in the app ↓</button>
  </div>
</div>

<div id="cloudwipe"></div>
<canvas id="confetti"></canvas>
<button id="mute" title="Sound on/off">🔊</button>

<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.js"></script>
<script>
const $ = s => document.querySelector(s), T = [];
const later = (ms, fn) => T.push(setTimeout(fn, ms));

// ---------- sound: everything is synthesized (no audio files). Starts on the first click. ----------
let ac, master, musicTimer, nextBar = 0, muted = false, noiseBuf;
function initAudio() {
  if (ac) { if (ac.state === 'suspended') ac.resume(); return; }
  const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return;
  ac = new AC(); master = ac.createGain(); master.gain.value = muted ? 0 : .9; master.connect(ac.destination);
  noiseBuf = ac.createBuffer(1, ac.sampleRate * 2, ac.sampleRate);
  const d = noiseBuf.getChannelData(0); for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  nextBar = ac.currentTime + .1; scheduleMusic(); musicTimer = setInterval(scheduleMusic, 400);
}
function tone(f, t, dur, type = 'sine', vol = .15, f2 = null) {
  if (!ac) return;
  const o = ac.createOscillator(), g = ac.createGain();
  o.type = type; o.frequency.setValueAtTime(f, t);
  if (f2) o.frequency.exponentialRampToValueAtTime(f2, t + dur);
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(vol, t + .015);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g); g.connect(master); o.start(t); o.stop(t + dur + .05);
}
function noise(t, dur, f1, f2, vol = .25) {
  if (!ac) return;
  const s = ac.createBufferSource(), fl = ac.createBiquadFilter(), g = ac.createGain();
  s.buffer = noiseBuf; fl.type = 'bandpass'; fl.Q.value = 1.2;
  fl.frequency.setValueAtTime(f1, t); fl.frequency.exponentialRampToValueAtTime(f2, t + dur);
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(vol, t + dur * .35);
  g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  s.connect(fl); fl.connect(g); g.connect(master); s.start(t); s.stop(t + dur + .05);
}
// soft background loop: C - Am - F - G, gentle arpeggio + bass + pad
const CH = [[261.6, 329.6, 392], [220, 261.6, 329.6], [174.6, 220, 261.6], [196, 246.9, 293.7]];
function scheduleMusic() {
  if (!ac) return;
  const beat = .42;
  while (nextBar < ac.currentTime + 1.2) {
    const bar = Math.round(nextBar / (beat * 8)) % 4, c = CH[bar];
    tone(c[0] / 2, nextBar, beat * 7.5, 'triangle', .07);
    c.forEach(f => tone(f, nextBar, beat * 7.5, 'sine', .018));
    for (let i = 0; i < 8; i++) {
      const f = c[[0, 1, 2, 1, 0, 2, 1, 2][i]] * 2;
      tone(f, nextBar + i * beat, beat * 1.4, 'sine', .05);
    }
    nextBar += beat * 8;
  }
}
const sfx = {
  click() { initAudio(); if (ac) tone(660, ac.currentTime, .1, 'square', .05, 990); },
  whoosh() { if (ac) noise(ac.currentTime, 1, 300, 3200, .3); },
  whooshUp() { if (ac) noise(ac.currentTime, .7, 3000, 300, .22); },
  pin(delay = 0) { if (!ac) return; const t = ac.currentTime + delay; tone(180, t, .18, 'sine', .35, 55); tone(520, t + .02, .15, 'triangle', .12, 880); },
  line(ms) {                                       // rising ticks while the red line is drawn
    if (!ac) return; const t0 = ac.currentTime, n = Math.floor(ms / 110);
    for (let i = 0; i < n; i++) tone(500 + i * 28, t0 + i * .11, .05, 'square', .035);
    tone(220, t0, ms / 1000, 'sawtooth', .02, 880);
  },
  boxPop() { if (!ac) return; const t = ac.currentTime; tone(160, t, .35, 'sine', .3, 520); tone(330, t + .12, .3, 'triangle', .1, 660); },
  boxOpen() {
    if (!ac) return; const t = ac.currentTime;
    tone(140, t, .8, 'sawtooth', .05, 260);                       // lid creak
    noise(t, .8, 500, 1800, .08);
    [784, 988, 1175, 1568].forEach((f, i) => tone(f, t + .75 + i * .09, .7, 'sine', .12));   // ta-da chime
    tone(2093, t + 1.15, .9, 'sine', .05);
  },
  done() { if (ac) { const t = ac.currentTime; tone(880, t, .25, 'sine', .1); tone(1318, t + .1, .35, 'sine', .1); } },
  ride(ms, kind) {
    if (!ac) return; const t0 = ac.currentTime;
    if (kind === 'bicycle') {                      // bell ding-ding, then light chain ticks
      tone(2400, t0, .25, 'sine', .08); tone(2400, t0 + .2, .3, 'sine', .08);
      for (let i = 0; i < Math.floor(ms / 70); i++) tone(1500 + (i % 3) * 60, t0 + .5 + i * .07, .02, 'square', .012);
      return;
    }
    const n = Math.floor(ms / 95), base = kind === 'motorcycle' ? 55 : 70;   // engine putt-putt
    for (let i = 0; i < n; i++) tone(base + (i % 2) * 14, t0 + i * .095, .07, 'sawtooth', .035);
  },
  thunder() { if (!ac) return; const t = ac.currentTime + .15; noise(t, 1.8, 180, 50, .5); tone(55, t, 1.5, 'sine', .3, 35); },
  party() { if (!ac) return; const t = ac.currentTime; noise(t, .25, 2000, 6000, .2); [523, 659, 784, 1047, 1319].forEach((f, i) => tone(f, t + i * .07, .4, 'triangle', .1)); }
};
$('#mute').onclick = () => {
  initAudio(); muted = !muted; $('#mute').textContent = muted ? '🔇' : '🔊';
  if (master) master.gain.setTargetAtTime(muted ? 0 : .9, ac.currentTime, .05);
};
// every real click starts/resumes audio (browsers require a user gesture)
document.addEventListener('click', e => { initAudio(); if (e.target.closest('button') && e.target.id !== 'mute') sfx.click(); }, true);

// ---------- imaginary towns (original): themes tint the map and sprinkle decorations ----------
const TOWNS = [
  { cls: 'town-sakura', name: '🌸 Sakura Harbor', big: ['⛩️', '🏮', '🌸'], small: ['🌸', '🌳', '🏠', '🎏', '🍡', '🏮', '🌉'],
    pal: { grass: '#fbd7e4', park: '#c9ecb4', road: '#b4a9b8', line: '#fff3c4', walk: '#fff0f5', win: 'rgba(255,255,255,.8)', water: '#81d4fa',
      trees: ['#f48fb1', '#ec407a', '#f8bbd0', '#81c784'], bld: ['#ffffff', '#ffcdd2', '#f8bbd0', '#d1c4e9', '#b2ebf2', '#ffe0b2'], roofs: ['#c62828', '#ad1457', '#6a1b9a'] } },
  { cls: 'town-cloud', name: '☁️ Cloud Meadow', big: ['🎡', '🏰', '🎈'], small: ['☁️', '🌷', '🌳', '🏡', '🪁', '🎈', '🌈'],
    pal: { grass: '#d3f2e0', park: '#a5e3b5', road: '#aeb7c4', line: '#ffffff', walk: '#f2fbff', win: 'rgba(255,255,255,.8)', water: '#4fc3f7',
      trees: ['#66bb6a', '#4db6ac', '#9ccc65'], bld: ['#ffffff', '#e3f2fd', '#bbdefb', '#fff9c4', '#e1bee7', '#ffccbc'], roofs: ['#1e88e5', '#5c6bc0', '#26a69a'] } },
  { cls: 'town-crayon', name: '🖍️ Crayon Hills', big: ['🏫', '⛲', '🎪'], small: ['🌻', '🌳', '🏘️', '🚲', '🍎', '🌼', '🐓'],
    pal: { grass: '#b5ef3c', park: '#8fdc3c', road: '#a7a39d', line: '#f5d76e', walk: '#d8d6d0', win: 'rgba(255,255,255,.75)', water: '#26d0a0',
      trees: ['#3fb04a', '#2e9c3c', '#6fd05a'], bld: ['#ff8a65', '#ffd54f', '#4dd0e1', '#ba68c8', '#f06292', '#fff176', '#90caf9'], roofs: ['#c62828', '#ef6c00', '#6d4c41'] } }
];
let townIdx = Math.floor(Math.random() * TOWNS.length), town = TOWNS[townIdx];
const DECOS = [];   // every decorated map, so the picker can re-skin them all
const hash = (a, b) => { let h = (a * 374761393 + b * 668265263) | 0; h = (h ^ (h >>> 13)) * 1274126177 | 0; return ((h ^ (h >>> 16)) >>> 0) / 4294967296; };
function decorate(m, box, tagHost = box) {
  const tag = document.createElement('div'); tag.className = 'townname'; tagHost.appendChild(tag);
  const layer = L.layerGroup().addTo(m);
  const redraw = () => {
    if (!m._loaded) return;   // map has no centre/zoom yet (this was the crash that skipped the whole animation)
    layer.clearLayers();
    const z = m.getZoom(), cell = 360 / Math.pow(2, z) * 0.55, b = m.getBounds().pad(.15);
    const i0 = Math.floor(b.getWest() / cell), i1 = Math.floor(b.getEast() / cell);
    const j0 = Math.floor(b.getSouth() / cell), j1 = Math.floor(b.getNorth() / cell);
    if ((i1 - i0) * (j1 - j0) > 220) return;
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
      const r = hash(i, j); if (r > .5) continue;
      const big = r < .06, set = big ? town.big : town.small, e = set[Math.floor(hash(j, i) * set.length)];
      const ll = [(j + hash(i + 7, j)) * cell, (i + hash(i, j + 7)) * cell];
      L.marker(ll, { icon: L.divIcon({ html: `<div class="deco ${big ? 'big' : ''}">${e}</div>`, className: '', iconSize: [30, 30], iconAnchor: [15, 15] }), interactive: false, keyboard: false, zIndexOffset: -1000 }).addTo(layer);
    }
  };
  const skin = () => { TOWNS.forEach(t => box.classList.remove(t.cls)); box.classList.add(town.cls); tag.textContent = town.name; };
  skin(); m.on('moveend zoomend', redraw); redraw();
  DECOS.push({ skin, redraw });
}
function setTown(i) { townIdx = i; town = TOWNS[i]; DECOS.forEach(d => { d.skin(); d.redraw(); }); }
$('#townSel').value = townIdx;
$('#townSel').onchange = e => setTown(+e.target.value);
let role, map, rest, cust, line, scoot, raf;
const TRAFFIC = ['Low', 'Medium', 'High', 'Jam'];

// skyline (built once, duplicated so the loop is seamless)
(() => {
  let h = '';
  for (let i = 0; i < 60; i++) h += `<i style="width:${30 + Math.random() * 50}px;height:${40 + Math.random() * 130}px;opacity:${.7 + Math.random() * .3}"></i>`;
  $('#skyline').innerHTML = h + h;
  for (let h = 0; h < 24; h++) $('#hour').add(new Option(String(h).padStart(2, '0') + ':00', h));
})();

later(2000, () => $('#choice').hidden = false);
document.querySelectorAll('#choice button').forEach(b => b.onclick = () => startPOV(b.dataset.r));


function startPOV(r) {
  role = r;
  $('#choice').hidden = true; $('#intro').hidden = true;
  const pov = $('#pov'); pov.hidden = false; pov.className = 'full ' + r;
  
  later(60, () => pov.classList.add('open'));
  if (r === 'customer') {
    $('#caption').textContent = '7:42 PM. You are starving...';
    later(2600, () => pov.classList.add('cap'));
    later(4600, showPhone);
  } else {
    $('#caption').textContent = 'Another night. Another delivery.';
    later(1200, () => pov.classList.add('reach'));
    later(2700, () => pov.classList.add('grab'));
    later(3900, () => { pov.classList.add('wear', 'cap'); });
    later(5600, showPhone);
  }
}

function showPhone() {
  if (!$('#phoneWrap').hidden) return;
  T.forEach(clearTimeout);
  
  const pov = $('#pov'); pov.classList.add('open'); if (role === 'rider') pov.classList.add('wear');
  $('#riderOnly').hidden = role !== 'rider';
  $('#who').textContent = role === 'rider' ? 'Rider' : 'Customer';
  $('#phone').classList.toggle('ipad', role === 'customer');   // customer gets a spacious iPad, rider a phone
  const w = $('#phoneWrap'); w.hidden = false;
  requestAnimationFrame(() => requestAnimationFrame(() => w.classList.add('up')));
  setTimeout(initMap, 400);
}

// ---------- map ----------
const icon = (e, flip = 1) => L.divIcon({ html: `<div style="font-size:26px;line-height:30px;transform:scaleX(${flip})">${e}</div>`, className: '', iconSize: [30, 30], iconAnchor: [15, 15] });

function initMap() {
  if (map) { map.invalidateSize(); return; }
  map = L.map('map', { zoomControl: false }).setView([12.9716, 77.5946], 12);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OSM' }).addTo(map);
  decorate(map, $('#map'));
  map.on('click', e => {
    if (rest && cust) { [rest, cust, line, scoot].forEach(l => l && map.removeLayer(l)); rest = cust = line = scoot = null; cancelAnimationFrame(raf); }
    if (!rest) { rest = L.marker(e.latlng, { icon: icon('🍕') }).addTo(map); $('#hint').textContent = 'Now tap the delivery spot 🏠.'; }
    else {
      cust = L.marker(e.latlng, { icon: icon('🏠') }).addTo(map);
      line = L.polyline([rest.getLatLng(), e.latlng], { color: '#e8501a', weight: 4, dashArray: '8 8' }).addTo(map);
      const km = rest.getLatLng().distanceTo(e.latlng) / 1000;
      $('#dist').value = km.toFixed(1);
      $('#hint').textContent = `Straight-line distance: ${km.toFixed(1)} km. Tap again to start over.`;
    }
  });
}

// ---------- Streamlit bridge (replaces the FastAPI calls) ----------
const sendST = (type, extra = {}) => window.parent.postMessage({ isStreamlitMessage: true, type, ...extra }, '*');
sendST('streamlit:componentReady', { apiVersion: 1 });
sendST('streamlit:setFrameHeight', { height: 800 });
let seen, waitT;   // id of the last result shown (the first render only records it)
window.addEventListener('message', ev => {
  if (!ev.data || ev.data.type !== 'streamlit:render') return;
  const r = ev.data.args && ev.data.args.results, id = r ? r.id : null;
  if (seen === undefined) { seen = id; return; }
  if (!r || id === seen) return;
  seen = id; clearTimeout(waitT);
  const btn = $('#go'); btn.disabled = false; btn.textContent = 'Predict delivery time';
  if (r.error) { $('#err').textContent = 'Could not get a prediction: ' + r.error; return; }
  render(r.main, r.best, r.traffic);
  try { reveal(r.main); } catch (e) { console.error(e); $('#err').textContent = 'Map animation problem: ' + e.message; }
});

function payload() {
  const p = {
    role, Distance: +$('#dist').value, Weatherconditions: $('#weather').value, Road_traffic_density: $('#traffic').value,
    Type_of_order: $('#otype').value, Festival: $('#fest').value, City: $('#city').value,
    Type_of_vehicle: role === 'rider' ? $('#veh').value : 'motorcycle',
    hour: $('#hour').value !== '' ? +$('#hour').value : new Date().getHours()   // "Now" = the visitor's own local hour
  };
  if (role === 'rider') Object.assign(p, {
    Delivery_person_Age: +$('#age').value, Delivery_person_Ratings: +$('#rating').value,
    Vehicle_condition: +$('#vcond').value, multiple_deliveries: +$('#multi').value
  });
  return p;
}

$('#go').onclick = () => {
  const d = +$('#dist').value, err = $('#err'); err.textContent = '';
  if (!(d >= 0.5 && d <= 25)) { err.textContent = 'Distance must be between 0.5 and 25 km (the model learned deliveries up to about 21 km).'; return; }
  const btn = $('#go'); btn.disabled = true; btn.textContent = 'Thinking...';
  sendST('streamlit:setComponentValue', { value: { ...payload(), id: Date.now() }, dataType: 'json' });
  waitT = setTimeout(() => { btn.disabled = false; btn.textContent = 'Predict delivery time'; err.textContent = 'No answer from the model. Try again.'; }, 20000);
};

function render(m, best, tr) {
  const v = Object.values(best.by_hour), mx = Math.max(...v), mn = Math.min(...v);
  const bars = Object.entries(best.by_hour).map(([h, t]) =>
    `<div class="bar ${h === best.best_time ? 'best' : ''}" title="${h}: ${t} min"><i style="height:${20 + 80 * (t - mn) / ((mx - mn) || 1)}%"></i>${h.slice(0, 2)}</div>`).join('');
  const cards = TRAFFIC.map((t, i) => `<div class="${t === $('#traffic').value ? 'on' : ''}">${t}<b>${tr[i].estimated_time}</b>min</div>`).join('');
  const acts = role === 'rider' ? `<div class="acts"><button class="ok" onclick="this.parentNode.innerHTML='✅ Order accepted. Ride safe!'">Accept</button><button class="no" onclick="this.parentNode.innerHTML='Order declined.'">Decline</button></div>` : '';
  const r = $('#result'); r.hidden = false;
  r.innerHTML = `<div class="big">${m.estimated_time}<small> min</small></div>
    <p class="sub">${m.display_text}</p>${acts}
    <h4>Best time to order</h4><div class="bars">${bars}</div>
    <p class="note">Shorter bar = faster. Fastest around ${best.best_time} (about ${best.by_hour[best.best_time]} min).</p>
    <h4>If traffic changed</h4><div class="cards">${cards}</div>
    <p class="note">Typical error of the model is about 3 minutes. The map preview is a simulation, not live tracking.</p>`;
  $('#screen').scrollTo({ top: 150, behavior: 'smooth' });
}

// ---------- reveal sequence ----------
const RT = []; let map2, pins = [], line2, rafLine, scoot2, rafScoot, cfRaf;
const FAST_MIN = 20;   // deliveries this quick (or quicker) get confetti
const LINE_MS = 2000;  // how long the red dotted line takes to draw
const RIDE_MS = 5000;  // how long the vehicle takes to drive restaurant -> home
const rlater = (ms, fn) => RT.push(setTimeout(fn, ms));
const rpin = e => L.divIcon({ html: `<div class="pw"><div class="rpin"><span>${e}</span></div></div>`, className: '', iconSize: [44, 44], iconAnchor: [22, 53] });

const cf = $('#confetti'), cx = cf.getContext('2d');
function confetti() {
  cf.width = innerWidth; cf.height = innerHeight; cancelAnimationFrame(cfRaf);
  const cols = ['#e8501a', '#ffd27a', '#16a34a', '#e11d2e', '#4a90e2', '#f06292'];
  const ps = Array.from({ length: 150 }, () => ({
    x: innerWidth / 2 + (Math.random() - .5) * 140, y: innerHeight * .7,
    vx: (Math.random() - .5) * 16, vy: -9 - Math.random() * 12, w: 6 + Math.random() * 6, h: 4 + Math.random() * 6,
    r: Math.random() * 6, vr: (Math.random() - .5) * .4, c: cols[Math.floor(Math.random() * cols.length)]
  }));
  const t0 = performance.now(), life = 3500;
  const step = now => {
    const age = now - t0; cx.clearRect(0, 0, cf.width, cf.height);
    if (age >= life) return;
    ps.forEach(p => {
      p.vy += .35; p.vx *= .99; p.x += p.vx; p.y += p.vy; p.r += p.vr;
      cx.save(); cx.translate(p.x, p.y); cx.rotate(p.r); cx.globalAlpha = Math.max(0, 1 - age / life);
      cx.fillStyle = p.c; cx.fillRect(-p.w / 2, -p.h / 2, p.w, p.h); cx.restore();
    });
    cfRaf = requestAnimationFrame(step);
  };
  cfRaf = requestAnimationFrame(step);
}

// ---------- delivery vehicles (all red, drawn as SVG, facing right) ----------
const WHEEL = (x, y, r) => `<circle cx="${x}" cy="${y}" r="${r}" fill="#1f1f24"/><circle cx="${x}" cy="${y}" r="${r * .45}" fill="#d1d5db"/>`;
const BOX = `<rect x="3" y="9" width="15" height="11" rx="2" fill="#f5a623" stroke="#8a5a00" stroke-width="1"/><text x="10.5" y="18" font-size="8" text-anchor="middle">🍕</text>`;
const RIDER = `<circle cx="36" cy="5" r="5" fill="#991b1b"/><circle cx="38" cy="7" r="3" fill="#f2c9a0"/><path d="M34 11 L30 20" stroke="#1d4ed8" stroke-width="5" stroke-linecap="round"/><path d="M34 12 L54 9" stroke="#1d4ed8" stroke-width="3" stroke-linecap="round"/>`;
const SCOOTER = `${BOX}${WHEEL(16, 35, 8)}${WHEEL(56, 35, 8)}<path d="M8 30 C8 21 16 19 24 19 H38 L42 31 H18 Z" fill="#dc2626"/><rect x="22" y="14" width="17" height="5" rx="2.5" fill="#1f1f24"/><rect x="28" y="29" width="22" height="4" rx="2" fill="#b91c1c"/><path d="M52 31 L56 9" stroke="#dc2626" stroke-width="5" stroke-linecap="round"/><path d="M50 8 H62" stroke="#1f1f24" stroke-width="3.5" stroke-linecap="round"/>${RIDER}`;
const VEH = {
  scooter: SCOOTER,
  electric_scooter: SCOOTER + `<polygon points="31,21 27,28 31,28 29,34 36,26 32,26 34,21" fill="#fde047" stroke="#a16207" stroke-width=".8"/>`,
  motorcycle: `${BOX}${WHEEL(14, 34, 9)}${WHEEL(58, 34, 9)}<path d="M14 34 L28 22 H44 L58 34" fill="none" stroke="#b91c1c" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/><ellipse cx="33" cy="19" rx="11" ry="5.5" fill="#dc2626"/><rect x="18" y="14" width="14" height="4" rx="2" fill="#1f1f24"/><rect x="30" y="24" width="12" height="8" rx="2" fill="#374151"/><path d="M58 34 L52 12" stroke="#6b7280" stroke-width="3.5" stroke-linecap="round"/><path d="M46 10 H57" stroke="#1f1f24" stroke-width="3.5" stroke-linecap="round"/>${RIDER}`,
  bicycle: `${BOX}<circle cx="14" cy="33" r="10" fill="none" stroke="#1f1f24" stroke-width="3"/><circle cx="58" cy="33" r="10" fill="none" stroke="#1f1f24" stroke-width="3"/><path d="M14 33 L32 33 L26 14 Z M26 14 L50 12 L32 33 M50 12 L58 33" fill="none" stroke="#dc2626" stroke-width="3.5" stroke-linejoin="round" stroke-linecap="round"/><path d="M22 12 H31" stroke="#1f1f24" stroke-width="3.5" stroke-linecap="round"/><path d="M46 8 H55" stroke="#1f1f24" stroke-width="3.5" stroke-linecap="round"/><circle cx="43" cy="3" r="5" fill="#991b1b"/><circle cx="45" cy="5" r="3" fill="#f2c9a0"/><path d="M28 13 L40 7" stroke="#1d4ed8" stroke-width="5" stroke-linecap="round"/><path d="M40 8 L51 9" stroke="#1d4ed8" stroke-width="3" stroke-linecap="round"/><path d="M28 14 L38 24 L32 33" fill="none" stroke="#1d4ed8" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>`
};
const vehicleIcon = (kind, flip) => L.divIcon({
  html: `<div style="transform:scaleX(${flip});transform-origin:50% 100%"><div class="bobv"><svg viewBox="0 0 72 44" width="108" height="66">${VEH[kind] || VEH.scooter}</svg></div></div>`,
  className: '', iconSize: [108, 66], iconAnchor: [54, 63]
});

// the vehicle drives from the restaurant to the home along the red line, then the pizza box appears
function ride(a, b, kind, done) {
  if (scoot2) map2.removeLayer(scoot2);
  scoot2 = L.marker(a, { icon: vehicleIcon(kind, b.lng >= a.lng ? 1 : -1), zIndexOffset: 2000, interactive: false }).addTo(map2);
  const t0 = performance.now(); sfx.ride(RIDE_MS, kind);
  const step = now => {
    const k = Math.min((now - t0) / RIDE_MS, 1), e = k < .5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    scoot2.setLatLng([a.lat + (b.lat - a.lat) * e, a.lng + (b.lng - a.lng) * e]);
    if (k < 1) rafScoot = requestAnimationFrame(step);
    else { sfx.done(); rlater(450, () => { if (scoot2) { map2.removeLayer(scoot2); scoot2 = null; } done(); }); }
  };
  rafScoot = requestAnimationFrame(step);
}

// ---------- weather sky at the top of the reveal map ----------
const wxc = $('#wxfx'), wx = wxc.getContext('2d');
let wxRaf, wxFlash = -9999, wxNext = 0, wxBolt = [], wxP = [];
const SKYG = { Sunny: ['#2f8fff', '#8fd0ff', '#fff3c4'], Cloudy: ['#6f7f93', '#a3b0bf', '#d8dee6'], Fog: ['#c4ccd0', '#dde3e5', '#f1f4f4'],
  Stormy: ['#121828', '#2b3652', '#4a5876'], Sandstorms: ['#9a6a2a', '#cf9d55', '#ecd09a'], Windy: ['#4fa8cf', '#9bd6e6', '#e4f4f7'] };
const WXICON = { Sunny: '☀️ Sunny', Cloudy: '☁️ Cloudy', Fog: '🌫️ Foggy', Stormy: '⛈️ Stormy', Sandstorms: '🏜️ Sandstorm', Windy: '💨 Windy' };
function cloud(x, y, s, col, a) {
  wx.fillStyle = col; wx.globalAlpha = a;
  [[0, 0, 28], [26, -10, 34], [56, 0, 28], [30, 8, 30]].forEach(([dx, dy, r]) => { wx.beginPath(); wx.arc(x + dx * s, y + dy * s, r * s, 0, 7); wx.fill(); });
  wx.globalAlpha = 1;
}
function drawFx(kind, W, H, now, t) {
  const drift = (p, speed, span) => ((p.x + t * speed * p.s) % (W + span)) - span / 2;
  if (kind === 'Sunny') {
    const sx = W * .82, sy = H * .3, g = wx.createRadialGradient(sx, sy, 10, sx, sy, H * .95);
    g.addColorStop(0, 'rgba(255,240,170,.9)'); g.addColorStop(.25, 'rgba(255,220,120,.35)'); g.addColorStop(1, 'rgba(255,220,120,0)');
    wx.fillStyle = g; wx.fillRect(0, 0, W, H);
    wx.save(); wx.translate(sx, sy); wx.rotate(t * .15); wx.strokeStyle = 'rgba(255,240,170,.4)'; wx.lineWidth = 10; wx.lineCap = 'round';
    for (let i = 0; i < 12; i++) { wx.rotate(Math.PI / 6); wx.beginPath(); wx.moveTo(72, 0); wx.lineTo(150 + 30 * Math.sin(t * 2 + i), 0); wx.stroke(); }
    wx.restore();
    wx.fillStyle = '#fff7c2'; wx.beginPath(); wx.arc(sx, sy, 48, 0, 7); wx.fill();
    wxP.forEach(p => cloud(drift(p, 12, 240), 20 + p.y * .5, .6 * p.s, '#fff', .75));
  } else if (kind === 'Cloudy') {
    wxP.forEach(p => cloud(drift(p, 14, 360), 10 + p.y * .6, 1.1 + p.s * .5, '#f1f4f8', .9));
  } else if (kind === 'Fog') {
    wxP.forEach(p => { wx.fillStyle = 'rgba(255,255,255,.22)'; wx.beginPath(); wx.ellipse(drift(p, 18, 600), p.y, 300 * p.s, 34 + 20 * p.s, 0, 0, 7); wx.fill(); });
    const g = wx.createLinearGradient(0, 0, 0, H); g.addColorStop(0, 'rgba(255,255,255,.5)'); g.addColorStop(1, 'rgba(255,255,255,0)'); wx.fillStyle = g; wx.fillRect(0, 0, W, H);
  } else if (kind === 'Stormy') {
    wxP.slice(0, 7).forEach(p => cloud(drift(p, 20, 360), 4 + p.y * .35, 1.3 + p.s * .5, '#1d2438', .95));
    wx.strokeStyle = 'rgba(190,210,255,.55)'; wx.lineWidth = 1.5; wx.beginPath();
    wxP.forEach(p => { p.y += 16 * p.s; p.x -= 3; if (p.y > H) { p.y = -10; p.x = Math.random() * W; } wx.moveTo(p.x, p.y); wx.lineTo(p.x - 4, p.y + 14); });
    wx.stroke();
    if (now > wxNext) { wxFlash = now; wxNext = now + 2600 + Math.random() * 3000; let x = W * (.2 + Math.random() * .6), y = 0; wxBolt = [[x, y]]; while (y < H * .85) { x += (Math.random() - .5) * 50; y += 22 + Math.random() * 24; wxBolt.push([x, y]); } sfx.thunder(); }
    const age = now - wxFlash;
    if (age >= 0 && age < 280) {
      wx.fillStyle = `rgba(255,255,255,${.5 * (1 - age / 280)})`; wx.fillRect(0, 0, W, H);
      wx.save(); wx.strokeStyle = 'rgba(255,255,255,.95)'; wx.lineWidth = 3; wx.shadowBlur = 18; wx.shadowColor = '#b8c8ff'; wx.beginPath();
      wxBolt.forEach(([x, y], i) => i ? wx.lineTo(x, y) : wx.moveTo(x, y)); wx.stroke(); wx.restore();
    }
  } else if (kind === 'Sandstorms') {
    const g = wx.createLinearGradient(0, 0, 0, H); g.addColorStop(0, 'rgba(176,120,50,.55)'); g.addColorStop(1, 'rgba(176,120,50,0)'); wx.fillStyle = g; wx.fillRect(0, 0, W, H);
    wx.fillStyle = 'rgba(222,178,106,.85)';
    wxP.forEach(p => { p.x += 9 * p.s + 3; p.y += Math.sin(t * 3 + p.r * 9) * 1.2; if (p.x > W + 10) { p.x = -10; p.y = Math.random() * H; } wx.fillRect(p.x, p.y, 2 + p.r * 3, 2 + p.r * 1.5); });
  } else if (kind === 'Windy') {
    wx.strokeStyle = 'rgba(255,255,255,.6)'; wx.lineWidth = 2; wx.lineCap = 'round';
    wxP.forEach(p => {
      const x = ((p.x + t * 260 * p.s) % (W + 300)) - 150, y = p.y;
      wx.beginPath(); wx.moveTo(x, y); wx.quadraticCurveTo(x + 40, y - 10 - p.r * 10, x + 90 + p.r * 60, y); wx.stroke();
      if (p.r > .6) { wx.font = '22px serif'; wx.fillText('🍃', x + 30, y + Math.sin(t * 4 + p.r * 10) * 12); }
    });
  }
}
function startSky(kind) {
  stopSky();
  const g = SKYG[kind] || SKYG.Sunny;
  $('#sky').style.background = `linear-gradient(${g[0]},${g[1]} 55%,${g[2]})`;
  $('#wxtag').textContent = WXICON[kind] || kind;
  const W = wxc.width = innerWidth, H = wxc.height = Math.round(innerHeight * .5);
  const N = { Stormy: 170, Sandstorms: 220, Windy: 26, Cloudy: 6, Fog: 7, Sunny: 5 }[kind] || 6;
  wxP = Array.from({ length: N }, () => ({ x: Math.random() * W, y: Math.random() * H, s: .5 + Math.random(), r: Math.random() }));
  wxFlash = -9999; wxNext = performance.now() + 1800;
  const t0 = performance.now();
  const frame = now => { wx.clearRect(0, 0, W, H); drawFx(kind, W, H, now, (now - t0) / 1000); wxRaf = requestAnimationFrame(frame); };
  wxRaf = requestAnimationFrame(frame);
}
function stopSky() { cancelAnimationFrame(wxRaf); wx.clearRect(0, 0, wxc.width, wxc.height); }

function closeReveal() {
  if ($('#reveal').classList.contains('down')) sfx.whooshUp();
  stopSky(); $('#cloudwipe').classList.remove('in');
  RT.forEach(clearTimeout); RT.length = 0; cancelAnimationFrame(rafLine); cancelAnimationFrame(rafScoot); cancelAnimationFrame(cfRaf);
  cx.clearRect(0, 0, cf.width, cf.height);
  if (scoot2 && map2) { map2.removeLayer(scoot2); scoot2 = null; }
  $('#reveal').classList.remove('down');
}
$('#back').onclick = closeReveal; $('#more').onclick = closeReveal;

function drawLine(a, b, ms, done) {
  line2 = L.polyline([a, a], { color: '#e11d2e', weight: 5, dashArray: '1 12', lineCap: 'round' }).addTo(map2);
  const t0 = performance.now();
  const step = now => {
    const k = Math.min((now - t0) / ms, 1);
    line2.setLatLngs([a, L.latLng(a.lat + (b.lat - a.lat) * k, a.lng + (b.lng - a.lng) * k)]);
    if (k < 1) rafLine = requestAnimationFrame(step); else done();
  };
  rafLine = requestAnimationFrame(step);
}

function reveal(m) {
  closeReveal();
  const R = $('#reveal'); R.hidden = false;
  let a, b;
  if (rest && cust) { a = rest.getLatLng(); b = cust.getLatLng(); }
  else { // distance was typed, not tapped: place two points that distance apart
    const c = map.getCenter(), k = (+$('#dist').value) / (111.32 * Math.cos(c.lat * Math.PI / 180));
    a = L.latLng(c.lat, c.lng - k / 2); b = L.latLng(c.lat, c.lng + k / 2);
  }
  if (!map2) {
    map2 = L.map('map2', { zoomControl: false });
    if (REVEAL_MAP === 'cartoon') {
      cartoon = new CartoonLayer({ maxZoom: 19 }).addTo(map2);
      const tag = document.createElement('div'); tag.className = 'townname'; $('#reveal').appendChild(tag);
      const skin = () => { tag.textContent = town.name; if (cartoon) cartoon.redraw(); };
      skin(); DECOS.push({ skin, redraw: () => {} });
    } else {
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '© OSM' }).addTo(map2);
      decorate(map2, $('#map2'), $('#reveal'));
    }
  }
  pins.forEach(p => map2.removeLayer(p)); pins = []; if (line2) map2.removeLayer(line2);
  $('#box').hidden = true; $('#box').className = '';
  map2.setView(a, 13, { animate: false });
  map2.invalidateSize(); map2.fitBounds([a, b], { paddingTopLeft: [50, 150], paddingBottomRight: [50, 330], animate: false });
  const veh = role === 'rider' ? $('#veh').value : 'scooter';   // riders see their own vehicle, customers a scooter
  const C = $('#cloudwipe');
  C.classList.add('in'); sfx.whoosh();                                                       // 1. clouds cover the whole screen
  rlater(750, () => { R.classList.add('down'); startSky($('#weather').value); });            //    the map drops in behind them (weather sky on top)
  rlater(1500, () => { C.classList.remove('in'); sfx.whooshUp(); });                         // 2. clouds part: the map is there
  rlater(2400, () => { pins.push(L.marker(a, { icon: rpin('🍕') }).addTo(map2)); sfx.pin(.45); rlater(300, () => { pins.push(L.marker(b, { icon: rpin('🏠') }).addTo(map2)); sfx.pin(.45); }); });   // 3. restaurant + home drop
  rlater(2900, () => { sfx.line(LINE_MS); drawLine(a, b, LINE_MS, () => {                   // 4. red dotted line, restaurant -> home (2 s)
    rlater(300, () => ride(a, b, veh, () => showBox(m)));                                    // 5. red vehicle drives it (5 s), vanishes, 6. pizza box with the time
  }); });
}

// ---------- cartoon town map for the reveal: drawn in code, coloured per town ----------
const REVEAL_MAP = 'cartoon';   // 'cartoon' = illustrated town map, 'real' = OpenStreetMap
let cartoon;
const rnd = (i, j, k) => hash(i * 37 + k * 11, j * 91 + k * 5);
const CAR_COLS = ['#e53935', '#fdd835', '#1e88e5', '#43a047', '#8e24aa', '#fb8c00'];
function rr(g, x, y, w, h, r) { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); }
function tree(g, x, y, P, r) {
  g.fillStyle = 'rgba(0,0,0,.18)'; g.beginPath(); g.ellipse(x + 3, y + 9, 10, 5, 0, 0, 7); g.fill();
  g.fillStyle = P.trees[Math.floor(r * P.trees.length)]; g.beginPath(); g.arc(x, y, 10, 0, 7); g.fill();
  g.fillStyle = 'rgba(255,255,255,.25)'; g.beginPath(); g.arc(x - 3, y - 3, 4.5, 0, 7); g.fill();
}
function car(g, x, y, vert, col) {
  g.fillStyle = col; vert ? rr(g, x, y, 11, 22, 4) : rr(g, x, y, 22, 11, 4); g.fill();
  g.fillStyle = 'rgba(30,40,60,.75)'; vert ? g.fillRect(x + 1.5, y + 4, 8, 5) : g.fillRect(x + 13, y + 1.5, 5, 8);
}
function house(g, x, y, w, h, P, r) {
  g.fillStyle = P.bld[Math.floor(r * P.bld.length)]; g.fillRect(x, y + h * .38, w, h * .62);
  g.fillStyle = P.roofs[Math.floor(rnd(Math.floor(x), Math.floor(y), 3) * P.roofs.length)];
  g.beginPath(); g.moveTo(x - 4, y + h * .4); g.lineTo(x + w / 2, y); g.lineTo(x + w + 4, y + h * .4); g.fill();
  g.fillStyle = P.win; g.fillRect(x + w * .15, y + h * .55, w * .22, h * .2); g.fillRect(x + w * .63, y + h * .55, w * .22, h * .2);
  g.fillStyle = 'rgba(0,0,0,.25)'; g.fillRect(x + w * .42, y + h * .66, w * .16, h * .34);
}
function tower(g, x, y, w, h, P, r) {
  g.fillStyle = P.bld[Math.floor(r * P.bld.length)]; rr(g, x, y, w, h, 5); g.fill();
  g.fillStyle = 'rgba(0,0,0,.12)'; g.fillRect(x + w * .8, y, w * .2, h);
  g.fillStyle = P.win;
  for (let wx = x + 7; wx < x + w * .75 - 6; wx += 12) for (let wy = y + 8; wy < y + h - 10; wy += 13) g.fillRect(wx, wy, 7, 7);
}
function block(g, i, j, x, y, w, h, P) {
  const r = rnd(i, j, 7);
  g.fillStyle = P.walk; rr(g, x - 3, y - 3, w + 6, h + 6, 10); g.fill();
  if (r < .25) {                                   // park with a pond
    g.fillStyle = P.park; rr(g, x, y, w, h, 8); g.fill();
    if (rnd(i, j, 8) < .55) { g.fillStyle = P.water; g.beginPath(); g.ellipse(x + w * .5, y + h * .5, w * .24, h * .16, 0, 0, 7); g.fill(); g.strokeStyle = 'rgba(255,255,255,.7)'; g.lineWidth = 3; g.stroke(); }
    for (let k = 0; k < 9; k++) tree(g, x + 14 + rnd(i, j, 20 + k) * (w - 28), y + 14 + rnd(i, j, 40 + k) * (h - 28), P, rnd(i, j, 60 + k));
  } else if (r < .55) {                            // little houses
    g.fillStyle = P.park; rr(g, x, y, w, h, 8); g.fill();
    const cw = w / 2, ch = h / 2;
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) house(g, x + a * cw + 9, y + b * ch + 8, cw - 18, ch - 14, P, rnd(i, j, 80 + a * 2 + b));
    tree(g, x + w * .5, y + h * .5, P, rnd(i, j, 90));
  } else {                                         // tall buildings
    g.fillStyle = P.walk; rr(g, x, y, w, h, 8); g.fill();
    const cw = w / 2, ch = h / 2;
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) tower(g, x + a * cw + 6, y + b * ch + 6, cw - 12, ch - 12, P, rnd(i, j, 100 + a * 2 + b));
  }
}
function drawCity(g, ox, oy, P) {
  const B = 180, RW = 34, S = 256;
  const i0 = Math.floor(ox / B), i1 = Math.floor((ox + S) / B), j0 = Math.floor(oy / B), j1 = Math.floor((oy + S) / B);
  g.fillStyle = P.grass; g.fillRect(ox, oy, S, S);
  g.fillStyle = P.road;
  for (let i = i0; i <= i1; i++) g.fillRect(i * B, oy, RW, S);
  for (let j = j0; j <= j1; j++) g.fillRect(ox, j * B, S, RW);
  g.strokeStyle = P.line; g.lineWidth = 3; g.setLineDash([14, 12]);
  for (let i = i0; i <= i1; i++) { g.lineDashOffset = ((oy % 26) + 26) % 26; g.beginPath(); g.moveTo(i * B + RW / 2, oy); g.lineTo(i * B + RW / 2, oy + S); g.stroke(); }
  for (let j = j0; j <= j1; j++) { g.lineDashOffset = ((ox % 26) + 26) % 26; g.beginPath(); g.moveTo(ox, j * B + RW / 2); g.lineTo(ox + S, j * B + RW / 2); g.stroke(); }
  g.setLineDash([]);
  for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
    const bx = i * B, by = j * B;
    g.fillStyle = P.road; g.fillRect(bx, by, RW, RW);
    g.fillStyle = '#f7f7f7';
    for (let k = 0; k < 5; k++) { g.fillRect(bx + 4 + k * 6, by + RW + 3, 3.5, 11); g.fillRect(bx + RW + 3, by + 4 + k * 6, 11, 3.5); }
    if (rnd(i, j, 1) < .8) car(g, bx + (rnd(i, j, 2) < .5 ? 6 : 19), by + RW + 20 + rnd(i, j, 3) * (B - RW - 70), true, CAR_COLS[Math.floor(rnd(i, j, 11) * 6)]);
    if (rnd(i, j, 4) < .8) car(g, bx + RW + 20 + rnd(i, j, 5) * (B - RW - 70), by + (rnd(i, j, 6) < .5 ? 6 : 19), false, CAR_COLS[Math.floor(rnd(i, j, 12) * 6)]);
    block(g, i, j, bx + RW, by + RW, B - RW, B - RW, P);
  }
}
const CartoonLayer = L.GridLayer.extend({
  createTile(c) {
    const t = document.createElement('canvas'); t.width = t.height = 256;
    const g = t.getContext('2d'); g.translate(-c.x * 256, -c.y * 256);
    drawCity(g, c.x * 256, c.y * 256, town.pal);
    return t;
  }
});

// ---------- clouds that cover the screen while the map arrives ----------
(() => {
  const C = $('#cloudwipe');
  for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) {
    const d = document.createElement('div'); d.className = 'cl ' + ((r + c) % 2 ? 'r' : 'l');
    d.style.left = (c * 22 - 12 + Math.random() * 6) + 'vw'; d.style.top = (r * 28 - 10 + Math.random() * 6) + 'vh';
    d.style.transitionDelay = (Math.random() * .25) + 's'; C.appendChild(d);
  }
})();

function showBox(m) {
  $('#tm').textContent = m.estimated_time; $('#rg').textContent = `${m.min_time} to ${m.max_time} min`;
  $('#boxtxt').textContent = m.display_text + (m.estimated_time <= FAST_MIN ? ' ⚡ Speedy delivery!' : '');
  const bx = $('#box'); bx.hidden = false;
  requestAnimationFrame(() => requestAnimationFrame(() => { bx.classList.add('in'); sfx.boxPop(); }));   // box appears
  rlater(1100, () => { bx.classList.add('open'); sfx.boxOpen(); });                                    // ...and opens
  rlater(1900, () => { if (m.estimated_time <= FAST_MIN) { confetti(); sfx.party(); } });              // quick delivery: confetti
}

function preview() {
  if (!rest || !cust) return;
  const a = rest.getLatLng(), b = cust.getLatLng();
  map.fitBounds([a, b], { padding: [30, 30] });
  if (scoot) map.removeLayer(scoot);
  scoot = L.marker(a, { icon: icon('🛵', b.lng > a.lng ? -1 : 1), zIndexOffset: 500 }).addTo(map);
  const t0 = performance.now(), dur = 7000; cancelAnimationFrame(raf);
  const step = now => {
    const k = Math.min((now - t0) / dur, 1);
    scoot.setLatLng([a.lat + (b.lat - a.lat) * k, a.lng + (b.lng - a.lng) * k]);
    if (k < 1) raf = requestAnimationFrame(step);
  };
  raf = requestAnimationFrame(step);
}
</script>
</body>
</html>
"""

# write it to a folder and register it as a Streamlit component
_dir = Path(tempfile.gettempdir()) / "dotg_ui"
_dir.mkdir(exist_ok=True)
(_dir / "index.html").write_text(HTML, encoding="utf-8")
ui = components.declare_component("dotg_ui", path=str(_dir))

ss = st.session_state
ss.setdefault("results", None)
ss.setdefault("last_req", None)

# the page sends its inputs here when Predict is pressed
req = ui(results=ss.results, key="ui", default=None)
if req and req.get("id") != ss.last_req:
    ss.last_req = req["id"]
    try:
        ss.results = {"id": req["id"], **run_request(req)}
    except Exception as e:
        ss.results = {"id": req["id"], "error": str(e)}
    # hand the answer back to the page
    st.rerun()
