import streamlit as st
import numpy as np
import datetime
import json
import os
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

# --- Load Outcomes from Scraper (JSON) ---
SCRAPER_JSON_PATH = "data/outcomes.json"

def load_scraped_games():
    if os.path.exists(SCRAPER_JSON_PATH):
        with open(SCRAPER_JSON_PATH, "r") as f:
            data = json.load(f)
        return data
    return []

# --- Data Utilities ---
def encode_day(day):
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return days.index(day)

def get_prnd_features(results, i):
    prnd = []
    if i >= 1:
        prnd.append(results[i - 1])
    else:
        prnd.append(-1)
    if i >= 2:
        prnd.append(results[i - 1] - results[i - 2])
    else:
        prnd.append(0)
    return prnd

def build_dataset(games, window=2):
    X, y = [], []
    for game in games:
        results = game["results"]
        if None in results:
            continue
        day = encode_day(game["day"])
        hour = game["hour"]
        for i in range(window, 10):
            features = results[i-window:i] + get_prnd_features(results, i) + [day, hour]
            X.append(features)
            y.append(results[i])
    return np.array(X), np.array(y)

def train_rf(games, window=2):
    X, y = build_dataset(games, window)
    if len(X) == 0:
        return None
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf

def build_markov(games, order=2):
    tables = [defaultdict(lambda: [0, 0]) for _ in range(10)]
    for game in games:
        results = game["results"]
        if None in results:
            continue
        for i in range(order, 10):
            key = tuple(results[i-order:i]) + tuple(get_prnd_features(results, i))
            tables[i][key][results[i]] += 1
    return tables

def markov_predict(game_so_far, tables, order=2):
    preds = []
    for i in range(10):
        if i < order:
            preds.append(None)
            continue
        key = tuple(game_so_far[i-order:i]) + tuple(get_prnd_features(game_so_far, i))
        if None in key:
            preds.append(None)
            continue
        counts = tables[i][key]
        if sum(counts) == 0:
            preds.append(None)
        else:
            preds.append(int(counts[1] > counts[0]))
    return preds

def get_stats(games):
    left = np.zeros(10)
    right = np.zeros(10)
    for game in games:
        results = game["results"]
        if None in results:
            continue
        for i, x in enumerate(results):
            if x == 1:
                left[i] += 1
            else:
                right[i] += 1
    return left, right

def stat_predict(left, right):
    return [1 if left[i] > right[i] else 0 for i in range(10)]

def predict_next(game_so_far, rf, markov_tables, stats, day, hour, window=2):
    markov_preds = markov_predict(game_so_far, markov_tables, order=window)
    left, right = stats
    stat_preds = stat_predict(left, right)
    X_pred = []
    for i in range(window, 10):
        features = game_so_far[i-window:i] + get_prnd_features(game_so_far, i) + [encode_day(day), hour]
        X_pred.append(features)
    rf_preds = rf.predict(X_pred) if X_pred and rf is not None else []
    preds = []
    for i in range(10):
        votes = []
        if i >= window and rf is not None and i - window < len(rf_preds):
            votes.append(rf_preds[i - window])
        if markov_preds[i] is not None:
            votes.append(markov_preds[i])
        votes.append(stat_preds[i])
        pred = int(round(np.mean(votes)))
        preds.append(pred)
    return preds

# --- UI Setup ---
st.set_page_config(page_title="PRND Game Predictor", layout="wide")
st.markdown(
    '''
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100% !important;
    }
    .stButton > button {
        font-size: 1.1rem;
        padding: 1.2rem;
    }
    .bottom-controls {
        position: sticky;
        bottom: 0; left: 0; right: 0;
        background-color: #f0f0f0;
        padding: 1em;
        border-top: 1px solid #ccc;
        z-index: 999;
    }
    </style>
    ''', unsafe_allow_html=True
)

st.title("PRND-enhanced Mobile Game Predictor (Live-Integrated)")

with st.expander("ℹ️ How to use"):
    st.markdown("""
    - Tap once per round.
    - Model prediction is shown in green.
    - Tap Save & Retrain to add your game.
    - Reset clears current round.
    - Live Mode below shows the latest game scraped.
    """)

# --- State Init ---
if "current_game" not in st.session_state:
    st.session_state.current_game = [None] * 10

# --- Load and Train ---
all_games = load_scraped_games()
saved_games = [g for g in all_games if len(g["results"]) == 10 and None not in g["results"]]
now = datetime.datetime.now()
day = now.strftime("%A")
hour = now.hour

rf = train_rf(saved_games)
markov_tables = build_markov(saved_games)
stats = get_stats(saved_games)

# --- Manual Game Prediction ---
st.subheader("Manual Entry Game")
progress = sum(x is not None for x in st.session_state.current_game) / 10
st.progress(progress)
game_so_far = [x if x is not None else 0 for x in st.session_state.current_game]
predictions = predict_next(game_so_far, rf, markov_tables, stats, day, hour)

for i in range(10):
    st.markdown(f"### Round {i+1}")
    cols = st.columns(2)
    for j, side in enumerate([0, 1]):
        with cols[j]:
            is_pred = predictions[i] == side
            is_selected = st.session_state.current_game[i] == side
            bg = "#e6ffe6" if is_pred else "#fff"
            border = "3px solid green" if is_selected else "1px solid #ccc"
            symbol = "✔️" if is_selected else ("❌" if is_pred else "")
            if st.button(" ", key=f"btn_{i}_{side}"):
                st.session_state.current_game[i] = side
            st.markdown(
                f"<div style='background:{bg};border:{border};padding:1.5em;text-align:center;border-radius:12px;font-size:1.5rem'>{symbol}</div>",
                unsafe_allow_html=True
            )

# --- Bottom Bar ---
st.markdown("<div class='bottom-controls'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Reset Game"):
        st.session_state.current_game = [None]*10
        st.experimental_rerun()
with col2:
    if st.button("✅ Save & Retrain"):
        if None not in st.session_state.current_game:
            new_game = {
                "results": st.session_state.current_game.copy(),
                "day": day,
                "hour": hour,
                "timestamp": now.isoformat()
            }
            all_games.append(new_game)
            with open(SCRAPER_JSON_PATH, "w") as f:
                json.dump(all_games, f, indent=2)
            st.session_state.current_game = [None]*10
            st.success("Game saved.")
            st.experimental_rerun()
        else:
            st.warning("Complete all rounds.")
st.markdown("</div>", unsafe_allow_html=True)

# --- Live Sync ---
st.subheader("Live Game Sync (Last Tracked Game)")
if all_games:
    live_game = all_games[-1]
    game_day = live_game["day"]
    game_hour = live_game["hour"]
    live_results = live_game["results"]
    padded_results = [r if r is not None else 0 for r in live_results]
    live_preds = predict_next(padded_results, rf, markov_tables, stats, game_day, game_hour)

    for i in range(10):
        actual = live_results[i]
        pred = live_preds[i]
        col = st.columns(1)[0]
        with col:
            if actual is None:
                st.markdown(f"<div style='padding:1em;border:1px solid #ccc;border-radius:8px;'>Round {i+1}: ⬜ Pending</div>", unsafe_allow_html=True)
            else:
                icon = "✔️" if actual == pred else "❌"
                st.markdown(f"<div style='padding:1em;border:1px solid #ccc;border-radius:8px;'>Round {i+1}: {icon} (Actual: {actual}, Pred: {pred})</div>", unsafe_allow_html=True)
else:
    st.info("No games found. Start playing to see live sync.")

# --- Stats ---
with st.expander("Model Stats"):
    total = len(saved_games)
    st.write(f"Total games learned: {total}")
    left, right = stats
    stat_summary = {f"Round {i+1}": ("Left" if left[i] > right[i] else "Right") for i in range(10)}
    st.write("Most common picks:", stat_summary)

with st.expander("PRND Pattern Analysis"):
    switch_rates = []
    for i in range(1, 10):
        same, switch = 0, 0
        for game in saved_games:
            if game["results"][i] == game["results"][i-1]:
                same += 1
            else:
                switch += 1
        total_switch = same + switch
        if total_switch:
            switch_rates.append(switch / total_switch)
        else:
            switch_rates.append(0.0)
    st.line_chart({"Switch Rate per Round": switch_rates})
