"""
predict_2026.py
===============
Generates win-probability predictions for each driver in the 2026 F1 season
using the trained stacking ensemble.

Run AFTER data_pipeline.py and models.py --train:
    python src/predict_2026.py
"""

import os
import pandas as pd
import numpy as np
import pickle
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 2026 Driver lineup (confirmed / projected as of late 2025)
# ---------------------------------------------------------------------------

DRIVERS_2026 = [
    # driver_id                  display_name              constructor_id     elo     grid_avg  home
    ("max_verstappen",           "Max Verstappen",         "red_bull",        1680,   1.8,      0),
    ("lando_norris",             "Lando Norris",           "mclaren",         1640,   2.5,      0),
    ("charles_leclerc",          "Charles Leclerc",        "ferrari",         1620,   2.8,      1),  # Monaco
    ("george_russell",           "George Russell",         "mercedes",        1580,   3.2,      0),
    ("oscar_piastri",            "Oscar Piastri",          "mclaren",         1590,   3.4,      0),
    ("carlos_sainz",             "Carlos Sainz",           "ferrari",         1570,   3.1,      0),
    ("fernando_alonso",          "Fernando Alonso",        "aston_martin",    1560,   5.5,      1),  # Barcelona
    ("lewis_hamilton",           "Lewis Hamilton",         "ferrari",         1610,   2.9,      1),  # Silverstone
    ("lance_stroll",             "Lance Stroll",           "aston_martin",    1430,   9.0,      1),  # Montreal
    ("esteban_ocon",             "Esteban Ocon",           "haas",            1460,   8.5,      0),
    ("pierre_gasly",             "Pierre Gasly",           "alpine",          1470,   8.0,      0),
    ("valtteri_bottas",          "Valtteri Bottas",        "sauber",          1480,   9.5,      0),
    ("nico_hulkenberg",          "Nico Hulkenberg",        "sauber",          1440,   11.0,     1),  # Germany
    ("yuki_tsunoda",             "Yuki Tsunoda",           "rb",              1450,   10.5,     0),
    ("alexander_albon",          "Alexander Albon",        "williams",        1420,   13.0,     0),
    ("oliver_bearman",           "Oliver Bearman",         "haas",            1400,   14.0,     0),
    ("kimi_antonelli",           "Kimi Antonelli",         "mercedes",        1390,   12.0,     0),
    ("isack_hadjar",             "Isack Hadjar",           "rb",              1380,   13.5,     0),
    ("jack_doohan",              "Jack Doohan",            "alpine",          1370,   14.5,     0),
    ("gabriel_bortoleto",        "Gabriel Bortoleto",      "sauber",          1360,   15.0,     0),
]

# Constructor momentum (2024-end proxy, points per race)
CONSTRUCTOR_PPR = {
    "red_bull": 24.0,
    "mclaren": 28.0,    # 2024 constructors leader
    "ferrari": 21.0,
    "mercedes": 16.0,
    "aston_martin": 8.0,
    "alpine": 4.0,
    "haas": 5.0,
    "rb": 6.0,
    "williams": 5.0,
    "sauber": 2.0,
}

# 2026 calendar (22 races)
CALENDAR_2026 = [
    {"round": 1,  "gp": "Bahrain",        "circuit_id": "bahrain",       "type": "mixed",      "ot": 5, "alt": 10},
    {"round": 2,  "gp": "Saudi Arabia",   "circuit_id": "jeddah",        "type": "street",     "ot": 5, "alt": 15},
    {"round": 3,  "gp": "Australia",      "circuit_id": "albert_park",   "type": "street",     "ot": 6, "alt": 10},
    {"round": 4,  "gp": "Japan",          "circuit_id": "suzuka",        "type": "technical",  "ot": 7, "alt": 45},
    {"round": 5,  "gp": "China",          "circuit_id": "shanghai",      "type": "mixed",      "ot": 5, "alt": 5},
    {"round": 6,  "gp": "Miami",          "circuit_id": "miami",         "type": "street",     "ot": 6, "alt": 0},
    {"round": 7,  "gp": "Emilia-Romagna", "circuit_id": "imola",         "type": "technical",  "ot": 7, "alt": 26},
    {"round": 8,  "gp": "Monaco",         "circuit_id": "monaco",        "type": "street",     "ot": 9, "alt": 7},
    {"round": 9,  "gp": "Spain",          "circuit_id": "catalunya",     "type": "mixed",      "ot": 5, "alt": 109},
    {"round": 10, "gp": "Canada",         "circuit_id": "montreal",      "type": "mixed",      "ot": 5, "alt": 16},
    {"round": 11, "gp": "Austria",        "circuit_id": "red_bull_ring", "type": "high_speed", "ot": 4, "alt": 698},
    {"round": 12, "gp": "Britain",        "circuit_id": "silverstone",   "type": "high_speed", "ot": 5, "alt": 126},
    {"round": 13, "gp": "Belgium",        "circuit_id": "spa",           "type": "mixed",      "ot": 4, "alt": 400},
    {"round": 14, "gp": "Hungary",        "circuit_id": "hungaroring",   "type": "technical",  "ot": 7, "alt": 264},
    {"round": 15, "gp": "Netherlands",    "circuit_id": "zandvoort",     "type": "technical",  "ot": 8, "alt": 10},
    {"round": 16, "gp": "Italy",          "circuit_id": "monza",         "type": "high_speed", "ot": 4, "alt": 162},
    {"round": 17, "gp": "Azerbaijan",     "circuit_id": "baku",          "type": "street",     "ot": 6, "alt": 0},
    {"round": 18, "gp": "Singapore",      "circuit_id": "singapore",     "type": "street",     "ot": 7, "alt": 15},
    {"round": 19, "gp": "USA",            "circuit_id": "americas",      "type": "mixed",      "ot": 5, "alt": 251},
    {"round": 20, "gp": "Mexico",         "circuit_id": "rodriguez",     "type": "mixed",      "ot": 5, "alt": 2240},
    {"round": 21, "gp": "Brazil",         "circuit_id": "interlagos",    "type": "mixed",      "ot": 5, "alt": 785},
    {"round": 22, "gp": "Las Vegas",      "circuit_id": "las_vegas",     "type": "street",     "ot": 4, "alt": 610},
    {"round": 23, "gp": "Qatar",          "circuit_id": "losail",        "type": "mixed",      "ot": 5, "alt": 10},
    {"round": 24, "gp": "Abu Dhabi",      "circuit_id": "yas_marina",    "type": "mixed",      "ot": 6, "alt": 3},
]

HOME_CIRCUIT_MAP = {
    "max_verstappen": "zandvoort",
    "charles_leclerc": "monaco",
    "fernando_alonso": "catalunya",
    "lewis_hamilton": "silverstone",
    "lance_stroll": "montreal",
    "nico_hulkenberg": None,
    "george_russell": "silverstone",
    "lando_norris": "silverstone",
}


def build_2026_feature_rows() -> pd.DataFrame:
    """
    Build a synthetic feature matrix representing every driver in every 2026 race.
    We use projected ELO, historical qualifying averages, and constructor momentum.
    """
    rows = []
    for race in CALENDAR_2026:
        circuit_type = race["type"]
        for driver in DRIVERS_2026:
            d_id, d_name, constructor, elo, grid_avg, _ = driver

            # Home race flag
            home_circuit = HOME_CIRCUIT_MAP.get(d_id, None)
            home = int(home_circuit is not None and home_circuit in race["circuit_id"])

            # Add some realistic qualifying jitter (~±1 position)
            quali_delta = max(0.0, (grid_avg - 1) * 0.18 + np.random.normal(0, 0.1))

            row = {
                "round": race["round"],
                "gp_name": race["gp"],
                "circuit_id": race["circuit_id"],
                "driver_id": d_id,
                "driver_name": d_name,
                "constructor_id": constructor,
                "grid_pos": round(grid_avg + np.random.normal(0, 1.5)),
                "quali_pos": round(grid_avg + np.random.normal(0, 1.5)),
                "quali_delta_s": round(quali_delta, 3),
                "rolling_win_rate_5": _win_rate_from_elo(elo),
                "rolling_win_rate_10": _win_rate_from_elo(elo) * 0.95,
                "rolling_points_5": _points_from_elo(elo),
                "driver_dnf_rate": 0.07 if elo > 1550 else 0.12,
                "constructor_momentum_5": CONSTRUCTOR_PPR.get(constructor, 5.0),
                "driver_elo": elo,
                "overtaking_difficulty": race["ot"],
                "altitude_m": race["alt"],
                "home_race": home,
                "circuit_street": int(circuit_type == "street"),
                "circuit_high_speed": int(circuit_type == "high_speed"),
                "circuit_technical": int(circuit_type == "technical"),
                "circuit_mixed": int(circuit_type == "mixed"),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def _win_rate_from_elo(elo: float) -> float:
    """Approximate 5-race win rate from ELO (heuristic)."""
    # ELO 1680 ≈ 0.35 win rate (Verstappen era), 1400 ≈ 0.01
    return max(0.0, min(0.5, (elo - 1350) / 1000))


def _points_from_elo(elo: float) -> float:
    return max(0.5, (elo - 1350) / 40)


def predict_season(n_simulations: int = 1000) -> pd.DataFrame:
    """
    Runs n_simulations of the 2026 season and aggregates win probabilities.
    Returns a DataFrame with season-level and per-race win probabilities.
    """
    try:
        with open(os.path.join(MODEL_DIR, "ensemble.pkl"), "rb") as f:
            model = pickle.load(f)
        log.info("Using trained ensemble model.")
    except FileNotFoundError:
        log.warning("Ensemble not found — falling back to XGBoost.")
        try:
            with open(os.path.join(MODEL_DIR, "xgboost.pkl"), "rb") as f:
                model = pickle.load(f)
        except FileNotFoundError:
            log.error("No trained model found. Run: python src/models.py --train")
            return pd.DataFrame()

    feat_cols = [
        "grid_pos", "quali_pos", "quali_delta_s",
        "rolling_win_rate_5", "rolling_win_rate_10", "rolling_points_5",
        "driver_dnf_rate", "constructor_momentum_5", "driver_elo",
        "overtaking_difficulty", "altitude_m", "home_race",
        "circuit_street", "circuit_high_speed", "circuit_technical", "circuit_mixed",
    ]

    all_results = []

    for sim in range(n_simulations):
        np.random.seed(sim)
        df = build_2026_feature_rows()
        X = df[feat_cols].fillna(0)
        probas = model.predict_proba(X)[:, 1]
        df["win_prob_raw"] = probas

        # Normalise per race so probabilities sum to 1
        for race_round in df["round"].unique():
            mask = df["round"] == race_round
            total = df.loc[mask, "win_prob_raw"].sum()
            if total > 0:
                df.loc[mask, "win_prob"] = df.loc[mask, "win_prob_raw"] / total
            else:
                df.loc[mask, "win_prob"] = 1 / 20

        all_results.append(df[["round", "gp_name", "driver_id", "driver_name",
                                "constructor_id", "win_prob"]])

    combined = pd.concat(all_results)
    season_probs = (
        combined.groupby(["driver_id", "driver_name", "constructor_id"])["win_prob"]
        .mean()
        .reset_index()
        .sort_values("win_prob", ascending=False)
    )
    season_probs["season_win_prob_pct"] = (season_probs["win_prob"] * 100).round(1)
    season_probs["expected_wins"] = (
        season_probs["win_prob"] * len(CALENDAR_2026)
    ).round(1)
    season_probs = season_probs.drop("win_prob", axis=1)

    # Per-race predictions
    per_race = (
        combined.groupby(["round", "gp_name", "driver_id", "driver_name"])["win_prob"]
        .mean()
        .reset_index()
        .sort_values(["round", "win_prob"], ascending=[True, False])
    )
    per_race["win_prob_pct"] = (per_race["win_prob"] * 100).round(1)

    return season_probs, per_race


def print_predictions(season_probs: pd.DataFrame, per_race: pd.DataFrame):
    print("\n" + "=" * 60)
    print("🏆  F1 2026 — SEASON WIN PROBABILITY RANKING")
    print("=" * 60)
    print(f"{'Pos':<4} {'Driver':<25} {'Team':<18} {'Season %':>9} {'Exp. Wins':>10}")
    print("-" * 60)
    for i, row in season_probs.head(10).iterrows():
        pos = season_probs.index.get_loc(i) + 1
        print(f"{pos:<4} {row['driver_name']:<25} {row['constructor_id']:<18} "
              f"{row['season_win_prob_pct']:>8.1f}% {row['expected_wins']:>9.1f}")
    print("=" * 60)

    print("\n📅  PER-RACE FAVOURITE (First 5 rounds)")
    print("-" * 50)
    for rnd in sorted(per_race["round"].unique())[:5]:
        race_data = per_race[per_race["round"] == rnd].head(3)
        gp = race_data.iloc[0]["gp_name"]
        print(f"\nRound {rnd} — {gp}")
        for _, r in race_data.iterrows():
            bar = "█" * int(r["win_prob_pct"] / 2)
            print(f"  {r['driver_name']:<22} {bar:<20} {r['win_prob_pct']:.1f}%")


if __name__ == "__main__":
    log.info("Generating 2026 season predictions...")
    result = predict_season(n_simulations=200)

    if isinstance(result, tuple):
        season_probs, per_race = result
        print_predictions(season_probs, per_race)

        season_probs.to_csv(os.path.join(OUTPUT_DIR, "season_predictions_2026.csv"), index=False)
        per_race.to_csv(os.path.join(OUTPUT_DIR, "per_race_predictions_2026.csv"), index=False)
        log.info("Saved predictions to outputs/")
    else:
        log.error("Prediction failed.")
