"""
data_pipeline.py
================
Fetches F1 historical data from the Ergast API and builds a clean
feature matrix ready for model training.

Usage:
    python src/data_pipeline.py
"""

import requests
import pandas as pd
import numpy as np
import time
import os
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ERGAST_BASE = "http://ergast.com/api/f1"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Ergast API helpers
# ---------------------------------------------------------------------------

def _get(url: str, retries: int = 3) -> dict:
    """GET wrapper with retry logic."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch: {url}")


def fetch_race_results(year_start: int = 2000, year_end: int = 2024) -> pd.DataFrame:
    """
    Fetch race-by-race results from Ergast for a range of seasons.
    Returns a DataFrame with one row per driver per race.
    """
    cache_path = os.path.join(DATA_DIR, f"results_{year_start}_{year_end}.parquet")
    if os.path.exists(cache_path):
        log.info("Loading cached race results.")
        return pd.read_parquet(cache_path)

    rows = []
    for year in range(year_start, year_end + 1):
        log.info(f"Fetching {year} season...")
        url = f"{ERGAST_BASE}/{year}/results.json?limit=1000"
        data = _get(url)
        races = data["MRData"]["RaceTable"]["Races"]

        for race in races:
            circuit = race["Circuit"]["circuitId"]
            round_num = int(race["round"])
            gp_name = race["raceName"]
            race_date = race["date"]

            for result in race["Results"]:
                rows.append({
                    "year": year,
                    "round": round_num,
                    "gp_name": gp_name,
                    "race_date": race_date,
                    "circuit_id": circuit,
                    "driver_id": result["Driver"]["driverId"],
                    "driver_name": f"{result['Driver']['givenName']} {result['Driver']['familyName']}",
                    "constructor_id": result["Constructor"]["constructorId"],
                    "grid_pos": int(result.get("grid", 0)),
                    "finish_pos": int(result["position"]),
                    "points": float(result.get("points", 0)),
                    "laps": int(result.get("laps", 0)),
                    "status": result.get("status", ""),
                    "fastest_lap_rank": int(result.get("FastestLap", {}).get("rank", 99)),
                })
        time.sleep(0.3)   # be polite to the API

    df = pd.DataFrame(rows)
    df["race_date"] = pd.to_datetime(df["race_date"])
    df["won"] = (df["finish_pos"] == 1).astype(int)
    df["dnf"] = df["status"].str.contains(
        r"Accident|Collision|Engine|Gearbox|Hydraulics|Retired|Mechanical|DNF",
        case=False, na=False
    ).astype(int)
    df.to_parquet(cache_path)
    log.info(f"Saved {len(df)} rows to {cache_path}")
    return df


def fetch_qualifying(year_start: int = 2000, year_end: int = 2024) -> pd.DataFrame:
    """Fetch qualifying results (lap-time delta to pole)."""
    cache_path = os.path.join(DATA_DIR, f"qualifying_{year_start}_{year_end}.parquet")
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    rows = []
    for year in range(year_start, year_end + 1):
        url = f"{ERGAST_BASE}/{year}/qualifying.json?limit=1000"
        try:
            data = _get(url)
        except RuntimeError:
            continue
        races = data["MRData"]["RaceTable"]["Races"]

        for race in races:
            pole_q3 = None
            for q in race.get("QualifyingResults", []):
                if int(q["position"]) == 1:
                    pole_q3 = q.get("Q3") or q.get("Q2") or q.get("Q1")
                    break

            for q in race.get("QualifyingResults", []):
                best_time = q.get("Q3") or q.get("Q2") or q.get("Q1")
                delta = _parse_laptime_delta(best_time, pole_q3)
                rows.append({
                    "year": year,
                    "round": int(race["round"]),
                    "driver_id": q["Driver"]["driverId"],
                    "quali_pos": int(q["position"]),
                    "quali_delta_s": delta,
                })
        time.sleep(0.3)

    df = pd.DataFrame(rows)
    df.to_parquet(cache_path)
    return df


def _parse_laptime_delta(t1: Optional[str], t2: Optional[str]) -> float:
    """Return t1 - t2 in seconds. Returns 5.0 as a penalty if either is missing."""
    def to_sec(t):
        if not t:
            return None
        try:
            parts = t.split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(parts[0])
        except ValueError:
            return None

    s1, s2 = to_sec(t1), to_sec(t2)
    if s1 is None or s2 is None:
        return 5.0
    return round(s1 - s2, 3)


# ---------------------------------------------------------------------------
# Circuit metadata
# ---------------------------------------------------------------------------

CIRCUIT_META = {
    "monaco": {"type": "street", "overtaking_difficulty": 9, "altitude_m": 7},
    "baku": {"type": "street", "overtaking_difficulty": 6, "altitude_m": 0},
    "singapore": {"type": "street", "overtaking_difficulty": 7, "altitude_m": 15},
    "monza": {"type": "high_speed", "overtaking_difficulty": 4, "altitude_m": 162},
    "spa": {"type": "mixed", "overtaking_difficulty": 4, "altitude_m": 400},
    "silverstone": {"type": "high_speed", "overtaking_difficulty": 5, "altitude_m": 126},
    "suzuka": {"type": "technical", "overtaking_difficulty": 7, "altitude_m": 45},
    "albert_park": {"type": "street", "overtaking_difficulty": 6, "altitude_m": 10},
    "bahrain": {"type": "mixed", "overtaking_difficulty": 5, "altitude_m": 10},
    "miami": {"type": "street", "overtaking_difficulty": 6, "altitude_m": 0},
    "las_vegas": {"type": "street", "overtaking_difficulty": 4, "altitude_m": 610},
    "rodriguez": {"type": "mixed", "overtaking_difficulty": 5, "altitude_m": 2240},
    "interlagos": {"type": "mixed", "overtaking_difficulty": 5, "altitude_m": 785},
}

DEFAULT_CIRCUIT = {"type": "mixed", "overtaking_difficulty": 5, "altitude_m": 50}


def enrich_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add circuit type and difficulty columns."""
    df = df.copy()
    df["circuit_type"] = df["circuit_id"].map(
        lambda c: CIRCUIT_META.get(c, DEFAULT_CIRCUIT)["type"]
    )
    df["overtaking_difficulty"] = df["circuit_id"].map(
        lambda c: CIRCUIT_META.get(c, DEFAULT_CIRCUIT)["overtaking_difficulty"]
    )
    df["altitude_m"] = df["circuit_id"].map(
        lambda c: CIRCUIT_META.get(c, DEFAULT_CIRCUIT)["altitude_m"]
    )
    # One-hot circuit type
    circuit_dummies = pd.get_dummies(df["circuit_type"], prefix="circuit")
    df = pd.concat([df, circuit_dummies], axis=1)
    return df


# ---------------------------------------------------------------------------
# Build final feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(year_start: int = 2000, year_end: int = 2024) -> pd.DataFrame:
    """
    Merges race results + qualifying and computes all rolling/lagged features.
    Returns a clean DataFrame ready for model.fit().
    """
    log.info("Fetching race results...")
    results = fetch_race_results(year_start, year_end)

    log.info("Fetching qualifying data...")
    quali = fetch_qualifying(year_start, year_end)

    df = results.merge(quali, on=["year", "round", "driver_id"], how="left")
    df = enrich_circuit_features(df)
    df = df.sort_values(["year", "round", "finish_pos"]).reset_index(drop=True)

    # --- Rolling features (computed per driver, no future leakage) ---
    df = _add_rolling_features(df)

    # --- Constructor momentum ---
    df = _add_constructor_momentum(df)

    # --- ELO ratings ---
    df = _add_elo_ratings(df)

    # --- Home race flag ---
    df["home_race"] = _compute_home_race(df).astype(int)

    # --- Grid delta (finish - grid) ---
    df["positions_gained"] = df["grid_pos"] - df["finish_pos"]

    # Drop rows with NaN features (first few races of each driver)
    feature_cols = [
        "grid_pos", "quali_pos", "quali_delta_s",
        "rolling_win_rate_5", "rolling_win_rate_10",
        "rolling_points_5", "driver_dnf_rate",
        "constructor_momentum_5",
        "driver_elo",
        "overtaking_difficulty", "altitude_m",
        "home_race",
    ]
    for col in ["circuit_street", "circuit_high_speed", "circuit_technical", "circuit_mixed"]:
        if col in df.columns:
            feature_cols.append(col)

    df_clean = df.dropna(subset=feature_cols).copy()
    log.info(f"Feature matrix: {df_clean.shape[0]} rows, {len(feature_cols)} features")
    df_clean.to_parquet(os.path.join(DATA_DIR, "features.parquet"))
    return df_clean


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for window in [5, 10]:
        df[f"rolling_win_rate_{window}"] = (
            df.groupby("driver_id")["won"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
        )
        df[f"rolling_points_{window}"] = (
            df.groupby("driver_id")["points"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
        )
    df["driver_dnf_rate"] = (
        df.groupby("driver_id")["dnf"]
        .transform(lambda x: x.shift(1).rolling(20, min_periods=5).mean())
    )
    return df


def _add_constructor_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Sum of constructor points in last 5 races."""
    constructor_pts = (
        df.groupby(["year", "round", "constructor_id"])["points"].sum().reset_index()
    )
    constructor_pts = constructor_pts.sort_values(["year", "round"])
    constructor_pts["constructor_momentum_5"] = (
        constructor_pts.groupby("constructor_id")["points"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
    )
    df = df.merge(
        constructor_pts[["year", "round", "constructor_id", "constructor_momentum_5"]],
        on=["year", "round", "constructor_id"], how="left"
    )
    return df


def _add_elo_ratings(df: pd.DataFrame, k: float = 32.0) -> pd.DataFrame:
    """
    Simplified pairwise ELO: each race treats every driver pair as a match.
    More wins against higher-rated opponents = bigger ELO gain.
    """
    elo = {}
    elo_col = []

    for _, race in df.groupby(["year", "round"]):
        drivers = race["driver_id"].tolist()
        for d in drivers:
            if d not in elo:
                elo[d] = 1500.0
        elo_col.extend([elo.get(d, 1500.0) for d in drivers])

        # Pairwise update (simplified: compare each driver to the winner)
        winner = race.loc[race["finish_pos"] == 1, "driver_id"]
        if winner.empty:
            continue
        winner_id = winner.iloc[0]
        for d in drivers:
            if d == winner_id:
                continue
            r_w, r_l = elo[winner_id], elo[d]
            expected_w = 1 / (1 + 10 ** ((r_l - r_w) / 400))
            elo[winner_id] = r_w + k * (1 - expected_w)
            elo[d] = r_l + k * (0 - (1 - expected_w))

    df = df.copy()
    df["driver_elo"] = elo_col
    return df


HOME_RACES = {
    "max_verstappen": ["zandvoort"],
    "lewis_hamilton": ["silverstone"],
    "charles_leclerc": ["monaco"],
    "lando_norris": ["silverstone"],
    "carlos_sainz": ["catalunya"],
    "fernando_alonso": ["catalunya"],
    "george_russell": ["silverstone"],
    "lance_stroll": ["montreal"],
    "sebastian_vettel": ["hockenheim", "nurburgring"],
    "michael_schumacher": ["hockenheim", "nurburgring"],
    "kimi_raikkonen": ["sepang"],
}


def _compute_home_race(df: pd.DataFrame) -> pd.Series:
    """Flag 1 if a driver is racing at their home circuit."""
    def is_home(row):
        home = HOME_RACES.get(row["driver_id"], [])
        return int(any(h in row["circuit_id"] for h in home))
    return df.apply(is_home, axis=1)


if __name__ == "__main__":
    df = build_feature_matrix(year_start=2003, year_end=2024)
    print(df.head())
    print(df.describe())
