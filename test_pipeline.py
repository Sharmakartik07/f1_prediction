"""
tests/test_pipeline.py
======================
Unit tests for data pipeline and feature engineering logic.

Run:
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_pipeline import (
    _parse_laptime_delta,
    enrich_circuit_features,
    _add_rolling_features,
    _add_constructor_momentum,
    _add_elo_ratings,
    _compute_home_race,
)


# ---------------------------------------------------------------------------
# Lap time parsing
# ---------------------------------------------------------------------------

class TestLapTimeDelta:
    def test_normal_gap(self):
        delta = _parse_laptime_delta("1:18.543", "1:18.000")
        assert abs(delta - 0.543) < 0.01

    def test_pole_position(self):
        """Pole should return 0.0 gap."""
        delta = _parse_laptime_delta("1:18.000", "1:18.000")
        assert delta == 0.0

    def test_missing_time_returns_penalty(self):
        """Missing lap time → 5.0s penalty."""
        assert _parse_laptime_delta(None, "1:18.000") == 5.0
        assert _parse_laptime_delta("1:18.000", None) == 5.0
        assert _parse_laptime_delta(None, None) == 5.0

    def test_seconds_only_format(self):
        delta = _parse_laptime_delta("78.543", "78.000")
        assert abs(delta - 0.543) < 0.01


# ---------------------------------------------------------------------------
# Circuit feature enrichment
# ---------------------------------------------------------------------------

class TestCircuitFeatures:
    def _make_df(self, circuit_ids):
        return pd.DataFrame({"circuit_id": circuit_ids})

    def test_known_circuit_type(self):
        df = enrich_circuit_features(self._make_df(["monaco"]))
        assert df.iloc[0]["circuit_type"] == "street"

    def test_unknown_circuit_defaults(self):
        df = enrich_circuit_features(self._make_df(["unknown_track"]))
        assert df.iloc[0]["circuit_type"] == "mixed"
        assert df.iloc[0]["overtaking_difficulty"] == 5

    def test_one_hot_columns_created(self):
        df = enrich_circuit_features(self._make_df(["monaco", "monza", "suzuka", "bahrain"]))
        assert "circuit_street" in df.columns
        assert "circuit_high_speed" in df.columns
        assert "circuit_technical" in df.columns

    def test_altitude_monza(self):
        df = enrich_circuit_features(self._make_df(["monza"]))
        assert df.iloc[0]["altitude_m"] == 162


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

def _make_race_df(n_races: int = 25) -> pd.DataFrame:
    """Minimal single-driver race history."""
    rows = []
    for i in range(n_races):
        rows.append({
            "year": 2020 + i // 20,
            "round": (i % 20) + 1,
            "driver_id": "test_driver",
            "constructor_id": "test_team",
            "finish_pos": 1 if i % 5 == 0 else (i % 10) + 2,
            "grid_pos": (i % 10) + 1,
            "won": 1 if i % 5 == 0 else 0,
            "dnf": 1 if i % 7 == 0 else 0,
            "points": 25 if i % 5 == 0 else max(0, 18 - (i % 10)),
        })
    return pd.DataFrame(rows)


class TestRollingFeatures:
    def test_rolling_win_rate_computed(self):
        df = _make_race_df(25)
        result = _add_rolling_features(df)
        assert "rolling_win_rate_5" in result.columns
        assert "rolling_win_rate_10" in result.columns

    def test_no_future_leakage(self):
        """Rolling features must be shifted by 1 (no peeking at current race)."""
        df = _make_race_df(20)
        result = _add_rolling_features(df)
        # The rolling average should not reflect the current race's outcome
        # We check that race 0 has NaN (no prior data) rather than a value
        first_val = result.iloc[0]["rolling_win_rate_5"]
        assert np.isnan(first_val)

    def test_dnf_rate_computed(self):
        df = _make_race_df(25)
        result = _add_rolling_features(df)
        assert "driver_dnf_rate" in result.columns
        non_nan = result["driver_dnf_rate"].dropna()
        assert (non_nan >= 0).all() and (non_nan <= 1).all()


# ---------------------------------------------------------------------------
# ELO ratings
# ---------------------------------------------------------------------------

class TestEloRatings:
    def _make_two_driver_race_df(self) -> pd.DataFrame:
        rows = []
        for year in [2020, 2021]:
            for rnd in range(1, 11):
                rows.append({"year": year, "round": rnd, "driver_id": "fast_driver",
                              "constructor_id": "team_a", "finish_pos": 1, "won": 1,
                              "grid_pos": 1, "points": 25, "dnf": 0})
                rows.append({"year": year, "round": rnd, "driver_id": "slow_driver",
                              "constructor_id": "team_b", "finish_pos": 2, "won": 0,
                              "grid_pos": 2, "points": 18, "dnf": 0})
        return pd.DataFrame(rows)

    def test_elo_column_created(self):
        df = _add_elo_ratings(self._make_two_driver_race_df())
        assert "driver_elo" in df.columns

    def test_winner_gains_elo(self):
        """Consistent winner should have higher ELO than consistent loser."""
        df = _add_elo_ratings(self._make_two_driver_race_df())
        fast_elo = df[df["driver_id"] == "fast_driver"]["driver_elo"].iloc[-1]
        slow_elo = df[df["driver_id"] == "slow_driver"]["driver_elo"].iloc[-1]
        assert fast_elo > slow_elo

    def test_initial_elo_is_1500(self):
        df = _add_elo_ratings(self._make_two_driver_race_df())
        # Only the very first year + first round has the initialised ELO
        first_race = df[(df["year"] == 2020) & (df["round"] == 1)]
        assert (first_race["driver_elo"] == 1500.0).all()


# ---------------------------------------------------------------------------
# Home race detection
# ---------------------------------------------------------------------------

class TestHomeRace:
    def test_verstappen_at_zandvoort(self):
        df = pd.DataFrame([{"driver_id": "max_verstappen", "circuit_id": "zandvoort"}])
        result = _compute_home_race(df)
        assert result.iloc[0] == 1

    def test_verstappen_at_monza(self):
        df = pd.DataFrame([{"driver_id": "max_verstappen", "circuit_id": "monza"}])
        result = _compute_home_race(df)
        assert result.iloc[0] == 0

    def test_unknown_driver(self):
        df = pd.DataFrame([{"driver_id": "mystery_driver", "circuit_id": "spa"}])
        result = _compute_home_race(df)
        assert result.iloc[0] == 0
