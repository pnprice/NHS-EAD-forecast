"""
utils.py
========
Shared utilities for NHS EAD forecasting scripts.

Constants, data-cleaning helpers, weather-loading functions, and the
PartialElasticNetCV class are centralised here to avoid duplication
across NHS_basic_forecast.py, NHS_global_forecast.py, and
NHS_rolling_forecast.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import ElasticNet

# ---------------------------------------------------------------------------
# Shared paths
# ---------------------------------------------------------------------------

WEATHER_PATH     = Path("data/bristol_weather.csv")
FORECAST_WX_PATH = Path("data/bristol_forecast_weather.csv")

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

BRISTOL_LAT = 51.45
BRISTOL_LON = -2.59

OUTCOME     = "estimated_avoidable_deaths"
DEV_END     = pd.Timestamp("2025-09-30")
MAX_FC_LEAD = 7   # Previous Runs API archives up to 7 days ahead

# Weather input caps (99th percentile of observed training data).
RAIN_CAP          = 25.0   # mm   — wx_rain_sum, observed and NWP
WIND_CAP          = 42.0   # km/h — wx_wind_max, observed and NWP
COLD_THRESH2      =  5.0   # °C   — wx_coldness2 = max(0, COLD_THRESH2 − T)
HEAVY_RAIN_THRESH = 20.0   # mm   — wx_heavy_rain = (rain > threshold)

# Prediction output bounds
PRED_MAX       = 10.0   # effectively no upper cap (actual EAD never exceeds ~2.2)
FLOOR_LOOKBACK = 90     # days of recent EAD history used for lower floor (5th-percentile)

# England bank holidays covering the development and assessment periods.
# Source: https://www.gov.uk/bank-holidays (England and Wales).
# Update if the assessment period is extended beyond Apr 2026.
_ENGLAND_BANK_HOLIDAYS: set[pd.Timestamp] = {pd.Timestamp(d) for d in [
    "2023-04-07", "2023-04-10", "2023-05-01", "2023-05-08",
    "2023-05-29", "2023-08-28", "2023-12-25", "2023-12-26",
    "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-06",
    "2024-05-27", "2024-08-26", "2024-12-25", "2024-12-26",
    "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-05",
    "2025-05-26", "2025-08-25", "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-04-03", "2026-04-06",
]}

# ---------------------------------------------------------------------------
# Weather — observed history
# ---------------------------------------------------------------------------

_WEATHER_DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_min",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
]


def fetch_weather(
    start: str = "2023-03-01",
    end: str   = "2025-09-30",
    lat: float = BRISTOL_LAT,
    lon: float = BRISTOL_LON,
    save_path: Path = WEATHER_PATH,
) -> pd.DataFrame:
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start,
        "end_date":   end,
        "daily":      ",".join(_WEATHER_DAILY_VARS),
        "timezone":   "Europe/London",
    }
    print(f"Fetching weather from Open-Meteo ({start} → {end})…")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    weather = pd.DataFrame({"date": pd.to_datetime(raw["daily"]["time"])})
    for var in _WEATHER_DAILY_VARS:
        weather[var] = raw["daily"][var]
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        weather.to_csv(save_path, index=False)
        print(f"  Saved to {save_path}")
    return weather


def load_weather(path: Path = WEATHER_PATH) -> pd.DataFrame:
    if path.exists():
        weather = pd.read_csv(path, parse_dates=["date"])
        if set(weather.columns) - {"date"} == set(_WEATHER_DAILY_VARS):
            print(f"Loaded weather from {path} ({len(weather)} days)")
            return weather
        print("  Cache columns differ — re-fetching.")
    return fetch_weather(save_path=path)


# ---------------------------------------------------------------------------
# Weather — NWP forecast (Open-Meteo Previous Runs API)
# ---------------------------------------------------------------------------

_FC_BASE_VARS = ["temperature_2m", "rain", "snowfall", "wind_speed_10m"]


def _fetch_fc_wx_chunk(start: str, end: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetch one chunk of hourly NWP forecast data and aggregate to daily."""
    hourly_vars = [
        f"{var}_previous_day{d}"
        for var in _FC_BASE_VARS
        for d in range(1, MAX_FC_LEAD + 1)
    ]
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "hourly":     ",".join(hourly_vars),
        "start_date": start,
        "end_date":   end,
        "timezone":   "Europe/London",
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    raw = resp.json()

    times = pd.to_datetime(raw["hourly"]["time"])
    df_h  = pd.DataFrame({"datetime": times})
    df_h["date"] = df_h["datetime"].dt.normalize()
    for var in hourly_vars:
        vals = raw["hourly"].get(var)
        df_h[var] = pd.array(vals, dtype="Float64") if vals is not None else pd.NA

    records = []
    for d in range(1, MAX_FC_LEAD + 1):
        grp = df_h.groupby("date").agg(
            t_mean=(f"temperature_2m_previous_day{d}", "mean"),
            t_min =(f"temperature_2m_previous_day{d}", "min"),
            rain  =(f"rain_previous_day{d}",           "sum"),
            snow  =(f"snowfall_previous_day{d}",        "sum"),
            wind  =(f"wind_speed_10m_previous_day{d}",  "max"),
        ).reset_index()
        grp["lead_day"] = d
        records.append(grp)
    return pd.concat(records, ignore_index=True)


def fetch_forecast_weather(
    start: str  = "2024-01-01",
    end: str    = "2026-04-30",
    lat: float  = BRISTOL_LAT,
    lon: float  = BRISTOL_LON,
    save_path: Path = FORECAST_WX_PATH,
    chunk_months: int = 3,
) -> pd.DataFrame:
    """Fetch NWP forecast data at lead days 1–7 from Open-Meteo Previous Runs API.

    Returns a long DataFrame with columns: date, lead_day, wx_coldness,
    wx_hotness, wx_below_freezing, wx_rain_sum, wx_snowfall_sum, wx_wind_max.
    """
    periods = pd.date_range(start=start, end=end, freq=f"{chunk_months}MS")
    chunks  = []
    for i, period_start in enumerate(periods):
        chunk_start = period_start.strftime("%Y-%m-%d")
        chunk_end   = (
            (periods[i + 1] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if i + 1 < len(periods) else end
        )
        chunks.append((chunk_start, chunk_end))

    print(f"Fetching NWP forecast weather ({start} → {end}, {len(chunks)} chunks)…")
    all_chunks = []
    for idx, (cs, ce) in enumerate(chunks):
        print(f"  chunk {idx + 1}/{len(chunks)}: {cs} → {ce}", end="", flush=True)
        df_chunk = _fetch_fc_wx_chunk(cs, ce, lat, lon)
        all_chunks.append(df_chunk)
        print(f"  ({len(df_chunk)//MAX_FC_LEAD} days)")

    df = pd.concat(all_chunks, ignore_index=True)
    df["t_mean"] = df["t_mean"].astype(float)
    df["t_min"]  = df["t_min"].astype(float)
    df["rain"]   = df["rain"].fillna(0).astype(float)
    df["snow"]   = df["snow"].fillna(0).astype(float)
    df["wind"]   = df["wind"].astype(float)

    df["wx_coldness"]       = (10 - df["t_mean"]).clip(lower=0)
    df["wx_hotness"]        = (df["t_mean"] - 25).clip(lower=0)
    df["wx_below_freezing"] = (df["t_min"] < 0).astype(float)
    df["wx_rain_sum"]       = df["rain"]
    df["wx_snowfall_sum"]   = df["snow"]
    df["wx_wind_max"]       = df["wind"]

    out_cols = ["date", "lead_day",
                "wx_coldness", "wx_hotness", "wx_below_freezing",
                "wx_rain_sum", "wx_snowfall_sum", "wx_wind_max"]
    df_out = df[out_cols].copy()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(save_path, index=False)
        print(f"  Saved {len(df_out)} rows ({len(df_out)//MAX_FC_LEAD} valid dates × "
              f"{MAX_FC_LEAD} lead days) to {save_path}")
    return df_out


def load_forecast_weather(path: Path = FORECAST_WX_PATH) -> pd.DataFrame | None:
    """Load NWP forecast weather cache, fetching from API if absent.
    Returns None on failure so callers can fall back to actual weather.
    """
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"])
        n_leads = df["lead_day"].nunique()
        print(f"  Loaded NWP forecast weather from {path} "
              f"({len(df)//n_leads} valid dates × {n_leads} lead days)")
        return df
    try:
        return fetch_forecast_weather(save_path=path)
    except Exception as exc:
        print(f"  WARNING: could not fetch forecast weather: {exc}")
        print("  Falling back to actual weather for all rows.")
        return None


# Base columns stored in the forecast-weather CSV.
# Derived columns (wx_coldness2, wx_heavy_rain) are added by build_fc_wx_wide.
_WX_BASE_COLS = [
    "wx_coldness", "wx_hotness", "wx_below_freezing",
    "wx_rain_sum", "wx_snowfall_sum", "wx_wind_max",
]

# 3-day mean is computed for these base weather columns.
_MEAN3_BASE_COLS = ["wx_coldness", "wx_hotness", "wx_coldness2"]


def build_fc_wx_wide(fc_wx_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long forecast-weather DataFrame to wide format indexed by date.

    Builds columns {wx_col}_L{lead} for each base column and lead day 1–MAX_FC_LEAD,
    then applies winsorization and derives wx_coldness2/wx_heavy_rain at each lead.
    """
    parts = []
    for lead in range(1, MAX_FC_LEAD + 1):
        sub = (
            fc_wx_df[fc_wx_df["lead_day"] == lead]
            .set_index("date")[_WX_BASE_COLS]
            .rename(columns={c: f"{c}_L{lead}" for c in _WX_BASE_COLS})
        )
        parts.append(sub)
    wide = pd.concat(parts, axis=1)
    for lead in range(1, MAX_FC_LEAD + 1):
        wide[f"wx_rain_sum_L{lead}"]   = wide[f"wx_rain_sum_L{lead}"].clip(upper=RAIN_CAP)
        wide[f"wx_wind_max_L{lead}"]   = wide[f"wx_wind_max_L{lead}"].clip(upper=WIND_CAP)
        # wx_coldness2 = max(0, COLD_THRESH2 − T) = max(0, wx_coldness − (10 − COLD_THRESH2))
        wide[f"wx_coldness2_L{lead}"]  = (wide[f"wx_coldness_L{lead}"] - (10 - COLD_THRESH2)).clip(lower=0)
        wide[f"wx_heavy_rain_L{lead}"] = (wide[f"wx_rain_sum_L{lead}"] > HEAVY_RAIN_THRESH).astype(int)
    return wide


# All wx feature columns present after build_fc_wx_wide (observed + derived).
wx_feature_cols = _WX_BASE_COLS + ["wx_coldness2", "wx_heavy_rain"]


def build_wx_train(
    df: pd.DataFrame,
    wx_cols: list[str],
    h: int,
    fc_wx_wide: "pd.DataFrame | None",
) -> np.ndarray:
    """Weather feature matrix for training horizon h.

    For each origin row (date d) the target is d+h.  Uses NWP forecast at
    lead=min(h, MAX_FC_LEAD) where available; falls back to actual weather.
    """
    n_rows   = len(df) - h
    lead     = min(h, MAX_FC_LEAD)
    X_actual = df[wx_cols].shift(-h).iloc[:n_rows].to_numpy(dtype=float)
    if fc_wx_wide is None:
        return X_actual
    fc_cols      = [f"{c}_L{lead}" for c in wx_cols]
    target_dates = (df["date"] + pd.Timedelta(days=h)).iloc[:n_rows]
    merged = pd.DataFrame({"date": target_dates.values}).merge(
        fc_wx_wide[fc_cols].reset_index(), on="date", how="left",
    )
    X_fc = merged[fc_cols].to_numpy(dtype=float)
    return np.where(np.isnan(X_fc), X_actual, X_fc)


def get_wx_pred(
    df: pd.DataFrame,
    wx_cols: list[str],
    origin_idx: int,
    h: int,
    fc_wx_wide: "pd.DataFrame | None",
) -> np.ndarray:
    """Weather features for a single prediction at origin_idx, horizon h."""
    target_date = df["date"].iloc[origin_idx + h]
    actual      = df[wx_cols].iloc[origin_idx + h].to_numpy(dtype=float).reshape(1, -1)
    if fc_wx_wide is None:
        return actual
    lead    = min(h, MAX_FC_LEAD)
    fc_cols = [f"{c}_L{lead}" for c in wx_cols]
    if target_date in fc_wx_wide.index:
        fc_vals = fc_wx_wide.loc[target_date, fc_cols].to_numpy(dtype=float).reshape(1, -1)
        return np.where(np.isnan(fc_vals), actual, fc_vals)
    return actual


def _wx_at_offset(
    df: pd.DataFrame,
    cols: list[str],
    offset: int,
    fc_wx_wide: "pd.DataFrame | None",
    *,
    n_rows: int | None = None,
    origin_idx: int | None = None,
) -> np.ndarray:
    """Weather at (date + offset) for either the training slice or a single prediction.

    Supply n_rows for training (returns shape (n_rows, len(cols)));
    supply origin_idx for prediction (returns shape (1, len(cols))).
    offset > 0 means future — uses NWP where available.
    """
    if n_rows is not None:
        # Training slice
        X_actual = df[cols].shift(-offset).iloc[:n_rows].to_numpy(dtype=float)
        if offset > 0 and fc_wx_wide is not None:
            lead    = min(offset, MAX_FC_LEAD)
            fc_cols = [f"{c}_L{lead}" for c in cols]
            target_dates = (df["date"] + pd.Timedelta(days=offset)).iloc[:n_rows]
            merged = pd.DataFrame({"date": target_dates.values}).merge(
                fc_wx_wide[fc_cols].reset_index(), on="date", how="left",
            )
            X_fc = merged[fc_cols].to_numpy(dtype=float)
            return np.where(np.isnan(X_fc), X_actual, X_fc)
        return X_actual
    else:
        # Single prediction row
        row_idx = origin_idx + offset  # type: ignore[operator]
        if row_idx < 0 or row_idx >= len(df):
            return np.full((1, len(cols)), np.nan)
        actual = df[cols].iloc[row_idx].to_numpy(dtype=float).reshape(1, -1)
        if offset > 0 and fc_wx_wide is not None:
            lead     = min(offset, MAX_FC_LEAD)
            fc_cols  = [f"{c}_L{lead}" for c in cols]
            tgt_date = df["date"].iloc[row_idx]
            if tgt_date in fc_wx_wide.index:
                fc_vals = fc_wx_wide.loc[tgt_date, fc_cols].to_numpy(dtype=float).reshape(1, -1)
                return np.where(np.isnan(fc_vals), actual, fc_vals)
        return actual


def build_wx_mean3_train(
    df: pd.DataFrame,
    h: int,
    fc_wx_wide: "pd.DataFrame | None",
) -> np.ndarray:
    """3-day mean of _MEAN3_BASE_COLS for training horizon h.

    Averages target day (d+h) with the two preceding days (d+h-1, d+h-2).
    Returns shape (len(df) - h, len(_MEAN3_BASE_COLS)).
    """
    n_rows = len(df) - h
    layers = [
        _wx_at_offset(df, _MEAN3_BASE_COLS, h,     fc_wx_wide, n_rows=n_rows),
        _wx_at_offset(df, _MEAN3_BASE_COLS, h - 1, fc_wx_wide, n_rows=n_rows),
        _wx_at_offset(df, _MEAN3_BASE_COLS, h - 2, fc_wx_wide, n_rows=n_rows),
    ]
    return np.nanmean(np.stack(layers, axis=2), axis=2)


def get_wx_mean3_pred(
    df: pd.DataFrame,
    origin_idx: int,
    h: int,
    fc_wx_wide: "pd.DataFrame | None",
) -> np.ndarray:
    """3-day mean of _MEAN3_BASE_COLS for a single prediction.
    Returns shape (1, len(_MEAN3_BASE_COLS)).
    """
    layers = [
        _wx_at_offset(df, _MEAN3_BASE_COLS, h,     fc_wx_wide, origin_idx=origin_idx),
        _wx_at_offset(df, _MEAN3_BASE_COLS, h - 1, fc_wx_wide, origin_idx=origin_idx),
        _wx_at_offset(df, _MEAN3_BASE_COLS, h - 2, fc_wx_wide, origin_idx=origin_idx),
    ]
    return np.nanmean(np.stack(layers, axis=2), axis=2)


# 3-day mean column names (one per entry in _MEAN3_BASE_COLS).
_WX_MEAN3_COLS = [f"{c}_mean3" for c in _MEAN3_BASE_COLS]


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def _clean_name(name: str) -> str:
    """Normalise a raw metric|coverage column key to a valid Python identifier."""
    name = re.sub(r"\d",              "",  name)
    name = re.sub(r"[()]",            "",  name)
    name = re.sub(r"[ \-|]",         "_", name)
    name = name.replace("%", "pct")
    name = re.sub(r"[^a-zA-Z0-9_]",  "",  name)
    return name.lower().strip("_")


def clean_column_names(cols: list[str]) -> list[str]:
    """Apply _clean_name to each column, deduplicating clashes with a numeric suffix."""
    seen: dict[str, int] = {}
    deduped: list[str]   = []
    for name in (_clean_name(c) for c in cols):
        if name in seen:
            seen[name] += 1
            deduped.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            deduped.append(name)
    return deduped


def _mse(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = ~(np.isnan(actual) | np.isnan(pred))
    return float(np.mean((actual[mask] - pred[mask]) ** 2))


# ---------------------------------------------------------------------------
# PartialElasticNetCV — block coordinate-descent solver
# ---------------------------------------------------------------------------


class PartialElasticNetCV:
    """Elastic Net where structural features (DOW, holidays, weather) are
    unpenalised and operational features receive the standard L1+L2 penalty.

    Solved by block coordinate descent: alternate between
      (1) OLS on partial residuals for structural features (exact, no penalty), and
      (2) sklearn ElasticNet on partial residuals for operational features.
    These two blocks are equivalent to joint CD with zero penalty on structural
    features — unlike sequential two-stage estimation, this gives the correct
    partially-penalised solution.

    Cross-validation uses contiguous time-series folds; alpha is chosen on a
    log-spaced grid from alpha_max (all op. coefs zero) to alpha_max * 1e-3.
    """

    def __init__(
        self,
        penalty_mask: np.ndarray,
        l1_ratio: float  = 0.5,
        n_alphas: int    = 50,
        cv: int          = 5,
        max_iter: int    = 10_000,
        max_outer: int   = 20,
        tol: float       = 1e-4,
    ):
        self.penalty_mask = np.asarray(penalty_mask, dtype=bool)
        self.l1_ratio  = l1_ratio
        self.n_alphas  = n_alphas
        self.cv        = cv
        self.max_iter  = max_iter
        self.max_outer = max_outer
        self.tol       = tol
        self.coef_:      np.ndarray | None = None
        self.intercept_: float             = 0.0
        self.alpha_:     float | None      = None

    def _alpha_grid(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Log-spaced grid with alpha_max computed over penalised features only.

        Restricting to penalised features prevents structural predictors (DOW,
        weather) from inflating alpha_max, which would shift the grid into a
        range where the operational features are all zeroed out.
        """
        y_c = y - y.mean()
        alpha_max = (
            float(np.max(np.abs(X[:, self.penalty_mask].T @ y_c)))
            / (len(y) * self.l1_ratio)
        )
        return np.exp(np.linspace(np.log(alpha_max), np.log(alpha_max * 1e-3), self.n_alphas))

    def _fit_path(self, X: np.ndarray, y: np.ndarray, alphas: np.ndarray) -> list[np.ndarray]:
        """Block-CD along an alpha path with warm starts; returns list of full coef vectors."""
        X_s  = X[:, ~self.penalty_mask]
        X_op = X[:,  self.penalty_mask]
        y_c  = y - y.mean()

        beta_s  = np.zeros(X_s.shape[1])
        beta_op = np.zeros(X_op.shape[1])
        model_op = ElasticNet(
            l1_ratio=self.l1_ratio, max_iter=self.max_iter,
            fit_intercept=False, warm_start=True, tol=1e-5,
        )

        coef_path = []
        for alpha in alphas:
            model_op.alpha = alpha
            for _ in range(self.max_outer):
                old_s, old_op = beta_s.copy(), beta_op.copy()
                beta_s  = np.linalg.lstsq(X_s, y_c - X_op @ beta_op, rcond=None)[0]
                model_op.fit(X_op, y_c - X_s @ beta_s)
                beta_op = model_op.coef_.copy()
                if (np.max(np.abs(beta_s - old_s)) < self.tol and
                        np.max(np.abs(beta_op - old_op)) < self.tol):
                    break
            full = np.zeros(X.shape[1])
            full[~self.penalty_mask] = beta_s
            full[self.penalty_mask]  = beta_op
            coef_path.append(full.copy())
        return coef_path

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PartialElasticNetCV":
        """CV to select alpha, then refit on full data at best alpha.

        Not used in the current hybrid approach (scripts call fit_alpha with
        alpha from sklearn ElasticNetCV), but retained for standalone use.
        """
        alphas    = self._alpha_grid(X, y)
        n         = len(y)
        fold_size = n // self.cv
        fold_mse  = np.zeros((self.cv, len(alphas)))

        for k in range(self.cv):
            v0  = k * fold_size
            v1  = (k + 1) * fold_size if k < self.cv - 1 else n
            val = np.zeros(n, dtype=bool)
            val[v0:v1] = True
            path = self._fit_path(X[~val], y[~val], alphas)
            for a, coef in enumerate(path):
                pred = X[val] @ coef + y[~val].mean()
                fold_mse[k, a] = float(np.mean((y[val] - pred) ** 2))

        best            = int(np.argmin(fold_mse.mean(axis=0)))
        self.alpha_     = float(alphas[best])
        path            = self._fit_path(X, y, alphas[: best + 1])
        self.coef_      = path[-1]
        self.intercept_ = float(y.mean())
        return self

    def fit_alpha(self, X: np.ndarray, y: np.ndarray, alpha: float) -> "PartialElasticNetCV":
        """Fit at a specific alpha with warm start along the grid path."""
        alphas      = self._alpha_grid(X, y)
        path_alphas = np.array([a for a in alphas if a >= alpha] + [alpha])
        path        = self._fit_path(X, y, path_alphas)
        self.coef_      = path[-1]
        self.intercept_ = float(y.mean())
        self.alpha_     = alpha
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_
