"""
run_forecast.py
===============
Single entry point for the NHS EAD forecasting ensemble.

Trains on all available development data (up to 30 Sep 2025), then generates
predictions for each origin in the target window as the 50/50 ensemble of:
  - Global Partial-Penalty model (Phase 1 feature selection + Phase 2 global fit)
  - Rolling 90-day ElasticNet (refitted at each origin)

No weather or bank-holiday features are used.

Usage
-----
    python -u run_forecast.py            # assessment period: Oct 2025 – Mar 2026
    python -u run_forecast.py --validate # holdout period:    Oct 2024 – Sep 2025

Outputs
-------
    submission/pred_matrix.csv   forecast_id, day_1 … day_10
    submission/mse_summary.csv   forecast_id, mse_1_5, mse_6_10
    model_outputs/ensemble_feature_selection.csv
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from utils import (
    OUTCOME, FLOOR_LOOKBACK, PRED_MAX,
    clean_column_names, _mse,
    PartialElasticNetCV,
)

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/turingAI_forecasting_challenge_dataset.csv")
SUB_DIR   = Path("submission")
OUT_DIR   = Path("model_outputs")

HORIZON       = 10
L1_RATIO      = 0.5
TRAIN_WIN     = 90      # basic model rolling window (days)
SELECT_WIN    = 120     # Phase 1 feature-selection window
STRIDE_SEL    = 14
N_TOP         = 20      # seed top-N for feature expansion
FREQ_FLOOR    = 0.05    # minimum Phase 1 selection frequency

DEV_END      = pd.Timestamp("2025-09-30")
VALIDATE     = "--validate" in sys.argv
ORIGIN_START   = pd.Timestamp("2024-10-01") if VALIDATE else pd.Timestamp("2025-10-01")
ORIGIN_END     = pd.Timestamp("2025-09-30") if VALIDATE else pd.Timestamp("2026-03-31")
# In --validate mode cut global-model training at Sep 2024 (same holdout design as methods_summary
# Phase 3) so the validation period Oct 2024–Sep 2025 is genuinely out-of-sample for the global model.
GLOBAL_DEV_END = pd.Timestamp("2024-09-30") if VALIDATE else DEV_END

_DOW_NAMES     = ["dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat"]
_FOURIER_NAMES = ["sin1_annual", "cos1_annual", "sin2_annual", "cos2_annual"]
_CAL_COLS      = set(_DOW_NAMES) | set(_FOURIER_NAMES)

SUB_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. LOAD & PREPROCESS
# ---------------------------------------------------------------------------

t0 = time.time()
print("Loading data…")

raw = pd.read_csv(DATA_PATH, low_memory=False)
raw["dt"]   = pd.to_datetime(raw["dt"], format="mixed", errors="coerce")
raw["date"] = raw["dt"].dt.normalize()
raw = raw[raw["value"] != -9999].copy()

sod = raw["dt"].dt.hour * 3600 + raw["dt"].dt.minute * 60 + raw["dt"].dt.second
raw["midday_day"] = pd.to_datetime(
    np.where(sod <= 43200, raw["date"], raw["date"] + pd.Timedelta(days=1))
).normalize()

target_daily = (
    raw[raw["metric_name"] == OUTCOME]
    .groupby("midday_day", as_index=False)["value"]
    .mean()
    .rename(columns={"midday_day": "date", "value": OUTCOME})
)
target_daily["date"] = pd.to_datetime(target_daily["date"]).dt.normalize()

features_long = (
    raw[raw["metric_name"] != OUTCOME]
    .assign(col_key=lambda d: d["metric_name"] + "|" + d["coverage_label"])
    .groupby(["midday_day", "col_key"], as_index=False)["value"]
    .mean()
)
features_wide = (
    features_long
    .pivot(index="midday_day", columns="col_key", values="value")
    .reset_index()
    .rename(columns={"midday_day": "date"})
)
features_wide.columns.name = None
features_wide["date"] = pd.to_datetime(features_wide["date"]).dt.normalize()

df = features_wide.merge(target_daily, on="date", how="left")

# Clean column names; build metric → name map for feature expansion
cols_to_rename = [c for c in df.columns if c not in ("date", OUTCOME)]
deduped = clean_column_names(cols_to_rename)
df = df.rename(columns=dict(zip(cols_to_rename, deduped)))
metric_name_map: dict[str, str] = {
    clean: (orig.split("|")[0] if "|" in orig else orig)
    for orig, clean in zip(cols_to_rename, deduped)
}
print(f"  {len(deduped)} feature columns after per-hospital pivot")

# Calendar + annual Fourier terms (evaluated at each row's date)
for i, name in enumerate(_DOW_NAMES):
    df[name] = (df["date"].dt.dayofweek == i).astype(float)
_doy = df["date"].dt.dayofyear
df["sin1_annual"] = np.sin(2 * np.pi * _doy / 365.25)
df["cos1_annual"] = np.cos(2 * np.pi * _doy / 365.25)
df["sin2_annual"] = np.sin(4 * np.pi * _doy / 365.25)
df["cos2_annual"] = np.cos(4 * np.pi * _doy / 365.25)

# Impute operational columns
predictor_cols = [c for c in df.columns if c not in ("date", OUTCOME)]
for col in predictor_cols:
    df[col] = (
        df[col].interpolate(method="linear", limit_direction="both").ffill().bfill()
    )
df = df.dropna(subset=[OUTCOME]).reset_index(drop=True)

# Rolling summary features + lagged EAD
roll_candidates = [c for c in predictor_cols if c not in _CAL_COLS]
rolling_cols: dict[str, pd.Series] = {}
for var in roll_candidates:
    rolling_cols[f"{var}_roll_mean7"] = df[var].rolling(7).mean()
    rolling_cols[f"{var}_roll_sd7"]   = df[var].rolling(7).std()
df = pd.concat([df, pd.DataFrame(rolling_cols, index=df.index)], axis=1)
df[f"{OUTCOME}_lag3"]    = df[OUTCOME].shift(3)
df[f"{OUTCOME}_mean7_3"] = df[OUTCOME].shift(3).rolling(5).mean()
df = df.dropna().reset_index(drop=True)

predictors = [c for c in df.columns if c not in ("date", OUTCOME)]
print(f"  Total predictors (incl. rolling + lag): {len(predictors)}")

# Extend metric_name_map to rolling variants and EAD lags
metric_name_map[f"{OUTCOME}_lag3"]    = OUTCOME
metric_name_map[f"{OUTCOME}_mean7_3"] = OUTCOME
for base_col in list(metric_name_map.keys()):
    for suffix in ("_roll_mean7", "_roll_sd7"):
        if base_col + suffix in df.columns:
            metric_name_map[base_col + suffix] = metric_name_map[base_col]

# Skewness correction (operational columns only; calendar/Fourier exempt)
skip_transform = {c for c in predictors if c in _CAL_COLS}
for col in predictors:
    if col in skip_transform:
        continue
    x = df[col].to_numpy(dtype=float)
    skew_val = float(stats.skew(x, nan_policy="omit"))
    if abs(skew_val) > 1:
        if skew_val > 1 and np.all(x > 0):
            df[col] = np.log1p(x)
        elif skew_val > 1:
            df[col] = np.sqrt(x - x.min() + 1)
        else:
            df[col] = x ** 2

# ---------------------------------------------------------------------------
# 2. PHASE 1 — FEATURE SELECTION (development data only)
# ---------------------------------------------------------------------------

df_dev = df[df["date"] <= GLOBAL_DEV_END].reset_index(drop=True)
n_dev  = len(df_dev)

np.random.seed(123)
window_starts_sel = list(range(0, n_dev - (SELECT_WIN + HORIZON) + 1, STRIDE_SEL))
n_sel        = len(window_starts_sel)
total_slots  = n_sel * HORIZON

print(f"\n=== Phase 1: Feature selection ===")
print(f"  Windows: {n_sel}  (train={SELECT_WIN}, stride={STRIDE_SEL})")

selection_counts: dict[str, int] = {}
for w, i in enumerate(window_starts_sel):
    if w % 10 == 0:
        print(f"  [{w + 1}/{n_sel}]  elapsed {time.time() - t0:.0f}s")
    train       = df_dev.iloc[i : i + SELECT_WIN]
    valid_preds = [p for p in predictors if train[p].std() > 0]
    for h in range(1, HORIZON + 1):
        X = train[valid_preds].iloc[: SELECT_WIN - h].to_numpy(dtype=float)
        y = train[OUTCOME].iloc[h:].to_numpy(dtype=float)
        scaler = StandardScaler()
        model  = ElasticNetCV(
            l1_ratio=L1_RATIO, cv=5, max_iter=10_000, n_jobs=-1, random_state=123,
        )
        model.fit(scaler.fit_transform(X), y)
        for p, c in zip(valid_preds, model.coef_):
            if c != 0.0:
                selection_counts[p] = selection_counts.get(p, 0) + 1

print(f"  Phase 1 complete. {len(selection_counts)} predictors ever selected.")

# Select feature set: structural (forced) + top operational (expanded)
all_structural  = [p for p in predictors if p in _CAL_COLS]
candidate_preds = [p for p in predictors if p not in _CAL_COLS]
seed_preds      = sorted(candidate_preds,
                         key=lambda p: selection_counts.get(p, 0), reverse=True)[:N_TOP]
seed_metrics    = {metric_name_map.get(p) for p in seed_preds if metric_name_map.get(p)}
freq_thresh     = FREQ_FLOOR * total_slots
expanded_ops    = [
    p for p in candidate_preds
    if metric_name_map.get(p) in seed_metrics
    and selection_counts.get(p, 0) >= freq_thresh
]
expanded_ops.sort(key=lambda p: selection_counts.get(p, 0), reverse=True)

# Always include mean7_3 (smoothed EAD lag) even if below frequency floor
_mean7_3 = f"{OUTCOME}_mean7_3"
if _mean7_3 not in expanded_ops:
    expanded_ops.append(_mean7_3)

selected_preds = all_structural + expanded_ops

print(f"  Forced structural: {len(all_structural)}")
print(f"  Operational selected: {len(expanded_ops)}")
print(f"  Total: {len(selected_preds)}")

# Save feature selection for inspection
sel_rows = [
    {
        "predictor":    p,
        "metric_name":  metric_name_map.get(p, ""),
        "n_selected":   selection_counts.get(p, 0),
        "total_slots":  total_slots,
        "pct_selected": round(selection_counts.get(p, 0) / total_slots * 100, 2),
        "forced":       p in all_structural,
        "in_model":     p in selected_preds,
    }
    for p in predictors
]
pd.DataFrame(sel_rows).sort_values("n_selected", ascending=False).to_csv(
    OUT_DIR / "ensemble_feature_selection.csv", index=False,
)

# ---------------------------------------------------------------------------
# 3. PHASE 2 — GLOBAL FIT (development data only)
# ---------------------------------------------------------------------------

valid_sel    = [p for p in selected_preds if df_dev[p].std() > 0]
cal_sel      = [p for p in valid_sel if p in _CAL_COLS]
op_sel       = [p for p in valid_sel if p not in _CAL_COLS]
feature_cols = op_sel + cal_sel
penalty_mask = np.array([p not in _CAL_COLS for p in feature_cols])

print(f"\n=== Phase 2: Global fit ({n_dev} rows, {len(feature_cols)} features) ===")

global_models: dict = {}
alpha_floor = 0.0
for h in range(1, HORIZON + 1):
    X_op  = df_dev[op_sel].iloc[: n_dev - h].to_numpy(dtype=float)
    X_cal = df_dev[cal_sel].iloc[h:].to_numpy(dtype=float)
    X_all = np.hstack([X_op, X_cal])
    y_all = df_dev[OUTCOME].iloc[h:].to_numpy(dtype=float)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    cv_model = ElasticNetCV(
        l1_ratio=L1_RATIO, cv=5, max_iter=10_000, n_jobs=-1, random_state=123,
    )
    cv_model.fit(X_scaled, y_all)
    cv_alpha = cv_model.alpha_

    if h == 1:
        alpha_floor = cv_alpha
    alpha = max(cv_alpha, alpha_floor)
    flag  = f" → floored from {cv_alpha:.5f}" if alpha > cv_alpha else ""

    model = PartialElasticNetCV(
        penalty_mask=penalty_mask, l1_ratio=L1_RATIO, max_iter=10_000,
    ).fit_alpha(X_scaled, y_all, alpha)
    global_models[h] = (model, scaler)

    n_nz = int(np.sum(model.coef_[penalty_mask] != 0))
    print(f"  h={h:2d}: alpha={alpha:.5f}{flag}, nonzero={n_nz}/{penalty_mask.sum()}"
          f"  ({time.time() - t0:.0f}s elapsed)")

print("  Global fitting complete.")

# ---------------------------------------------------------------------------
# 4. PREDICTION LOOP
# ---------------------------------------------------------------------------

n = len(df)
origin_rows = [
    i for i in range(TRAIN_WIN - 1, n - HORIZON)
    if ORIGIN_START <= df["date"].iloc[i] <= ORIGIN_END
]

print(f"\n=== Predictions: {len(origin_rows)} origins"
      f" ({ORIGIN_START.date()} – {ORIGIN_END.date()}) ===")

pred_mat   = np.full((len(origin_rows), HORIZON), np.nan)
actual_mat = np.full((len(origin_rows), HORIZON), np.nan)
origin_dates: list = []

for j, origin_idx in enumerate(origin_rows):
    if j % 50 == 0:
        print(f"  [{j + 1}/{len(origin_rows)}]  elapsed {time.time() - t0:.0f}s")

    origin_dates.append(df["date"].iloc[origin_idx])
    actual_mat[j] = df[OUTCOME].iloc[origin_idx + 1 : origin_idx + HORIZON + 1].to_numpy(dtype=float)

    floor_end   = origin_idx - 2
    floor_start = max(0, floor_end - FLOOR_LOOKBACK)
    obs_floor   = float(df[OUTCOME].iloc[floor_start:floor_end].quantile(0.05))

    # Global: operational features at origin, calendar features at each target day
    X_op_g = df[op_sel].iloc[[origin_idx]].to_numpy(dtype=float)

    # Basic: operational features at end of 90-day window
    win_start  = origin_idx - TRAIN_WIN + 1
    train_b    = df.iloc[win_start : origin_idx + 1]
    op_b       = [p for p in expanded_ops if train_b[p].std() > 0]
    X_op_b_org = train_b[op_b].iloc[[-1]].to_numpy(dtype=float)

    for h in range(1, HORIZON + 1):
        # --- Global prediction ---
        mdl_g, scl_g = global_models[h]
        X_cal_g      = df[cal_sel].iloc[[origin_idx + h]].to_numpy(dtype=float)
        pred_g       = mdl_g.predict(scl_g.transform(np.hstack([X_op_g, X_cal_g])))[0]

        # --- Basic rolling prediction ---
        X_op_b  = train_b[op_b].iloc[: TRAIN_WIN - h].to_numpy(dtype=float)
        X_cal_b = df[cal_sel].iloc[win_start + h : origin_idx + 1].to_numpy(dtype=float)
        y_b     = train_b[OUTCOME].iloc[h:].to_numpy(dtype=float)

        X_cal_b_o = df[cal_sel].iloc[[origin_idx + h]].to_numpy(dtype=float)
        X_or_b    = np.hstack([X_op_b_org, X_cal_b_o])

        scaler_b = StandardScaler()
        enet_b   = ElasticNetCV(
            l1_ratio=L1_RATIO, cv=5, max_iter=10_000, n_jobs=-1, random_state=123,
        )
        enet_b.fit(scaler_b.fit_transform(np.nan_to_num(np.hstack([X_op_b, X_cal_b]))), y_b)
        pred_b = enet_b.predict(scaler_b.transform(np.nan_to_num(X_or_b)))[0]

        # --- Ensemble ---
        raw_pred = (pred_g + pred_b) / 2
        pred_mat[j, h - 1] = float(np.clip(raw_pred, obs_floor, PRED_MAX))

print("  Prediction complete.")

# ---------------------------------------------------------------------------
# 5. OUTPUT
# ---------------------------------------------------------------------------

n_fc   = len(origin_rows)
day_cols = [f"day_{d + 1}" for d in range(HORIZON)]

pred_df = pd.DataFrame(pred_mat, columns=day_cols)
pred_df.insert(0, "forecast_id", range(1, n_fc + 1))
pred_df.to_csv(SUB_DIR / "pred_matrix.csv", index=False)

mse_rows = [
    {
        "forecast_id": j + 1,
        "mse_1_5":     _mse(actual_mat[j, :5], pred_mat[j, :5]),
        "mse_6_10":    _mse(actual_mat[j, 5:],  pred_mat[j, 5:]),
    }
    for j in range(n_fc)
]
mse_df = pd.DataFrame(mse_rows)
mse_df.to_csv(SUB_DIR / "mse_summary.csv", index=False)

elapsed = time.time() - t0
print(f"\nTotal elapsed: {elapsed:.0f}s")
print(f"Outputs written to {SUB_DIR}/")

ho = mse_df.dropna(subset=["mse_1_5", "mse_6_10"])
if len(ho) > 0:
    label = "holdout" if VALIDATE else "assessment"
    print(f"\n*** MSE — {label} ({len(ho)} windows) ***")
    print(f"  Mean MSE   days 1–5:  {ho['mse_1_5'].mean():.4f}")
    print(f"  Median MSE days 1–5:  {ho['mse_1_5'].median():.4f}")
    print(f"  Mean MSE   days 6–10: {ho['mse_6_10'].mean():.4f}")
    print(f"  Median MSE days 6–10: {ho['mse_6_10'].median():.4f}")
    print(f"  Periods: {origin_dates[0].date()} – {origin_dates[-1].date()}")
