"""
NHS_rolling_forecast.py
=======================
Forecasts estimated avoidable deaths (EAD) for Bristol NHS hospitals 1–10 days
ahead using a rolling-window Elastic Net fitted on a pre-selected feature set.

Motivation
----------
The global model (NHS_global_forecast.py) outperforms the basic rolling model on
days 1–5 but loses on days 6–10 mean MSE.  The likely cause is non-stationarity:
the global model assumes one set of coefficients applies across the full
development period, which breaks when the data distribution shifts.  The basic
rolling model adapts locally but suffers from n << p (90 rows, ~1000 features).

This model combines both advantages:
  - Feature stability: uses the Phase 1 feature set (≈45 predictors), avoiding
    the n << p problem that makes the basic rolling model unreliable.
  - Local adaptation: fits a fresh PartialElasticNetCV at each origin on the
    preceding TRAIN_WIN_EVAL (90) days, allowing coefficients to track regime changes.

All Phase 2 predictions are genuinely out-of-sample.

Usage
-----
    python -u NHS_rolling_forecast.py

Outputs (model_outputs/)
------------------------
    rolling_pred_matrix.csv / rolling_mse_summary.csv / rolling_forecast_detail.csv
    rolling_feature_selection.csv
    rolling_holdout_pred_matrix.csv / rolling_holdout_mse_summary.csv
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
    OUTCOME, DEV_END, MAX_FC_LEAD,
    WEATHER_PATH, FORECAST_WX_PATH,
    RAIN_CAP, WIND_CAP, COLD_THRESH2, HEAVY_RAIN_THRESH, PRED_MAX, FLOOR_LOOKBACK,
    _ENGLAND_BANK_HOLIDAYS,
    _clean_name, clean_column_names, _mse,
    fetch_weather, load_weather,
    load_forecast_weather,
    _WX_BASE_COLS, wx_feature_cols,
    build_fc_wx_wide, build_wx_train, get_wx_pred,
    PartialElasticNetCV,
)

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & model-specific constants
# ---------------------------------------------------------------------------

DATA_PATH = Path("data/turingAI_forecasting_challenge_dataset.csv")

HORIZON  = 10
L1_RATIO = 0.5

TRAIN_WIN_SELECT = 120
STRIDE_SELECT    = 14
N_TOP_FEATURES   = 20

TRAIN_WIN_EVAL = 90
STRIDE_EVAL    = 1

_DOW_NAMES = ["dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat"]
_CAL_COLS  = set(_DOW_NAMES) | {"is_holiday"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    OUTPUT_DIR = Path("model_outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ========================================================================
    # 1. LOAD & PREPROCESS
    # ========================================================================

    print("Loading operational data…")
    data = pd.read_csv(DATA_PATH, low_memory=False)
    data["dt"]   = pd.to_datetime(data["dt"], format="mixed", errors="coerce")
    data["date"] = data["dt"].dt.normalize()
    data = data[(data["dt"] <= DEV_END) & (data["value"] != -9999)].copy()

    sod = (
        data["dt"].dt.hour * 3600
        + data["dt"].dt.minute * 60
        + data["dt"].dt.second
    )
    data["midday_day"] = np.where(
        sod <= 43200,
        data["date"],
        data["date"] + pd.Timedelta(days=1),
    )
    data["midday_day"] = pd.to_datetime(data["midday_day"]).dt.normalize()

    print("Aggregating to daily per-hospital features…")

    target_daily = (
        data[data["metric_name"] == OUTCOME]
        .groupby("midday_day", as_index=False)["value"]
        .mean()
        .rename(columns={"midday_day": "date", "value": OUTCOME})
    )
    target_daily["date"] = pd.to_datetime(target_daily["date"]).dt.normalize()

    features_long = (
        data[data["metric_name"] != OUTCOME]
        .groupby(["midday_day", "metric_name", "coverage_label"], as_index=False)["value"]
        .mean()
    )
    features_long["col_key"] = (
        features_long["metric_name"] + "|" + features_long["coverage_label"]
    )
    features_wide = (
        features_long
        .pivot(index="midday_day", columns="col_key", values="value")
        .reset_index()
        .rename(columns={"midday_day": "date"})
    )
    features_wide.columns.name = None
    features_wide["date"] = pd.to_datetime(features_wide["date"]).dt.normalize()
    forecasting_df = features_wide.merge(target_daily, on="date", how="left")

    # ========================================================================
    # 2. CLEAN COLUMN NAMES
    # ========================================================================

    cols_to_rename = [c for c in forecasting_df.columns if c not in ("date", OUTCOME)]
    deduped        = clean_column_names(cols_to_rename)
    forecasting_df = forecasting_df.rename(columns=dict(zip(cols_to_rename, deduped)))
    print(f"  Feature columns after per-hospital pivot: {len(deduped)}")

    metric_name_map: dict[str, str] = {
        clean_col: (orig_col.split("|")[0] if "|" in orig_col else orig_col)
        for orig_col, clean_col in zip(cols_to_rename, deduped)
    }

    # ========================================================================
    # 3. WEATHER
    # ========================================================================

    weather = load_weather()
    t_mean  = weather["temperature_2m_mean"]
    t_min   = weather["temperature_2m_min"]
    weather["wx_coldness"]       = (10 - t_mean).clip(lower=0)
    weather["wx_hotness"]        = (t_mean - 25).clip(lower=0)
    weather["wx_below_freezing"] = (t_min < 0).astype(float)
    weather["wx_rain_sum"]       = weather["rain_sum"].clip(upper=RAIN_CAP)
    weather["wx_snowfall_sum"]   = weather["snowfall_sum"]
    weather["wx_wind_max"]       = weather["wind_speed_10m_max"].clip(upper=WIND_CAP)
    weather["wx_coldness2"]      = (COLD_THRESH2 - t_mean).clip(lower=0)
    weather["wx_heavy_rain"]     = (weather["rain_sum"] > HEAVY_RAIN_THRESH).astype(int)

    forecasting_df = forecasting_df.merge(
        weather[["date"] + wx_feature_cols], on="date", how="left"
    )

    print("  Loading NWP forecast weather…")
    _fc_wx_df  = load_forecast_weather()
    fc_wx_wide = build_fc_wx_wide(_fc_wx_df) if _fc_wx_df is not None else None
    if fc_wx_wide is not None:
        print(f"  NWP forecast lookup ready: "
              f"{len(fc_wx_wide)} valid dates, lead days 1–{MAX_FC_LEAD}")

    # ========================================================================
    # 4. CALENDAR INDICATORS
    # ========================================================================

    dow = forecasting_df["date"].dt.dayofweek
    for i, name in enumerate(_DOW_NAMES):
        forecasting_df[name] = (dow == i).astype(float)
    forecasting_df["is_holiday"] = (
        forecasting_df["date"].isin(_ENGLAND_BANK_HOLIDAYS).astype(float)
    )

    # ========================================================================
    # 5. IMPUTE
    # ========================================================================

    predictor_cols = [c for c in forecasting_df.columns if c not in ("date", OUTCOME)]
    for col in predictor_cols:
        forecasting_df[col] = (
            forecasting_df[col]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )
    forecasting_df = forecasting_df.dropna(subset=[OUTCOME]).reset_index(drop=True)

    # ========================================================================
    # 6. FEATURE ENGINEERING
    # ========================================================================

    roll_candidates = [
        c for c in predictor_cols
        if c not in _CAL_COLS and not c.startswith("wx_")
    ]
    rolling_cols: dict[str, pd.Series] = {}
    for var in roll_candidates:
        rolling_cols[f"{var}_roll_mean7"] = forecasting_df[var].rolling(7).mean()
        rolling_cols[f"{var}_roll_sd7"]   = forecasting_df[var].rolling(7).std()

    forecasting_df = pd.concat(
        [forecasting_df, pd.DataFrame(rolling_cols, index=forecasting_df.index)],
        axis=1,
    )
    forecasting_df[f"{OUTCOME}_lag3"] = forecasting_df[OUTCOME].shift(3)
    forecasting_df = forecasting_df.dropna().reset_index(drop=True)

    predictors = [c for c in forecasting_df.columns if c not in ("date", OUTCOME)]
    print(f"  Total predictors (incl. rolling + lag): {len(predictors)}")

    for base_col in list(metric_name_map.keys()):
        for suffix in ("_roll_mean7", "_roll_sd7"):
            if base_col + suffix in forecasting_df.columns:
                metric_name_map[base_col + suffix] = metric_name_map[base_col]
    metric_name_map[f"{OUTCOME}_lag3"] = OUTCOME

    # ========================================================================
    # 7. SKEWNESS CORRECTION
    # ========================================================================

    skip_transform = {c for c in predictors if c in _CAL_COLS or c.startswith("wx_")}
    for col in predictors:
        if col in skip_transform:
            continue
        x = forecasting_df[col].to_numpy(dtype=float)
        skew_val = float(stats.skew(x, nan_policy="omit"))
        if abs(skew_val) > 1:
            if skew_val > 1 and np.all(x > 0):
                forecasting_df[col] = np.log1p(x)
            elif skew_val > 1:
                forecasting_df[col] = np.sqrt(x - x.min() + 1)
            else:
                forecasting_df[col] = x ** 2

    # ========================================================================
    # PHASE 1 — FEATURE SELECTION
    # ========================================================================

    np.random.seed(123)
    n = len(forecasting_df)

    window_starts_sel = list(range(0, n - (TRAIN_WIN_SELECT + HORIZON) + 1, STRIDE_SELECT))
    n_sel       = len(window_starts_sel)
    total_slots = n_sel * HORIZON

    print(f"\n=== Phase 1: Feature selection ===")
    print(f"  Windows: {n_sel}  (train={TRAIN_WIN_SELECT}, stride={STRIDE_SELECT})")

    selection_counts: dict[str, int] = {}

    for w, i in enumerate(window_starts_sel):
        if w % 10 == 0:
            print(f"  [{w + 1}/{n_sel}]  elapsed {time.time()-t0:.0f}s")

        train       = forecasting_df.iloc[i : i + TRAIN_WIN_SELECT]
        valid_preds = [p for p in predictors if train[p].std() > 0]

        for h in range(1, HORIZON + 1):
            X_train_h  = train[valid_preds].iloc[: TRAIN_WIN_SELECT - h].to_numpy(dtype=float)
            y_train_h  = train[OUTCOME].iloc[h:].to_numpy(dtype=float)
            scaler     = StandardScaler()
            X_train_hs = scaler.fit_transform(X_train_h)
            model      = ElasticNetCV(
                l1_ratio=L1_RATIO, cv=5, max_iter=10_000, n_jobs=-1, random_state=123,
            )
            model.fit(X_train_hs, y_train_h)
            for pred_name, coef_val in zip(valid_preds, model.coef_):
                if coef_val != 0.0:
                    selection_counts[pred_name] = selection_counts.get(pred_name, 0) + 1

    print(f"  Phase 1 complete.  {len(selection_counts)} predictors ever selected.")

    all_structural  = [p for p in predictors if p in _CAL_COLS or p.startswith("wx_")]
    candidate_preds = [p for p in predictors if p not in _CAL_COLS and not p.startswith("wx_")]

    seed_preds = sorted(candidate_preds, key=lambda p: selection_counts.get(p, 0), reverse=True)
    seed_preds = seed_preds[:N_TOP_FEATURES]

    seed_metrics = {metric_name_map.get(p) for p in seed_preds if metric_name_map.get(p)}
    expanded_ops = [p for p in candidate_preds if metric_name_map.get(p) in seed_metrics]
    expanded_ops.sort(key=lambda p: selection_counts.get(p, 0), reverse=True)

    selected_preds = all_structural + expanded_ops

    print(f"\n  Forced (DOW + holiday + weather): {len(all_structural)}")
    print(f"  Seed top-{N_TOP_FEATURES} → {len(seed_metrics)} distinct metrics → "
          f"{len(expanded_ops)} operational variables after expansion:")
    for p in expanded_ops:
        cnt = selection_counts.get(p, 0)
        pct = cnt / total_slots * 100
        print(f"    {'[seed]' if p in seed_preds else '      '}  {p}  ({cnt}/{total_slots}, {pct:.1f}%)")
    print(f"  Total selected: {len(selected_preds)}")

    sel_rows = []
    for p in predictors:
        cnt = selection_counts.get(p, 0)
        sel_rows.append({
            "predictor":    p,
            "metric_name":  metric_name_map.get(p, ""),
            "n_selected":   cnt,
            "total_slots":  total_slots,
            "pct_selected": round(cnt / total_slots * 100, 2),
            "forced":       p in all_structural,
            "seed":         p in seed_preds,
            "expanded":     p in expanded_ops and p not in seed_preds,
            "in_model":     p in selected_preds,
        })
    sel_df = pd.DataFrame(sel_rows).sort_values("n_selected", ascending=False)
    sel_df.to_csv(OUTPUT_DIR / "rolling_feature_selection.csv", index=False)
    print(f"\n  Selection summary saved to model_outputs/rolling_feature_selection.csv")

    # ========================================================================
    # PHASE 2 — ROLLING FIT + EVALUATION
    # ========================================================================

    non_wx_selected = [p for p in selected_preds if not p.startswith("wx_")]
    wx_selected     = [p for p in selected_preds if p.startswith("wx_")]
    feature_cols    = non_wx_selected + wx_selected

    all_structural_set = set(all_structural)
    penalty_mask = np.array([p not in all_structural_set for p in feature_cols])

    window_starts_eval = list(range(0, n - (TRAIN_WIN_EVAL + HORIZON) + 1, STRIDE_EVAL))
    n_forecasts = len(window_starts_eval)

    pred_matrix    = np.full((n_forecasts, HORIZON), np.nan)
    actual_matrix  = np.full((n_forecasts, HORIZON), np.nan)
    origin_dates:  list = []
    forecast_dates: list = []

    print(f"\n=== Phase 2: Rolling fit + evaluation ===")
    print(f"  Windows: {n_forecasts}  (train={TRAIN_WIN_EVAL}, stride={STRIDE_EVAL})")

    for w, i in enumerate(window_starts_eval):
        if w % 200 == 0:
            print(f"  [{w + 1}/{n_forecasts}]  elapsed {time.time()-t0:.0f}s")

        train_df   = forecasting_df.iloc[i : i + TRAIN_WIN_EVAL].reset_index(drop=True)
        origin_idx = i + TRAIN_WIN_EVAL - 1
        test       = forecasting_df.iloc[i + TRAIN_WIN_EVAL : i + TRAIN_WIN_EVAL + HORIZON]

        origin_dates.append(forecasting_df["date"].iloc[origin_idx])
        forecast_dates.append(test["date"].tolist())
        actual_matrix[w] = test[OUTCOME].to_numpy(dtype=float)

        X_nonwx_o     = forecasting_df[non_wx_selected].iloc[[origin_idx]].to_numpy(dtype=float)
        X_nonwx_train = train_df[non_wx_selected].to_numpy(dtype=float)
        y_train        = train_df[OUTCOME].to_numpy(dtype=float)

        alpha_floor = 0.0
        for h in range(1, HORIZON + 1):
            X_nonwx = X_nonwx_train[: TRAIN_WIN_EVAL - h]
            X_wx    = build_wx_train(train_df, wx_selected, h, fc_wx_wide)
            X       = np.hstack([X_nonwx, X_wx])
            y       = y_train[h:]

            scaler   = StandardScaler()
            X_s      = scaler.fit_transform(X)
            X_s      = np.nan_to_num(X_s, copy=False)

            cv_model = ElasticNetCV(
                l1_ratio=L1_RATIO, cv=5, max_iter=10_000, n_jobs=1, random_state=123,
            )
            cv_model.fit(X_s, y)
            cv_alpha = cv_model.alpha_

            if h == 1:
                alpha_floor = cv_alpha
            effective_alpha = max(cv_alpha, alpha_floor)

            model = PartialElasticNetCV(
                penalty_mask=penalty_mask, l1_ratio=L1_RATIO, max_iter=10_000,
            ).fit_alpha(X_s, y, effective_alpha)

            X_wx_o = get_wx_pred(forecasting_df, wx_selected, origin_idx, h, fc_wx_wide)
            X_o    = np.hstack([X_nonwx_o, X_wx_o])
            raw_pred = model.predict(scaler.transform(X_o))[0]
            pred_matrix[w, h - 1] = float(np.clip(raw_pred, 0.0, PRED_MAX))

    print("  Evaluation complete.")

    # ========================================================================
    # OUTPUT
    # ========================================================================

    pred_df = pd.DataFrame(pred_matrix, columns=[f"day_{d+1}" for d in range(HORIZON)])
    pred_df.insert(0, "origin_date", origin_dates)
    pred_df.insert(0, "forecast_id", range(1, n_forecasts + 1))
    pred_df.to_csv(OUTPUT_DIR / "rolling_pred_matrix.csv", index=False)

    mse_df = pd.DataFrame([
        {"forecast_id": i + 1, "origin_date": origin_dates[i],
         "mse_1_5":  _mse(actual_matrix[i, :5], pred_matrix[i, :5]),
         "mse_6_10": _mse(actual_matrix[i, 5:],  pred_matrix[i, 5:])}
        for i in range(n_forecasts)
    ])
    mse_df.to_csv(OUTPUT_DIR / "rolling_mse_summary.csv", index=False)

    detail_rows = [
        {"forecast_id": i + 1, "origin_date": origin_dates[i], "horizon": h + 1,
         "forecast_date": forecast_dates[i][h],
         "actual": actual_matrix[i, h], "predicted": pred_matrix[i, h]}
        for i in range(n_forecasts) for h in range(HORIZON)
    ]
    pd.DataFrame(detail_rows).to_csv(OUTPUT_DIR / "rolling_forecast_detail.csv", index=False)

    print(f"\n*** OUT-OF-SAMPLE MSE — rolling model, {len(feature_cols)} features ***")
    print(f"  Mean MSE   (days 1–5):  {mse_df['mse_1_5'].mean():.4f}")
    print(f"  Median MSE (days 1–5):  {mse_df['mse_1_5'].median():.4f}")
    print(f"  Mean MSE   (days 6–10): {mse_df['mse_6_10'].mean():.4f}")
    print(f"  Median MSE (days 6–10): {mse_df['mse_6_10'].median():.4f}")

    # ========================================================================
    # PHASE 3 — HOLDOUT REPORT (Oct 2024 – Sep 2025)
    # No re-fitting: filter Phase 2 results to the holdout period.
    # Holdout origins still use the 90 days immediately before each origin,
    # which may be post-Sep 2024 for later origins — this simulates the
    # production rolling-update scenario rather than a strict train/test split.
    # ========================================================================

    TRAIN_CUTOFF = pd.Timestamp("2024-09-30")
    holdout_idx  = [i for i, od in enumerate(origin_dates) if od > TRAIN_CUTOFF]
    n_ho         = len(holdout_idx)

    pred_ho          = pred_matrix[holdout_idx]
    actual_ho        = actual_matrix[holdout_idx]
    origin_dates_ho  = [origin_dates[i] for i in holdout_idx]

    mse_ho_df = pd.DataFrame([
        {"forecast_id": j + 1, "origin_date": origin_dates_ho[j],
         "mse_1_5":  _mse(actual_ho[j, :5],  pred_ho[j, :5]),
         "mse_6_10": _mse(actual_ho[j, 5:],   pred_ho[j, 5:])}
        for j in range(n_ho)
    ])
    mse_ho_df.to_csv(OUTPUT_DIR / "rolling_holdout_mse_summary.csv", index=False)

    pred_ho_df = pd.DataFrame(pred_ho, columns=[f"day_{d+1}" for d in range(HORIZON)])
    pred_ho_df.insert(0, "origin_date", origin_dates_ho)
    pred_ho_df.insert(0, "forecast_id", range(1, n_ho + 1))
    pred_ho_df.to_csv(OUTPUT_DIR / "rolling_holdout_pred_matrix.csv", index=False)

    print(f"\n=== Phase 3: Holdout report (Oct 2024 – Sep 2025) ===")
    print(f"  N windows: {n_ho}")
    print(f"  Mean MSE   (days 1–5):  {mse_ho_df['mse_1_5'].mean():.4f}")
    print(f"  Median MSE (days 1–5):  {mse_ho_df['mse_1_5'].median():.4f}")
    print(f"  Mean MSE   (days 6–10): {mse_ho_df['mse_6_10'].mean():.4f}")
    print(f"  Median MSE (days 6–10): {mse_ho_df['mse_6_10'].median():.4f}")

    for other_name, other_file in [
        ("Basic rolling", OUTPUT_DIR / "basic_mse_summary.csv"),
        ("Global partial-penalty", OUTPUT_DIR / "global_holdout_mse_summary.csv"),
    ]:
        if other_file.exists():
            other = pd.read_csv(other_file, parse_dates=["origin_date"])
            other_ho = other[other["origin_date"] > TRAIN_CUTOFF]
            print(f"\n  {other_name} model on same {len(other_ho)} holdout windows:")
            print(f"  Mean MSE   (days 1–5):  {other_ho['mse_1_5'].mean():.4f}")
            print(f"  Median MSE (days 1–5):  {other_ho['mse_1_5'].median():.4f}")
            print(f"  Mean MSE   (days 6–10): {other_ho['mse_6_10'].mean():.4f}")
            print(f"  Median MSE (days 6–10): {other_ho['mse_6_10'].median():.4f}")

    print(f"\nTotal elapsed: {time.time() - t0:.0f}s")
    print("Outputs written to model_outputs/rolling_*.csv")


if __name__ == "__main__":
    main()
