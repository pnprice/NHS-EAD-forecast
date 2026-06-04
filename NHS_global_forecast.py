"""
NHS_global_forecast.py
======================
Forecasts estimated avoidable deaths (EAD) for Bristol NHS hospitals 1–10 days
ahead using a global Elastic Net model.  This is the main model script for the
SPHERE-PPL forecasting contest (assessment period Oct 2025–Mar 2026).

Motivation
----------
A naive rolling-window Elastic Net fitted on ~1000 predictors produces wildly
unstable coefficients because n (≈90-day window) << p.  This script addresses
that by first selecting a small, stable feature set and then fitting a single
model on the full development period.

Approach
--------
Phase 1 — Feature selection
    Roll 120-day windows with a 14-day stride.  For each window × horizon, fit
    an ElasticNetCV and record which predictors receive a non-zero coefficient.
    Keep DOW, bank holidays, and all weather features; keep the top-N operational
    predictors by selection frequency, expanding to include all hospital/service
    variants of any selected metric.

Phase 2 — Global fit
    Fit one model per horizon on the full development period.  Alpha is selected
    by sklearn ElasticNetCV; the actual fit uses PartialElasticNetCV (block
    coordinate descent) so that structural features (DOW, holidays, weather) are
    estimated by OLS without penalty while operational features get L1+L2.
    A monotone-alpha constraint ensures longer horizons are at least as
    regularised as h=1.

Phase 3 — Holdout evaluation
    Re-fit Phase 2 using only data up to Sep 2024, evaluate on Oct 2024–Sep 2025.

Usage
-----
    python -u NHS_global_forecast.py

Outputs (model_outputs/)
------------------------
    global_pred_matrix.csv / global_mse_summary.csv / global_forecast_detail.csv
    global_coef_detail.csv / global_feature_selection.csv
    global_holdout_pred_matrix.csv / global_holdout_mse_summary.csv
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
    _WX_BASE_COLS, wx_feature_cols, _MEAN3_BASE_COLS, _WX_MEAN3_COLS,
    build_fc_wx_wide, build_wx_train, get_wx_pred,
    build_wx_mean3_train, get_wx_mean3_pred,
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

TRAIN_WIN_SELECT  = 120
STRIDE_SELECT     = 14
N_TOP_FEATURES    = 20
EXPAND_FREQ_FLOOR = 0.05

TRAIN_WIN_EVAL = 90
STRIDE_EVAL    = 1

_DAY_AFTER_HOLIDAYS = {d + pd.Timedelta(days=1) for d in _ENGLAND_BANK_HOLIDAYS}
_DOW_NAMES = ["dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat"]
_CAL_COLS  = set(_DOW_NAMES) | {"is_holiday", "is_day_after_holiday"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    NO_WX      = "--no-wx" in sys.argv
    MODEL_TAG  = "nwx_global" if NO_WX else "global"
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

    # Map each cleaned column name back to its original metric (part before "|").
    # Used to expand the selected feature set to all hospital/service variants.
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
    forecasting_df["is_day_after_holiday"] = (
        forecasting_df["date"].isin(_DAY_AFTER_HOLIDAYS).astype(float)
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
    if NO_WX:
        predictors = [p for p in predictors
                      if not p.startswith("wx_") and p not in ("is_holiday", "is_day_after_holiday")]
        print(f"  [--no-wx] Excluding weather and bank holiday features")
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

    # Determine selected feature set
    all_structural = [p for p in predictors if p in _CAL_COLS or p.startswith("wx_")]
    candidate_preds = [p for p in predictors if p not in _CAL_COLS and not p.startswith("wx_")]

    seed_preds = sorted(candidate_preds, key=lambda p: selection_counts.get(p, 0), reverse=True)
    seed_preds = seed_preds[:N_TOP_FEATURES]

    seed_metrics = {metric_name_map.get(p) for p in seed_preds if metric_name_map.get(p)}
    freq_floor   = EXPAND_FREQ_FLOOR * total_slots
    expanded_ops = [
        p for p in candidate_preds
        if metric_name_map.get(p) in seed_metrics
        and selection_counts.get(p, 0) >= freq_floor
    ]
    expanded_ops.sort(key=lambda p: selection_counts.get(p, 0), reverse=True)

    selected_preds = all_structural + expanded_ops

    print(f"\n  Forced (DOW + holiday + weather): {len(all_structural)}")
    print(f"  Seed top-{N_TOP_FEATURES} → {len(seed_metrics)} distinct metrics → "
          f"{len(expanded_ops)} operational variables after expansion "
          f"(≥{EXPAND_FREQ_FLOOR:.0%} freq floor):")
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
    sel_df.to_csv(OUTPUT_DIR / f"{MODEL_TAG}_feature_selection.csv", index=False)
    print(f"\n  Selection summary saved to model_outputs/global_feature_selection.csv")

    # ========================================================================
    # PHASE 2 — GLOBAL FIT
    # ========================================================================

    print(f"\n=== Phase 2: Global fit ===")

    valid_selected  = [p for p in selected_preds if forecasting_df[p].std() > 0]
    if len(valid_selected) < len(selected_preds):
        dropped = set(selected_preds) - set(valid_selected)
        print(f"  Dropped {len(dropped)} zero-variance features: {dropped}")

    non_wx_selected = [p for p in valid_selected if not p.startswith("wx_")]
    wx_selected     = [p for p in valid_selected if p.startswith("wx_")]
    cal_selected    = [p for p in non_wx_selected if p in _CAL_COLS]
    op_selected     = [p for p in non_wx_selected if p not in _CAL_COLS]
    mean3_cols_use  = [] if NO_WX else _WX_MEAN3_COLS
    feature_cols    = op_selected + cal_selected + wx_selected + mean3_cols_use

    all_structural_set = set(all_structural) | set(mean3_cols_use)
    penalty_mask = np.array([p not in all_structural_set for p in feature_cols])

    print(f"  Training on all {n} rows using {len(feature_cols)} features…")

    global_models: dict = {}
    coef_records:  list = []
    alpha_floor = 0.0

    for h in range(1, HORIZON + 1):
        X_op     = forecasting_df[op_selected].iloc[: n - h].to_numpy(dtype=float)
        X_cal    = forecasting_df[cal_selected].iloc[h:].to_numpy(dtype=float)
        X_wx     = build_wx_train(forecasting_df, wx_selected, h, fc_wx_wide)
        X_wx_m3  = build_wx_mean3_train(forecasting_df, h, fc_wx_wide) if mean3_cols_use else np.empty((len(X_op), 0))
        X_all    = np.hstack([X_op, X_cal, X_wx, X_wx_m3])
        y_all    = forecasting_df[OUTCOME].iloc[h:].to_numpy(dtype=float)

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)

        cv_sklearn = ElasticNetCV(l1_ratio=L1_RATIO, cv=5, max_iter=10_000, n_jobs=-1, random_state=123)
        cv_sklearn.fit(X_scaled, y_all)
        cv_alpha = cv_sklearn.alpha_

        if h == 1:
            alpha_floor = cv_alpha
        effective_alpha = max(cv_alpha, alpha_floor)
        flag = f" → floored from {cv_alpha:.5f}" if effective_alpha > cv_alpha else ""

        model = PartialElasticNetCV(
            penalty_mask=penalty_mask, l1_ratio=L1_RATIO, max_iter=10_000,
        ).fit_alpha(X_scaled, y_all, effective_alpha)

        global_models[h] = (model, scaler)
        n_nz = int(np.sum(model.coef_[penalty_mask] != 0))
        print(f"  h={h:2d}: alpha={effective_alpha:.5f}{flag}, nonzero={n_nz}/{penalty_mask.sum()}  "
              f"({time.time()-t0:.0f}s elapsed)")

        for pred_name, coef_val in zip(feature_cols, model.coef_):
            if coef_val != 0.0:
                coef_records.append({"horizon": h, "predictor": pred_name, "coefficient": coef_val})

    print("  Global fitting complete.")

    # ========================================================================
    # EVALUATION — stride through same windows as basic model (IN-SAMPLE)
    # ========================================================================

    def _predict_windows(
        win_starts: list[int],
        models: dict,
        label: str,
    ) -> tuple[np.ndarray, np.ndarray, list, list]:
        """Build prediction and actual matrices for a list of window start indices."""
        n_fc          = len(win_starts)
        pred_mat      = np.full((n_fc, HORIZON), np.nan)
        actual_mat    = np.full((n_fc, HORIZON), np.nan)
        orig_dates:   list = []
        fc_dates:     list = []

        print(f"\n  Evaluating {n_fc} windows ({label})…")
        for j, i in enumerate(win_starts):
            if j % 200 == 0:
                print(f"  [{j + 1}/{n_fc}]")

            origin_idx = i + TRAIN_WIN_EVAL - 1
            test       = forecasting_df.iloc[i + TRAIN_WIN_EVAL : i + TRAIN_WIN_EVAL + HORIZON]

            orig_dates.append(forecasting_df["date"].iloc[origin_idx])
            fc_dates.append(test["date"].tolist())
            actual_mat[j] = test[OUTCOME].to_numpy(dtype=float)

            floor_end   = origin_idx - 2
            floor_start = max(0, floor_end - FLOOR_LOOKBACK)
            obs_floor   = float(forecasting_df[OUTCOME].iloc[floor_start:floor_end].min())

            X_op_o = forecasting_df[op_selected].iloc[[origin_idx]].to_numpy(dtype=float)
            for h in range(1, HORIZON + 1):
                mdl, scl  = models[h]
                X_cal_o   = forecasting_df[cal_selected].iloc[[origin_idx + h]].to_numpy(dtype=float)
                X_wx_o    = get_wx_pred(forecasting_df, wx_selected, origin_idx, h, fc_wx_wide)
                X_wx_m3_o = get_wx_mean3_pred(forecasting_df, origin_idx, h, fc_wx_wide) if mean3_cols_use else np.empty((1, 0))
                X_o       = np.hstack([X_op_o, X_cal_o, X_wx_o, X_wx_m3_o])
                raw_pred  = mdl.predict(scl.transform(X_o))[0]
                pred_mat[j, h - 1] = float(np.clip(raw_pred, obs_floor, PRED_MAX))

        print("  Evaluation complete.")
        return pred_mat, actual_mat, orig_dates, fc_dates

    window_starts_eval = list(range(0, n - (TRAIN_WIN_EVAL + HORIZON) + 1, STRIDE_EVAL))
    pred_matrix, actual_matrix, origin_dates, forecast_dates = _predict_windows(
        window_starts_eval, global_models, f"stride={STRIDE_EVAL}, IN-SAMPLE"
    )
    n_forecasts = len(window_starts_eval)

    # ========================================================================
    # OUTPUT — Phase 2
    # ========================================================================

    mse_df = pd.DataFrame([
        {"forecast_id": i + 1, "origin_date": origin_dates[i],
         "mse_1_5":  _mse(actual_matrix[i, :5], pred_matrix[i, :5]),
         "mse_6_10": _mse(actual_matrix[i, 5:],  pred_matrix[i, 5:])}
        for i in range(n_forecasts)
    ])
    mse_df.to_csv(OUTPUT_DIR / f"{MODEL_TAG}_mse_summary.csv", index=False)

    pred_df = pd.DataFrame(pred_matrix, columns=[f"day_{d+1}" for d in range(HORIZON)])
    pred_df.insert(0, "origin_date", origin_dates)
    pred_df.insert(0, "forecast_id", range(1, n_forecasts + 1))
    pred_df.to_csv(OUTPUT_DIR / f"{MODEL_TAG}_pred_matrix.csv", index=False)

    detail_rows = [
        {"forecast_id": i + 1, "origin_date": origin_dates[i], "horizon": h + 1,
         "forecast_date": forecast_dates[i][h],
         "actual": actual_matrix[i, h], "predicted": pred_matrix[i, h]}
        for i in range(n_forecasts) for h in range(HORIZON)
    ]
    pd.DataFrame(detail_rows).to_csv(OUTPUT_DIR / f"{MODEL_TAG}_forecast_detail.csv", index=False)
    pd.DataFrame(coef_records).to_csv(OUTPUT_DIR / f"{MODEL_TAG}_coef_detail.csv", index=False)

    print(f"\n*** IN-SAMPLE MSE — global model, {len(feature_cols)} features, weather at target day ***")
    print(f"  Mean MSE   (days 1–5):  {mse_df['mse_1_5'].mean():.4f}")
    print(f"  Median MSE (days 1–5):  {mse_df['mse_1_5'].median():.4f}")
    print(f"  Mean MSE   (days 6–10): {mse_df['mse_6_10'].mean():.4f}")
    print(f"  Median MSE (days 6–10): {mse_df['mse_6_10'].median():.4f}")

    # ========================================================================
    # PHASE 3 — HOLDOUT EVALUATION (out-of-sample)
    # ========================================================================

    TRAIN_CUTOFF = pd.Timestamp("2024-09-30")

    print(f"\n=== Phase 3: Holdout evaluation ===")
    print(f"  Training cutoff: {TRAIN_CUTOFF.date()}  (holdout: Oct 2024 – Sep 2025)")

    df_train = forecasting_df[forecasting_df["date"] <= TRAIN_CUTOFF].reset_index(drop=True)
    n_tr     = len(df_train)
    print(f"  Training rows: {n_tr}  Holdout rows: {n - n_tr}")

    holdout_models: dict = {}
    alpha_floor_ho = 0.0

    for h in range(1, HORIZON + 1):
        X_op_h    = df_train[op_selected].iloc[: n_tr - h].to_numpy(dtype=float)
        X_cal_h   = df_train[cal_selected].iloc[h:].to_numpy(dtype=float)
        X_wx_h    = build_wx_train(df_train, wx_selected, h, fc_wx_wide)
        X_wx_m3_h = build_wx_mean3_train(df_train, h, fc_wx_wide) if mean3_cols_use else np.empty((len(X_op_h), 0))
        X_h       = np.hstack([X_op_h, X_cal_h, X_wx_h, X_wx_m3_h])
        y_h       = df_train[OUTCOME].iloc[h:].to_numpy(dtype=float)

        scaler_h  = StandardScaler()
        X_h_s     = scaler_h.fit_transform(X_h)

        cv_h = ElasticNetCV(l1_ratio=L1_RATIO, cv=5, max_iter=10_000, n_jobs=-1, random_state=123)
        cv_h.fit(X_h_s, y_h)
        cv_alpha_h = cv_h.alpha_

        if h == 1:
            alpha_floor_ho = cv_alpha_h
        effective_alpha_h = max(cv_alpha_h, alpha_floor_ho)
        flag = f" → floored from {cv_alpha_h:.5f}" if effective_alpha_h > cv_alpha_h else ""

        model_h = PartialElasticNetCV(
            penalty_mask=penalty_mask, l1_ratio=L1_RATIO, max_iter=10_000,
        ).fit_alpha(X_h_s, y_h, effective_alpha_h)
        holdout_models[h] = (model_h, scaler_h)

        n_nz = int(np.sum(model_h.coef_[penalty_mask] != 0))
        print(f"  h={h:2d}: alpha={effective_alpha_h:.5f}{flag}, nonzero={n_nz}/{penalty_mask.sum()}")

    holdout_starts = [
        i for i in window_starts_eval
        if forecasting_df["date"].iloc[i + TRAIN_WIN_EVAL - 1] > TRAIN_CUTOFF
    ]
    pred_ho, actual_ho, origin_dates_ho, forecast_dates_ho = _predict_windows(
        holdout_starts, holdout_models, "holdout"
    )
    n_ho = len(holdout_starts)

    mse_ho_df = pd.DataFrame([
        {"forecast_id": j + 1, "origin_date": origin_dates_ho[j],
         "mse_1_5":  _mse(actual_ho[j, :5],  pred_ho[j, :5]),
         "mse_6_10": _mse(actual_ho[j, 5:],   pred_ho[j, 5:])}
        for j in range(n_ho)
    ])
    mse_ho_df.to_csv(OUTPUT_DIR / f"{MODEL_TAG}_holdout_mse_summary.csv", index=False)

    pred_ho_df = pd.DataFrame(pred_ho, columns=[f"day_{d+1}" for d in range(HORIZON)])
    pred_ho_df.insert(0, "origin_date", origin_dates_ho)
    pred_ho_df.insert(0, "forecast_id", range(1, n_ho + 1))
    pred_ho_df.to_csv(OUTPUT_DIR / f"{MODEL_TAG}_holdout_pred_matrix.csv", index=False)

    print(f"\n*** OUT-OF-SAMPLE MSE — global model, holdout Oct 2024 – Sep 2025 ***")
    print(f"  N windows: {n_ho}")
    print(f"  Mean MSE   (days 1–5):  {mse_ho_df['mse_1_5'].mean():.4f}")
    print(f"  Median MSE (days 1–5):  {mse_ho_df['mse_1_5'].median():.4f}")
    print(f"  Mean MSE   (days 6–10): {mse_ho_df['mse_6_10'].mean():.4f}")
    print(f"  Median MSE (days 6–10): {mse_ho_df['mse_6_10'].median():.4f}")

    basic_mse_path = OUTPUT_DIR / "basic_mse_summary.csv"
    if basic_mse_path.exists():
        basic_all = pd.read_csv(basic_mse_path, parse_dates=["origin_date"])
        basic_ho  = basic_all[basic_all["origin_date"] > TRAIN_CUTOFF]
        print(f"\n  Basic rolling model on same {len(basic_ho)} holdout windows:")
        print(f"  Mean MSE   (days 1–5):  {basic_ho['mse_1_5'].mean():.4f}")
        print(f"  Median MSE (days 1–5):  {basic_ho['mse_1_5'].median():.4f}")
        print(f"  Mean MSE   (days 6–10): {basic_ho['mse_6_10'].mean():.4f}")
        print(f"  Median MSE (days 6–10): {basic_ho['mse_6_10'].median():.4f}")

    print(f"\nTotal elapsed: {time.time() - t0:.0f}s")
    print("Outputs written to model_outputs/global_*.csv")


if __name__ == "__main__":
    main()
