"""
NHS_basic_forecast.py
=====================
Rolling-window Elastic Net for forecasting estimated avoidable deaths (EAD)
1–10 days ahead.  Sweeps TRAIN_WINS to find the best training window length.

Design choices
--------------
- Per-hospital features: pivot on (metric_name × coverage_label) so BRI, NBT,
  and WGH become distinct columns rather than being averaged together.
- Calendar features (DOW dummies, is_holiday, is_day_after_holiday) evaluated
  at the target day, not the origin day, so they encode the day being forecast.
- Weather at target day: NWP forecast values (Open-Meteo Previous Runs API,
  lead 1–7 days), falling back to observed where NWP is unavailable.  A 3-day
  mean of coldness/hotness/coldness2 captures sustained cold/heat spells.
- Operational features restricted to those selected by Phase 1 of
  NHS_global_forecast.py, preventing noise-chasing in short windows.
- One ElasticNet (l1_ratio=0.5, 5-fold CV) per horizon per window.

Usage
-----
    python NHS_basic_forecast.py
"""

from __future__ import annotations

import sys
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
    wx_feature_cols, _MEAN3_BASE_COLS,
    load_forecast_weather, build_fc_wx_wide,
    PartialElasticNetCV,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths & model-specific constants
# ---------------------------------------------------------------------------

DATA_PATH  = Path("data/turingAI_forecasting_challenge_dataset.csv")
OUTPUT_DIR = Path("model_outputs")

HOLDOUT_START = pd.Timestamp("2024-10-01")
HORIZON       = 10
STRIDE        = 1
L1_RATIO      = 0.5

_argv_nums  = [x for x in sys.argv[1:] if not x.startswith("-")]
TRAIN_WINS  = [int(x) for x in _argv_nums] or [70, 90, 110, 130]
USE_PARTIAL = "--partial"    in sys.argv
NO_WX       = "--no-wx"     in sys.argv
EAD_SMOOTH  = "--ead-smooth" in sys.argv

_sel_tag        = "nwx_global" if NO_WX else "global"
GLOBAL_SEL_PATH = Path(f"model_outputs/{_sel_tag}_feature_selection.csv")

_DOW_NAMES = ["dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat"]
_CAL_NAMES = _DOW_NAMES + ["is_holiday", "is_day_after_holiday"]

_DAY_AFTER_HOLIDAYS = {d + pd.Timedelta(days=1) for d in _ENGLAND_BANK_HOLIDAYS}

# ---------------------------------------------------------------------------
# Rolling-loop weather/mean3 helpers
# These reference fc_wx_wide as a module-level global (set in __main__ block)
# ---------------------------------------------------------------------------


def _wx_train(df: pd.DataFrame, wx_cols: list[str], i: int, h: int, train_win: int) -> np.ndarray:
    """Weather at target training rows [i+h : i+train_win], NWP-corrected where available."""
    X_act = df[wx_cols].iloc[i + h : i + train_win].to_numpy(dtype=float)
    if fc_wx_wide is None:
        return X_act
    fc_cols = [f"{c}_L{min(h, MAX_FC_LEAD)}" for c in wx_cols]
    td  = df["date"].iloc[i + h : i + train_win]
    mg  = pd.DataFrame({"date": td.values}).merge(fc_wx_wide[fc_cols].reset_index(), on="date", how="left")
    X_fc = mg[fc_cols].to_numpy(dtype=float)
    return np.where(np.isnan(X_fc), X_act, X_fc)


def _wx_pred(df: pd.DataFrame, wx_cols: list[str], origin_idx: int, h: int) -> np.ndarray:
    """Weather at a single target row for prediction, NWP-corrected where available."""
    X_act = df[wx_cols].iloc[[origin_idx + h]].to_numpy(dtype=float)
    if fc_wx_wide is None:
        return X_act
    fc_cols = [f"{c}_L{min(h, MAX_FC_LEAD)}" for c in wx_cols]
    td = df["date"].iloc[origin_idx + h]
    if td not in fc_wx_wide.index:
        return X_act
    X_fc = fc_wx_wide.loc[td, fc_cols].to_numpy(dtype=float).reshape(1, -1)
    return np.where(np.isnan(X_fc), X_act, X_fc)


def _mean3_train(df: pd.DataFrame, base_cols: list[str], i: int, h: int, train_win: int) -> np.ndarray:
    """3-day mean of base_cols at training target rows, NWP-corrected where available.

    Averages offsets [h, h-1, h-2] relative to each origin row.  When offset < 0
    (only for h=1, off=-1, i=0) the first row is repeated to fill the gap.
    """
    layers = []
    for off in [h, h - 1, h - 2]:
        start, end = i + off, i + train_win - h + off
        if start < 0:
            avail = df[base_cols].iloc[0:end].to_numpy(dtype=float)
            X_act = np.vstack([np.tile(avail[[0]], (-start, 1)), avail])
        else:
            X_act = df[base_cols].iloc[start:end].to_numpy(dtype=float)
        if off > 0 and fc_wx_wide is not None:
            fc_cols = [f"{c}_L{min(off, MAX_FC_LEAD)}" for c in base_cols]
            td  = df["date"].iloc[start:end]
            mg  = pd.DataFrame({"date": td.values}).merge(
                fc_wx_wide[fc_cols].reset_index(), on="date", how="left"
            )
            X_fc = mg[fc_cols].to_numpy(dtype=float)
            layers.append(np.where(np.isnan(X_fc), X_act, X_fc))
        else:
            layers.append(X_act)
    return np.nanmean(np.stack(layers, axis=2), axis=2)


def _mean3_pred(df: pd.DataFrame, base_cols: list[str], origin_idx: int, h: int) -> np.ndarray:
    """3-day mean of base_cols at a single prediction target row, NWP-corrected."""
    layers = []
    for off in [h, h - 1, h - 2]:
        X_act = df[base_cols].iloc[[origin_idx + off]].to_numpy(dtype=float)
        if off > 0 and fc_wx_wide is not None:
            fc_cols = [f"{c}_L{min(off, MAX_FC_LEAD)}" for c in base_cols]
            td = df["date"].iloc[origin_idx + off]
            if td in fc_wx_wide.index:
                X_fc = fc_wx_wide.loc[td, fc_cols].to_numpy(dtype=float).reshape(1, -1)
                layers.append(np.where(np.isnan(X_fc), X_act, X_fc))
                continue
        layers.append(X_act)
    return np.nanmean(np.stack(layers, axis=2), axis=2)


# ---------------------------------------------------------------------------
# Rolling multi-horizon Elastic Net
# ---------------------------------------------------------------------------


def run_rolling(train_win: int, use_partial: bool = False, use_wx: bool = True, use_ead_smooth: bool = False):
    """Fit rolling ElasticNet (or PartialElasticNetCV) for every window.

    Returns (pred_matrix, actual_matrix, origin_dates, forecast_dates, coef_records).
    """
    wx_cols    = wx_feature_cols  if use_wx else []
    mean3_cols = _MEAN3_BASE_COLS if use_wx else []
    cal_cols   = _CAL_NAMES      if use_wx else _DOW_NAMES
    n             = len(forecasting_df)
    window_starts = range(0, n - (train_win + HORIZON) + 1, STRIDE)
    n_fc          = len(window_starts)

    pred_matrix    = np.full((n_fc, HORIZON), np.nan)
    actual_matrix  = np.full((n_fc, HORIZON), np.nan)
    origin_dates:   list = []
    forecast_dates: list = []
    coef_records:   list = []

    print(f"\nRunning {n_fc} windows × {HORIZON} horizons  (train_win={train_win}, stride={STRIDE})…")

    for w, i in enumerate(window_starts):
        if w % 50 == 0:
            print(f"  [{w + 1}/{n_fc}]")

        train       = forecasting_df.iloc[i : i + train_win]
        test        = forecasting_df.iloc[i + train_win : i + train_win + HORIZON]
        origin_date = train["date"].iloc[-1]
        origin_idx  = i + train_win - 1
        origin_dates.append(origin_date)
        forecast_dates.append(test["date"].tolist())
        actual_matrix[w] = test[OUTCOME].to_numpy(dtype=float)

        op_preds = [p for p in GLOBAL_OP_FEATURES if train[p].std() > 0]
        if use_ead_smooth:
            _col = f"{OUTCOME}_mean7_3"
            if _col not in op_preds and train[_col].std() > 0:
                op_preds = op_preds + [_col]
        feat_names  = op_preds + cal_cols + wx_cols + [f"{c}_mean3" for c in mean3_cols]
        X_op_origin = train[op_preds].iloc[[-1]].to_numpy(dtype=float)

        if use_partial:
            n_struct     = len(cal_cols) + len(wx_cols) + len(mean3_cols)
            penalty_mask = np.concatenate([np.ones(len(op_preds), dtype=bool),
                                           np.zeros(n_struct, dtype=bool)])

        # Lower floor: min observable EAD over FLOOR_LOOKBACK days before origin.
        # 3-day reporting lag means the last observable EAD is at origin_idx − 3.
        floor_end   = origin_idx - 2          # exclusive; last included = origin_idx − 3
        floor_start = max(0, floor_end - FLOOR_LOOKBACK)
        obs_floor   = float(forecasting_df[OUTCOME].iloc[floor_start:floor_end].min())

        for h in range(1, HORIZON + 1):
            X_op  = train[op_preds].iloc[: train_win - h].to_numpy(dtype=float)
            X_cal = forecasting_df[cal_cols].iloc[i + h : i + train_win].to_numpy(dtype=float)
            X_wx  = _wx_train(forecasting_df, wx_cols, i, h, train_win) if wx_cols else np.empty((train_win - h, 0))
            X_m3  = _mean3_train(forecasting_df, mean3_cols, i, h, train_win) if mean3_cols else np.empty((train_win - h, 0))
            X_tr  = np.nan_to_num(np.hstack([X_op, X_cal, X_wx, X_m3]))
            y_tr  = train[OUTCOME].iloc[h:].to_numpy(dtype=float)

            X_cal_o = forecasting_df[cal_cols].iloc[[origin_idx + h]].to_numpy(dtype=float)
            X_wx_o  = _wx_pred(forecasting_df, wx_cols, origin_idx, h) if wx_cols else np.empty((1, 0))
            X_m3_o  = _mean3_pred(forecasting_df, mean3_cols, origin_idx, h) if mean3_cols else np.empty((1, 0))
            X_or    = np.hstack([X_op_origin, X_cal_o, X_wx_o, X_m3_o])

            scaler  = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_tr)
            X_or_s  = scaler.transform(X_or)
            if use_partial:
                enet_cv = ElasticNetCV(
                    l1_ratio=L1_RATIO, cv=5, max_iter=10_000, n_jobs=-1, random_state=123,
                )
                enet_cv.fit(X_tr_s, y_tr)
                model = PartialElasticNetCV(
                    penalty_mask=penalty_mask, l1_ratio=L1_RATIO, max_iter=10_000,
                ).fit_alpha(X_tr_s, y_tr, enet_cv.alpha_)
            else:
                model = ElasticNetCV(
                    l1_ratio=L1_RATIO, cv=5, max_iter=10_000, n_jobs=-1, random_state=123,
                )
                model.fit(X_tr_s, y_tr)
            raw_pred = model.predict(X_or_s)[0]
            pred_matrix[w, h - 1] = float(np.clip(raw_pred, obs_floor, PRED_MAX))

            coef_records.extend(
                {"forecast_id": w + 1, "origin_date": origin_date, "horizon": h,
                 "predictor": p, "coefficient": c}
                for p, c in zip(feat_names, model.coef_) if c != 0.0
            )

    n_coef = len(coef_records)
    print(f"  Done. {n_coef:,} coefficient records (avg {n_coef / n_fc:.1f} non-zero/window).")
    return pred_matrix, actual_matrix, origin_dates, forecast_dates, coef_records


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_outputs(
    tag: str,
    pred_matrix:    np.ndarray,
    actual_matrix:  np.ndarray,
    origin_dates:   list,
    forecast_dates: list,
    coef_records:   list,
) -> None:
    """Write model outputs to model_outputs/{tag}_*.csv and print holdout summary."""
    n_fc = pred_matrix.shape[0]
    OUTPUT_DIR.mkdir(exist_ok=True)

    pd.DataFrame(
        pred_matrix, columns=[f"day_{d + 1}" for d in range(HORIZON)]
    ).assign(origin_date=origin_dates, forecast_id=range(1, n_fc + 1))[
        ["forecast_id", "origin_date"] + [f"day_{d + 1}" for d in range(HORIZON)]
    ].to_csv(OUTPUT_DIR / f"{tag}_pred_matrix.csv", index=False)

    mse_df = pd.DataFrame([
        {"forecast_id": w + 1, "origin_date": origin_dates[w],
         "mse_1_5":  _mse(actual_matrix[w, :5], pred_matrix[w, :5]),
         "mse_6_10": _mse(actual_matrix[w, 5:], pred_matrix[w, 5:])}
        for w in range(n_fc)
    ])
    mse_df.to_csv(OUTPUT_DIR / f"{tag}_mse_summary.csv", index=False)

    pd.DataFrame([
        {"forecast_id": w + 1, "origin_date": origin_dates[w], "horizon": h + 1,
         "forecast_date": forecast_dates[w][h],
         "actual": actual_matrix[w, h], "predicted": pred_matrix[w, h]}
        for w in range(n_fc) for h in range(HORIZON)
    ]).to_csv(OUTPUT_DIR / f"{tag}_forecast_detail.csv", index=False)

    pd.DataFrame(coef_records).to_csv(OUTPUT_DIR / f"{tag}_coef_detail.csv", index=False)

    ho = mse_df[mse_df["origin_date"] >= HOLDOUT_START]
    print(f"  Holdout ({len(ho)} windows): "
          f"days 1–5  mean={ho['mse_1_5'].mean():.4f} median={ho['mse_1_5'].median():.4f}  "
          f"days 6–10 mean={ho['mse_6_10'].mean():.4f} median={ho['mse_6_10'].median():.4f}")
    print(f"  Outputs → {OUTPUT_DIR}/{tag}_*.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)

    print("Loading operational data…")
    data = pd.read_csv(DATA_PATH, low_memory=False)
    data["dt"]   = pd.to_datetime(data["dt"], format="mixed", errors="coerce")
    data["date"] = data["dt"].dt.normalize()
    data = data[(data["dt"] <= DEV_END) & (data["value"] != -9999)].copy()

    sod = data["dt"].dt.hour * 3600 + data["dt"].dt.minute * 60 + data["dt"].dt.second
    data["midday_day"] = pd.to_datetime(
        np.where(sod <= 43200, data["date"], data["date"] + pd.Timedelta(days=1))
    ).normalize()

    print("Aggregating to daily per-hospital features…")

    target_daily = (
        data[data["metric_name"] == OUTCOME]
        .groupby("midday_day", as_index=False)["value"]
        .mean()
        .rename(columns={"midday_day": "date", "value": OUTCOME})
    )
    target_daily["date"] = pd.to_datetime(target_daily["date"]).dt.normalize()

    features_long = data[data["metric_name"] != OUTCOME].copy()
    features_long["col_key"] = features_long["metric_name"] + "|" + features_long["coverage_label"]
    features_long = features_long.groupby(["midday_day", "col_key"], as_index=False)["value"].mean()

    features_wide = (
        features_long
        .pivot(index="midday_day", columns="col_key", values="value")
        .reset_index()
        .rename(columns={"midday_day": "date"})
    )
    features_wide.columns.name = None
    features_wide["date"] = pd.to_datetime(features_wide["date"]).dt.normalize()

    forecasting_df = features_wide.merge(target_daily, on="date", how="left")

    # ---- Clean column names ------------------------------------------------
    cols_to_rename = [c for c in forecasting_df.columns if c not in ("date", OUTCOME)]
    deduped = clean_column_names(cols_to_rename)
    forecasting_df = forecasting_df.rename(columns=dict(zip(cols_to_rename, deduped)))
    print(f"  Feature columns after per-hospital pivot: {len(deduped)}")

    # ---- Weather -----------------------------------------------------------
    weather = load_weather()
    t_mean = weather["temperature_2m_mean"]
    t_min  = weather["temperature_2m_min"]
    weather["wx_coldness"]       = (10 - t_mean).clip(lower=0)
    weather["wx_hotness"]        = (t_mean - 25).clip(lower=0)
    weather["wx_below_freezing"] = (t_min < 0).astype(int)
    weather["wx_rain_sum"]       = weather["rain_sum"].clip(upper=RAIN_CAP)
    weather["wx_snowfall_sum"]   = weather["snowfall_sum"]
    weather["wx_wind_max"]       = weather["wind_speed_10m_max"].clip(upper=WIND_CAP)
    weather["wx_coldness2"]      = (COLD_THRESH2 - t_mean).clip(lower=0)
    weather["wx_heavy_rain"]     = (weather["rain_sum"] > HEAVY_RAIN_THRESH).astype(int)

    forecasting_df = forecasting_df.merge(
        weather[["date"] + wx_feature_cols], on="date", how="left"
    )
    print(f"  Weather features added: {wx_feature_cols}")

    fc_wx_wide = None
    if not NO_WX:
        _fc_wx_df  = load_forecast_weather(FORECAST_WX_PATH)
        fc_wx_wide = build_fc_wx_wide(_fc_wx_df) if _fc_wx_df is not None else None

    # ---- NWP weather log ---------------------------------------------------
    if fc_wx_wide is not None:
        print(f"  NWP forecast lookup ready: "
              f"{len(fc_wx_wide)} valid dates, lead days 1–{MAX_FC_LEAD}")

    # ---- Calendar features -------------------------------------------------
    dow = forecasting_df["date"].dt.dayofweek
    for idx, name in enumerate(_DOW_NAMES):
        forecasting_df[name] = (dow == idx).astype(float)
    forecasting_df["is_holiday"]           = forecasting_df["date"].isin(_ENGLAND_BANK_HOLIDAYS).astype(float)
    forecasting_df["is_day_after_holiday"] = forecasting_df["date"].isin(_DAY_AFTER_HOLIDAYS).astype(float)

    # ---- Impute ------------------------------------------------------------
    predictor_cols = [c for c in forecasting_df.columns if c not in ("date", OUTCOME)]
    for col in predictor_cols:
        forecasting_df[col] = (
            forecasting_df[col].interpolate(method="linear", limit_direction="both").ffill().bfill()
        )
    forecasting_df = forecasting_df.dropna(subset=[OUTCOME]).reset_index(drop=True)

    # ---- Feature engineering -----------------------------------------------
    roll_candidates = [
        c for c in predictor_cols
        if not c.startswith("dow_") and not c.startswith("wx_")
        and c not in ("is_holiday", "is_day_after_holiday")
    ]
    rolling_cols: dict[str, pd.Series] = {}
    for var in roll_candidates:
        rolling_cols[f"{var}_roll_mean7"] = forecasting_df[var].rolling(7).mean()
        rolling_cols[f"{var}_roll_sd7"]   = forecasting_df[var].rolling(7).std()

    forecasting_df = pd.concat(
        [forecasting_df, pd.DataFrame(rolling_cols, index=forecasting_df.index)], axis=1
    )
    forecasting_df[f"{OUTCOME}_lag3"]     = forecasting_df[OUTCOME].shift(3)
    forecasting_df[f"{OUTCOME}_mean7_3"]  = forecasting_df[OUTCOME].shift(3).rolling(5).mean()
    forecasting_df[f"{OUTCOME}_mean28_3"] = forecasting_df[OUTCOME].shift(3).rolling(28).mean()
    forecasting_df = forecasting_df.dropna().reset_index(drop=True)

    predictors = [c for c in forecasting_df.columns if c not in ("date", OUTCOME)]
    print(f"  Total predictors (incl. rolling + lag): {len(predictors)}")

    # ---- Skewness correction -----------------------------------------------
    skip_transform = {c for c in predictors
                      if c.startswith("dow_") or c.startswith("wx_")
                      or c in ("is_holiday", "is_day_after_holiday")}
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

    # ---- Load global feature selection -------------------------------------
    _sel_df = pd.read_csv(GLOBAL_SEL_PATH)
    GLOBAL_OP_FEATURES = [
        p for p in _sel_df.loc[_sel_df["in_model"] & ~_sel_df["forced"], "predictor"]
        if p in forecasting_df.columns
    ]
    print(f"  Global-model operational features available: {len(GLOBAL_OP_FEATURES)}")

    # ---- Main sweep --------------------------------------------------------
    np.random.seed(123)

    tag_prefix  = "partial" if USE_PARTIAL else "basic"
    if NO_WX:
        tag_prefix = "nwx_" + tag_prefix
    if EAD_SMOOTH:
        tag_prefix = tag_prefix + "_esmooth"
    all_results: dict[int, tuple[np.ndarray, np.ndarray, list]] = {}
    for tw in TRAIN_WINS:
        pm, am, od, fd, cr = run_rolling(tw, use_partial=USE_PARTIAL, use_wx=not NO_WX,
                                         use_ead_smooth=EAD_SMOOTH)
        save_outputs(f"{tag_prefix}_w{tw}", pm, am, od, fd, cr)
        all_results[tw] = (pm, am, od)

    print("\n=== Per-horizon holdout MSE by training window (Oct 2024 – Sep 2025) ===")
    print(f"{'':5}" + "".join(f"  w={tw:3d}" for tw in TRAIN_WINS))
    for h in range(1, HORIZON + 1):
        row = f"h={h:2d} "
        for tw in TRAIN_WINS:
            pm, am, od = all_results[tw]
            ho = np.array([d >= HOLDOUT_START for d in od])
            row += f"  {_mse(am[ho, h - 1], pm[ho, h - 1]):.4f}"
        print(row)
