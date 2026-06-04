"""
NHS EAD Forecasting — Python baseline (Elastic Net, rolling window)
Equivalent to NHS_example_forecast.R

Pipeline
--------
1. Load & preprocess (midday aggregation, long → wide)
2. Clean column names
3. Impute missing values (linear interpolation; R uses na_kalman)
4. Feature engineering: 7-day rolling mean/SD, lag-3 target
5. Skewness correction (log1p / sqrt / squared)
6. Rolling 90-day Elastic Net (l1_ratio=0.5), 5-fold CV, 10-day horizon
7. Write model_outputs/example_pred_matrix.csv and model_outputs/example_mse_summary.csv
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUTCOME = "estimated_avoidable_deaths"


def _clean_name(name: str) -> str:
    """Replicates the R gsub chain applied after abbreviation."""
    name = re.sub(r"\d", "", name)
    name = re.sub(r"[()]", "", name)
    name = re.sub(r"[ \-]", "_", name)
    name = name.replace("%", "pct")
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)
    return name.lower().strip("_")


def _mse(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = ~(np.isnan(actual) | np.isnan(pred))
    return float(np.mean((actual[mask] - pred[mask]) ** 2))


def main() -> None:
    # ========================================================================
    # 1. LOAD DATA + PREPROCESSING
    # ========================================================================

    data = pd.read_csv("data/turingAI_forecasting_challenge_dataset.csv")
    data["dt"] = pd.to_datetime(data["dt"], format="mixed", errors="coerce")
    data["date"] = data["dt"].dt.normalize()

    # Development period only; -9999 are sentinel values for the assessment period.
    data = data[(data["dt"] <= pd.Timestamp("2025-09-30")) & (data["value"] != -9999)].copy()

    seconds_of_day = (
        data["dt"].dt.hour * 3600
        + data["dt"].dt.minute * 60
        + data["dt"].dt.second
    )
    data["midday_day"] = np.where(
        seconds_of_day <= 43200,
        data["date"].dt.date,
        (data["date"] + pd.Timedelta(days=1)).dt.date,
    )
    data["midday_day"] = pd.to_datetime(data["midday_day"])

    forecasting_df = (
        data.groupby(["midday_day", "metric_name"], as_index=False)["value"]
        .mean()
        .pivot(index="midday_day", columns="metric_name", values="value")
        .reset_index()
    )
    forecasting_df.columns.name = None

    # ========================================================================
    # 2. CLEAN COLUMN NAMES
    # ========================================================================

    cols_to_rename = [c for c in forecasting_df.columns if c not in ("midday_day", OUTCOME)]
    cleaned = [_clean_name(c) for c in cols_to_rename]

    seen: dict[str, int] = {}
    deduped: list[str]   = []
    for name in cleaned:
        if name in seen:
            seen[name] += 1
            deduped.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            deduped.append(name)

    forecasting_df = forecasting_df.rename(columns=dict(zip(cols_to_rename, deduped)))

    # ========================================================================
    # 3. IMPUTE MISSING VALUES
    # ========================================================================

    predictor_cols = [c for c in forecasting_df.columns if c not in ("midday_day", OUTCOME)]

    for col in predictor_cols:
        forecasting_df[col] = (
            forecasting_df[col]
            .interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
        )

    forecasting_df = forecasting_df.dropna().reset_index(drop=True)

    # ========================================================================
    # 4. FEATURE ENGINEERING: ROLLING FEATURES + LAGGED TARGET
    # ========================================================================

    predictors = [c for c in forecasting_df.columns if c not in ("midday_day", OUTCOME)]

    rolling_cols: dict[str, pd.Series] = {}
    for var in predictors:
        rolling_cols[f"{var}_roll_mean_7"] = forecasting_df[var].rolling(7).mean()
        rolling_cols[f"{var}_roll_sd_7"]   = forecasting_df[var].rolling(7).std()

    forecasting_df = pd.concat(
        [forecasting_df, pd.DataFrame(rolling_cols, index=forecasting_df.index)],
        axis=1,
    )
    forecasting_df[f"{OUTCOME}_lag3"] = forecasting_df[OUTCOME].shift(3)
    forecasting_df = forecasting_df.dropna().reset_index(drop=True)

    predictors = [c for c in forecasting_df.columns if c not in ("midday_day", OUTCOME)]

    # ========================================================================
    # 5. SKEWNESS CORRECTION
    # ========================================================================

    for col in predictors:
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
    # 6. ROLLING MULTI-HORIZON FORECASTING WITH ELASTIC NET
    # ========================================================================

    np.random.seed(123)

    n             = len(forecasting_df)
    TRAIN_WIN     = 90
    HORIZON       = 10
    STRIDE        = 10
    L1_RATIO      = 0.5
    window_starts = list(range(0, n - (TRAIN_WIN + HORIZON) + 1, STRIDE))
    n_forecasts   = len(window_starts)

    pred_matrix    = np.full((n_forecasts, HORIZON), np.nan)
    actual_matrix  = np.full((n_forecasts, HORIZON), np.nan)
    origin_dates:  list = []
    forecast_dates: list = []
    coef_records:  list = []

    print(f"Running {n_forecasts} rolling forecasts × {HORIZON} horizons (stride={STRIDE})...")

    for w, i in enumerate(window_starts):
        if w % 10 == 0:
            print(f"  [{w + 1}/{n_forecasts}]")

        train = forecasting_df.iloc[i : i + TRAIN_WIN]
        test  = forecasting_df.iloc[i + TRAIN_WIN : i + TRAIN_WIN + HORIZON]

        origin_date = train["midday_day"].iloc[-1]
        origin_dates.append(origin_date)
        forecast_dates.append(test["midday_day"].tolist())
        actual_matrix[w] = test[OUTCOME].to_numpy(dtype=float)

        valid_preds = [p for p in predictors if train[p].std() > 0]
        X_origin    = train[valid_preds].iloc[[-1]].to_numpy(dtype=float)

        for h in range(1, HORIZON + 1):
            X_train_h  = train[valid_preds].iloc[: TRAIN_WIN - h].to_numpy(dtype=float)
            y_train_h  = train[OUTCOME].iloc[h:].to_numpy(dtype=float)
            scaler     = StandardScaler()
            X_train_hs = scaler.fit_transform(X_train_h)
            X_origin_s = scaler.transform(X_origin)
            model      = ElasticNetCV(
                l1_ratio=L1_RATIO, cv=5, max_iter=10_000, n_jobs=-1, random_state=123,
            )
            model.fit(X_train_hs, y_train_h)
            pred_matrix[w, h - 1] = model.predict(X_origin_s)[0]

            for pred_name, coef_val in zip(valid_preds, model.coef_):
                if coef_val != 0.0:
                    coef_records.append({
                        "forecast_id": w + 1,
                        "origin_date": origin_date,
                        "horizon":     h,
                        "predictor":   pred_name,
                        "coefficient": coef_val,
                    })

    print("Forecasting complete.")

    # ========================================================================
    # 7. OUTPUT
    # ========================================================================

    MODEL_NAME = "example"
    OUTPUT_DIR = Path("model_outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)

    pred_df = pd.DataFrame(pred_matrix, columns=[f"day_{d + 1}" for d in range(HORIZON)])
    pred_df.insert(0, "origin_date", origin_dates)
    pred_df.insert(0, "forecast_id", range(1, n_forecasts + 1))
    pred_df.to_csv(OUTPUT_DIR / f"{MODEL_NAME}_pred_matrix.csv", index=False)

    mse_df = pd.DataFrame([
        {"forecast_id": i + 1, "origin_date": origin_dates[i],
         "mse_1_5":  _mse(actual_matrix[i, :5], pred_matrix[i, :5]),
         "mse_6_10": _mse(actual_matrix[i, 5:],  pred_matrix[i, 5:])}
        for i in range(n_forecasts)
    ])
    mse_df.to_csv(OUTPUT_DIR / f"{MODEL_NAME}_mse_summary.csv", index=False)

    detail_rows = [
        {"forecast_id": i + 1, "origin_date": origin_dates[i], "horizon": h + 1,
         "forecast_date": forecast_dates[i][h],
         "actual": actual_matrix[i, h], "predicted": pred_matrix[i, h]}
        for i in range(n_forecasts) for h in range(HORIZON)
    ]
    pd.DataFrame(detail_rows).to_csv(OUTPUT_DIR / f"{MODEL_NAME}_forecast_detail.csv", index=False)

    coef_df = pd.DataFrame(coef_records)
    coef_df.to_csv(OUTPUT_DIR / f"{MODEL_NAME}_coef_detail.csv", index=False)
    print(f"  Coefficient records saved: {len(coef_df):,} "
          f"(avg {len(coef_df)/n_forecasts:.1f} non-zero per window)")

    print(f"\nMean MSE (days 1–5):    {mse_df['mse_1_5'].mean():.4f}")
    print(f"Median MSE (days 1–5):  {mse_df['mse_1_5'].median():.4f}")
    print(f"Mean MSE (days 6–10):   {mse_df['mse_6_10'].mean():.4f}")
    print(f"Median MSE (days 6–10): {mse_df['mse_6_10'].median():.4f}")
    print(f"\nOutputs written to model_outputs/{MODEL_NAME}_*.csv")


if __name__ == "__main__":
    main()
