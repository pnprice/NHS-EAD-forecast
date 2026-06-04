"""
Summer residual analysis for basic_w90 model.
Plots residuals (actual - predicted) vs weather variables for Jun–Aug 2025
to check whether wx_hotness needs a second kink.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

COLD_THRESH = 10.0
COLD_THRESH2 = 5.0
HOT_THRESH = 10.0
HOT_THRESH2 = 25.0
RAIN_CAP = 25.0
WIND_CAP = 42.0
HEAVY_RAIN_THRESH = 20.0

SUMMER_START = pd.Timestamp("2025-06-01")
SUMMER_END   = pd.Timestamp("2025-08-31")

DETAIL_PATH = Path("model_outputs/basic_w90_forecast_detail.csv")
WX_PATH     = Path("data/bristol_weather.csv")
OUT_PATH    = Path("Plots/summer_residuals.png")

# --------------------------------------------------------------------------
# Load forecast detail, compute residuals
# --------------------------------------------------------------------------
detail = pd.read_csv(DETAIL_PATH, parse_dates=["origin_date", "forecast_date"])
detail["residual"] = detail["actual"] - detail["predicted"]

# Filter to summer 2025 origins
summer = detail[
    detail["origin_date"].between(SUMMER_START, SUMMER_END)
].copy()
print(f"Summer rows: {len(summer):,}  "
      f"({summer['origin_date'].nunique()} unique origins, "
      f"{summer['forecast_date'].nunique()} unique forecast dates)")

# --------------------------------------------------------------------------
# Load & derive weather features for forecast_date (outcome day)
# --------------------------------------------------------------------------
wx = pd.read_csv(WX_PATH, parse_dates=["date"])
wx = wx.rename(columns={
    "temperature_2m_mean": "t_mean",
    "temperature_2m_min":  "t_min",
    "rain_sum":            "rain",
    "snowfall_sum":        "snow",
    "wind_speed_10m_max":  "wind",
})
wx["wx_coldness"]      = (COLD_THRESH - wx["t_mean"]).clip(lower=0)
wx["wx_hotness"]       = (wx["t_mean"] - HOT_THRESH).clip(lower=0)
wx["wx_coldness2"]     = (wx["wx_coldness"] - (COLD_THRESH - COLD_THRESH2)).clip(lower=0)
wx["wx_hotness2"]      = (wx["wx_hotness"]  - (HOT_THRESH2 - HOT_THRESH)).clip(lower=0)
wx["wx_rain_sum"]      = wx["rain"].clip(upper=RAIN_CAP)
wx["wx_heavy_rain"]    = (wx["rain"] > HEAVY_RAIN_THRESH).astype(int)
wx["wx_below_freezing"] = (wx["t_min"] < 0).astype(int)
wx["wx_wind_max"]      = wx["wind"].clip(upper=WIND_CAP)

# Merge on forecast_date
summer = summer.merge(
    wx[["date","wx_coldness","wx_hotness","wx_coldness2","wx_hotness2",
        "wx_rain_sum","wx_heavy_rain","wx_wind_max","t_mean"]],
    left_on="forecast_date", right_on="date", how="left"
)

# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
weather_vars = [
    ("wx_hotness",   "wx_hotness (°C above 10°C)"),
    ("wx_hotness2",  "wx_hotness2 (°C above 25°C)"),
    ("t_mean",       "Mean temperature (°C)"),
    ("wx_coldness",  "wx_coldness (°C below 10°C)"),
    ("wx_coldness2", "wx_coldness2 (°C below 5°C)"),
    ("wx_rain_sum",  "Daily rain (mm, cap 25)"),
    ("wx_wind_max",  "Daily wind max (km/h)"),
    ("wx_heavy_rain","Heavy rain flag (>20mm)"),
]

n_cols = 4
n_rows = 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
axes = axes.flatten()

for ax, (var, label) in zip(axes, weather_vars):
    sub = summer.dropna(subset=[var, "residual"])
    x = sub[var].to_numpy(dtype=float)
    y = sub["residual"].to_numpy(dtype=float)

    ax.scatter(x, y, alpha=0.15, s=12, color="steelblue", label="_nolegend_")

    # LOWESS-style smoothing using rolling quantiles on sorted data
    if len(np.unique(x)) > 5:
        sort_idx = np.argsort(x)
        xs, ys = x[sort_idx], y[sort_idx]
        window = max(20, len(xs) // 8)
        smooth_x, smooth_y = [], []
        for k in range(len(xs) - window + 1):
            smooth_x.append(xs[k : k + window].mean())
            smooth_y.append(ys[k : k + window].mean())
        ax.plot(smooth_x, smooth_y, color="crimson", lw=2, label="rolling mean")

    r, p = pearsonr(x, y) if len(x) > 2 else (np.nan, np.nan)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(label, fontsize=9)
    ax.set_ylabel("Residual (actual − pred)", fontsize=9)
    ax.set_title(f"r = {r:.3f}" + (" *" if p < 0.05 else ""), fontsize=10)
    ax.tick_params(labelsize=8)

fig.suptitle(
    "Basic w90: residuals vs weather  |  Jun–Aug 2025 origins",
    fontsize=12, fontweight="bold",
)
fig.tight_layout()

OUT_PATH.parent.mkdir(exist_ok=True)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
plt.show()
