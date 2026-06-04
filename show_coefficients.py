"""
show_coefficients.py
====================
Print a coefficient table from any model_outputs/*_coef_detail.csv.

Rows are predictors (only those with at least one non-zero coefficient across
all horizons).  Columns are h1..h10.  Rows are sorted by mean absolute
coefficient descending so the most influential predictors appear first.

Usage
-----
    python show_coefficients.py                          # global model (default)
    python show_coefficients.py model_outputs/global_coef_detail.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("model_outputs/global_coef_detail.csv")

if not path.exists():
    sys.exit(f"File not found: {path}")

df = pd.read_csv(path)

# Pivot to wide: rows = predictor, columns = horizon
wide = (
    df.pivot(index="predictor", columns="horizon", values="coefficient")
    .fillna(0.0)
    .rename(columns=lambda h: f"h{h:02d}")
)

# Sort by mean absolute coefficient descending
wide["_mean_abs"] = wide.filter(like="h").abs().mean(axis=1)
wide = wide.sort_values("_mean_abs", ascending=False).drop(columns="_mean_abs")

# Format: right-align numbers, truncate long predictor names
pd.set_option("display.float_format", "{:+.4f}".format)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 160)

print(f"\nCoefficients from: {path}")
print(f"Predictors with non-zero coef: {len(wide)}  |  Horizons: {len(wide.columns)}\n")
print(wide.to_string())
print()
