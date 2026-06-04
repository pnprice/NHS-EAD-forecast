"""
NHS EAD Forecasting — EDA plotting utilities
=============================================
Reusable functions for exploring the long-format NHS dataset with plotnine.

Typical workflow
----------------
    from eda_plots import load_raw, get_daily, plot_panels, plot_timeseries

    df = load_raw()

    # Daily ED occupancy at the three adult acute hospitals
    daily = get_daily(df,
                      metrics=["Patients in A&E"],
                      coverages=["BRI", "NBT", "WGH"],
                      agg="mean")
    p = plot_timeseries(daily, color_by="coverage", title="Patients in A&E")
    p.show()

    # Multi-panel: several ED metrics stacked on a shared date axis
    daily2 = get_daily(df,
                       metrics=["Patients in A&E", "No. of DTAs",
                                "4hr Breach Performance"],
                       coverages=["BRI", "NBT", "WGH"])
    p2 = plot_panels(daily2, title="ED operational metrics")
    p2.show()

Public API
----------
    load_raw(path)              → raw long-format DataFrame
    get_daily(df, metrics, coverages, agg, date_range)
                                → tidy daily DataFrame
    plot_timeseries(daily_df, …)→ ggplot  (single panel, series as colour)
    plot_panels(daily_df, …)    → ggplot  (one panel per metric, shared x-axis)
    plot_with_target(df, …)     → ggplot  (target + features as stacked panels)

Metric / coverage quick-reference constants are defined at the bottom of this
file (ED_METRICS, BED_METRICS, AMBULANCE_METRICS, SYSTEM_METRICS, HOSPITALS).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_grid,
    facet_wrap,
    geom_line,
    geom_smooth,
    ggplot,
    labs,
    scale_alpha_manual,
    scale_color_manual,
    scale_size_manual,
    scale_color_brewer,
    scale_x_datetime,
    theme,
    theme_bw,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
DATA_PATH = _HERE / "data" / "turingAI_forecasting_challenge_dataset.csv"

TARGET = "estimated_avoidable_deaths"

# ---------------------------------------------------------------------------
# Convenience metric / coverage groups (edit freely)
# ---------------------------------------------------------------------------

HOSPITALS = ["BRI", "NBT", "WGH"]

ED_METRICS = [
    "Patients in A&E",
    "New Arrivals in Last Hour",
    "No. of DTAs",
    "No. of DTAs (> 8hrs)",
    "4hr Breach Performance",
    "Total Breaches Since Midnight",
    "Average Time to Triage",
    "Average Time to Assessment",
    "A&E Discharges in Last Hour",
    "Number of Admissions",
    "Number of Discharges",
]

BED_METRICS = [
    "G&A Bed occupancy",
    "Bed Occupancy Adult",
    "% of beds occupied by patients with NCtR",
    "Beds occupied by long-stay patients (21+ days)",
    "Escalation beds open",
    "% of open beds that are escalation beds",
    "Number of Outliers (excluding paediatrics) at 1000",
]

AMBULANCE_METRICS = [
    "Ambulance Queue",
    "Ambulances Conveyed to Hospital (Since Midnight)",
    "Ambulance Handovers 30mins (Since Midnight)",
    "Ambulance Handovers 60mins (Since Midnight)",
    "Handover Time Lost Since Midnight (hh:mm)",
    "Category 1 - BNSSG Mean Response (Since Midnight)",
    "Category 2 - BNSSG Mean Response (Since Midnight)",
]

SYSTEM_METRICS = [
    "OPEL",
    "Aggregated NHSE OPEL Score",
    "Automated OPEL",
    "ED all-type 4-hour performance",
    "% of patients spending >12 hours in ED",
]

DISCHARGE_METRICS = [
    "BRI NCtR Patients",
    "NBT NCtR Patients",
    "WGH NCtR Patients",
    "BRI NCtR Beddays",
    "NBT NCtR Beddays",
    "WGH NCtR Beddays",
    "BRI P0 Discharges",
    "NBT P0 Discharges",
    "WGH P0 Discharges",
]

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

_PALETTE_QUALITATIVE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def _nhs_theme(figure_size: tuple[float, float] = (12, 4), **kwargs) -> theme:
    """Clean, print-friendly plotnine theme."""
    return theme_bw() + theme(
        figure_size=figure_size,
        axis_text_x=element_text(rotation=30, hjust=1, size=8),
        axis_text_y=element_text(size=8),
        axis_title=element_text(size=9),
        strip_text=element_text(size=8, face="bold"),
        legend_title=element_text(size=8),
        legend_text=element_text(size=8),
        panel_grid_minor=element_blank(),
        panel_grid_major_x=element_blank(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def load_raw(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset once, parse datetimes, cache in memory.

    Strips assessment-period dummy rows (-9999) and adds a ``date`` column
    (date only, no time).  The original ``dt`` column is preserved.

    Returns
    -------
    pd.DataFrame
        Columns: dt, metric_name, coverage, value, coverage_label,
                 variable_type, date
    """
    df = pd.read_csv(path, low_memory=False)
    df["dt"] = pd.to_datetime(df["dt"], format="mixed", errors="coerce")
    df["date"] = df["dt"].dt.normalize()
    # Remove dummy assessment-period rows
    df = df[df["value"] != -9999].copy()
    return df


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

_AGG_FUNCS = {
    "mean": "mean",
    "max": "max",
    "min": "min",
    "sum": "sum",
    "median": "median",
    # 'midday' handled separately below
}


def _midday_agg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the reading closest to (but not after) midday each day.
    Readings after noon are attributed to the following date, matching the
    contest's midday-snapshot convention.
    """
    sod = (
        df["dt"].dt.hour * 3600
        + df["dt"].dt.minute * 60
        + df["dt"].dt.second
    )
    df = df.copy()
    df["_snap_date"] = np.where(
        sod <= 43200,
        df["date"],
        df["date"] + pd.Timedelta(days=1),
    )
    # For each (snap_date, metric, coverage) keep the reading closest to noon
    df["_dist_noon"] = (sod - 43200).abs()
    idx = df.groupby(
        ["_snap_date", "metric_name", "coverage_label"]
    )["_dist_noon"].idxmin()
    result = df.loc[idx].copy()
    result["date"] = pd.to_datetime(result["_snap_date"]).dt.normalize()
    return result[["date", "metric_name", "coverage_label", "value"]]


def get_daily(
    df: pd.DataFrame,
    metrics: Sequence[str],
    coverages: Sequence[str] | None = None,
    agg: Literal["mean", "max", "min", "sum", "median", "midday"] = "mean",
    date_range: tuple[str, str] | None = None,
    dev_only: bool = True,
) -> pd.DataFrame:
    """Filter, aggregate and return a tidy daily DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw long-format data from :func:`load_raw`.
    metrics : list[str]
        Metric names to include.  Use ``TARGET`` for the outcome.
    coverages : list[str] | None
        Subset of ``coverage_label`` values.  ``None`` → all coverages that
        carry any of the requested metrics.
    agg : str
        Aggregation applied to sub-daily readings:
        ``'mean'`` | ``'max'`` | ``'min'`` | ``'sum'`` | ``'median'``
        | ``'midday'`` (nearest reading to noon).
    date_range : (str, str) | None
        Optional ``(start, end)`` ISO date strings to restrict output.
    dev_only : bool
        If True (default) restrict to development period (≤ 2025-09-30).

    Returns
    -------
    pd.DataFrame
        Columns: date, metric_name, coverage_label, value
    """
    mask = df["metric_name"].isin(metrics)
    if coverages is not None:
        mask &= df["coverage_label"].isin(coverages)
    if dev_only:
        mask &= df["date"] <= pd.Timestamp("2025-09-30")

    sub = df[mask].copy()

    if agg == "midday":
        daily = _midday_agg(sub)
    else:
        fn = _AGG_FUNCS[agg]
        daily = (
            sub.groupby(["date", "metric_name", "coverage_label"], as_index=False)["value"]
            .agg(fn)
        )

    if date_range is not None:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        daily = daily[(daily["date"] >= start) & (daily["date"] <= end)]

    daily = daily.sort_values(["metric_name", "coverage_label", "date"]).reset_index(drop=True)
    return daily


def _make_series_label(daily: pd.DataFrame) -> pd.DataFrame:
    """Add a ``series`` column combining metric + coverage for legend labels."""
    daily = daily.copy()
    n_metrics = daily["metric_name"].nunique()
    n_cov = daily["coverage_label"].nunique()
    if n_metrics == 1:
        daily["series"] = daily["coverage_label"]
    elif n_cov == 1:
        daily["series"] = daily["metric_name"]
    else:
        daily["series"] = daily["metric_name"] + " · " + daily["coverage_label"]
    return daily


def _date_scale(breaks: str = "3 months") -> scale_x_datetime:
    return scale_x_datetime(date_breaks=breaks, date_labels="%b %Y")


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def _color_scale(n: int):
    if n <= 8:
        return scale_color_brewer(type="qual", palette="Dark2")
    return scale_color_manual(values=_PALETTE_QUALITATIVE[:n])


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_timeseries(
    daily_df: pd.DataFrame,
    color_by: Literal["series", "coverage", "metric"] = "series",
    smooth: bool = False,
    title: str | None = None,
    y_label: str = "value",
    date_breaks: str = "3 months",
    fig_size: tuple[float, float] = (12, 4),
) -> ggplot:
    """Single-panel time-series plot.

    All series are drawn as lines on one shared y-axis.  Use this when all
    series have comparable scales (e.g. attendance counts for three hospitals).

    Parameters
    ----------
    daily_df :
        Output of :func:`get_daily` — tidy daily long DataFrame.
    color_by :
        ``'series'``: auto-label combining metric + coverage.
        ``'coverage'``: colour by ``coverage_label`` only.
        ``'metric'``: colour by ``metric_name`` only.
    smooth :
        Overlay a LOESS smooth line.
    title :
        Plot title string.
    y_label :
        Y-axis label.
    date_breaks :
        Passed to :func:`scale_x_datetime` (e.g. ``"3 months"``).
    fig_size :
        ``(width, height)`` in inches.

    Returns
    -------
    ggplot
        Add more layers or call ``.show()`` / ``.save()``.

    Examples
    --------
    >>> from eda_plots import load_raw, get_daily, plot_timeseries
    >>> df = load_raw()
    >>> daily = get_daily(df, ["Patients in A&E"], ["BRI", "NBT", "WGH"])
    >>> plot_timeseries(daily, title="Patients in A&E").show()
    """
    if color_by == "series":
        daily_df = _make_series_label(daily_df)
        color_col = "series"
    elif color_by == "coverage":
        color_col = "coverage_label"
    else:
        color_col = "metric_name"

    n_colors = daily_df[color_col].nunique()

    p = (
        ggplot(daily_df, aes(x="date", y="value", color=color_col))
        + geom_line(size=0.6, alpha=0.85)
        + _date_scale(date_breaks)
        + _color_scale(n_colors)
        + _nhs_theme(figure_size=fig_size)
        + labs(
            title=title or "",
            x="Date",
            y=y_label,
            color="",
        )
    )
    if smooth:
        p = p + geom_smooth(method="loess", se=False, size=1.1)
    return p


def plot_panels(
    daily_df: pd.DataFrame,
    facet_by: Literal["metric", "coverage"] = "metric",
    color_by: Literal["coverage", "metric", "none"] = "coverage",
    scales: Literal["free_y", "fixed", "free"] = "free_y",
    smooth: bool = False,
    title: str | None = None,
    date_breaks: str = "3 months",
    fig_height_per_panel: float = 2.5,
    fig_width: float = 12,
) -> ggplot:
    """Faceted plot — one panel per metric (or coverage) on a shared x-axis.

    This is the primary multi-metric view.  Panels share the x-axis so you
    can visually align patterns across metrics.

    Parameters
    ----------
    daily_df :
        Output of :func:`get_daily`.
    facet_by :
        ``'metric'``: one panel per ``metric_name``.
        ``'coverage'``: one panel per ``coverage_label``.
    color_by :
        ``'coverage'``: lines coloured by hospital/site.
        ``'metric'``: lines coloured by metric name.
        ``'none'``: single colour (useful when faceting already separates series).
    scales :
        ``'free_y'`` (default) lets each panel have its own y-axis range.
        Use ``'fixed'`` when comparing absolute magnitudes.
    smooth :
        Add LOESS smooth per series.
    title :
        Overall plot title.
    date_breaks :
        X-axis break frequency (e.g. ``"3 months"``).
    fig_height_per_panel :
        Height in inches allocated per facet row.
    fig_width :
        Figure width in inches.
    Returns
    -------
    ggplot

    Examples
    --------
    >>> daily = get_daily(df,
    ...     metrics=["Patients in A&E", "No. of DTAs", "4hr Breach Performance"],
    ...     coverages=["BRI", "NBT", "WGH"])
    >>> plot_panels(daily, title="ED operational metrics").show()
    """
    facet_col = "metric_name" if facet_by == "metric" else "coverage_label"
    n_panels = daily_df[facet_col].nunique()
    fig_h = max(3.0, fig_height_per_panel * n_panels)

    if color_by == "coverage":
        color_col = "coverage_label"
    elif color_by == "metric":
        color_col = "metric_name"
    else:
        color_col = None

    base_aes = aes(x="date", y="value") if color_col is None else aes(x="date", y="value", color=color_col)

    p = (
        ggplot(daily_df, base_aes)
        + geom_line(size=0.55, alpha=0.85)
        + facet_wrap(f"~{facet_col}", ncol=1, scales=scales)
        + _date_scale(date_breaks)
        + _nhs_theme(figure_size=(fig_width, fig_h))
        + labs(
            title=title or "",
            x="Date",
            y="value",
            color="",
        )
    )
    if color_col is not None:
        n_colors = daily_df[color_col].nunique()
        p = p + _color_scale(n_colors)
    if smooth:
        p = p + geom_smooth(method="loess", se=False, size=1.1)
    return p


def plot_with_target(
    df: pd.DataFrame,
    feature_metrics: Sequence[str],
    coverages: Sequence[str] | None = None,
    agg: str = "mean",
    date_range: tuple[str, str] | None = None,
    scales: str = "free_y",
    title: str | None = None,
    smooth: bool = False,
    date_breaks: str = "3 months",
) -> ggplot:
    """Stacked panels: target (estimated_avoidable_deaths) + feature metrics.

    The target is always shown in its own panel at the top; feature metrics
    each get their own panel below.  Useful for checking visual correlation
    between predictors and the outcome.

    Parameters
    ----------
    df :
        Raw long-format data from :func:`load_raw`.
    feature_metrics :
        List of feature metric names to show beneath the target.
    coverages :
        Filter to these ``coverage_label`` values for feature metrics.
        Target is always BNSSG and is added automatically.
    agg :
        Daily aggregation method (see :func:`get_daily`).
    date_range :
        Optional ``(start, end)`` ISO date strings.
    scales, title, smooth, date_breaks :
        Passed through to :func:`plot_panels`.

    Returns
    -------
    ggplot

    Examples
    --------
    >>> p = plot_with_target(df,
    ...     feature_metrics=["Patients in A&E", "Ambulance Queue"],
    ...     coverages=["BRI", "NBT", "WGH"])
    >>> p.show()
    """
    # Target: always BNSSG, no coverage filter needed
    target_daily = get_daily(
        df,
        metrics=[TARGET],
        coverages=None,
        agg=agg,
        date_range=date_range,
    )

    feature_daily = get_daily(
        df,
        metrics=list(feature_metrics),
        coverages=coverages,
        agg=agg,
        date_range=date_range,
    )

    combined = pd.concat([target_daily, feature_daily], ignore_index=True)

    # Fix panel ordering: target first
    all_metrics = [TARGET] + [m for m in feature_metrics if m != TARGET]
    combined["metric_name"] = pd.Categorical(combined["metric_name"], categories=all_metrics, ordered=True)

    color_col = "coverage_label"
    n_colors = combined[color_col].nunique()
    n_panels = combined["metric_name"].nunique()
    fig_h = max(3.0, 2.5 * n_panels)

    p = (
        ggplot(combined, aes(x="date", y="value", color=color_col))
        + geom_line(size=0.55, alpha=0.85)
        + facet_wrap("~metric_name", ncol=1, scales=scales)
        + _date_scale(date_breaks)
        + _color_scale(n_colors)
        + _nhs_theme(figure_size=(12, fig_h))
        + labs(
            title=title or "Features vs. target",
            x="Date",
            y="value",
            color="",
        )
    )
    if smooth:
        p = p + geom_smooth(method="loess", se=False, size=1.1)
    return p


# ---------------------------------------------------------------------------
# Convenience wrappers for common views
# ---------------------------------------------------------------------------


def plot_ed_overview(
    df: pd.DataFrame,
    hospitals: Sequence[str] = HOSPITALS,
    date_range: tuple[str, str] | None = None,
    agg: str = "mean",
) -> ggplot:
    """Standard ED overview: 4 key metrics faceted, coloured by hospital."""
    metrics = [
        "Patients in A&E",
        "No. of DTAs",
        "4hr Breach Performance",
        "Total Breaches Since Midnight",
    ]
    daily = get_daily(df, metrics, coverages=list(hospitals), agg=agg, date_range=date_range)
    return plot_panels(daily, title="ED overview — key metrics", date_breaks="3 months")


def plot_boarding_pressure(
    df: pd.DataFrame,
    hospitals: Sequence[str] = HOSPITALS,
    date_range: tuple[str, str] | None = None,
    agg: str = "mean",
) -> ggplot:
    """NCtR (boarding/stranded) patients and bed-days by hospital."""
    metrics = [
        f"{h} NCtR Patients" for h in ["BRI", "NBT", "WGH"]
    ] + [
        f"{h} NCtR Beddays" for h in ["BRI", "NBT", "WGH"]
    ]
    # These metrics have coverage=BNSSG; no hospital filter needed
    daily = get_daily(df, metrics, coverages=None, agg=agg, date_range=date_range)
    return plot_panels(
        daily,
        color_by="none",
        title="NCtR (boarding) patients and bed-days",
        date_breaks="3 months",
    )


def plot_system_pressure(
    df: pd.DataFrame,
    date_range: tuple[str, str] | None = None,
    agg: str = "mean",
) -> ggplot:
    """System-level pressure: OPEL scores + 4-hr performance + target."""
    return plot_with_target(
        df,
        feature_metrics=["OPEL", "Aggregated NHSE OPEL Score", "ED all-type 4-hour performance"],
        coverages=None,
        agg=agg,
        date_range=date_range,
        title="System pressure vs. avoidable deaths",
    )


# ---------------------------------------------------------------------------
# Forecast comparison plots
# ---------------------------------------------------------------------------


def load_forecast_detail(path: str | Path) -> pd.DataFrame:
    """Load a *_forecast_detail.csv produced by any forecast script."""
    df = pd.read_csv(path, parse_dates=["origin_date", "forecast_date"])
    return df


def plot_forecast_ts(
    models: dict[str, str | Path],
    horizons: Sequence[int] = (1, 5, 10),
    date_range: tuple[str, str] | None = None,
    date_breaks: str = "3 months",
    fig_width: float = 13,
    fig_height_per_panel: float = 2.8,
) -> ggplot:
    """Plot actual vs. model forecasts as time series, faceted by horizon.

    Each panel shows one forecast horizon.  Within each panel the actual
    observed value is drawn as a bold black line; each model's predictions
    for that horizon are drawn as coloured lines.

    At a given horizon h, the prediction shown for calendar date D is the
    forecast made h days earlier (origin date D − h).  This gives one
    non-overlapping prediction per date per model.

    Parameters
    ----------
    models : dict[str, path]
        ``{label: path_to_forecast_detail_csv}`` — e.g.
        ``{"Example": "submission/example_forecast_detail.csv",
           "Basic":   "submission/basic_forecast_detail.csv"}``
    horizons : sequence of int
        Which forecast horizons (1–10) to show as separate panels.
    date_range : (start, end) ISO strings or None
    date_breaks : x-axis tick frequency
    fig_width, fig_height_per_panel : figure dimensions

    Returns
    -------
    ggplot

    Examples
    --------
    >>> p = plot_forecast_ts(
    ...     {"Example": "model_outputs/example_forecast_detail.csv",
    ...      "Basic":   "model_outputs/basic_forecast_detail.csv"},
    ...     horizons=[1, 5, 10],
    ... )
    >>> p.show()
    """
    frames = []
    actual_added = False

    for label, path in models.items():
        detail = load_forecast_detail(path)

        for h in horizons:
            sub = detail[detail["horizon"] == h].copy()

            if date_range is not None:
                start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
                sub = sub[(sub["forecast_date"] >= start) & (sub["forecast_date"] <= end)]

            # One prediction per calendar date at this horizon
            pred_rows = sub[["forecast_date", "predicted"]].copy()
            pred_rows = pred_rows.rename(columns={"forecast_date": "date", "predicted": "value"})
            pred_rows["series"] = label
            pred_rows["horizon_label"] = f"Horizon = {h}"
            frames.append(pred_rows)

            # Actual values — only need to add once (same across models)
            if not actual_added:
                actual_rows = sub[["forecast_date", "actual"]].copy()
                actual_rows = actual_rows.rename(columns={"forecast_date": "date", "actual": "value"})
                actual_rows["series"] = "Actual"
                actual_rows["horizon_label"] = f"Horizon = {h}"
                frames.append(actual_rows)

        actual_added = True  # added for all horizons in first model iteration

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    # Order: Actual first so it's the reference in the legend
    model_labels = list(models.keys())
    all_series = ["Actual"] + model_labels
    combined["series"] = pd.Categorical(combined["series"], categories=all_series, ordered=True)

    # Horizon panels in requested order
    horizon_labels = [f"Horizon = {h}" for h in horizons]
    combined["horizon_label"] = pd.Categorical(
        combined["horizon_label"], categories=horizon_labels, ordered=True
    )

    # Colour: black for actual, then qualitative palette for models
    n_models = len(model_labels)
    colours = {"Actual": "#111111"} | dict(zip(model_labels, _PALETTE_QUALITATIVE[:n_models]))

    # Line weight and opacity: actual is heavier
    sizes   = {"Actual": 0.9} | {m: 0.65 for m in model_labels}
    alphas  = {"Actual": 1.0} | {m: 0.80 for m in model_labels}

    fig_h = max(3.0, fig_height_per_panel * len(horizons))

    p = (
        ggplot(combined, aes(x="date", y="value", color="series",
                             size="series", alpha="series"))
        + geom_line()
        + facet_wrap("~horizon_label", ncol=1, scales="fixed")
        + scale_color_manual(values=colours)
        + scale_size_manual(values=sizes)
        + scale_alpha_manual(values=alphas)
        + _date_scale(date_breaks)
        + _nhs_theme(figure_size=(fig_width, fig_h))
        + labs(
            title="Estimated avoidable deaths — actual vs. forecasts",
            x="Date",
            y="Estimated avoidable deaths",
            color="",
            size="",
            alpha="",
        )
    )
    return p
