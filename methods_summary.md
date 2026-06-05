# NHS Estimated Avoidable Deaths — Forecasting Methods

## 1. Problem and Target Variable

The task is to forecast *estimated avoidable deaths* (EAD) at Bristol NHS hospitals (BRI, NBT, WGH) for each of the 10 days following a forecast origin date D, using only information available by midday on day D. EAD is a pre-computed daily count of 30-day deaths attributable to ED boarding delays of 4+ hours. The variable carries a **3-day reporting lag**, so the latest observable EAD at origin D is EAD(D−3).

The assessment period runs from October 2025 to March 2026, spanning 173 rolling 10-day forecast periods. Performance is evaluated separately for days 1–5 and days 6–10, using mean squared error (MSE) averaged across windows.

---

## 2. Data Sources

### 2.1 Operational Metrics

The primary dataset (`data/turingAI_forecasting_challenge_dataset.csv`) is provided in long format with columns `dt`, `metric_name`, `value`, `coverage_label`, and `variable_type`. It contains approximately 220 distinct metrics at frequencies ranging from 15-minute to daily, covering ED attendances, bed occupancy, ambulance activity, and system escalation levels. The development period is 16 March 2023 – 30 September 2025.

**Midday aggregation.** Observations with `dt ≤ midday` are assigned to date D; observations after midday to D+1. Per-day means are computed within each metric-date cell.

**Per-hospital pivoting.** Each `(metric_name, coverage_label)` pair becomes a distinct column, separating BRI, NBT, and WGH where labels differ, yielding 348 operational columns.

### 2.2 Weather and Bank Holidays

Daily observed weather for Bristol (51.45°N, 2.59°W) — temperature, rainfall, snowfall, wind speed — is fetched from the Open-Meteo historical API and cached in `data/bristol_weather.csv`. Numerical weather prediction (NWP) forecasts at lead days 1–7 are fetched from the Open-Meteo Previous Runs API and cached in `data/bristol_forecast_weather.csv`, replacing observed values at prediction time. England and Wales bank holidays through 2026 are stored in `utils.py`.

**These features are computed but not included in the submitted model** (see Sections 3.5 and 3.6 for implementation details).

---

## 3. Feature Engineering

### 3.1 Imputation

Each operational column is imputed by linear interpolation in both directions, followed by forward-fill and backward-fill. Rows with a missing EAD target are dropped.

### 3.2 Rolling Summary Features

For each operational column $x$, two rolling features are computed:

$$\bar{x}^{(7)}_t = \frac{1}{7}\sum_{k=0}^{6} x_{t-k}, \qquad s^{(7)}_t = \text{SD}(x_{t-6}, \ldots, x_t)$$

### 3.3 Lagged Target Features

Two EAD lag features are computed, both respecting the 3-day reporting lag:

$$\text{EAD\_lag3}_t = \text{EAD}_{t-3}$$

$$\text{EAD\_mean7\_3}_t = \frac{1}{5}\sum_{k=3}^{7}\text{EAD}_{t-k}$$


### 3.4 Skewness Correction

For each operational feature, if $|\hat{\gamma}| > 1$: right-skewed with $x > 0$ → $\log(1 + x)$; right-skewed otherwise → $\sqrt{x - \min(x) + 1}$; left-skewed → $x^2$. Calendar features are exempt.

### 3.5 Weather Features (available but not included in submitted model)

From temperature $T$, rainfall $R$, and wind $W$, the following features are derived:

| Feature | Definition |
|---|---|
| `wx_coldness` | $\max(0,\; 10 - T)$ |
| `wx_hotness` | $\max(0,\; T - 25)$ |
| `wx_coldness2` | $\max(0,\; 5 - T)$ — second kink below 5°C |
| `wx_below_freezing` | $\mathbf{1}[T_{\min} < 0]$ |
| `wx_rain_sum` | $\min(R,\; 25)$ — winsorised |
| `wx_heavy_rain` | $\mathbf{1}[R > 20\;\text{mm}]$ |
| `wx_snowfall_sum` | Snowfall (cm) |
| `wx_wind_max` | $\min(W,\; 42)$ — winsorised |

The double-kink temperature specification captures nonlinear cold-weather mortality risk. At prediction time, NWP forecasts replace observed weather for lead days 1–7. These features did not improve out-of-sample MSE and are excluded from the submitted model; the implementation is retained for future use.

### 3.6 Calendar and Seasonality Features

Six day-of-week indicators (Monday–Saturday; Sunday is reference) and four annual Fourier terms are evaluated at the **target day** $t+h$:

$$\sin\!\left(\frac{2\pi(t+h)}{365.25}\right),\; \cos\!\left(\frac{2\pi(t+h)}{365.25}\right),\; \sin\!\left(\frac{4\pi(t+h)}{365.25}\right),\; \cos\!\left(\frac{4\pi(t+h)}{365.25}\right)$$

These capture smooth annual seasonality without discretising by month. Bank holiday indicators (`is_holiday`, `is_day_after_holiday`) were also computed and evaluated but did not emerge as independently selected features across any model configuration and are excluded from the submitted model.

---

## 4. Feature Selection (Phase 1)

To prevent noise-chasing with $p \approx 1{,}060$ predictors, a stable operational feature set is identified before fitting final models.

A standard ElasticNet with 5-fold cross-validation ($\rho = 0.5$) is fitted on 57 rolling 120-day windows (14-day stride) for each of the 10 forecast horizons. The frequency with which each predictor receives a non-zero coefficient is recorded across all 570 fits.

**Selection.** The top 20 operational predictors by frequency seed an expansion: all hospital variants (BRI/NBT/WGH) of any selected metric name are included, subject to a 5% frequency floor. This yields approximately 36 operational predictors. Calendar and Fourier features are always forced in [and potentially weather variables].

---

## 5. Models

### 5.1 Basic Rolling Model

One ElasticNet is fit per (window, horizon) pair on a rolling window of $W = 90$ days.

**Feature vector at origin $t$, horizon $h$:**
$$\mathbf{x}^{(h)}_t = \bigl[\underbrace{\mathbf{x}^{\text{op}}_{t}}_{\text{operational [+ weather]}},\; \underbrace{\mathbf{x}^{\text{cal}}_{t+h}}_{\text{calendar + Fourier}}\bigr]$$

where $\mathbf{x}^{\text{op}}_t$ contains the selected operational features (raw values, rolling means, rolling SDs, EAD\_lag3, EAD\_mean7\_3) at origin day $t$, and calendar/Fourier features are evaluated at the target day $t+h$. Features are standardised per window; alpha is selected by 5-fold CV. Predictions are clipped to a floor equal to the 5th percentile of observed EAD over the 90 days preceding origin.

### 5.2 Global Partial-Penalty Model

The global model fits on the full development period (~923 days), partitioning features into **structural** (calendar + Fourier [+ weather]) and **operational** groups. Structural features are estimated by OLS on the full history; operational features receive an ElasticNet penalty. This is solved by block coordinate descent (**PartialElasticNetCV**):

$$\boldsymbol{\beta}_s \leftarrow \arg\min \|\mathbf{y} - \mathbf{X}_s\boldsymbol{\beta}_s - \mathbf{X}_{\text{op}}\boldsymbol{\beta}_{\text{op}}\|_2^2 \quad \text{(OLS)}$$

$$\boldsymbol{\beta}_{\text{op}} \leftarrow \arg\min \|\mathbf{y} - \mathbf{X}_s\boldsymbol{\beta}_s - \mathbf{X}_{\text{op}}\boldsymbol{\beta}_{\text{op}}\|_2^2 + \alpha\!\left[\tfrac{1-\rho}{2}\|\boldsymbol{\beta}_{\text{op}}\|_2^2 + \rho\|\boldsymbol{\beta}_{\text{op}}\|_1\right]$$

$\alpha$ is initialised from ElasticNetCV; a **monotone floor** ($\alpha_h \geq \alpha_1$) ensures longer-horizon models are at least as regularised as the 1-day model.

### 5.3 Ensemble

$$\hat{y}^{\text{ens}}_{t+h} = \tfrac{1}{2}\hat{y}^{\text{basic}}_{t+h} + \tfrac{1}{2}\hat{y}^{\text{global}}_{t+h}$$

The rolling model adapts to recent operational regime shifts; the global model contributes stable seasonal estimates.

---

## 6. Results

Holdout evaluation on 355 windows, October 2024 – September 2025 (global model trained on data through September 2024):

| Model | Mean MSE 1–5 | Median MSE 1–5 | Mean MSE 6–10 | Median MSE 6–10 |
|---|---|---|---|---|
| Basic w90 | 0.0899 | 0.0549 | 0.1122 | 0.0725 |
| Global (Phase 3) | 0.1075 | 0.0392 | 0.1281 | 0.0467 |
| **Ensemble** | **0.0965** | **0.0490** | **0.1068** | **0.0559** |

The mean/median gap (~2×) reflects a fat-tailed error distribution: a small number of winter-crisis windows (cold snaps, flu/norovirus surges) produce large underpredictions that inflate the mean while the median tracks typical-day performance.

---

## 7. Algorithm

The algorithm is implemented as a single script, `run_forecast.py`. It loads the dataset, applies all preprocessing and feature engineering, performs Phase 1 selection and Phase 2 global fitting on development data, then for each origin date refits the basic rolling model and averages the two predictions. `python run_forecast.py` generates assessment forecasts; `python run_forecast.py --validate` evaluates on the holdout period.
