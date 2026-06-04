# NHS Estimated Avoidable Deaths — Forecasting Methods

## 1. Problem and Target Variable

The task is to forecast *estimated avoidable deaths* (EAD) at Bristol NHS hospitals (BRI, NBT, WGH) for each of the 10 days following a forecast origin date D, using only information available by midday on day D. EAD is a pre-computed daily count of 30-day deaths attributable to ED boarding delays of 4+ hours. The variable carries a **3-day reporting lag**, so the latest observable EAD at origin D is EAD(D−3).

The assessment period runs from October 2025 to March 2026, spanning 173 rolling 10-day forecast periods. Performance is evaluated separately for days 1–5 and days 6–10, using mean squared error (MSE) averaged across windows.

---

## 2. Data Sources

### 2.1 Operational Metrics

The primary dataset (`data/turingAI_forecasting_challenge_dataset.csv`) is provided in long format with columns `dt`, `metric_name`, `value`, `coverage_label`, and `variable_type`. It contains approximately 220 distinct metrics recorded at frequencies ranging from 15-minute to daily, covering ED attendances, bed occupancy, ambulance activity, community referrals, and system escalation levels. The development period is 16 March 2023 – 30 September 2025.

**Midday aggregation.** Observations with `dt ≤ midday` are assigned to date D; observations after midday are assigned to D+1. This reflects the contest protocol of "data available by midday on day D". Per-day means are computed within each metric-date cell.

**Per-hospital pivoting.** Rather than averaging across hospitals, each `(metric_name, coverage_label)` pair becomes a distinct column, separating BRI, NBT, and WGH where coverage labels differ. This yields 348 operational columns before any further processing.

### 2.2 Observed Weather

Daily weather for Bristol (51.45°N, 2.59°W) is fetched from the Open-Meteo historical API and cached in `data/bristol_weather.csv`. Variables used: mean temperature (°C), minimum temperature, daily rainfall (mm), snowfall (cm), and maximum wind speed (km/h).

### 2.3 NWP Forecast Weather

Numerical weather prediction forecasts for lead days 1–7 are fetched from the Open-Meteo Previous Runs API and cached in `data/bristol_forecast_weather.csv`. These are used at prediction time in place of observed weather, since future observed weather is unavailable.

### 2.4 Bank Holidays

England and Wales bank holidays from 2023–2026 are stored as a hardcoded set. A "day after holiday" flag is derived from this set.

---

## 3. Feature Engineering

### 3.1 Imputation

Each operational column is imputed by linear interpolation in both directions, followed by forward-fill and backward-fill for any remaining gaps. Rows with a missing EAD target are dropped.

### 3.2 Rolling Summary Features

For each operational column $x$, two rolling features are computed:

$$\bar{x}^{(7)}_t = \frac{1}{7}\sum_{k=0}^{6} x_{t-k}, \qquad s^{(7)}_t = \text{SD}(x_{t-6}, \ldots, x_t)$$

These capture recent mean levels and volatility.

### 3.3 Lagged Target

$$\text{EAD\_lag3}_t = \text{EAD}_{t-3}$$

This is the most recent EAD value observable at midday on day $t$.

### 3.4 Skewness Correction

For each operational feature, the sample skewness $\hat{\gamma}$ is computed. If $|\hat{\gamma}| > 1$:
- Right-skewed with $x > 0$: replace with $\log(1 + x)$
- Right-skewed otherwise: replace with $\sqrt{x - \min(x) + 1}$
- Left-skewed: replace with $x^2$

Calendar and weather features are exempt.

### 3.5 Weather Features

From observed (and NWP forecast) temperature $T$ and rainfall $R$, the following are derived:

| Feature | Definition |
|---|---|
| `wx_coldness` | $\max(0,\; 10 - T)$ |
| `wx_hotness` | $\max(0,\; T - 10)$ |
| `wx_coldness2` | $\max(0,\; 5 - T)$ — second kink below 5°C |
| `wx_below_freezing` | $\mathbf{1}[T_{\min} < 0]$ |
| `wx_rain_sum` | $\min(R,\; 25)$ — winsorised at 25 mm |
| `wx_heavy_rain` | $\mathbf{1}[R > 20\;\text{mm}]$ |
| `wx_snowfall_sum` | Snowfall (cm) |
| `wx_wind_max` | $\min(W,\; 42)$ — winsorised at 42 km/h |

The double-kink temperature specification (coldness, coldness2, hotness) is motivated by evidence that mortality risk from cold increases nonlinearly below 5°C, and that heat effects appear only at high temperatures.

At prediction time, NWP forecasts replace observed weather for future days; for lead days beyond 7 the lead-7 NWP value is used as the best available forecast. A **3-day mean** of coldness, hotness, and coldness2 centred on the target day captures sustained spells:

$$\overline{\text{cold3}}_{t+h} = \frac{1}{3}\sum_{k=0}^{2}\text{wx\_coldness}_{t+h-k}$$

### 3.6 Calendar Features

Six day-of-week indicator variables (Monday–Saturday; Sunday is the reference), an `is_holiday` flag, and an `is_day_after_holiday` flag are evaluated at the **target day** $t+h$, not the origin day. Day-of-week and the holiday calendar are known in advance, and NWP forecasts supply forward-looking weather.

---

## 4. Feature Selection (Phase 1)

To prevent noise-chasing with $p \approx 1{,}060$ predictors, a stable operational feature set is identified before fitting final models.

**Procedure.** A standard ElasticNet with 5-fold cross-validation ($\alpha_{1/2} = 0.5$, i.e., equal L1/L2) is fitted on 57 rolling 120-day windows with a 14-day stride, for each of the 10 forecast horizons. Each fitting uses all 1,060+ predictors with StandardScaler pre-processing. The frequency with which each predictor receives a non-zero coefficient is recorded across all 570 (windows × horizons) fits.

**Selection.** The top 20 operational predictors by frequency are identified. To avoid splitting related metrics across hospitals, all variants of any selected metric name (BRI, NBT, WGH versions) are included. Predictors appearing in fewer than 5% of fits are dropped. The result is approximately 24 operational predictors. Calendar and weather features are always forced in.

---

## 5. Models

### 5.1 Basic Rolling Model

The basic model fits one ElasticNet per (window, horizon) pair on a rolling training window of width $W = 90$ days, with a stride of 1 day.

**Feature vector at origin $t$, horizon $h$:**
$$\mathbf{x}^{(h)}_t = \bigl[\underbrace{\mathbf{x}^{\text{op}}_{t}}_{\text{operational}},\; \underbrace{\mathbf{x}^{\text{cal}}_{t+h}}_{\text{calendar}},\; \underbrace{\mathbf{x}^{\text{wx}}_{t+h}}_{\text{weather}},\; \underbrace{\bar{\mathbf{x}}^{\text{wx3}}_{t+h}}_{\text{3-day means}}\bigr]$$

where $\mathbf{x}^{\text{op}}_{t}$ contains the 24 selected operational features (raw values, rolling means, rolling SDs, and EAD\_lag3) at origin day $t$, and calendar/weather features are evaluated at the target day.

**Training.** For horizon $h$, the training pairs are $\{(\mathbf{x}^{(h)}_{t-k},\; \text{EAD}_{t-k+h})\}_{k=1}^{W-h}$. Features are standardised with a per-window StandardScaler. Alpha is selected by 5-fold time-ordered cross-validation:

$$\hat{\alpha}_h = \arg\min_{\alpha} \text{CV-MSE}\!\left(\alpha;\, \mathbf{X}^{\text{train}}_h,\, \mathbf{y}^{\text{train}}_h\right)$$

**Prediction.** $\hat{y}_{t+h} = \hat{\boldsymbol{\beta}}_h^\top \tilde{\mathbf{x}}^{(h)}_t$, where $\tilde{\cdot}$ denotes standardisation with training-window statistics. Predictions are bounded below by the minimum observed EAD over the 60 days preceding the origin (acknowledging the 3-day lag), and capped above at 1.75 deaths/day.

### 5.2 Global Partial-Penalty Model

The global model fits on the full development period (~924 days) but partitions features into **structural** (calendar + weather) and **operational** groups, applying fundamentally different regularisation to each.

**Motivation.** Calendar and weather effects change slowly and can be estimated precisely from multi-year data. Penalising them equally with rapidly-varying operational metrics shrinks coefficients that should be large and stable. In a short rolling window, the opposite problem holds — seasonal features have limited variation, making unpenalised OLS degenerate. The global model resolves this by using all available history.

**Block coordinate descent (PartialElasticNetCV).** Let $\mathbf{X}_s$ be the structural feature matrix (unpenalised) and $\mathbf{X}_{\text{op}}$ the operational matrix (penalised). With penalty mask $\mathbf{m} \in \{0,1\}^p$, the objective alternates between:

$$\boldsymbol{\beta}_s \leftarrow \arg\min_{\boldsymbol{\beta}_s} \|\mathbf{y} - \mathbf{X}_s\boldsymbol{\beta}_s - \mathbf{X}_{\text{op}}\boldsymbol{\beta}_{\text{op}}\|_2^2 \quad \text{(OLS)}$$

$$\boldsymbol{\beta}_{\text{op}} \leftarrow \arg\min_{\boldsymbol{\beta}_{\text{op}}} \|\mathbf{y} - \mathbf{X}_s\boldsymbol{\beta}_s - \mathbf{X}_{\text{op}}\boldsymbol{\beta}_{\text{op}}\|_2^2 + \alpha\left[\tfrac{1}{2}(1-\rho)\|\boldsymbol{\beta}_{\text{op}}\|_2^2 + \rho\|\boldsymbol{\beta}_{\text{op}}\|_1\right]$$

with $\rho = 0.5$ and convergence tested by $\max(\|\Delta\boldsymbol{\beta}_s\|, \|\Delta\boldsymbol{\beta}_{\text{op}}\|) < 10^{-4}$.

**Alpha selection.** A standard ElasticNetCV on the full (scaled) feature set yields a candidate $\tilde{\alpha}_h$ for each horizon. A **monotone floor** is applied:

$$\alpha_h = \max(\tilde{\alpha}_h,\; \alpha_1), \quad h = 2,\ldots,10$$

ensuring longer-horizon models are at least as regularised as the 1-day-ahead model.

**Phase 3 (holdout refitting).** For evaluation, the global model is refit using only data up to September 2024 and evaluated on October 2024 – September 2025, matching the assessment period structure.

### 5.3 Ensemble Model

The final submission averages predictions from the Basic Rolling model and the Global model:

$$\hat{y}^{\text{ens}}_{t+h} = \frac{1}{2}\hat{y}^{\text{basic}}_{t+h} + \frac{1}{2}\hat{y}^{\text{global}}_{t+h}$$

The basic model adapts to recent operational patterns (small rolling window); the global model provides stable structural-feature coefficients (full training history, unpenalised OLS for calendar and weather). The ensemble captures the complementary strengths of both: the basic model limits damage in windows where the global model's static coefficients are stale; the global model's precise structural estimates reduce systematic bias on typical days.

---

## 6. Results

Holdout evaluation on 355 windows, October 2024 – September 2025:

| Model | Mean MSE 1–5 | Median MSE 1–5 | Mean MSE 6–10 | Median MSE 6–10 |
|---|---|---|---|---|
| Example (R baseline) | 0.1102 | 0.0693 | 0.1258 | 0.1048 |
| Global | 0.1190 | 0.0556 | 0.1334 | 0.0572 |
| Basic w90 | 0.0937 | 0.0571 | 0.1087 | 0.0703 |
| **Ensemble** | **0.0882** | **0.0514** | **0.1023** | **0.0574** |

The ensemble improves on the basic model by 5.9% on mean days 1–5, 10% on median days 1–5, and 5.9% on mean days 6–10 relative to the next-best single model, while also achieving near-global median performance on days 6–10.

The global model's lower typical-day error (median) reflects precise estimation of structural effects over the full training history. The basic model's lower mean error reflects rolling adaptation to operational regime shifts that the static global model cannot accommodate. Simple averaging exploits both properties without requiring a more complex meta-learner.

---

## 7. Submission Pipeline

At assessment time, for each origin date D in the assessment window:

1. Load operational metrics and aggregate to midday day D.
2. Fetch observed weather through day D; fetch NWP forecasts for days D+1 through D+7.
3. Compute all derived features. Calendar features for D+1…D+10 are always known.
4. **Basic w90**: refit 10 ElasticNetCV models on the 90-day window ending at D; produce $\hat{y}^{\text{basic}}_{D+1},\ldots,\hat{y}^{\text{basic}}_{D+10}$.
5. **Global**: apply the pre-fitted models (fixed coefficients, trained through Sep 2025) to the feature vector at D; produce $\hat{y}^{\text{global}}_{D+1},\ldots,\hat{y}^{\text{global}}_{D+10}$.
6. **Ensemble**: $\hat{y}^{\text{ens}}_{D+h} = \tfrac{1}{2}(\hat{y}^{\text{basic}}_{D+h} + \hat{y}^{\text{global}}_{D+h})$.
7. Write `submission/pred_matrix.csv` and `submission/mse_summary.csv`.
