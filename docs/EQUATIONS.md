# Equation Reference for EN3100

This document collects the mathematical definitions used throughout the EN3100 dissertation codebase so every transformation, feature, target, metric, and risk measure is explicit. Variables assume a per-ticker daily time series unless noted.

## Notation
- $P_t^{\text{open}}, P_t^{\text{high}}, P_t^{\text{low}}, P_t^{\text{close}}$: OHLC prices on day $t$
- $V_t$: volume on day $t$
- $r_t$: simple return on day $t$
- $\log r_t$: log return on day $t$
- $\hat{r}_{t+1}$: model-predicted next-day return
- $\sigma_t$: rolling volatility estimate
- $\mu$: mean, $\sigma$: standard deviation, $\text{EMA}_\lambda$: exponential moving average with span $\lambda$

## Data handling and alignment
- **Forward fill for missing values:** $x_t = \begin{cases}x_t & \text{if observed} \\ x_{t-1} & \text{if missing}\end{cases}$
- **Standard scaling:** $x_t^{\text{scaled}} = \dfrac{x_t - \mu_{\text{train}}}{\sigma_{\text{train}}}$ (fit on train, applied to train/test).
- **Timestamp alignment:** All auxiliary feeds (sentiment/order book) are left-joined to price bars and forward-filled to daily frequency.

## Returns and targets
- **Simple return** (`return_1d`): $r_t = \dfrac{P_t^{\text{close}} - P_{t-1}^{\text{close}}}{P_{t-1}^{\text{close}}}$
- **Log return:** $\log r_t = \log P_t^{\text{close}} - \log P_{t-1}^{\text{close}}$
- **Next-day return (target):** $r_{t+1}^{\text{target}} = \dfrac{P_{t+1}^{\text{close}} - P_t^{\text{close}}}{P_t^{\text{close}}}$
- **Next-day direction:** $\text{dir}_{t+1} = \mathbb{1}(r_{t+1}^{\text{target}} > 0)$

## Technical indicators
- **Exponential Moving Average (EMA):** $\text{EMA}_\lambda(t) = \alpha P_t^{\text{close}} + (1-\alpha)\,\text{EMA}_\lambda(t-1)$ where $\alpha = \dfrac{2}{\lambda + 1}$
- **Simple Moving Average (MA):** $\text{MA}_N(t) = \dfrac{1}{N} \sum_{i=0}^{N-1} P_{t-i}^{\text{close}}$
- **MACD** (`macd_line`, `macd_signal`, `macd_hist`): $\text{MACD}_t = \text{EMA}_{12}(t) - \text{EMA}_{26}(t)$; signal line $= \text{EMA}_{9}(\text{MACD}_t)$
- **RSI** (`rsi_14`): $\text{RS}_t = \dfrac{\text{EMA}_{14}(\text{gains})}{\text{EMA}_{14}(\text{losses})}$, $\text{RSI}_t = 100 - \dfrac{100}{1 + \text{RS}_t}$
- **Rolling volatility** (`volatility_21`): $\sigma_t = \sqrt{\dfrac{1}{N-1} \sum_{i=0}^{N-1} (r_{t-i} - \bar{r})^2}$ with $N=21$
- **Rolling volume z-score** (`volume_zscore_63`): $z_t = \dfrac{V_t - \mu_{V, N}}{\sigma_{V, N}}$ with window $N=63$
- **Time-Series Momentum** (`tsmom_252`): $\text{TSMOM}_t = \dfrac{P_t^{\text{close}}}{P_{t-L}^{\text{close}}} - 1$ with lookback $L=252$
- **VWAP/TWAP placeholders:** $\text{VWAP} = \dfrac{\sum p_i v_i}{\sum v_i}$, $\text{TWAP} = \dfrac{1}{n}\sum_{i=1}^n p_i$ (implemented as TODO for intraday data)

## Market structure and pattern features
- **Swing high/low** (`swing_high`, `swing_low`): $\text{swing\_high}_t = \mathbb{1}(P_t^{\text{high}} = \max(P_{t-w:t+w}^{\text{high}}))$, similarly for swing lows. Also stored as binary flags (`swing_high_flag`, `swing_low_flag`).
- **MA crossover flags** (`ma_bullish_crossover`, `ma_bearish_crossover`):
  - Bullish: $\mathbb{1}(\text{MA}_{10}(P^{\text{close}})_t > \text{MA}_{50}(P^{\text{close}})_t \land \text{MA}_{10}(P^{\text{close}})_{t-1} \le \text{MA}_{50}(P^{\text{close}})_{t-1})$
  - Bearish: analogous with the inequality reversed.
- **Head-and-shoulders detection** (`pattern_head_shoulders`): Identifies triplets of consecutive peaks where the middle peak (head) exceeds both shoulders by at least a `tolerance` threshold (default 2%). The pattern is marked at the right shoulder.
- **Double-top/bottom detection** (`pattern_double_top`, `pattern_double_bottom`): Identifies pairs of consecutive peaks (or troughs) at similar heights (within `tolerance`), with a significant trough (or peak) between them.
- **Liquidity grab:** Detects volume spikes (above `volume_threshold` × rolling average) combined with price reversal patterns (close position > 0.7 for bullish or < 0.3 for bearish) and significant price movement.
- **Fair Value Gap (FVG):** Bullish when $P_{t-2}^{\text{high}} < P_t^{\text{low}}$; bearish when $P_{t-2}^{\text{low}} > P_t^{\text{high}}$. Gap must exceed `min_gap_percent` (default 0.1%) and remain unfilled for `fill_lookforward` bars.
- **Asia session breakout** (`ict_smt_asia`): For daily data, uses previous day's range as proxy. Bullish breakout when $P_t^{\text{close}} > P_{t-1}^{\text{high}}$; bearish when $P_t^{\text{close}} < P_{t-1}^{\text{low}}$.

## Order flow and liquidity features
- **Order Flow Imbalance** (`ofi`): $\text{OFI}_t = \dfrac{\text{bidVol}_t - \text{askVol}_t}{\text{bidVol}_t + \text{askVol}_t + \varepsilon}$
- **Depth ratio** (`depth_ratio`): $\text{Depth}_t = \dfrac{\text{bidVol}_t}{\text{bidVol}_t + \text{askVol}_t + \varepsilon}$
- **Bid-ask spread proxy** (`bid_ask_spread`): $\text{Spread}_t = \dfrac{\text{ask}_t - \text{bid}_t}{P_t^{\text{close}} + \varepsilon}$
- **Realised volatility buckets** (`realised_vol_bucket` → one-hot encoded as `regime_low`, `regime_medium`, `regime_high`): label as $\{\text{low}, \text{medium}, \text{high}\}$ by quantiles of $\sigma_t$; one-hot encoded for modelling.
- **Index drawdown** (`drawdown`): $\text{DD}_t = \dfrac{\max_{\tau \le t} P_\tau^{\text{index}} - P_t^{\text{index}}}{\max_{\tau \le t} P_\tau^{\text{index}}}$

## Feature scaling for models
- **StandardScaler (per split):** $x^{\text{scaled}} = \dfrac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$ applied after dropping non-numeric columns.
- **Rolling skewness:** $\text{Skew}_t = \dfrac{1}{N} \sum_{i=0}^{N-1} \left(\dfrac{r_{t-i} - \bar{r}}{\sigma_r}\right)^3$
- **Rolling kurtosis (excess):** $\text{Kurt}_t = \dfrac{1}{N} \sum_{i=0}^{N-1} \left(\dfrac{r_{t-i} - \bar{r}}{\sigma_r}\right)^4 - 3$
- **Rolling correlation:** $\rho_{XY}(t) = \dfrac{\sum_{i=0}^{N-1} (x_{t-i}-\bar{x})(y_{t-i}-\bar{y})}{\sqrt{\sum_{i=0}^{N-1} (x_{t-i}-\bar{x})^2 \sum_{i=0}^{N-1} (y_{t-i}-\bar{y})^2}}$

## Model training and evaluation
- **Walk-forward splits:** for split $k$, train on $[0, t_k]$, test on $(t_k, t_{k+1}]$ with chronological ordering; no shuffling.
- **SVR objective:** minimise $\dfrac{1}{2}\lVert w \rVert^2 + C \sum \xi_i$ subject to $|y_i - (w \cdot \phi(x_i) + b)| \le \epsilon + \xi_i$.
- **LightGBM objective:** gradient-boosted decision trees minimising squared error; each tree fits residuals $r_i^{(m)} = y_i - \hat{y}_i^{(m-1)}$ with shrinkage $\eta$.

### Metrics
- **RMSE:** $\text{RMSE} = \sqrt{\dfrac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$
- **MAE:** $\text{MAE} = \dfrac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$
- **$R^2$:** $R^2 = 1 - \dfrac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
- **Directional accuracy:** $\text{DA} = \dfrac{1}{n}\sum \mathbb{1}(\text{sign}(\hat{y}_i) = \text{sign}(y_i))$
- **Sharpe ratio (daily to annualised):** $\text{Sharpe} = \dfrac{\mathbb{E}[r]}{\sigma_r} \sqrt{252}$
- **Max drawdown:** $\text{MDD} = \max_t \left(\dfrac{\max_{\tau \le t} E_\tau - E_t}{\max_{\tau \le t} E_\tau}\right)$ where $E_t$ is equity.

## Strategy layer (Iteration 5)
- **Position sizing:** $\text{pos}_t = \text{clip}\left( \dfrac{\hat{r}_{t+1}}{\sigma^{\text{unscaled}}_t + 10^{-6}},\, -L,\, L \right)$ with leverage cap $L$.
- **Strategy return:** $r_t^{\text{strat}} = \text{pos}_t \cdot r_{t+1}$ (after costs when applied).
- **Equity curve:** $E_t = E_{t-1} (1 + r_t^{\text{strat}})$, starting $E_0 = 1$.

## Monte Carlo risk analysis
- **i.i.d. bootstrap:** sample returns $\{r_t^{\text{strat}}\}$ with replacement to form synthetic path $\tilde{r}_{1:T}$.
- **Block bootstrap:** sample contiguous blocks of length $B$ with replacement until length $T$, then truncate.
- **Simulated equity:** $\tilde{E}_t = \prod_{i=1}^t (1 + \tilde{r}_i)$
- **Simulated max drawdown:** same MDD formula applied to $\tilde{E}_t$.

## Sentiment and macro placeholders
- **Sentiment alignment:** daily sentiment score $s_t$ forward-filled to trading days; currently neutral $s_t=0$ unless CSV/API provided.
- **Macro benchmarks:** placeholder to merge yields/VIX; treat as additional columns $m_t$ merged and scaled as above.
