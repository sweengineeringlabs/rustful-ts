# SVHMA: A Unified Framework for Step-Based Moving Averages

## 1. Introduction

The **Step Variable Horizontal Moving Average (SVHMA)** is a non-classical filtering technique used in time series analysis and quantitative trading. Unlike traditional moving averages that update continuously at every time step, the SVHMA updates **only when a significant event occurs**. Between such events, the indicator remains constant, producing horizontal (step-like) segments.

The primary purpose of SVHMA is **noise suppression through state persistence**, rather than smoothing via continuous averaging.

---

## 2. Motivation

Financial time series are:
- Noisy
- Non-stationary
- Regime-based
- Event-driven

Classical smoothers (SMA, EMA) react to every price change, often producing whipsaws. SVHMA addresses this by:
- Ignoring insignificant price movements
- Reacting only to meaningful deviations
- Preserving market structure

---

## 3. Signal Definition

Let:
- \( x_t \in \mathbb{R} \) be the observed price or signal at time \( t \)
- \( y_t \in \mathbb{R} \) be the SVHMA output

---

## 4. Baseline Moving Average

A standard moving average is defined as:

\[
\text{MA}_t = \frac{1}{N} \sum_{i=0}^{N-1} x_{t-i}
\]

This serves as the reference value used during step updates.

---

## 5. Event-Triggered State Update

SVHMA follows a **sample-and-hold** mechanism:

\[
y_t =
\begin{cases}
y_{t-1}, & \text{if no update condition is met} \\
\phi(x_{1:t}), & \text{if update condition is met}
\end{cases}
\]

Where \( \phi(\cdot) \) is the update function.

---

## 6. Deadband Update Condition

Define the deviation from the current state:

\[
d_t = |x_t - y_{t-1}|
\]

An update occurs only if:

\[
d_t > \theta_t
\]

where \( \theta_t \) is a threshold.

This structure defines a **non-linear deadband filter**.

---

## 7. Threshold Modes

The threshold \( \theta_t \) can be computed using various methods. Each mode offers different trade-offs between simplicity, responsiveness, and robustness.

### 7.1 Fixed Threshold

\[
\theta_t = c, \quad c > 0
\]

- Simplest implementation
- No adaptation to market conditions
- Useful for normalized or stationary signals

---

### 7.2 Percentage Threshold

\[
\theta_t = p \cdot |x_t|, \quad p \in (0, 1)
\]

- Scales with price level
- Handles instruments at different price magnitudes
- Common choice: \( p = 0.01 \) to \( 0.05 \) (1-5%)

---

### 7.3 ATR-Based Threshold (Default)

Let:
- \( \text{ATR}_t \) be the Average True Range over period \( n \)
- \( k > 0 \) be a sensitivity multiplier

\[
\theta_t = k \cdot \text{ATR}_t
\]

Where:

\[
\text{TR}_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
\]

\[
\text{ATR}_t = \frac{1}{n} \sum_{i=0}^{n-1} \text{TR}_{t-i} \quad \text{(SMA)} \quad \text{or} \quad \text{EMA}(\text{TR}, n)
\]

- Volatility-adaptive
- Industry standard for range-based filters
- Common choice: \( k = 1.0 \) to \( 3.0 \), \( n = 14 \)

---

### 7.4 Standard Deviation Threshold

Let \( \sigma_t \) be the rolling standard deviation of \( x \) over period \( n \):

\[
\sigma_t = \sqrt{\frac{1}{n} \sum_{i=0}^{n-1} (x_{t-i} - \bar{x}_t)^2}
\]

\[
\theta_t = k \cdot \sigma_t, \quad k > 0
\]

- Bollinger Band-style volatility measure
- More sensitive to recent price dispersion
- Common choice: \( k = 1.0 \) to \( 2.0 \), \( n = 20 \)

---

### 7.5 Donchian Range Threshold

Let \( H^{(n)}_t \) and \( L^{(n)}_t \) be the highest high and lowest low over period \( n \):

\[
\theta_t = k \cdot (H^{(n)}_t - L^{(n)}_t), \quad k \in (0, 1]
\]

- Based on price channel width
- Captures breakout-style volatility
- Common choice: \( k = 0.25 \) to \( 0.5 \), \( n = 20 \)

---

### 7.6 Interquartile Range (IQR) Threshold

Let \( Q_1 \) and \( Q_3 \) be the 25th and 75th percentiles of \( x \) over period \( n \):

\[
\text{IQR}_t = Q_3(x_{t-n+1:t}) - Q_1(x_{t-n+1:t})
\]

\[
\theta_t = k \cdot \text{IQR}_t, \quad k > 0
\]

- Robust to outliers and fat tails
- Suitable for noisy or spike-prone data
- Common choice: \( k = 1.0 \) to \( 1.5 \), \( n = 20 \)

---

### 7.7 Median Absolute Deviation (MAD) Threshold

Let \( \tilde{x}_t \) be the rolling median over period \( n \):

\[
\text{MAD}_t = \text{median}(|x_{t-i} - \tilde{x}_t|), \quad i = 0, \ldots, n-1
\]

\[
\theta_t = k \cdot \text{MAD}_t, \quad k > 0
\]

- Most robust to outliers
- Preferred for heavy-tailed distributions
- Common choice: \( k = 2.0 \) to \( 3.0 \), \( n = 20 \)

---

### 7.8 VHF-Based Threshold

The Vertical Horizontal Filter (VHF) measures trend strength as the ratio of net price movement to cumulative absolute movement.

Let:
- \( H^{(n)}_t \) and \( L^{(n)}_t \) be the highest high and lowest low over period \( n \)
- \( \Delta_i = |x_i - x_{i-1}| \) be the absolute price change

Define VHF:

\[
\text{VHF}_t = \frac{H^{(n)}_t - L^{(n)}_t}{\sum_{i=t-n+1}^{t} \Delta_i}
\]

Where \( \text{VHF}_t \in [0, 1] \):
- \( \text{VHF}_t \to 1 \): Strong trend (price moved efficiently)
- \( \text{VHF}_t \to 0 \): Choppy/ranging (much noise, little net movement)

**Threshold formulation:**

\[
\theta_t = k \cdot (1 - \text{VHF}_t) \cdot \text{ATR}_t, \quad k > 0
\]

Or inversely, for trend-following behavior:

\[
\theta_t = k \cdot \text{VHF}_t \cdot \text{ATR}_t
\]

- Combines trend strength with volatility
- Tighter threshold in trends, wider in chop (or vice versa)
- Common choice: \( k = 1.0 \) to \( 2.0 \), \( n = 14 \)

**Alternative: VHF-Adaptive Smoothing Factor**

VHF can also modulate the smoothing factor of the base moving average:

\[
\alpha_t = \alpha_{\text{base}} \cdot \text{VHF}_t \cdot 2
\]

\[
\text{VMA}_t = \text{VMA}_{t-1} + \alpha_t \cdot (x_t - \text{VMA}_{t-1})
\]

This creates a Variable Moving Average (VMA) that smooths more in choppy markets and responds faster in trends.

---

### 7.9 Change Volatility Threshold

This mode uses the standard deviation of price *changes* rather than price levels, filtering based on whether the current change is unusual relative to recent change behavior.

Let \( \Delta x_i = |x_i - x_{i-1}| \) be the absolute price change at time \( i \).

Define the mean change over period \( n \):

\[
\bar{\Delta}_t = \frac{1}{n} \sum_{i=t-n+1}^{t} \Delta x_i
\]

Define the standard deviation of changes:

\[
\sigma_{\Delta,t} = \sqrt{\frac{1}{n} \sum_{i=t-n+1}^{t} (\Delta x_i - \bar{\Delta}_t)^2}
\]

Threshold:

\[
\theta_t = k \cdot \sigma_{\Delta,t}, \quad k > 0
\]

Update condition (note: compares current *change* to threshold):

\[
\Delta x_t > \theta_t \implies \text{update}
\]

- Filters based on "unusualness" of current move relative to recent moves
- More sensitive to regime changes in volatility
- Distinct from Section 7.4 which uses \( \sigma(x) \) (price dispersion)
- Common choice: \( k = 2.0 \) to \( 3.0 \), \( n = 14 \)

**Comparison:**

| Mode | Measures | Filters When |
|------|----------|--------------|
| Std Dev (7.4) | \( \sigma(x) \) - price dispersion | Price far from mean price |
| Change Vol (7.9) | \( \sigma(\Delta x) \) - change dispersion | Change far from mean change |

---

### 7.10 Threshold Mode Summary

| Mode | Formula | Adaptive | Outlier Robust | Complexity |
|------|---------|----------|----------------|------------|
| Fixed | \( c \) | No | N/A | O(1) |
| Percentage | \( p \cdot |x_t| \) | Price-level | No | O(1) |
| ATR | \( k \cdot \text{ATR}_t \) | Volatility | No | O(n) |
| Std Dev | \( k \cdot \sigma_t \) | Volatility | No | O(n) |
| Donchian | \( k \cdot (H_n - L_n) \) | Range | No | O(n) |
| IQR | \( k \cdot \text{IQR}_t \) | Volatility | Yes | O(n log n) |
| MAD | \( k \cdot \text{MAD}_t \) | Volatility | Yes | O(n log n) |
| VHF | \( k \cdot (1 - \text{VHF}_t) \cdot \text{ATR}_t \) | Trend + Volatility | No | O(n) |
| Change Vol | \( k \cdot \sigma_{\Delta,t} \) | Change Volatility | No | O(n) |

---

## 8. Canonical SVHMA Equation

\[
\boxed{
 y_t =
 \begin{cases}
 y_{t-1}, & |x_t - y_{t-1}| \le \theta_t \\
 \phi(x_{1:t}), & |x_t - y_{t-1}| > \theta_t
 \end{cases}
}
\]

Where:
- \( \theta_t \) is the threshold (Section 7)
- \( \phi(\cdot) \) is the update function (Section 9)

This formulation is:
- Event-driven
- Non-linear
- Threshold-mode agnostic
- Piecewise constant

---

## 9. Update Function Variants

### 9.1 Snap-to-Price

\[
\phi(x_{1:t}) = x_t
\]

Fastest reaction, minimal lag.

---

### 9.2 Snap-to-Moving-Average (Default)

\[
\phi(x_{1:t}) = \text{MA}_t
\]

Stable and robust.

---

### 9.3 Damped Step Update

\[
\phi(x_{1:t}) = y_{t-1} + \alpha (x_t - y_{t-1}), \quad \alpha \in (0,1]
\]

Hybrid between step MA and EMA.

---

### 9.4 Quantized Step Update

Rather than snapping to a computed value, this variant moves in discrete increments of a fixed step size \( s \):

\[
\delta_t = \text{MA}_t - y_{t-1}
\]

\[
\phi(x_{1:t}) = y_{t-1} + \left\lfloor \frac{\delta_t}{s} \right\rfloor \cdot s
\]

Where \( \lfloor \cdot \rfloor \) is the floor function (truncation toward zero).

- Output is always a multiple of step size \( s \)
- Produces staircase-like output
- Useful for discretized signal levels or grid-based trading
- Common choice: \( s \) in pips or price units (e.g., 0.0001 for forex, 0.01 for stocks)

**Initialization:**

\[
y_0 = \left\lfloor \frac{x_0}{s} \right\rceil \cdot s
\]

Where \( \lfloor \cdot \rceil \) denotes rounding to nearest multiple.

---

### 9.5 VHF-Adaptive VMA Update

This variant uses a Variable Moving Average where the smoothing factor is modulated by the Vertical Horizontal Filter:

\[
\alpha_t = \alpha_{\text{base}} \cdot \text{VHF}_t \cdot 2, \quad \alpha_{\text{base}} = \frac{2}{n+1}
\]

\[
\phi(x_{1:t}) = \text{VMA}_t = \text{VMA}_{t-1} + \alpha_t \cdot (x_t - \text{VMA}_{t-1})
\]

Where \( \text{VHF}_t \) is defined in Section 7.8.

Behavior:
- High VHF (trending): Higher \( \alpha_t \), faster response
- Low VHF (choppy): Lower \( \alpha_t \), more smoothing

This can be combined with quantized step update:

\[
\phi(x_{1:t}) = y_{t-1} + \left\lfloor \frac{\text{VMA}_t - y_{t-1}}{s} \right\rfloor \cdot s
\]

---

### 9.6 Update Function Summary

| Variant | Formula | Behavior | Use Case |
|---------|---------|----------|----------|
| Snap-to-Price | \( x_t \) | Immediate | High-frequency, minimal lag |
| Snap-to-MA | \( \text{MA}_t \) | Stable | General purpose (default) |
| Damped Step | \( y_{t-1} + \alpha(x_t - y_{t-1}) \) | Gradual | Smooth transitions |
| Quantized Step | \( y_{t-1} + \lfloor\delta/s\rfloor \cdot s \) | Discrete | Grid trading, levels |
| VHF-Adaptive | \( \text{VMA}_t \) | Trend-aware | Regime-adaptive |

---

## 10. Directional (Trend-Constrained) SVHMA

Let direction:

\[
s_t = \text{sign}(x_t - y_{t-1})
\]

Directional update rule:

\[
y_t =
\begin{cases}
y_{t-1}, & |x_t - y_{t-1}| \le \theta_t \\
\max(y_{t-1}, \phi(x_{1:t})), & s_t > 0 \\
\min(y_{t-1}, \phi(x_{1:t})), & s_t < 0
\end{cases}
\]

This variant behaves similarly to trailing stop and SuperTrend logic.

---

## 11. Zero-Order Hold Representation

Define update times:

\[
\mathcal{T} = \{ t : |x_t - y_{t-1}| > \theta_t \}
\]

Then:

\[
y_t = y_{\tau}, \quad \tau = \max \{ s \in \mathcal{T} : s \le t \}
\]

This is a **zero-order hold over event times**.

---

## 12. Regime Interpretation

Define regime indicator:

\[
R_t = \mathbb{1}(|x_t - y_{t-1}| > \theta_t)
\]

- \( R_t = 0 \): consolidation / noise regime
- \( R_t = 1 \): trend / transition regime

SVHMA remains constant in regime 0 and updates in regime 1.

---

## 13. Classification

| Property | Classical MA | SVHMA |
|--------|-------------|-------|
| Linear | Yes | No |
| Continuous | Yes | No |
| Event-driven | No | Yes |
| State-holding | No | Yes |
| Threshold-adaptive | No | Yes (9 modes) |
| Update-configurable | No | Yes (5 modes) |

---

## 14. One-Sentence Definition

> The Step Variable Horizontal Moving Average is a **non-linear, event-triggered, threshold-adaptive, state-holding filter** that updates only when price deviations exceed a configurable dynamic threshold, remaining constant otherwise.

---

## 15. Applications

- Trend state detection
- Noise-robust entries and exits
- Dynamic support and resistance
- Regime labeling for machine learning
- Algorithmic trading filters

---

## 16. Conclusion

SVHMA represents a shift from continuous smoothing toward **event-based signal estimation**, aligning more closely with the structural and regime-driven nature of financial markets.

---

## 17. References and Prior Art

### 17.1 Step VHF Adaptive VMA (MQL5)

The VHF-based threshold mode (Section 7.8) and VHF-Adaptive VMA update function (Section 9.5) are derived from the "Step VHF Adaptive VMA" indicator by mladen (2018).

**Implementation characteristics:**
- Fixed step size threshold (pips)
- VHF-modulated EMA smoothing factor: \( \alpha_t = \alpha_{\text{base}} \cdot \text{VHF}_t \cdot 2 \)
- Quantized step output

**Reference:** `step_vhf_adaptive_vma.mq5` - MetaTrader 5 indicator

### 17.2 Deviation Filtered Average (MQL5)

The Change Volatility threshold mode (Section 7.9) is derived from the "Deviation Filtered Average" indicator by mladen (2018).

**Implementation characteristics:**
- Threshold based on standard deviation of price changes: \( \theta_t = k \cdot \sigma(\Delta x) \)
- Compares current change magnitude to threshold
- Snap-to-MA update function

**Reference:** `filtered_averages.mq5` - MetaTrader 5 indicator

### 17.3 Related Indicators

| Indicator | Relationship to SVHMA |
|-----------|----------------------|
| SuperTrend | ATR-based bands with directional state |
| Keltner Channels | ATR-based envelope (continuous, not stepped) |
| Parabolic SAR | Trailing stop with acceleration factor |
| Renko Charts | Fixed step price representation |
| Kaufman AMA | Efficiency-ratio adaptive smoothing |
| VHF (Vertical Horizontal Filter) | Trend strength measure used in Section 7.8 |

