# Step Variable Horizontal Moving Average (SVHMA)

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

## 7. Volatility-Adaptive Threshold (ATR-Based)

Let:
- \( \text{ATR}_t \) be the Average True Range
- \( k > 0 \) be a sensitivity parameter

Define:

\[
\theta_t = k \cdot \text{ATR}_t
\]

Update condition becomes:

\[
|x_t - y_{t-1}| > k \cdot \text{ATR}_t
\]

---

## 8. Canonical SVHMA Equation

\[
\boxed{
 y_t =
 \begin{cases}
 y_{t-1}, & |x_t - y_{t-1}| \le k \cdot \text{ATR}_t \\
 \text{MA}_t, & |x_t - y_{t-1}| > k \cdot \text{ATR}_t
 \end{cases}
}
\]

This formulation is:
- Event-driven
- Non-linear
- Volatility-aware
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

## 10. Directional (Trend-Constrained) SVHMA

Let direction:

\[
s_t = \text{sign}(x_t - y_{t-1})
\]

Directional update rule:

\[
y_t =
\begin{cases}
y_{t-1}, & |x_t - y_{t-1}| \le k \cdot \text{ATR}_t \\
\max(y_{t-1}, \text{MA}_t), & s_t > 0 \\
\min(y_{t-1}, \text{MA}_t), & s_t < 0
\end{cases}
\]

This variant behaves similarly to trailing stop and SuperTrend logic.

---

## 11. Zero-Order Hold Representation

Define update times:

\[
\mathcal{T} = \{ t : |x_t - y_{t-1}| > k \cdot \text{ATR}_t \}
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
R_t = \mathbb{1}(|x_t - y_{t-1}| > k \cdot \text{ATR}_t)
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
| Volatility-adaptive | Optional | Native |

---

## 14. One-Sentence Definition

> The Step Variable Horizontal Moving Average is a **non-linear, event-triggered, volatility-adaptive, state-holding filter** that updates only when price deviations exceed a dynamic threshold, remaining constant otherwise.

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

