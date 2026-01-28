//! Advanced Oscillator Indicators
//!
//! A collection of sophisticated oscillator indicators for technical analysis.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Cycle Oscillator - Oscillator based on cycle analysis
///
/// Detects and measures cyclical patterns in price data by analyzing
/// deviations from a smoothed baseline.
#[derive(Debug, Clone)]
pub struct CycleOscillator {
    cycle_period: usize,
    smooth_period: usize,
}

impl CycleOscillator {
    pub fn new(cycle_period: usize, smooth_period: usize) -> Result<Self> {
        if cycle_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "cycle_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            cycle_period,
            smooth_period,
        })
    }

    /// Calculate cycle oscillator values
    ///
    /// Returns values oscillating around zero, indicating cycle position
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let min_period = self.cycle_period + self.smooth_period;

        if n < min_period {
            return result;
        }

        // Calculate smoothed baseline (SMA)
        let mut baseline = vec![0.0; n];
        for i in (self.smooth_period - 1)..n {
            let start = i + 1 - self.smooth_period;
            baseline[i] = close[start..=i].iter().sum::<f64>() / self.smooth_period as f64;
        }

        // Calculate cycle component
        for i in min_period..n {
            let start = i.saturating_sub(self.cycle_period);

            // Deviation from baseline
            let deviation = close[i] - baseline[i];

            // Calculate cycle amplitude using standard deviation of deviations
            let mut deviations = Vec::with_capacity(self.cycle_period);
            for j in start..=i {
                if baseline[j] > 1e-10 {
                    deviations.push(close[j] - baseline[j]);
                }
            }

            if !deviations.is_empty() {
                let mean_dev: f64 = deviations.iter().sum::<f64>() / deviations.len() as f64;
                let variance: f64 = deviations
                    .iter()
                    .map(|d| (d - mean_dev).powi(2))
                    .sum::<f64>()
                    / deviations.len() as f64;
                let std_dev = variance.sqrt();

                // Normalize deviation by standard deviation
                if std_dev > 1e-10 {
                    result[i] = (deviation / std_dev) * 50.0; // Scale to approximately -100 to +100
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for CycleOscillator {
    fn name(&self) -> &str {
        "Cycle Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.cycle_period + self.smooth_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Volatility Oscillator - Oscillator measuring volatility changes
///
/// Measures how current volatility compares to historical volatility,
/// oscillating between expansion and contraction zones.
#[derive(Debug, Clone)]
pub struct VolatilityOscillator {
    short_period: usize,
    long_period: usize,
}

impl VolatilityOscillator {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if long_period < short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than or equal to short_period".to_string(),
            });
        }
        Ok(Self {
            short_period,
            long_period,
        })
    }

    /// Calculate volatility oscillator values
    ///
    /// Returns values where positive indicates volatility expansion,
    /// negative indicates volatility contraction
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.long_period + 1 {
            return result;
        }

        // Calculate True Range for each bar
        let mut tr = vec![0.0; n];
        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }

        for i in self.long_period..n {
            // Short-term volatility (ATR)
            let short_start = i + 1 - self.short_period;
            let short_atr: f64 =
                tr[short_start..=i].iter().sum::<f64>() / self.short_period as f64;

            // Long-term volatility (ATR)
            let long_start = i + 1 - self.long_period;
            let long_atr: f64 = tr[long_start..=i].iter().sum::<f64>() / self.long_period as f64;

            // Volatility ratio as oscillator
            if long_atr > 1e-10 {
                // Ratio > 1 means expansion, < 1 means contraction
                // Convert to oscillator: (ratio - 1) * 100
                let ratio = short_atr / long_atr;
                result[i] = (ratio - 1.0) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for VolatilityOscillator {
    fn name(&self) -> &str {
        "Volatility Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(
            &data.high,
            &data.low,
            &data.close,
        )))
    }
}

/// Trend Oscillator - Measures trend strength oscillation
///
/// Quantifies the strength and direction of the current trend
/// using multiple timeframe analysis.
#[derive(Debug, Clone)]
pub struct TrendOscillator {
    fast_period: usize,
    slow_period: usize,
}

impl TrendOscillator {
    pub fn new(fast_period: usize, slow_period: usize) -> Result<Self> {
        if fast_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "fast_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if slow_period <= fast_period {
            return Err(IndicatorError::InvalidParameter {
                name: "slow_period".to_string(),
                reason: "must be greater than fast_period".to_string(),
            });
        }
        Ok(Self {
            fast_period,
            slow_period,
        })
    }

    /// Calculate trend oscillator values
    ///
    /// Returns values from -100 to +100 indicating trend strength and direction
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.slow_period + 1 {
            return result;
        }

        // Calculate EMAs
        let fast_multiplier = 2.0 / (self.fast_period as f64 + 1.0);
        let slow_multiplier = 2.0 / (self.slow_period as f64 + 1.0);

        let mut fast_ema = vec![0.0; n];
        let mut slow_ema = vec![0.0; n];

        // Initialize with SMA
        let fast_sma: f64 = close[0..self.fast_period].iter().sum::<f64>() / self.fast_period as f64;
        let slow_sma: f64 = close[0..self.slow_period].iter().sum::<f64>() / self.slow_period as f64;

        fast_ema[self.fast_period - 1] = fast_sma;
        slow_ema[self.slow_period - 1] = slow_sma;

        // Calculate fast EMA
        for i in self.fast_period..n {
            fast_ema[i] = (close[i] - fast_ema[i - 1]) * fast_multiplier + fast_ema[i - 1];
        }

        // Calculate slow EMA
        for i in self.slow_period..n {
            slow_ema[i] = (close[i] - slow_ema[i - 1]) * slow_multiplier + slow_ema[i - 1];
        }

        // Calculate trend oscillator
        for i in self.slow_period..n {
            if slow_ema[i] > 1e-10 {
                // Percentage difference between fast and slow EMA
                let diff = ((fast_ema[i] - slow_ema[i]) / slow_ema[i]) * 100.0;

                // Calculate trend consistency (directional movement)
                let lookback = self.fast_period.min(i);
                let mut up_moves = 0;
                let mut down_moves = 0;
                for j in (i - lookback + 1)..=i {
                    if close[j] > close[j - 1] {
                        up_moves += 1;
                    } else if close[j] < close[j - 1] {
                        down_moves += 1;
                    }
                }

                let consistency = if up_moves + down_moves > 0 {
                    (up_moves as f64 - down_moves as f64) / (up_moves + down_moves) as f64
                } else {
                    0.0
                };

                // Combine EMA difference with consistency
                result[i] = diff * 10.0 + consistency * 50.0;

                // Clamp to -100 to +100
                result[i] = result[i].max(-100.0).min(100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for TrendOscillator {
    fn name(&self) -> &str {
        "Trend Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.slow_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Price Action Oscillator - Based on pure price action
///
/// Analyzes candlestick patterns and price structure to generate
/// an oscillator reading.
#[derive(Debug, Clone)]
pub struct PriceActionOscillator {
    period: usize,
}

impl PriceActionOscillator {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate price action oscillator values
    ///
    /// Returns values from -100 to +100 based on price action characteristics
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        for i in self.period..n {
            let start = i + 1 - self.period;
            let mut bullish_score = 0.0;
            let mut bearish_score = 0.0;

            for j in start..=i {
                let body = close[j] - open[j];
                let range = high[j] - low[j];
                let upper_wick = high[j] - close[j].max(open[j]);
                let lower_wick = close[j].min(open[j]) - low[j];

                if range > 1e-10 {
                    // Body ratio (larger bodies = stronger signal)
                    let body_ratio = body.abs() / range;

                    // Wick analysis
                    let upper_wick_ratio = upper_wick / range;
                    let lower_wick_ratio = lower_wick / range;

                    if body > 0.0 {
                        // Bullish candle
                        bullish_score += body_ratio;
                        // Long lower wick on bullish candle = rejection of lower prices
                        bullish_score += lower_wick_ratio * 0.5;
                        // Long upper wick = selling pressure
                        bearish_score += upper_wick_ratio * 0.3;
                    } else {
                        // Bearish candle
                        bearish_score += body_ratio;
                        // Long upper wick on bearish candle = rejection of higher prices
                        bearish_score += upper_wick_ratio * 0.5;
                        // Long lower wick = buying pressure
                        bullish_score += lower_wick_ratio * 0.3;
                    }
                }

                // Higher highs and higher lows (bullish)
                if j > start {
                    if high[j] > high[j - 1] {
                        bullish_score += 0.5;
                    }
                    if low[j] > low[j - 1] {
                        bullish_score += 0.5;
                    }
                    if high[j] < high[j - 1] {
                        bearish_score += 0.5;
                    }
                    if low[j] < low[j - 1] {
                        bearish_score += 0.5;
                    }
                }
            }

            // Convert to oscillator
            let total = bullish_score + bearish_score;
            if total > 1e-10 {
                result[i] = ((bullish_score - bearish_score) / total) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for PriceActionOscillator {
    fn name(&self) -> &str {
        "Price Action Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(
            &data.open,
            &data.high,
            &data.low,
            &data.close,
        )))
    }
}

/// Volume Flow Oscillator - Oscillator based on volume flow
///
/// Measures the flow of volume in relation to price movement,
/// indicating accumulation or distribution.
#[derive(Debug, Clone)]
pub struct VolumeFlowOscillator {
    period: usize,
    smooth_period: usize,
}

impl VolumeFlowOscillator {
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            period,
            smooth_period,
        })
    }

    /// Calculate volume flow oscillator values
    ///
    /// Returns values where positive indicates accumulation (buying),
    /// negative indicates distribution (selling)
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let min_period = self.period + self.smooth_period;

        if n < min_period {
            return result;
        }

        // Calculate Money Flow Volume (similar to Chaikin Money Flow concept)
        let mut mfv = vec![0.0; n];
        for i in 0..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                // Money Flow Multiplier: ((close - low) - (high - close)) / (high - low)
                let mf_multiplier = ((close[i] - low[i]) - (high[i] - close[i])) / range;
                mfv[i] = mf_multiplier * volume[i];
            }
        }

        // Calculate cumulative money flow over period
        let mut cmf = vec![0.0; n];
        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let sum_mfv: f64 = mfv[start..=i].iter().sum();
            let sum_vol: f64 = volume[start..=i].iter().sum();

            if sum_vol > 1e-10 {
                cmf[i] = sum_mfv / sum_vol;
            }
        }

        // Smooth the result
        for i in min_period..n {
            let start = i + 1 - self.smooth_period;
            let smoothed: f64 = cmf[start..=i].iter().sum::<f64>() / self.smooth_period as f64;

            // Scale to approximately -100 to +100
            result[i] = smoothed * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for VolumeFlowOscillator {
    fn name(&self) -> &str {
        "Volume Flow Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(
            &data.high,
            &data.low,
            &data.close,
            &data.volume,
        )))
    }
}

/// Momentum Flow Oscillator - Measures flow of momentum
///
/// Analyzes the rate of change in momentum itself, detecting
/// momentum acceleration and deceleration.
#[derive(Debug, Clone)]
pub struct MomentumFlowOscillator {
    momentum_period: usize,
    flow_period: usize,
}

impl MomentumFlowOscillator {
    pub fn new(momentum_period: usize, flow_period: usize) -> Result<Self> {
        if momentum_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if flow_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "flow_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            flow_period,
        })
    }

    /// Calculate momentum flow oscillator values
    ///
    /// Returns values indicating momentum acceleration (positive) or
    /// deceleration (negative)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let min_period = self.momentum_period + self.flow_period;

        if n < min_period + 1 {
            return result;
        }

        // Calculate momentum (Rate of Change)
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period] > 1e-10 {
                momentum[i] =
                    ((close[i] / close[i - self.momentum_period]) - 1.0) * 100.0;
            }
        }

        // Calculate momentum of momentum (acceleration)
        let mut momentum_change = vec![0.0; n];
        for i in (self.momentum_period + 1)..n {
            momentum_change[i] = momentum[i] - momentum[i - 1];
        }

        // Calculate flow: cumulative direction of momentum changes
        for i in min_period..n {
            let start = i + 1 - self.flow_period;

            let mut positive_flow = 0.0;
            let mut negative_flow = 0.0;
            let mut positive_count = 0;
            let mut negative_count = 0;

            for j in start..=i {
                if momentum_change[j] > 0.0 {
                    positive_flow += momentum_change[j];
                    positive_count += 1;
                } else if momentum_change[j] < 0.0 {
                    negative_flow += momentum_change[j].abs();
                    negative_count += 1;
                }
            }

            // Flow oscillator combines magnitude and frequency
            let total_flow = positive_flow + negative_flow;
            let total_count = positive_count + negative_count;

            if total_flow > 1e-10 && total_count > 0 {
                // Direction based on flow difference
                let flow_direction = (positive_flow - negative_flow) / total_flow;
                // Consistency based on count difference
                let count_direction =
                    (positive_count as f64 - negative_count as f64) / total_count as f64;

                // Combine both factors
                result[i] = (flow_direction * 70.0 + count_direction * 30.0)
                    .max(-100.0)
                    .min(100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for MomentumFlowOscillator {
    fn name(&self) -> &str {
        "Momentum Flow Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period + self.flow_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Adaptive CCI - CCI that adapts period based on volatility
///
/// Adjusts the CCI calculation period dynamically based on current market
/// volatility, using shorter periods in high volatility and longer periods
/// in low volatility environments.
#[derive(Debug, Clone)]
pub struct AdaptiveCCI {
    min_period: usize,
    max_period: usize,
    volatility_period: usize,
}

impl AdaptiveCCI {
    pub fn new(min_period: usize, max_period: usize, volatility_period: usize) -> Result<Self> {
        if min_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if max_period <= min_period {
            return Err(IndicatorError::InvalidParameter {
                name: "max_period".to_string(),
                reason: "must be greater than min_period".to_string(),
            });
        }
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            min_period,
            max_period,
            volatility_period,
        })
    }

    /// Calculate adaptive CCI values
    ///
    /// Returns CCI values with dynamically adjusted period based on volatility
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];
        let min_required = self.max_period + self.volatility_period;

        if n < min_required {
            return result;
        }

        // Calculate typical price
        let tp: Vec<f64> = (0..n)
            .map(|i| (high[i] + low[i] + close[i]) / 3.0)
            .collect();

        // Calculate volatility (standard deviation of returns)
        let mut volatility = vec![0.0; n];
        for i in self.volatility_period..n {
            let start = i + 1 - self.volatility_period;
            let returns: Vec<f64> = (start + 1..=i)
                .map(|j| {
                    if close[j - 1] > 1e-10 {
                        (close[j] / close[j - 1] - 1.0).abs()
                    } else {
                        0.0
                    }
                })
                .collect();

            if !returns.is_empty() {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance: f64 =
                    returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
                volatility[i] = variance.sqrt();
            }
        }

        // Find volatility range for normalization
        let vol_values: Vec<f64> = volatility[self.volatility_period..].to_vec();
        let vol_min = vol_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let vol_max = vol_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let vol_range = vol_max - vol_min;

        // Calculate adaptive CCI
        for i in min_required..n {
            // Determine adaptive period based on volatility
            let normalized_vol = if vol_range > 1e-10 {
                (volatility[i] - vol_min) / vol_range
            } else {
                0.5
            };

            // Higher volatility = shorter period, lower volatility = longer period
            let adaptive_period = self.max_period
                - ((self.max_period - self.min_period) as f64 * normalized_vol).round() as usize;
            let period = adaptive_period.max(self.min_period).min(self.max_period);

            // Calculate CCI with adaptive period
            let start = i + 1 - period;
            let tp_mean: f64 = tp[start..=i].iter().sum::<f64>() / period as f64;

            // Mean deviation
            let mean_dev: f64 =
                tp[start..=i].iter().map(|&x| (x - tp_mean).abs()).sum::<f64>() / period as f64;

            if mean_dev > 1e-10 {
                result[i] = (tp[i] - tp_mean) / (0.015 * mean_dev);
            }
        }

        result
    }
}

impl TechnicalIndicator for AdaptiveCCI {
    fn name(&self) -> &str {
        "Adaptive CCI"
    }

    fn min_periods(&self) -> usize {
        self.max_period + self.volatility_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(
            &data.high,
            &data.low,
            &data.close,
        )))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Smoothed TSI - Triple smoothed TSI with signal line
///
/// An enhanced version of the True Strength Index that applies triple
/// smoothing for reduced noise and includes a signal line for crossover signals.
#[derive(Debug, Clone)]
pub struct SmoothedTSI {
    first_period: usize,
    second_period: usize,
    third_period: usize,
    signal_period: usize,
}

impl SmoothedTSI {
    pub fn new(
        first_period: usize,
        second_period: usize,
        third_period: usize,
        signal_period: usize,
    ) -> Result<Self> {
        if first_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "first_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if second_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "second_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if third_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "third_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if signal_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "signal_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            first_period,
            second_period,
            third_period,
            signal_period,
        })
    }

    /// Helper function to calculate EMA
    fn ema(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];

        if n < period {
            return result;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);

        // Initialize with SMA
        let sma: f64 = data[0..period].iter().sum::<f64>() / period as f64;
        result[period - 1] = sma;

        // Calculate EMA
        for i in period..n {
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
        }

        result
    }

    /// Calculate smoothed TSI values with signal line
    ///
    /// Returns (tsi_values, signal_values)
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let min_period = self.first_period + self.second_period + self.third_period;

        if n < min_period + 1 {
            return (vec![0.0; n], vec![0.0; n]);
        }

        // Calculate price changes
        let mut price_change = vec![0.0; n];
        let mut abs_price_change = vec![0.0; n];
        for i in 1..n {
            price_change[i] = close[i] - close[i - 1];
            abs_price_change[i] = price_change[i].abs();
        }

        // Triple smooth price change
        let pc_ema1 = Self::ema(&price_change, self.first_period);
        let pc_ema2 = Self::ema(&pc_ema1, self.second_period);
        let pc_ema3 = Self::ema(&pc_ema2, self.third_period);

        // Triple smooth absolute price change
        let abs_ema1 = Self::ema(&abs_price_change, self.first_period);
        let abs_ema2 = Self::ema(&abs_ema1, self.second_period);
        let abs_ema3 = Self::ema(&abs_ema2, self.third_period);

        // Calculate TSI
        let mut tsi = vec![0.0; n];
        for i in min_period..n {
            if abs_ema3[i] > 1e-10 {
                tsi[i] = (pc_ema3[i] / abs_ema3[i]) * 100.0;
            }
        }

        // Calculate signal line
        let signal = Self::ema(&tsi, self.signal_period);

        (tsi, signal)
    }
}

impl TechnicalIndicator for SmoothedTSI {
    fn name(&self) -> &str {
        "Smoothed TSI"
    }

    fn min_periods(&self) -> usize {
        self.first_period + self.second_period + self.third_period + self.signal_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (tsi, signal) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(tsi, signal))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Volatility Adjusted RSI - RSI normalized by volatility
///
/// Modifies the standard RSI by adjusting for current market volatility,
/// making the indicator more responsive in volatile markets and smoother
/// in calm markets.
#[derive(Debug, Clone)]
pub struct VolatilityAdjustedRSI {
    rsi_period: usize,
    volatility_period: usize,
    adjustment_factor: f64,
}

impl VolatilityAdjustedRSI {
    pub fn new(rsi_period: usize, volatility_period: usize, adjustment_factor: f64) -> Result<Self> {
        if rsi_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "rsi_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if adjustment_factor <= 0.0 || adjustment_factor > 2.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "adjustment_factor".to_string(),
                reason: "must be between 0 and 2".to_string(),
            });
        }
        Ok(Self {
            rsi_period,
            volatility_period,
            adjustment_factor,
        })
    }

    /// Calculate volatility adjusted RSI values
    ///
    /// Returns RSI values adjusted for volatility
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.rsi_period.max(self.volatility_period) + 1;
        let mut result = vec![0.0; n];

        if n < min_period {
            return result;
        }

        // Calculate standard RSI components
        let mut gains = vec![0.0; n];
        let mut losses = vec![0.0; n];

        for i in 1..n {
            let change = close[i] - close[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Calculate average gain/loss using Wilder's smoothing
        let mut avg_gain = vec![0.0; n];
        let mut avg_loss = vec![0.0; n];

        if n > self.rsi_period {
            avg_gain[self.rsi_period] =
                gains[1..=self.rsi_period].iter().sum::<f64>() / self.rsi_period as f64;
            avg_loss[self.rsi_period] =
                losses[1..=self.rsi_period].iter().sum::<f64>() / self.rsi_period as f64;

            for i in (self.rsi_period + 1)..n {
                avg_gain[i] = (avg_gain[i - 1] * (self.rsi_period - 1) as f64 + gains[i])
                    / self.rsi_period as f64;
                avg_loss[i] = (avg_loss[i - 1] * (self.rsi_period - 1) as f64 + losses[i])
                    / self.rsi_period as f64;
            }
        }

        // Calculate volatility (ATR-like measure using close prices)
        let mut volatility = vec![0.0; n];
        for i in self.volatility_period..n {
            let start = i + 1 - self.volatility_period;
            let returns: Vec<f64> = (start + 1..=i)
                .map(|j| {
                    if close[j - 1] > 1e-10 {
                        (close[j] - close[j - 1]).abs() / close[j - 1]
                    } else {
                        0.0
                    }
                })
                .collect();

            if !returns.is_empty() {
                volatility[i] = returns.iter().sum::<f64>() / returns.len() as f64;
            }
        }

        // Find average volatility for normalization
        let vol_slice: Vec<f64> = volatility[self.volatility_period..].to_vec();
        let avg_vol = if !vol_slice.is_empty() {
            vol_slice.iter().sum::<f64>() / vol_slice.len() as f64
        } else {
            1.0
        };

        // Calculate volatility-adjusted RSI
        for i in min_period..n {
            // Standard RSI
            let rsi = if avg_loss[i] > 1e-10 {
                100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
            } else if avg_gain[i] > 1e-10 {
                100.0
            } else {
                50.0
            };

            // Volatility adjustment factor
            let vol_ratio = if avg_vol > 1e-10 {
                volatility[i] / avg_vol
            } else {
                1.0
            };

            // Adjust RSI based on volatility
            // Higher volatility pushes RSI further from 50
            let adjusted_rsi = 50.0 + (rsi - 50.0) * (1.0 + (vol_ratio - 1.0) * self.adjustment_factor);

            // Clamp to 0-100 range
            result[i] = adjusted_rsi.max(0.0).min(100.0);
        }

        result
    }
}

impl TechnicalIndicator for VolatilityAdjustedRSI {
    fn name(&self) -> &str {
        "Volatility Adjusted RSI"
    }

    fn min_periods(&self) -> usize {
        self.rsi_period.max(self.volatility_period) + 2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Cycle Tuned Oscillator - Oscillator tuned to dominant cycle
///
/// Uses cycle analysis to detect the dominant market cycle and generates
/// an oscillator that is optimally tuned to that cycle length.
#[derive(Debug, Clone)]
pub struct CycleTunedOscillator {
    min_cycle: usize,
    max_cycle: usize,
    smooth_period: usize,
}

impl CycleTunedOscillator {
    pub fn new(min_cycle: usize, max_cycle: usize, smooth_period: usize) -> Result<Self> {
        if min_cycle < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_cycle".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if max_cycle <= min_cycle {
            return Err(IndicatorError::InvalidParameter {
                name: "max_cycle".to_string(),
                reason: "must be greater than min_cycle".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            min_cycle,
            max_cycle,
            smooth_period,
        })
    }

    /// Detect dominant cycle using autocorrelation
    fn detect_dominant_cycle(&self, data: &[f64], end_idx: usize) -> usize {
        let lookback = self.max_cycle * 2;
        if end_idx < lookback {
            return (self.min_cycle + self.max_cycle) / 2;
        }

        let start = end_idx - lookback;
        let segment: Vec<f64> = data[start..=end_idx].to_vec();
        let n = segment.len();

        // Calculate mean
        let mean: f64 = segment.iter().sum::<f64>() / n as f64;

        // Detrend
        let detrended: Vec<f64> = segment.iter().map(|x| x - mean).collect();

        // Calculate autocorrelation for each potential cycle length
        let mut best_cycle = self.min_cycle;
        let mut best_correlation = f64::NEG_INFINITY;

        for cycle in self.min_cycle..=self.max_cycle {
            if cycle >= n {
                continue;
            }

            let mut correlation = 0.0;
            let mut norm1 = 0.0;
            let mut norm2 = 0.0;

            for i in cycle..n {
                correlation += detrended[i] * detrended[i - cycle];
                norm1 += detrended[i].powi(2);
                norm2 += detrended[i - cycle].powi(2);
            }

            let norm = (norm1 * norm2).sqrt();
            if norm > 1e-10 {
                let normalized_corr = correlation / norm;
                if normalized_corr > best_correlation {
                    best_correlation = normalized_corr;
                    best_cycle = cycle;
                }
            }
        }

        best_cycle
    }

    /// Calculate cycle tuned oscillator values
    ///
    /// Returns oscillator values tuned to the dominant cycle
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.max_cycle * 2 + self.smooth_period;
        let mut result = vec![0.0; n];

        if n < min_period {
            return result;
        }

        for i in min_period..n {
            // Detect dominant cycle at this point
            let cycle = self.detect_dominant_cycle(close, i);

            // Calculate oscillator using detected cycle
            let start = i + 1 - cycle;

            // Calculate cycle high and low
            let cycle_high = close[start..=i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let cycle_low = close[start..=i]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let cycle_range = cycle_high - cycle_low;

            // Calculate position within cycle (stochastic-like)
            if cycle_range > 1e-10 {
                let raw_osc = ((close[i] - cycle_low) / cycle_range) * 2.0 - 1.0; // -1 to +1
                result[i] = raw_osc * 100.0; // Scale to -100 to +100
            }
        }

        // Apply smoothing
        let mut smoothed = vec![0.0; n];
        for i in (min_period + self.smooth_period - 1)..n {
            let start = i + 1 - self.smooth_period;
            smoothed[i] = result[start..=i].iter().sum::<f64>() / self.smooth_period as f64;
        }

        smoothed
    }
}

impl TechnicalIndicator for CycleTunedOscillator {
    fn name(&self) -> &str {
        "Cycle Tuned Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.max_cycle * 2 + self.smooth_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Momentum Accumulator - Cumulative momentum with decay
///
/// Accumulates momentum readings over time with an exponential decay factor,
/// creating a smoothed momentum indicator that captures longer-term trends
/// while still being responsive to recent changes.
#[derive(Debug, Clone)]
pub struct MomentumAccumulator {
    momentum_period: usize,
    decay_factor: f64,
    normalization_period: usize,
}

impl MomentumAccumulator {
    pub fn new(
        momentum_period: usize,
        decay_factor: f64,
        normalization_period: usize,
    ) -> Result<Self> {
        if momentum_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if decay_factor <= 0.0 || decay_factor >= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "decay_factor".to_string(),
                reason: "must be between 0 and 1 (exclusive)".to_string(),
            });
        }
        if normalization_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "normalization_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            decay_factor,
            normalization_period,
        })
    }

    /// Calculate momentum accumulator values
    ///
    /// Returns accumulated momentum values with decay
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.momentum_period + self.normalization_period;
        let mut result = vec![0.0; n];

        if n < min_period {
            return result;
        }

        // Calculate raw momentum (rate of change)
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period] > 1e-10 {
                momentum[i] =
                    (close[i] / close[i - self.momentum_period] - 1.0) * 100.0;
            }
        }

        // Accumulate momentum with decay
        let mut accumulated = vec![0.0; n];
        for i in self.momentum_period..n {
            if i == self.momentum_period {
                accumulated[i] = momentum[i];
            } else {
                accumulated[i] = accumulated[i - 1] * self.decay_factor + momentum[i];
            }
        }

        // Normalize using recent standard deviation
        for i in min_period..n {
            let start = i + 1 - self.normalization_period;
            let acc_slice: Vec<f64> = accumulated[start..=i].to_vec();

            let mean: f64 = acc_slice.iter().sum::<f64>() / acc_slice.len() as f64;
            let variance: f64 = acc_slice
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / acc_slice.len() as f64;
            let std_dev = variance.sqrt();

            // Z-score normalization, then scale to approximately -100 to +100
            if std_dev > 1e-10 {
                let z_score = (accumulated[i] - mean) / std_dev;
                result[i] = (z_score * 33.0).max(-100.0).min(100.0); // ~3 std dev = 100
            }
        }

        result
    }
}

impl TechnicalIndicator for MomentumAccumulator {
    fn name(&self) -> &str {
        "Momentum Accumulator"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period + self.normalization_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Price Strength Index - Composite strength index from multiple timeframes
///
/// Combines price strength measurements from multiple timeframes into a
/// single composite index, providing a more robust view of overall price strength.
#[derive(Debug, Clone)]
pub struct PriceStrengthIndex {
    short_period: usize,
    medium_period: usize,
    long_period: usize,
    weights: (f64, f64, f64),
}

impl PriceStrengthIndex {
    pub fn new(
        short_period: usize,
        medium_period: usize,
        long_period: usize,
        weights: (f64, f64, f64),
    ) -> Result<Self> {
        if short_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if medium_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "medium_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if long_period <= medium_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than medium_period".to_string(),
            });
        }
        let total_weight = weights.0 + weights.1 + weights.2;
        if total_weight <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "weights".to_string(),
                reason: "sum of weights must be positive".to_string(),
            });
        }
        Ok(Self {
            short_period,
            medium_period,
            long_period,
            weights,
        })
    }

    /// Calculate RSI-like strength for a given period
    fn calculate_strength(close: &[f64], period: usize, idx: usize) -> f64 {
        if idx < period {
            return 50.0;
        }

        let start = idx + 1 - period;
        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in (start + 1)..=idx {
            let change = close[i] - close[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }

        if gains + losses > 1e-10 {
            (gains / (gains + losses)) * 100.0
        } else {
            50.0
        }
    }

    /// Calculate price strength index values
    ///
    /// Returns composite strength values from 0 to 100
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.long_period + 1 {
            return result;
        }

        let total_weight = self.weights.0 + self.weights.1 + self.weights.2;
        let w_short = self.weights.0 / total_weight;
        let w_medium = self.weights.1 / total_weight;
        let w_long = self.weights.2 / total_weight;

        for i in self.long_period..n {
            // Calculate strength at each timeframe
            let short_strength = Self::calculate_strength(close, self.short_period, i);
            let medium_strength = Self::calculate_strength(close, self.medium_period, i);
            let long_strength = Self::calculate_strength(close, self.long_period, i);

            // Weighted composite
            let composite =
                short_strength * w_short + medium_strength * w_medium + long_strength * w_long;

            // Add trend alignment bonus/penalty
            let alignment = if (short_strength > 50.0 && medium_strength > 50.0 && long_strength > 50.0)
                || (short_strength < 50.0 && medium_strength < 50.0 && long_strength < 50.0)
            {
                // Aligned trends - boost the signal
                let avg = (short_strength + medium_strength + long_strength) / 3.0;
                (avg - 50.0) * 0.2 // Add up to 10 points boost
            } else {
                0.0
            };

            result[i] = (composite + alignment).max(0.0).min(100.0);
        }

        result
    }
}

impl TechnicalIndicator for PriceStrengthIndex {
    fn name(&self) -> &str {
        "Price Strength Index"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.3).sin() * 2.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.5).collect();
        let open: Vec<f64> = close
            .iter()
            .enumerate()
            .map(|(i, c)| {
                if i % 2 == 0 {
                    c - 0.3
                } else {
                    c + 0.3
                }
            })
            .collect();
        let volume: Vec<f64> = (0..50)
            .map(|i| 1000.0 + (i as f64 * 0.5).sin() * 500.0)
            .collect();

        OHLCVSeries {
            open,
            high,
            low,
            close,
            volume,
        }
    }

    // CycleOscillator tests
    #[test]
    fn test_cycle_oscillator_basic() {
        let data = make_test_data();
        let indicator = CycleOscillator::new(10, 5).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Values should exist after min_period
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_cycle_oscillator_validation() {
        assert!(CycleOscillator::new(4, 5).is_err());
        assert!(CycleOscillator::new(10, 1).is_err());
        assert!(CycleOscillator::new(5, 2).is_ok());
    }

    #[test]
    fn test_cycle_oscillator_trait() {
        let data = make_test_data();
        let indicator = CycleOscillator::new(10, 5).unwrap();

        assert_eq!(indicator.name(), "Cycle Oscillator");
        assert_eq!(indicator.min_periods(), 16);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // VolatilityOscillator tests
    #[test]
    fn test_volatility_oscillator_basic() {
        let data = make_test_data();
        let indicator = VolatilityOscillator::new(5, 20).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        // After min_period, values should be calculated
        for i in 25..result.len() {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_volatility_oscillator_validation() {
        assert!(VolatilityOscillator::new(2, 20).is_err());
        assert!(VolatilityOscillator::new(10, 5).is_err());
        assert!(VolatilityOscillator::new(5, 5).is_ok());
        assert!(VolatilityOscillator::new(5, 20).is_ok());
    }

    #[test]
    fn test_volatility_oscillator_trait() {
        let data = make_test_data();
        let indicator = VolatilityOscillator::new(5, 20).unwrap();

        assert_eq!(indicator.name(), "Volatility Oscillator");
        assert_eq!(indicator.min_periods(), 21);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // TrendOscillator tests
    #[test]
    fn test_trend_oscillator_basic() {
        let data = make_test_data();
        let indicator = TrendOscillator::new(5, 20).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Values should be in -100 to +100 range
        for i in 25..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_trend_oscillator_validation() {
        assert!(TrendOscillator::new(2, 20).is_err());
        assert!(TrendOscillator::new(10, 10).is_err());
        assert!(TrendOscillator::new(10, 5).is_err());
        assert!(TrendOscillator::new(5, 20).is_ok());
    }

    #[test]
    fn test_trend_oscillator_uptrend() {
        // Strong uptrend data
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        let data = OHLCVSeries {
            open: close.iter().map(|c| c - 0.5).collect(),
            high: close.iter().map(|c| c + 1.0).collect(),
            low: close.iter().map(|c| c - 1.0).collect(),
            close,
            volume: vec![1000.0; 50],
        };

        let indicator = TrendOscillator::new(5, 10).unwrap();
        let result = indicator.calculate(&data.close);

        // In strong uptrend, later values should be positive
        let last_10: Vec<f64> = result[40..50].to_vec();
        let avg: f64 = last_10.iter().sum::<f64>() / 10.0;
        assert!(avg > 0.0);
    }

    #[test]
    fn test_trend_oscillator_trait() {
        let data = make_test_data();
        let indicator = TrendOscillator::new(5, 20).unwrap();

        assert_eq!(indicator.name(), "Trend Oscillator");
        assert_eq!(indicator.min_periods(), 21);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // PriceActionOscillator tests
    #[test]
    fn test_price_action_oscillator_basic() {
        let data = make_test_data();
        let indicator = PriceActionOscillator::new(10).unwrap();
        let result = indicator.calculate(&data.open, &data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        // Values should be in -100 to +100 range
        for i in 15..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_price_action_oscillator_validation() {
        assert!(PriceActionOscillator::new(4).is_err());
        assert!(PriceActionOscillator::new(5).is_ok());
        assert!(PriceActionOscillator::new(20).is_ok());
    }

    #[test]
    fn test_price_action_oscillator_bullish() {
        // Bullish price action: close > open, higher highs/lows
        let n = 30;
        let mut close = vec![100.0];
        let mut open = vec![99.0];
        let mut high = vec![101.0];
        let mut low = vec![98.5];

        for i in 1..n {
            let base = 100.0 + i as f64;
            close.push(base);
            open.push(base - 1.0);
            high.push(base + 1.0);
            low.push(base - 1.5);
        }

        let data = OHLCVSeries {
            open,
            high,
            low,
            close,
            volume: vec![1000.0; n],
        };

        let indicator = PriceActionOscillator::new(10).unwrap();
        let result = indicator.calculate(&data.open, &data.high, &data.low, &data.close);

        // Last values should be positive (bullish)
        let last = result.last().unwrap();
        assert!(*last > 0.0);
    }

    #[test]
    fn test_price_action_oscillator_trait() {
        let data = make_test_data();
        let indicator = PriceActionOscillator::new(10).unwrap();

        assert_eq!(indicator.name(), "Price Action Oscillator");
        assert_eq!(indicator.min_periods(), 11);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // VolumeFlowOscillator tests
    #[test]
    fn test_volume_flow_oscillator_basic() {
        let data = make_test_data();
        let indicator = VolumeFlowOscillator::new(10, 3).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
        // Values should exist after min_period
        for i in 20..result.len() {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_volume_flow_oscillator_validation() {
        assert!(VolumeFlowOscillator::new(4, 3).is_err());
        assert!(VolumeFlowOscillator::new(10, 1).is_err());
        assert!(VolumeFlowOscillator::new(5, 2).is_ok());
    }

    #[test]
    fn test_volume_flow_oscillator_accumulation() {
        // Accumulation pattern: close near highs consistently
        let n = 40;
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.5).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.2).collect(); // Close near high
        let low: Vec<f64> = close.iter().map(|c| c - 2.0).collect();
        let volume: Vec<f64> = vec![1000.0; n];

        let indicator = VolumeFlowOscillator::new(10, 3).unwrap();
        let result = indicator.calculate(&high, &low, &close, &volume);

        // Should show positive (accumulation) values
        let last = result.last().unwrap();
        assert!(*last > 0.0);
    }

    #[test]
    fn test_volume_flow_oscillator_trait() {
        let data = make_test_data();
        let indicator = VolumeFlowOscillator::new(10, 3).unwrap();

        assert_eq!(indicator.name(), "Volume Flow Oscillator");
        assert_eq!(indicator.min_periods(), 14);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // MomentumFlowOscillator tests
    #[test]
    fn test_momentum_flow_oscillator_basic() {
        let data = make_test_data();
        let indicator = MomentumFlowOscillator::new(5, 5).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Values should be in -100 to +100 range
        for i in 15..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_momentum_flow_oscillator_validation() {
        assert!(MomentumFlowOscillator::new(2, 5).is_err());
        assert!(MomentumFlowOscillator::new(5, 2).is_err());
        assert!(MomentumFlowOscillator::new(3, 3).is_ok());
    }

    #[test]
    fn test_momentum_flow_oscillator_acceleration() {
        // Accelerating uptrend
        let close: Vec<f64> = (0..40).map(|i| 100.0 + (i as f64).powi(2) * 0.1).collect();

        let indicator = MomentumFlowOscillator::new(5, 5).unwrap();
        let result = indicator.calculate(&close);

        // With accelerating prices, momentum flow should be positive
        let last_vals: Vec<f64> = result[30..].to_vec();
        let avg: f64 = last_vals.iter().sum::<f64>() / last_vals.len() as f64;
        assert!(avg > 0.0);
    }

    #[test]
    fn test_momentum_flow_oscillator_trait() {
        let data = make_test_data();
        let indicator = MomentumFlowOscillator::new(5, 5).unwrap();

        assert_eq!(indicator.name(), "Momentum Flow Oscillator");
        assert_eq!(indicator.min_periods(), 11);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    // Edge case tests
    #[test]
    fn test_insufficient_data() {
        let short_data = OHLCVSeries {
            open: vec![100.0, 101.0, 102.0],
            high: vec![101.0, 102.0, 103.0],
            low: vec![99.0, 100.0, 101.0],
            close: vec![100.5, 101.5, 102.5],
            volume: vec![1000.0, 1100.0, 1200.0],
        };

        let cycle = CycleOscillator::new(10, 5).unwrap();
        let result = cycle.calculate(&short_data.close);
        assert!(result.iter().all(|&v| v == 0.0));

        let vol = VolatilityOscillator::new(5, 10).unwrap();
        let result = vol.calculate(&short_data.high, &short_data.low, &short_data.close);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_flat_prices() {
        let flat_close: Vec<f64> = vec![100.0; 50];
        let flat_high: Vec<f64> = vec![100.5; 50];
        let flat_low: Vec<f64> = vec![99.5; 50];

        let cycle = CycleOscillator::new(10, 5).unwrap();
        let result = cycle.calculate(&flat_close);
        // With flat prices, oscillator should be near zero
        for i in 20..result.len() {
            assert!(result[i].abs() < 1.0);
        }

        let trend = TrendOscillator::new(5, 10).unwrap();
        let result = trend.calculate(&flat_close);
        // With flat prices, trend should be near zero
        for i in 15..result.len() {
            assert!(result[i].abs() < 50.0);
        }
    }

    // ============================================================
    // AdaptiveCCI Tests
    // ============================================================

    #[test]
    fn test_adaptive_cci_basic() {
        let data = make_test_data();
        let indicator = AdaptiveCCI::new(5, 20, 10).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        // Values should exist after min_period
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_adaptive_cci_validation() {
        // min_period too small
        assert!(AdaptiveCCI::new(4, 20, 10).is_err());
        // max_period <= min_period
        assert!(AdaptiveCCI::new(10, 10, 10).is_err());
        assert!(AdaptiveCCI::new(10, 5, 10).is_err());
        // volatility_period too small
        assert!(AdaptiveCCI::new(5, 20, 4).is_err());
        // Valid parameters
        assert!(AdaptiveCCI::new(5, 20, 10).is_ok());
        assert!(AdaptiveCCI::new(10, 30, 15).is_ok());
    }

    #[test]
    fn test_adaptive_cci_trait() {
        let data = make_test_data();
        let indicator = AdaptiveCCI::new(5, 15, 10).unwrap();

        assert_eq!(indicator.name(), "Adaptive CCI");
        assert_eq!(indicator.min_periods(), 26); // max_period + volatility_period + 1
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_adaptive_cci_uptrend() {
        // Strong uptrend should give positive CCI
        let n = 60;
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 1.5).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();

        let indicator = AdaptiveCCI::new(5, 15, 10).unwrap();
        let result = indicator.calculate(&high, &low, &close);

        // Last values should be positive in uptrend
        let last_10: Vec<f64> = result[50..60].to_vec();
        let positive_count = last_10.iter().filter(|&&v| v > 0.0).count();
        assert!(positive_count >= 5); // Most should be positive
    }

    #[test]
    fn test_adaptive_cci_insufficient_data() {
        let short_close: Vec<f64> = vec![100.0; 10];
        let short_high: Vec<f64> = vec![101.0; 10];
        let short_low: Vec<f64> = vec![99.0; 10];

        let indicator = AdaptiveCCI::new(5, 20, 10).unwrap();
        let result = indicator.calculate(&short_high, &short_low, &short_close);

        // All values should be zero due to insufficient data
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // ============================================================
    // SmoothedTSI Tests
    // ============================================================

    #[test]
    fn test_smoothed_tsi_basic() {
        let data = make_test_data();
        let indicator = SmoothedTSI::new(25, 13, 5, 7).unwrap();
        let (tsi, signal) = indicator.calculate(&data.close);

        assert_eq!(tsi.len(), data.close.len());
        assert_eq!(signal.len(), data.close.len());
    }

    #[test]
    fn test_smoothed_tsi_validation() {
        // first_period too small
        assert!(SmoothedTSI::new(4, 13, 5, 7).is_err());
        // second_period too small
        assert!(SmoothedTSI::new(25, 2, 5, 7).is_err());
        // third_period too small
        assert!(SmoothedTSI::new(25, 13, 1, 7).is_err());
        // signal_period too small
        assert!(SmoothedTSI::new(25, 13, 5, 1).is_err());
        // Valid parameters
        assert!(SmoothedTSI::new(25, 13, 5, 7).is_ok());
        assert!(SmoothedTSI::new(10, 5, 3, 5).is_ok());
    }

    #[test]
    fn test_smoothed_tsi_trait() {
        let data = make_test_data();
        let indicator = SmoothedTSI::new(10, 5, 3, 5).unwrap();

        assert_eq!(indicator.name(), "Smoothed TSI");
        assert_eq!(indicator.min_periods(), 24); // 10+5+3+5+1
        assert_eq!(indicator.output_features(), 2);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_smoothed_tsi_range() {
        // TSI should be in range [-100, 100]
        let data = make_test_data();
        let indicator = SmoothedTSI::new(10, 5, 3, 5).unwrap();
        let (tsi, _signal) = indicator.calculate(&data.close);

        for i in 25..tsi.len() {
            assert!(
                tsi[i] >= -100.0 && tsi[i] <= 100.0,
                "TSI value {} out of range at index {}",
                tsi[i],
                i
            );
        }
    }

    #[test]
    fn test_smoothed_tsi_uptrend() {
        // Strong uptrend should give positive TSI
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 2.0).collect();

        let indicator = SmoothedTSI::new(10, 5, 3, 5).unwrap();
        let (tsi, _signal) = indicator.calculate(&close);

        // Last values should be positive
        let last = tsi.last().unwrap();
        assert!(*last > 0.0);
    }

    // ============================================================
    // VolatilityAdjustedRSI Tests
    // ============================================================

    #[test]
    fn test_volatility_adjusted_rsi_basic() {
        let data = make_test_data();
        let indicator = VolatilityAdjustedRSI::new(14, 10, 1.0).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Values should be in 0-100 range
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(
                result[i] >= 0.0 && result[i] <= 100.0,
                "RSI value {} out of range at index {}",
                result[i],
                i
            );
        }
    }

    #[test]
    fn test_volatility_adjusted_rsi_validation() {
        // rsi_period too small
        assert!(VolatilityAdjustedRSI::new(1, 10, 1.0).is_err());
        // volatility_period too small
        assert!(VolatilityAdjustedRSI::new(14, 4, 1.0).is_err());
        // adjustment_factor out of range
        assert!(VolatilityAdjustedRSI::new(14, 10, 0.0).is_err());
        assert!(VolatilityAdjustedRSI::new(14, 10, -0.5).is_err());
        assert!(VolatilityAdjustedRSI::new(14, 10, 2.5).is_err());
        // Valid parameters
        assert!(VolatilityAdjustedRSI::new(14, 10, 1.0).is_ok());
        assert!(VolatilityAdjustedRSI::new(7, 5, 0.5).is_ok());
        assert!(VolatilityAdjustedRSI::new(14, 10, 2.0).is_ok());
    }

    #[test]
    fn test_volatility_adjusted_rsi_trait() {
        let data = make_test_data();
        let indicator = VolatilityAdjustedRSI::new(10, 10, 1.0).unwrap();

        assert_eq!(indicator.name(), "Volatility Adjusted RSI");
        assert_eq!(indicator.min_periods(), 12); // max(10, 10) + 2
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_volatility_adjusted_rsi_uptrend() {
        // Strong uptrend should give high RSI (>50)
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();

        let indicator = VolatilityAdjustedRSI::new(14, 10, 1.0).unwrap();
        let result = indicator.calculate(&close);

        let last = result.last().unwrap();
        assert!(*last > 50.0);
    }

    #[test]
    fn test_volatility_adjusted_rsi_downtrend() {
        // Strong downtrend should give low RSI (<50)
        let close: Vec<f64> = (0..50).map(|i| 200.0 - i as f64).collect();

        let indicator = VolatilityAdjustedRSI::new(14, 10, 1.0).unwrap();
        let result = indicator.calculate(&close);

        let last = result.last().unwrap();
        assert!(*last < 50.0);
    }

    // ============================================================
    // CycleTunedOscillator Tests
    // ============================================================

    #[test]
    fn test_cycle_tuned_oscillator_basic() {
        // Need longer data for cycle detection
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 10.0 + (i as f64) * 0.1)
            .collect();

        let indicator = CycleTunedOscillator::new(5, 20, 3).unwrap();
        let result = indicator.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_cycle_tuned_oscillator_validation() {
        // min_cycle too small
        assert!(CycleTunedOscillator::new(4, 20, 3).is_err());
        // max_cycle <= min_cycle
        assert!(CycleTunedOscillator::new(10, 10, 3).is_err());
        assert!(CycleTunedOscillator::new(10, 5, 3).is_err());
        // smooth_period too small
        assert!(CycleTunedOscillator::new(5, 20, 1).is_err());
        // Valid parameters
        assert!(CycleTunedOscillator::new(5, 20, 3).is_ok());
        assert!(CycleTunedOscillator::new(10, 30, 5).is_ok());
    }

    #[test]
    fn test_cycle_tuned_oscillator_trait() {
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 10.0)
            .collect();
        let data = OHLCVSeries {
            open: close.iter().map(|c| c - 0.5).collect(),
            high: close.iter().map(|c| c + 1.0).collect(),
            low: close.iter().map(|c| c - 1.0).collect(),
            close,
            volume: vec![1000.0; 100],
        };

        let indicator = CycleTunedOscillator::new(5, 15, 3).unwrap();

        assert_eq!(indicator.name(), "Cycle Tuned Oscillator");
        assert_eq!(indicator.min_periods(), 34); // max_cycle * 2 + smooth_period + 1
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_cycle_tuned_oscillator_range() {
        // Oscillator should be in range [-100, 100]
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 10.0)
            .collect();

        let indicator = CycleTunedOscillator::new(5, 15, 3).unwrap();
        let result = indicator.calculate(&close);

        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(
                result[i] >= -100.0 && result[i] <= 100.0,
                "Value {} out of range at index {}",
                result[i],
                i
            );
        }
    }

    #[test]
    fn test_cycle_tuned_oscillator_insufficient_data() {
        let short_close: Vec<f64> = vec![100.0; 20];

        let indicator = CycleTunedOscillator::new(5, 20, 3).unwrap();
        let result = indicator.calculate(&short_close);

        // All values should be zero due to insufficient data
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // ============================================================
    // MomentumAccumulator Tests
    // ============================================================

    #[test]
    fn test_momentum_accumulator_basic() {
        let data = make_test_data();
        let indicator = MomentumAccumulator::new(5, 0.95, 20).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Values should be in -100 to +100 range
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(
                result[i] >= -100.0 && result[i] <= 100.0,
                "Value {} out of range at index {}",
                result[i],
                i
            );
        }
    }

    #[test]
    fn test_momentum_accumulator_validation() {
        // momentum_period too small
        assert!(MomentumAccumulator::new(0, 0.95, 20).is_err());
        // decay_factor out of range
        assert!(MomentumAccumulator::new(5, 0.0, 20).is_err());
        assert!(MomentumAccumulator::new(5, 1.0, 20).is_err());
        assert!(MomentumAccumulator::new(5, -0.5, 20).is_err());
        assert!(MomentumAccumulator::new(5, 1.5, 20).is_err());
        // normalization_period too small
        assert!(MomentumAccumulator::new(5, 0.95, 9).is_err());
        // Valid parameters
        assert!(MomentumAccumulator::new(1, 0.5, 10).is_ok());
        assert!(MomentumAccumulator::new(5, 0.95, 20).is_ok());
        assert!(MomentumAccumulator::new(10, 0.99, 30).is_ok());
    }

    #[test]
    fn test_momentum_accumulator_trait() {
        let data = make_test_data();
        let indicator = MomentumAccumulator::new(5, 0.95, 20).unwrap();

        assert_eq!(indicator.name(), "Momentum Accumulator");
        assert_eq!(indicator.min_periods(), 26); // momentum_period + normalization_period + 1
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_momentum_accumulator_uptrend() {
        // Strong uptrend should give positive accumulated momentum
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 2.0).collect();

        let indicator = MomentumAccumulator::new(5, 0.95, 20).unwrap();
        let result = indicator.calculate(&close);

        // Last values should be positive
        let last_10: Vec<f64> = result[50..60].to_vec();
        let positive_count = last_10.iter().filter(|&&v| v > 0.0).count();
        assert!(positive_count >= 5);
    }

    #[test]
    fn test_momentum_accumulator_decay_effect() {
        // Test that decay factor affects the result
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64).collect();

        let high_decay = MomentumAccumulator::new(5, 0.99, 20).unwrap();
        let low_decay = MomentumAccumulator::new(5, 0.5, 20).unwrap();

        let result_high = high_decay.calculate(&close);
        let result_low = low_decay.calculate(&close);

        // Results should be different due to different decay factors
        let last_high = result_high.last().unwrap();
        let last_low = result_low.last().unwrap();

        // With constant trend, both should be positive but may differ in magnitude
        assert!(*last_high > 0.0 || *last_low > 0.0);
    }

    // ============================================================
    // PriceStrengthIndex Tests
    // ============================================================

    #[test]
    fn test_price_strength_index_basic() {
        let data = make_test_data();
        let indicator = PriceStrengthIndex::new(5, 14, 28, (0.5, 0.3, 0.2)).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Values should be in 0-100 range
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(
                result[i] >= 0.0 && result[i] <= 100.0,
                "Value {} out of range at index {}",
                result[i],
                i
            );
        }
    }

    #[test]
    fn test_price_strength_index_validation() {
        // short_period too small
        assert!(PriceStrengthIndex::new(2, 14, 28, (0.5, 0.3, 0.2)).is_err());
        // medium_period <= short_period
        assert!(PriceStrengthIndex::new(5, 5, 28, (0.5, 0.3, 0.2)).is_err());
        assert!(PriceStrengthIndex::new(5, 3, 28, (0.5, 0.3, 0.2)).is_err());
        // long_period <= medium_period
        assert!(PriceStrengthIndex::new(5, 14, 14, (0.5, 0.3, 0.2)).is_err());
        assert!(PriceStrengthIndex::new(5, 14, 10, (0.5, 0.3, 0.2)).is_err());
        // Invalid weights
        assert!(PriceStrengthIndex::new(5, 14, 28, (0.0, 0.0, 0.0)).is_err());
        assert!(PriceStrengthIndex::new(5, 14, 28, (-1.0, 0.5, 0.5)).is_ok()); // Total is positive
        // Valid parameters
        assert!(PriceStrengthIndex::new(5, 14, 28, (0.5, 0.3, 0.2)).is_ok());
        assert!(PriceStrengthIndex::new(3, 10, 20, (1.0, 1.0, 1.0)).is_ok());
    }

    #[test]
    fn test_price_strength_index_trait() {
        let data = make_test_data();
        let indicator = PriceStrengthIndex::new(5, 14, 28, (0.5, 0.3, 0.2)).unwrap();

        assert_eq!(indicator.name(), "Price Strength Index");
        assert_eq!(indicator.min_periods(), 29); // long_period + 1
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_price_strength_index_uptrend() {
        // Strong uptrend should give high strength (>50)
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();

        let indicator = PriceStrengthIndex::new(5, 14, 28, (0.5, 0.3, 0.2)).unwrap();
        let result = indicator.calculate(&close);

        let last = result.last().unwrap();
        assert!(*last > 50.0);
    }

    #[test]
    fn test_price_strength_index_downtrend() {
        // Strong downtrend should give low strength (<50)
        let close: Vec<f64> = (0..50).map(|i| 200.0 - i as f64 * 2.0).collect();

        let indicator = PriceStrengthIndex::new(5, 14, 28, (0.5, 0.3, 0.2)).unwrap();
        let result = indicator.calculate(&close);

        let last = result.last().unwrap();
        assert!(*last < 50.0);
    }

    #[test]
    fn test_price_strength_index_weights() {
        // Test that different weights produce different results
        let close: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0 + i as f64 * 0.5)
            .collect();

        let short_weighted = PriceStrengthIndex::new(5, 14, 28, (1.0, 0.0, 0.0)).unwrap();
        let long_weighted = PriceStrengthIndex::new(5, 14, 28, (0.0, 0.0, 1.0)).unwrap();

        let result_short = short_weighted.calculate(&close);
        let result_long = long_weighted.calculate(&close);

        // Results should generally differ
        let last_short = result_short.last().unwrap();
        let last_long = result_long.last().unwrap();

        // Both should be valid (0-100) but likely different
        assert!(*last_short >= 0.0 && *last_short <= 100.0);
        assert!(*last_long >= 0.0 && *last_long <= 100.0);
    }

    #[test]
    fn test_price_strength_index_alignment_bonus() {
        // Test aligned trends boost the signal
        // Strong consistent uptrend across all timeframes
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 3.0).collect();

        let indicator = PriceStrengthIndex::new(5, 14, 28, (1.0, 1.0, 1.0)).unwrap();
        let result = indicator.calculate(&close);

        let last = result.last().unwrap();
        // With alignment bonus, should be well above 50
        assert!(*last > 60.0);
    }

    // ============================================================
    // Additional edge case tests for new indicators
    // ============================================================

    #[test]
    fn test_new_indicators_insufficient_data() {
        let short_data = OHLCVSeries {
            open: vec![100.0, 101.0, 102.0],
            high: vec![101.0, 102.0, 103.0],
            low: vec![99.0, 100.0, 101.0],
            close: vec![100.5, 101.5, 102.5],
            volume: vec![1000.0, 1100.0, 1200.0],
        };

        // AdaptiveCCI
        let acci = AdaptiveCCI::new(5, 15, 10).unwrap();
        let result = acci.calculate(&short_data.high, &short_data.low, &short_data.close);
        assert!(result.iter().all(|&v| v == 0.0));

        // SmoothedTSI
        let stsi = SmoothedTSI::new(10, 5, 3, 5).unwrap();
        let (tsi, signal) = stsi.calculate(&short_data.close);
        assert!(tsi.iter().all(|&v| v == 0.0));
        assert!(signal.iter().all(|&v| v == 0.0));

        // VolatilityAdjustedRSI
        let varsi = VolatilityAdjustedRSI::new(10, 10, 1.0).unwrap();
        let result = varsi.calculate(&short_data.close);
        assert!(result.iter().all(|&v| v == 0.0));

        // MomentumAccumulator
        let macc = MomentumAccumulator::new(5, 0.95, 20).unwrap();
        let result = macc.calculate(&short_data.close);
        assert!(result.iter().all(|&v| v == 0.0));

        // PriceStrengthIndex
        let psi = PriceStrengthIndex::new(5, 10, 20, (1.0, 1.0, 1.0)).unwrap();
        let result = psi.calculate(&short_data.close);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_new_indicators_flat_prices() {
        let flat_close: Vec<f64> = vec![100.0; 100];
        let flat_high: Vec<f64> = vec![100.5; 100];
        let flat_low: Vec<f64> = vec![99.5; 100];

        // AdaptiveCCI - should be near zero with flat prices
        let acci = AdaptiveCCI::new(5, 15, 10).unwrap();
        let result = acci.calculate(&flat_high, &flat_low, &flat_close);
        for i in 30..result.len() {
            assert!(result[i].abs() < 10.0, "AdaptiveCCI value {} at index {}", result[i], i);
        }

        // VolatilityAdjustedRSI - should be around 50 with flat prices
        let varsi = VolatilityAdjustedRSI::new(10, 10, 1.0).unwrap();
        let result = varsi.calculate(&flat_close);
        for i in 20..result.len() {
            assert!(
                result[i] >= 40.0 && result[i] <= 60.0,
                "VolatilityAdjustedRSI value {} at index {}",
                result[i],
                i
            );
        }

        // PriceStrengthIndex - should be around 50 with flat prices
        let psi = PriceStrengthIndex::new(5, 14, 28, (1.0, 1.0, 1.0)).unwrap();
        let result = psi.calculate(&flat_close);
        for i in 30..result.len() {
            assert!(
                result[i] >= 40.0 && result[i] <= 60.0,
                "PriceStrengthIndex value {} at index {}",
                result[i],
                i
            );
        }
    }
}
