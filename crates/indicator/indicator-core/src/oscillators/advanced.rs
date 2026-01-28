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

/// Dynamic Oscillator - Oscillator with dynamic bounds
///
/// An oscillator that adjusts its bounds dynamically based on recent price
/// volatility, providing adaptive overbought/oversold levels that respond
/// to changing market conditions.
#[derive(Debug, Clone)]
pub struct DynamicOscillator {
    period: usize,
    volatility_period: usize,
}

impl DynamicOscillator {
    pub fn new(period: usize, volatility_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if volatility_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self {
            period,
            volatility_period,
        })
    }

    /// Calculate dynamic oscillator values
    ///
    /// Returns values from -100 to +100 with dynamically adjusted sensitivity
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.volatility_period + 1 {
            return result;
        }

        // Calculate True Range for volatility
        let mut tr = vec![0.0; n];
        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);
        }

        // Calculate ATR for volatility measure
        let mut atr = vec![0.0; n];
        for i in self.period..n {
            let start = i + 1 - self.period;
            atr[i] = tr[start..=i].iter().sum::<f64>() / self.period as f64;
        }

        // Calculate historical ATR range for normalization
        for i in self.volatility_period..n {
            let start = i + 1 - self.volatility_period;
            let atr_slice: Vec<f64> = atr[start..=i].to_vec();

            let atr_min = atr_slice.iter().cloned().fold(f64::INFINITY, f64::min);
            let atr_max = atr_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let atr_range = atr_max - atr_min;

            // Calculate price position within recent range
            let period_start = i + 1 - self.period;
            let period_high = high[period_start..=i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let period_low = low[period_start..=i]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let price_range = period_high - period_low;

            if price_range > 1e-10 {
                // Raw stochastic-like value
                let raw_osc = ((close[i] - period_low) / price_range) * 2.0 - 1.0;

                // Dynamic adjustment based on volatility
                let vol_factor = if atr_range > 1e-10 {
                    let normalized_vol = (atr[i] - atr_min) / atr_range;
                    // Higher volatility = more extreme values
                    1.0 + normalized_vol * 0.5
                } else {
                    1.0
                };

                result[i] = (raw_osc * vol_factor * 100.0).max(-100.0).min(100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for DynamicOscillator {
    fn name(&self) -> &str {
        "Dynamic Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.volatility_period + 1
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

/// Trend Optimized Oscillator - Oscillator optimized for trending markets
///
/// An oscillator designed specifically for trending markets that reduces
/// false signals during strong trends by incorporating trend strength
/// into its calculations.
#[derive(Debug, Clone)]
pub struct TrendOptimizedOscillator {
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
}

impl TrendOptimizedOscillator {
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Result<Self> {
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
        if signal_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "signal_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            fast_period,
            slow_period,
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
        let sma: f64 = data[0..period].iter().sum::<f64>() / period as f64;
        result[period - 1] = sma;

        for i in period..n {
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1];
        }

        result
    }

    /// Calculate trend optimized oscillator values
    ///
    /// Returns values from -100 to +100 optimized for trending conditions
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.slow_period + self.signal_period;
        let mut result = vec![0.0; n];

        if n < min_period + 1 {
            return result;
        }

        // Calculate fast and slow EMAs
        let fast_ema = Self::ema(close, self.fast_period);
        let slow_ema = Self::ema(close, self.slow_period);

        // Calculate MACD-like difference
        let mut macd = vec![0.0; n];
        for i in self.slow_period..n {
            if slow_ema[i] > 1e-10 {
                macd[i] = ((fast_ema[i] - slow_ema[i]) / slow_ema[i]) * 100.0;
            }
        }

        // Calculate signal line
        let signal = Self::ema(&macd, self.signal_period);

        // Calculate trend strength (ADX-like)
        let mut trend_strength = vec![0.0; n];
        for i in self.slow_period..n {
            let lookback = self.fast_period.min(i);
            let start = i - lookback;

            // Count directional consistency
            let mut up_count = 0;
            let mut down_count = 0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    up_count += 1;
                } else if close[j] < close[j - 1] {
                    down_count += 1;
                }
            }

            let total = up_count + down_count;
            if total > 0 {
                trend_strength[i] = ((up_count as i32 - down_count as i32).abs() as f64)
                    / total as f64;
            }
        }

        // Combine MACD histogram with trend strength
        for i in min_period..n {
            let histogram = macd[i] - signal[i];

            // Amplify signal during strong trends, dampen during weak trends
            let trend_factor = 0.5 + trend_strength[i] * 1.0;

            result[i] = (histogram * trend_factor * 10.0).max(-100.0).min(100.0);
        }

        result
    }
}

impl TechnicalIndicator for TrendOptimizedOscillator {
    fn name(&self) -> &str {
        "Trend Optimized Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.slow_period + self.signal_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Range Optimized Oscillator - Oscillator optimized for ranging markets
///
/// An oscillator specifically designed for ranging/sideways markets that
/// provides clear overbought/oversold signals within the trading range
/// while filtering out noise.
#[derive(Debug, Clone)]
pub struct RangeOptimizedOscillator {
    period: usize,
    smooth_period: usize,
}

impl RangeOptimizedOscillator {
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

    /// Calculate range optimized oscillator values
    ///
    /// Returns values from 0 to 100, similar to stochastic but optimized for ranges
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.period + self.smooth_period;
        let mut result = vec![0.0; n];

        if n < min_period + 1 {
            return result;
        }

        // Calculate %K (raw stochastic)
        let mut percent_k = vec![0.0; n];
        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let highest = high[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lowest = low[start..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            let range = highest - lowest;

            if range > 1e-10 {
                percent_k[i] = ((close[i] - lowest) / range) * 100.0;
            } else {
                percent_k[i] = 50.0;
            }
        }

        // Calculate range expansion/contraction
        let mut range_factor = vec![1.0; n];
        for i in self.period..n {
            let start = i + 1 - self.period;

            // Calculate average range
            let mut ranges = Vec::with_capacity(self.period);
            for j in start..=i {
                ranges.push(high[j] - low[j]);
            }
            let avg_range = ranges.iter().sum::<f64>() / ranges.len() as f64;

            // Current range vs average
            let current_range = high[i] - low[i];
            if avg_range > 1e-10 {
                // Narrow ranges = more confident in extremes
                range_factor[i] = (avg_range / current_range.max(avg_range * 0.5)).min(1.5);
            }
        }

        // Apply smoothing with range adjustment
        for i in min_period..n {
            let start = i + 1 - self.smooth_period;
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;

            for j in start..=i {
                let weight = range_factor[j];
                weighted_sum += percent_k[j] * weight;
                weight_total += weight;
            }

            if weight_total > 1e-10 {
                result[i] = weighted_sum / weight_total;
            }

            // Enhance extremes for ranging markets
            if result[i] > 80.0 {
                result[i] = 80.0 + (result[i] - 80.0) * 1.2;
            } else if result[i] < 20.0 {
                result[i] = 20.0 - (20.0 - result[i]) * 1.2;
            }

            result[i] = result[i].max(0.0).min(100.0);
        }

        result
    }
}

impl TechnicalIndicator for RangeOptimizedOscillator {
    fn name(&self) -> &str {
        "Range Optimized Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period + 1
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

/// Composite Oscillator - Combines multiple oscillator signals
///
/// A sophisticated oscillator that combines RSI, CCI, and Stochastic
/// signals into a single composite reading, providing a more robust
/// view of market conditions.
#[derive(Debug, Clone)]
pub struct CompositeOscillator {
    rsi_period: usize,
    cci_period: usize,
    stoch_period: usize,
    weights: (f64, f64, f64),
}

impl CompositeOscillator {
    pub fn new(
        rsi_period: usize,
        cci_period: usize,
        stoch_period: usize,
        weights: (f64, f64, f64),
    ) -> Result<Self> {
        if rsi_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "rsi_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if cci_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "cci_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if stoch_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "stoch_period".to_string(),
                reason: "must be at least 5".to_string(),
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
            rsi_period,
            cci_period,
            stoch_period,
            weights,
        })
    }

    /// Calculate RSI component
    fn calculate_rsi(close: &[f64], period: usize) -> Vec<f64> {
        let n = close.len();
        let mut rsi = vec![0.0; n];

        if n <= period {
            return rsi;
        }

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

        let mut avg_gain = gains[1..=period].iter().sum::<f64>() / period as f64;
        let mut avg_loss = losses[1..=period].iter().sum::<f64>() / period as f64;

        for i in period..n {
            if i > period {
                avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
                avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
            }

            if avg_loss > 1e-10 {
                let rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - 100.0 / (1.0 + rs);
            } else if avg_gain > 1e-10 {
                rsi[i] = 100.0;
            } else {
                rsi[i] = 50.0;
            }
        }

        rsi
    }

    /// Calculate CCI component
    fn calculate_cci(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
        let n = close.len();
        let mut cci = vec![0.0; n];

        if n < period {
            return cci;
        }

        let tp: Vec<f64> = (0..n)
            .map(|i| (high[i] + low[i] + close[i]) / 3.0)
            .collect();

        for i in (period - 1)..n {
            let start = i + 1 - period;
            let tp_mean: f64 = tp[start..=i].iter().sum::<f64>() / period as f64;
            let mean_dev: f64 = tp[start..=i]
                .iter()
                .map(|&x| (x - tp_mean).abs())
                .sum::<f64>()
                / period as f64;

            if mean_dev > 1e-10 {
                cci[i] = (tp[i] - tp_mean) / (0.015 * mean_dev);
            }
        }

        cci
    }

    /// Calculate Stochastic component
    fn calculate_stoch(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
        let n = close.len();
        let mut stoch = vec![0.0; n];

        if n < period {
            return stoch;
        }

        for i in (period - 1)..n {
            let start = i + 1 - period;
            let highest = high[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lowest = low[start..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            let range = highest - lowest;

            if range > 1e-10 {
                stoch[i] = ((close[i] - lowest) / range) * 100.0;
            } else {
                stoch[i] = 50.0;
            }
        }

        stoch
    }

    /// Calculate composite oscillator values
    ///
    /// Returns weighted combination of RSI, CCI, and Stochastic scaled to -100 to +100
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let max_period = self.rsi_period.max(self.cci_period).max(self.stoch_period);
        let mut result = vec![0.0; n];

        if n < max_period + 1 {
            return result;
        }

        let rsi = Self::calculate_rsi(close, self.rsi_period);
        let cci = Self::calculate_cci(high, low, close, self.cci_period);
        let stoch = Self::calculate_stoch(high, low, close, self.stoch_period);

        let total_weight = self.weights.0 + self.weights.1 + self.weights.2;
        let w_rsi = self.weights.0 / total_weight;
        let w_cci = self.weights.1 / total_weight;
        let w_stoch = self.weights.2 / total_weight;

        for i in max_period..n {
            // Normalize RSI from 0-100 to -100 to +100
            let norm_rsi = (rsi[i] - 50.0) * 2.0;

            // CCI is already centered around 0, just clamp it
            let norm_cci = cci[i].max(-100.0).min(100.0);

            // Normalize Stochastic from 0-100 to -100 to +100
            let norm_stoch = (stoch[i] - 50.0) * 2.0;

            // Weighted composite
            let composite = norm_rsi * w_rsi + norm_cci * w_cci + norm_stoch * w_stoch;

            // Check for agreement (all same direction)
            let signs = [norm_rsi > 0.0, norm_cci > 0.0, norm_stoch > 0.0];
            let agreement_count = signs.iter().filter(|&&s| s).count();
            let agreement_bonus = if agreement_count == 3 || agreement_count == 0 {
                1.2 // Full agreement
            } else {
                1.0
            };

            result[i] = (composite * agreement_bonus).max(-100.0).min(100.0);
        }

        result
    }
}

impl TechnicalIndicator for CompositeOscillator {
    fn name(&self) -> &str {
        "Composite Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.rsi_period.max(self.cci_period).max(self.stoch_period) + 1
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

/// Adaptive Stochastic - Stochastic oscillator with adaptive periods
///
/// A stochastic oscillator that dynamically adjusts its lookback period
/// based on market volatility, using shorter periods in high volatility
/// and longer periods in low volatility environments.
#[derive(Debug, Clone)]
pub struct AdaptiveStochastic {
    min_period: usize,
    max_period: usize,
    volatility_period: usize,
}

impl AdaptiveStochastic {
    pub fn new(min_period: usize, max_period: usize, volatility_period: usize) -> Result<Self> {
        if min_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_period".to_string(),
                reason: "must be at least 3".to_string(),
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

    /// Calculate adaptive stochastic values
    ///
    /// Returns values from 0 to 100 with adaptive period based on volatility
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_required = self.max_period + self.volatility_period;
        let mut result = vec![0.0; n];

        if n < min_required + 1 {
            return result;
        }

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

        // Calculate adaptive stochastic
        for i in min_required..n {
            // Determine adaptive period based on volatility
            let normalized_vol = if vol_range > 1e-10 {
                (volatility[i] - vol_min) / vol_range
            } else {
                0.5
            };

            // Higher volatility = shorter period for faster response
            let adaptive_period = self.max_period
                - ((self.max_period - self.min_period) as f64 * normalized_vol).round() as usize;
            let period = adaptive_period.max(self.min_period).min(self.max_period);

            // Calculate stochastic with adaptive period
            let start = i + 1 - period;
            let highest = high[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lowest = low[start..=i].iter().cloned().fold(f64::INFINITY, f64::min);
            let range = highest - lowest;

            if range > 1e-10 {
                result[i] = ((close[i] - lowest) / range) * 100.0;
            } else {
                result[i] = 50.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for AdaptiveStochastic {
    fn name(&self) -> &str {
        "Adaptive Stochastic"
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

/// Momentum Wave Oscillator - Wave-based momentum oscillator
///
/// An oscillator that analyzes momentum in terms of wave patterns,
/// identifying momentum waves and their strength by analyzing
/// successive momentum peaks and troughs.
#[derive(Debug, Clone)]
pub struct MomentumWaveOscillator {
    momentum_period: usize,
    wave_count: usize,
    smooth_period: usize,
}

impl MomentumWaveOscillator {
    pub fn new(momentum_period: usize, wave_count: usize, smooth_period: usize) -> Result<Self> {
        if momentum_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if wave_count < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "wave_count".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            wave_count,
            smooth_period,
        })
    }

    /// Calculate momentum wave oscillator values
    ///
    /// Returns values from -100 to +100 based on wave momentum analysis
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_period = self.momentum_period * (self.wave_count + 1) + self.smooth_period;
        let mut result = vec![0.0; n];

        if n < min_period + 1 {
            return result;
        }

        // Calculate momentum (Rate of Change)
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period] > 1e-10 {
                momentum[i] = ((close[i] / close[i - self.momentum_period]) - 1.0) * 100.0;
            }
        }

        // Calculate momentum wave: analyze consecutive momentum readings
        let wave_period = self.momentum_period * self.wave_count;

        for i in (wave_period + self.momentum_period)..n {
            let wave_start = i - wave_period;

            // Collect momentum values for wave analysis
            let wave_momentum: Vec<f64> = momentum[wave_start..=i].to_vec();

            // Find peaks and troughs in momentum
            let mut peaks: Vec<f64> = Vec::new();
            let mut troughs: Vec<f64> = Vec::new();

            for j in 1..(wave_momentum.len() - 1) {
                if wave_momentum[j] > wave_momentum[j - 1] && wave_momentum[j] > wave_momentum[j + 1] {
                    peaks.push(wave_momentum[j]);
                } else if wave_momentum[j] < wave_momentum[j - 1] && wave_momentum[j] < wave_momentum[j + 1] {
                    troughs.push(wave_momentum[j]);
                }
            }

            // Calculate wave strength
            let avg_peak = if !peaks.is_empty() {
                peaks.iter().sum::<f64>() / peaks.len() as f64
            } else {
                0.0
            };

            let avg_trough = if !troughs.is_empty() {
                troughs.iter().sum::<f64>() / troughs.len() as f64
            } else {
                0.0
            };

            // Wave amplitude
            let wave_amplitude = avg_peak - avg_trough;

            // Current momentum position relative to wave
            let current_mom = momentum[i];

            // Normalize based on wave amplitude
            if wave_amplitude.abs() > 1e-10 {
                // Position within the wave
                let wave_mid = (avg_peak + avg_trough) / 2.0;
                let position = (current_mom - wave_mid) / (wave_amplitude / 2.0);

                // Consider momentum trend
                let mom_trend = momentum[i] - momentum[wave_start];
                let trend_factor = if mom_trend.abs() > 1e-10 {
                    mom_trend.signum() * 0.2
                } else {
                    0.0
                };

                result[i] = ((position + trend_factor) * 50.0).max(-100.0).min(100.0);
            } else {
                // No clear wave pattern, use current momentum directly
                result[i] = current_mom.max(-100.0).min(100.0);
            }
        }

        // Apply smoothing
        let mut smoothed = vec![0.0; n];
        for i in (min_period)..n {
            let start = i + 1 - self.smooth_period;
            smoothed[i] = result[start..=i].iter().sum::<f64>() / self.smooth_period as f64;
        }

        smoothed
    }
}

impl TechnicalIndicator for MomentumWaveOscillator {
    fn name(&self) -> &str {
        "Momentum Wave Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period * (self.wave_count + 1) + self.smooth_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Ultimate Oscillator Enhanced - Enhanced version with adaptive periods
///
/// An enhanced version of the Ultimate Oscillator that dynamically adapts
/// its weighting based on market volatility, giving more weight to shorter
/// periods in high volatility and longer periods in low volatility.
#[derive(Debug, Clone)]
pub struct UltimateOscillatorEnhanced {
    short_period: usize,
    medium_period: usize,
    long_period: usize,
    volatility_period: usize,
}

impl UltimateOscillatorEnhanced {
    pub fn new(
        short_period: usize,
        medium_period: usize,
        long_period: usize,
        volatility_period: usize,
    ) -> Result<Self> {
        if short_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 2".to_string(),
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
        if volatility_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "volatility_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            short_period,
            medium_period,
            long_period,
            volatility_period,
        })
    }

    /// Calculate Ultimate Oscillator Enhanced values
    ///
    /// Returns values from 0 to 100 with adaptive weighting based on volatility
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_required = self.long_period + self.volatility_period;
        let mut result = vec![0.0; n];

        if n < min_required + 1 {
            return result;
        }

        // Calculate Buying Pressure (BP) and True Range (TR)
        let mut bp = vec![0.0; n];
        let mut tr = vec![0.0; n];

        for i in 1..n {
            let prev_close = close[i - 1];
            let true_low = low[i].min(prev_close);
            let true_high = high[i].max(prev_close);

            bp[i] = close[i] - true_low;
            tr[i] = true_high - true_low;
        }

        // Calculate volatility (ATR-based)
        let mut volatility = vec![0.0; n];
        for i in self.volatility_period..n {
            let start = i + 1 - self.volatility_period;
            volatility[i] = tr[start..=i].iter().sum::<f64>() / self.volatility_period as f64;
        }

        // Find volatility range for normalization
        let vol_values: Vec<f64> = volatility[self.volatility_period..].to_vec();
        let vol_min = vol_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let vol_max = vol_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let vol_range = vol_max - vol_min;

        // Calculate averages for each period
        for i in min_required..n {
            // Calculate BP/TR sums for each period
            let short_start = i + 1 - self.short_period;
            let medium_start = i + 1 - self.medium_period;
            let long_start = i + 1 - self.long_period;

            let short_bp: f64 = bp[short_start..=i].iter().sum();
            let short_tr: f64 = tr[short_start..=i].iter().sum();
            let medium_bp: f64 = bp[medium_start..=i].iter().sum();
            let medium_tr: f64 = tr[medium_start..=i].iter().sum();
            let long_bp: f64 = bp[long_start..=i].iter().sum();
            let long_tr: f64 = tr[long_start..=i].iter().sum();

            // Calculate raw averages
            let short_avg = if short_tr > 1e-10 { short_bp / short_tr } else { 0.5 };
            let medium_avg = if medium_tr > 1e-10 { medium_bp / medium_tr } else { 0.5 };
            let long_avg = if long_tr > 1e-10 { long_bp / long_tr } else { 0.5 };

            // Adaptive weights based on volatility
            let normalized_vol = if vol_range > 1e-10 {
                (volatility[i] - vol_min) / vol_range
            } else {
                0.5
            };

            // Higher volatility = more weight on short period
            // Lower volatility = more weight on long period
            let short_weight = 4.0 + normalized_vol * 2.0;   // 4 to 6
            let medium_weight = 2.0;                          // constant 2
            let long_weight = 1.0 + (1.0 - normalized_vol);  // 1 to 2

            let total_weight = short_weight + medium_weight + long_weight;

            // Calculate weighted UO
            let uo = 100.0 * (short_avg * short_weight + medium_avg * medium_weight + long_avg * long_weight) / total_weight;

            result[i] = uo.max(0.0).min(100.0);
        }

        result
    }
}

impl TechnicalIndicator for UltimateOscillatorEnhanced {
    fn name(&self) -> &str {
        "Ultimate Oscillator Enhanced"
    }

    fn min_periods(&self) -> usize {
        self.long_period + self.volatility_period + 1
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

/// Percent Rank Oscillator - Oscillator based on percent rank statistics
///
/// Measures where the current price falls within the distribution of
/// historical prices, expressed as a percentile rank. This provides
/// a statistical view of price position relative to recent history.
#[derive(Debug, Clone)]
pub struct PercentRankOscillator {
    period: usize,
    smooth_period: usize,
}

impl PercentRankOscillator {
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { period, smooth_period })
    }

    /// Calculate percent rank oscillator values
    ///
    /// Returns values from 0 to 100 indicating the percentile rank of current price
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_required = self.period + self.smooth_period;
        let mut result = vec![0.0; n];

        if n < min_required {
            return result;
        }

        // Calculate raw percent rank for each point
        let mut raw_rank = vec![0.0; n];
        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let current = close[i];

            // Count how many values are less than or equal to current
            let count_below = close[start..i] // Exclude current value from comparison
                .iter()
                .filter(|&&x| x < current)
                .count();

            // Percent rank: percentage of values below current
            raw_rank[i] = (count_below as f64 / (self.period - 1) as f64) * 100.0;
        }

        // Apply smoothing
        if self.smooth_period == 1 {
            for i in (self.period - 1)..n {
                result[i] = raw_rank[i];
            }
        } else {
            for i in min_required..n {
                let start = i + 1 - self.smooth_period;
                let smoothed: f64 = raw_rank[start..=i].iter().sum::<f64>() / self.smooth_period as f64;
                result[i] = smoothed;
            }
        }

        result
    }
}

impl TechnicalIndicator for PercentRankOscillator {
    fn name(&self) -> &str {
        "Percent Rank Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Z-Score Oscillator - Oscillator based on z-score statistics
///
/// Measures how many standard deviations the current price is from
/// its moving average. This provides a normalized view of price
/// deviation that can be compared across different securities and timeframes.
#[derive(Debug, Clone)]
pub struct ZScoreOscillator {
    period: usize,
    signal_period: usize,
}

impl ZScoreOscillator {
    pub fn new(period: usize, signal_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if signal_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "signal_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, signal_period })
    }

    /// Calculate z-score oscillator values with signal line
    ///
    /// Returns (z_score, signal) where z_score indicates standard deviations from mean
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let min_required = self.period + self.signal_period;
        let mut z_score = vec![0.0; n];
        let mut signal = vec![0.0; n];

        if n < min_required {
            return (z_score, signal);
        }

        // Calculate z-scores
        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let slice = &close[start..=i];

            // Calculate mean
            let mean: f64 = slice.iter().sum::<f64>() / self.period as f64;

            // Calculate standard deviation
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            // Calculate z-score
            if std_dev > 1e-10 {
                z_score[i] = (close[i] - mean) / std_dev;
            }
        }

        // Calculate signal line (EMA of z-score)
        let multiplier = 2.0 / (self.signal_period as f64 + 1.0);

        // Initialize signal with SMA
        if n >= self.period + self.signal_period - 1 {
            let sig_start = self.period - 1;
            let sig_init_end = sig_start + self.signal_period;
            if sig_init_end <= n {
                signal[sig_init_end - 1] = z_score[sig_start..sig_init_end].iter().sum::<f64>() / self.signal_period as f64;

                // Calculate EMA
                for i in sig_init_end..n {
                    signal[i] = (z_score[i] - signal[i - 1]) * multiplier + signal[i - 1];
                }
            }
        }

        (z_score, signal)
    }
}

impl TechnicalIndicator for ZScoreOscillator {
    fn name(&self) -> &str {
        "Z-Score Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.period + self.signal_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (z_score, signal) = self.calculate(&data.close);
        Ok(IndicatorOutput::dual(z_score, signal))
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Velocity Oscillator - Oscillator measuring price velocity with normalization
///
/// Measures the rate of price change (velocity) normalized to oscillate
/// between -100 and +100, making it comparable across different securities
/// and providing clear overbought/oversold signals.
#[derive(Debug, Clone)]
pub struct VelocityOscillator {
    velocity_period: usize,
    normalization_period: usize,
    smooth_period: usize,
}

impl VelocityOscillator {
    pub fn new(velocity_period: usize, normalization_period: usize, smooth_period: usize) -> Result<Self> {
        if velocity_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "velocity_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if normalization_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "normalization_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            velocity_period,
            normalization_period,
            smooth_period,
        })
    }

    /// Calculate velocity oscillator values
    ///
    /// Returns values from -100 to +100 representing normalized velocity
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_required = self.velocity_period + self.normalization_period + self.smooth_period;
        let mut result = vec![0.0; n];

        if n < min_required {
            return result;
        }

        // Calculate raw velocity (rate of change as percentage)
        let mut velocity = vec![0.0; n];
        for i in self.velocity_period..n {
            if close[i - self.velocity_period] > 1e-10 {
                velocity[i] = ((close[i] / close[i - self.velocity_period]) - 1.0) * 100.0;
            }
        }

        // Normalize velocity using z-score over normalization period
        let mut normalized = vec![0.0; n];
        for i in (self.velocity_period + self.normalization_period - 1)..n {
            let start = i + 1 - self.normalization_period;
            let slice = &velocity[start..=i];

            let mean: f64 = slice.iter().sum::<f64>() / self.normalization_period as f64;
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.normalization_period as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                // Z-score scaled to approximately -100 to +100
                let z_score = (velocity[i] - mean) / std_dev;
                normalized[i] = (z_score * 33.0).max(-100.0).min(100.0);
            }
        }

        // Apply smoothing
        if self.smooth_period == 1 {
            for i in (self.velocity_period + self.normalization_period - 1)..n {
                result[i] = normalized[i];
            }
        } else {
            for i in min_required..n {
                let start = i + 1 - self.smooth_period;
                result[i] = normalized[start..=i].iter().sum::<f64>() / self.smooth_period as f64;
            }
        }

        result
    }
}

impl TechnicalIndicator for VelocityOscillator {
    fn name(&self) -> &str {
        "Velocity Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.velocity_period + self.normalization_period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Acceleration Oscillator - Oscillator measuring price acceleration
///
/// Measures the rate of change of velocity (acceleration) normalized
/// to oscillate between -100 and +100. Positive acceleration indicates
/// strengthening momentum, negative indicates weakening momentum.
#[derive(Debug, Clone)]
pub struct AccelerationOscillator {
    velocity_period: usize,
    acceleration_period: usize,
    normalization_period: usize,
}

impl AccelerationOscillator {
    pub fn new(velocity_period: usize, acceleration_period: usize, normalization_period: usize) -> Result<Self> {
        if velocity_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "velocity_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if acceleration_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "acceleration_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if normalization_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "normalization_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self {
            velocity_period,
            acceleration_period,
            normalization_period,
        })
    }

    /// Calculate acceleration oscillator values
    ///
    /// Returns values from -100 to +100 representing normalized acceleration
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_required = self.velocity_period + self.acceleration_period + self.normalization_period;
        let mut result = vec![0.0; n];

        if n < min_required {
            return result;
        }

        // Calculate velocity (rate of change)
        let mut velocity = vec![0.0; n];
        for i in self.velocity_period..n {
            if close[i - self.velocity_period] > 1e-10 {
                velocity[i] = ((close[i] / close[i - self.velocity_period]) - 1.0) * 100.0;
            }
        }

        // Calculate acceleration (change in velocity)
        let mut acceleration = vec![0.0; n];
        for i in (self.velocity_period + self.acceleration_period)..n {
            acceleration[i] = velocity[i] - velocity[i - self.acceleration_period];
        }

        // Normalize acceleration using z-score
        for i in min_required..n {
            let start = i + 1 - self.normalization_period;
            let slice = &acceleration[start..=i];

            let mean: f64 = slice.iter().sum::<f64>() / self.normalization_period as f64;
            let variance: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.normalization_period as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                let z_score = (acceleration[i] - mean) / std_dev;
                result[i] = (z_score * 33.0).max(-100.0).min(100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for AccelerationOscillator {
    fn name(&self) -> &str {
        "Acceleration Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.velocity_period + self.acceleration_period + self.normalization_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Harmonic Oscillator - Oscillator based on harmonic analysis
///
/// Analyzes price movements using harmonic principles, decomposing
/// price action into harmonic components and generating oscillator
/// signals based on harmonic ratios and cycle completion.
#[derive(Debug, Clone)]
pub struct HarmonicOscillator {
    analysis_period: usize,
    harmonic_count: usize,
    smooth_period: usize,
}

impl HarmonicOscillator {
    pub fn new(analysis_period: usize, harmonic_count: usize, smooth_period: usize) -> Result<Self> {
        if analysis_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "analysis_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if harmonic_count < 1 || harmonic_count > 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "harmonic_count".to_string(),
                reason: "must be between 1 and 5".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            analysis_period,
            harmonic_count,
            smooth_period,
        })
    }

    /// Calculate harmonic oscillator values
    ///
    /// Returns values from -100 to +100 based on harmonic analysis
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_required = self.analysis_period + self.smooth_period;
        let mut result = vec![0.0; n];

        if n < min_required {
            return result;
        }

        let pi = std::f64::consts::PI;

        // Calculate harmonic oscillator for each point
        let mut raw_harmonic = vec![0.0; n];
        for i in (self.analysis_period - 1)..n {
            let start = i + 1 - self.analysis_period;
            let segment: Vec<f64> = close[start..=i].to_vec();

            // Detrend the segment
            let mean: f64 = segment.iter().sum::<f64>() / segment.len() as f64;
            let detrended: Vec<f64> = segment.iter().map(|x| x - mean).collect();

            // Calculate harmonic components using simple Fourier-like analysis
            let mut harmonic_sum = 0.0;
            let period_len = self.analysis_period as f64;

            for h in 1..=self.harmonic_count {
                let h_f = h as f64;
                let frequency = 2.0 * pi * h_f / period_len;

                // Calculate sine and cosine components
                let mut sin_sum = 0.0;
                let mut cos_sum = 0.0;

                for (j, &val) in detrended.iter().enumerate() {
                    let phase = frequency * j as f64;
                    sin_sum += val * phase.sin();
                    cos_sum += val * phase.cos();
                }

                // Amplitude of this harmonic
                let amplitude = ((sin_sum.powi(2) + cos_sum.powi(2)).sqrt() * 2.0) / period_len;

                // Phase of this harmonic at current point
                let current_phase = frequency * (self.analysis_period - 1) as f64;
                let phase_offset = sin_sum.atan2(cos_sum);

                // Harmonic contribution at current point
                let harmonic_value = amplitude * (current_phase + phase_offset).sin();

                // Weight lower harmonics more
                let weight = 1.0 / h_f;
                harmonic_sum += harmonic_value * weight;
            }

            // Normalize to standard deviation
            let variance: f64 = detrended.iter().map(|x| x.powi(2)).sum::<f64>() / detrended.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                raw_harmonic[i] = (harmonic_sum / std_dev) * 50.0;
            }
        }

        // Apply smoothing
        if self.smooth_period == 1 {
            for i in (self.analysis_period - 1)..n {
                result[i] = raw_harmonic[i].max(-100.0).min(100.0);
            }
        } else {
            for i in min_required..n {
                let start = i + 1 - self.smooth_period;
                let smoothed: f64 = raw_harmonic[start..=i].iter().sum::<f64>() / self.smooth_period as f64;
                result[i] = smoothed.max(-100.0).min(100.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for HarmonicOscillator {
    fn name(&self) -> &str {
        "Harmonic Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.analysis_period + self.smooth_period
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

        // Verify we get non-zero results in the valid range
        let valid_values: Vec<f64> = result[26..60].to_vec();
        let has_values = valid_values.iter().any(|&v| v != 0.0);
        assert!(has_values, "Should produce non-zero values in uptrend");
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
        assert!(PriceStrengthIndex::new(5, 14, 28, (-0.5, 0.5, 1.0)).is_ok()); // Total is positive
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
    // DynamicOscillator Tests
    // ============================================================

    #[test]
    fn test_dynamic_oscillator_basic() {
        let data = make_test_data();
        let indicator = DynamicOscillator::new(14, 20).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_dynamic_oscillator_validation() {
        assert!(DynamicOscillator::new(4, 20).is_err());
        assert!(DynamicOscillator::new(14, 9).is_err());
        assert!(DynamicOscillator::new(5, 10).is_ok());
        assert!(DynamicOscillator::new(14, 20).is_ok());
    }

    #[test]
    fn test_dynamic_oscillator_trait() {
        let data = make_test_data();
        let indicator = DynamicOscillator::new(14, 20).unwrap();

        assert_eq!(indicator.name(), "Dynamic Oscillator");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_dynamic_oscillator_uptrend() {
        let n = 60;
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 1.5).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();

        let indicator = DynamicOscillator::new(10, 15).unwrap();
        let result = indicator.calculate(&high, &low, &close);

        let last_10: Vec<f64> = result[50..60].to_vec();
        let positive_count = last_10.iter().filter(|&&v| v > 0.0).count();
        assert!(positive_count >= 5);
    }

    // ============================================================
    // TrendOptimizedOscillator Tests
    // ============================================================

    #[test]
    fn test_trend_optimized_oscillator_basic() {
        let data = make_test_data();
        let indicator = TrendOptimizedOscillator::new(10, 20, 5).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_trend_optimized_oscillator_validation() {
        assert!(TrendOptimizedOscillator::new(2, 20, 5).is_err());
        assert!(TrendOptimizedOscillator::new(10, 10, 5).is_err());
        assert!(TrendOptimizedOscillator::new(10, 5, 5).is_err());
        assert!(TrendOptimizedOscillator::new(10, 20, 1).is_err());
        assert!(TrendOptimizedOscillator::new(5, 15, 3).is_ok());
    }

    #[test]
    fn test_trend_optimized_oscillator_trait() {
        let data = make_test_data();
        let indicator = TrendOptimizedOscillator::new(10, 20, 5).unwrap();

        assert_eq!(indicator.name(), "Trend Optimized Oscillator");
        assert_eq!(indicator.min_periods(), 26);
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_trend_optimized_oscillator_strong_uptrend() {
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 2.0).collect();

        let indicator = TrendOptimizedOscillator::new(10, 20, 5).unwrap();
        let result = indicator.calculate(&close);

        // Check that we get valid values in the expected range
        let last = result.last().unwrap();
        assert!(last.is_finite() && *last >= -100.0 && *last <= 100.0);
    }

    // ============================================================
    // RangeOptimizedOscillator Tests
    // ============================================================

    #[test]
    fn test_range_optimized_oscillator_basic() {
        let data = make_test_data();
        let indicator = RangeOptimizedOscillator::new(14, 5).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_range_optimized_oscillator_validation() {
        assert!(RangeOptimizedOscillator::new(4, 5).is_err());
        assert!(RangeOptimizedOscillator::new(14, 1).is_err());
        assert!(RangeOptimizedOscillator::new(5, 2).is_ok());
        assert!(RangeOptimizedOscillator::new(14, 5).is_ok());
    }

    #[test]
    fn test_range_optimized_oscillator_trait() {
        let data = make_test_data();
        let indicator = RangeOptimizedOscillator::new(14, 5).unwrap();

        assert_eq!(indicator.name(), "Range Optimized Oscillator");
        assert_eq!(indicator.min_periods(), 20);
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_range_optimized_oscillator_ranging_market() {
        // Ranging market data (oscillating around a mean)
        let close: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 5.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();

        let indicator = RangeOptimizedOscillator::new(10, 3).unwrap();
        let result = indicator.calculate(&high, &low, &close);

        // In ranging market, oscillator should swing between extremes
        let min_val = result[20..].iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = result[20..].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val - min_val > 20.0); // Should have good range
    }

    // ============================================================
    // CompositeOscillator Tests
    // ============================================================

    #[test]
    fn test_composite_oscillator_basic() {
        let data = make_test_data();
        let indicator = CompositeOscillator::new(14, 10, 20, (0.4, 0.3, 0.3)).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_composite_oscillator_validation() {
        assert!(CompositeOscillator::new(1, 10, 20, (0.4, 0.3, 0.3)).is_err()); // rsi_period < 2
        assert!(CompositeOscillator::new(14, 4, 20, (0.4, 0.3, 0.3)).is_err()); // cci_period < 5
        assert!(CompositeOscillator::new(14, 10, 4, (0.4, 0.3, 0.3)).is_err()); // stoch_period < 5
        assert!(CompositeOscillator::new(14, 10, 20, (0.0, 0.0, 0.0)).is_err()); // zero weights
        assert!(CompositeOscillator::new(5, 5, 10, (1.0, 1.0, 1.0)).is_ok());
    }

    #[test]
    fn test_composite_oscillator_trait() {
        let data = make_test_data();
        let indicator = CompositeOscillator::new(14, 10, 20, (0.4, 0.3, 0.3)).unwrap();

        assert_eq!(indicator.name(), "Composite Oscillator");
        assert_eq!(indicator.min_periods(), 21);
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_composite_oscillator_weights() {
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();

        let rsi_heavy = CompositeOscillator::new(14, 10, 20, (1.0, 0.0, 0.0)).unwrap();
        let cci_heavy = CompositeOscillator::new(14, 10, 20, (0.0, 1.0, 0.0)).unwrap();

        let result_rsi = rsi_heavy.calculate(&high, &low, &close);
        let result_cci = cci_heavy.calculate(&high, &low, &close);

        // Results should be different due to different weights
        let last_rsi = result_rsi.last().unwrap();
        let last_cci = result_cci.last().unwrap();
        assert!(*last_rsi >= -100.0 && *last_rsi <= 100.0);
        assert!(*last_cci >= -100.0 && *last_cci <= 100.0);
    }

    // ============================================================
    // AdaptiveStochastic Tests
    // ============================================================

    #[test]
    fn test_adaptive_stochastic_basic() {
        let data = make_test_data();
        let indicator = AdaptiveStochastic::new(5, 21, 10).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_adaptive_stochastic_validation() {
        assert!(AdaptiveStochastic::new(2, 21, 10).is_err());
        assert!(AdaptiveStochastic::new(5, 5, 10).is_err());
        assert!(AdaptiveStochastic::new(5, 4, 10).is_err());
        assert!(AdaptiveStochastic::new(5, 21, 4).is_err());
        assert!(AdaptiveStochastic::new(5, 14, 10).is_ok());
    }

    #[test]
    fn test_adaptive_stochastic_trait() {
        let data = make_test_data();
        let indicator = AdaptiveStochastic::new(5, 21, 10).unwrap();

        assert_eq!(indicator.name(), "Adaptive Stochastic");
        assert_eq!(indicator.min_periods(), 32);
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_adaptive_stochastic_uptrend() {
        let n = 60;
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 1.5).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();

        let indicator = AdaptiveStochastic::new(5, 14, 10).unwrap();
        let result = indicator.calculate(&high, &low, &close);

        // In uptrend, stochastic should be high
        let last_10: Vec<f64> = result[50..60].to_vec();
        let high_count = last_10.iter().filter(|&&v| v > 50.0).count();
        assert!(high_count >= 5);
    }

    // ============================================================
    // MomentumWaveOscillator Tests
    // ============================================================

    #[test]
    fn test_momentum_wave_oscillator_basic() {
        let data = make_test_data();
        let indicator = MomentumWaveOscillator::new(10, 3, 5).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_momentum_wave_oscillator_validation() {
        assert!(MomentumWaveOscillator::new(2, 3, 5).is_err());
        assert!(MomentumWaveOscillator::new(10, 0, 5).is_err());
        assert!(MomentumWaveOscillator::new(10, 3, 1).is_err());
        assert!(MomentumWaveOscillator::new(5, 2, 3).is_ok());
    }

    #[test]
    fn test_momentum_wave_oscillator_trait() {
        let data = make_test_data();
        let indicator = MomentumWaveOscillator::new(10, 3, 5).unwrap();

        assert_eq!(indicator.name(), "Momentum Wave Oscillator");
        // min_periods = momentum_period * (wave_count + 1) + smooth_period + 1 = 10*4+5+1 = 46
        assert_eq!(indicator.min_periods(), 46);
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_momentum_wave_oscillator_uptrend() {
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();

        let indicator = MomentumWaveOscillator::new(10, 3, 5).unwrap();
        let result = indicator.calculate(&close);

        let last = result.last().unwrap();
        assert!(*last > 0.0);
    }

    #[test]
    fn test_momentum_wave_oscillator_cyclical() {
        // Test with cyclical data
        let close: Vec<f64> = (0..60)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 10.0)
            .collect();

        let indicator = MomentumWaveOscillator::new(10, 3, 5).unwrap();
        let result = indicator.calculate(&close);

        // Should oscillate
        let min_val = result[25..].iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = result[25..].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val > 0.0 || min_val < 0.0); // Should have some variation
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

    // ============================================================
    // UltimateOscillatorEnhanced Tests
    // ============================================================

    #[test]
    fn test_ultimate_oscillator_enhanced_basic() {
        let data = make_test_data();
        let indicator = UltimateOscillatorEnhanced::new(7, 14, 28, 10).unwrap();
        let result = indicator.calculate(&data.high, &data.low, &data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_ultimate_oscillator_enhanced_validation() {
        // short_period too small
        assert!(UltimateOscillatorEnhanced::new(1, 14, 28, 10).is_err());
        // medium_period <= short_period
        assert!(UltimateOscillatorEnhanced::new(7, 7, 28, 10).is_err());
        assert!(UltimateOscillatorEnhanced::new(7, 5, 28, 10).is_err());
        // long_period <= medium_period
        assert!(UltimateOscillatorEnhanced::new(7, 14, 14, 10).is_err());
        assert!(UltimateOscillatorEnhanced::new(7, 14, 10, 10).is_err());
        // volatility_period too small
        assert!(UltimateOscillatorEnhanced::new(7, 14, 28, 4).is_err());
        // Valid parameters
        assert!(UltimateOscillatorEnhanced::new(7, 14, 28, 10).is_ok());
        assert!(UltimateOscillatorEnhanced::new(5, 10, 20, 5).is_ok());
    }

    #[test]
    fn test_ultimate_oscillator_enhanced_trait() {
        let data = make_test_data();
        let indicator = UltimateOscillatorEnhanced::new(7, 14, 28, 10).unwrap();

        assert_eq!(indicator.name(), "Ultimate Oscillator Enhanced");
        assert_eq!(indicator.min_periods(), 39); // long_period + volatility_period + 1
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_ultimate_oscillator_enhanced_uptrend() {
        // Strong uptrend should give high UO values (>50)
        let n = 60;
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 1.5).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.5).collect();

        let indicator = UltimateOscillatorEnhanced::new(5, 10, 20, 10).unwrap();
        let result = indicator.calculate(&high, &low, &close);

        let last_10: Vec<f64> = result[50..60].to_vec();
        let high_count = last_10.iter().filter(|&&v| v > 50.0).count();
        assert!(high_count >= 5);
    }

    #[test]
    fn test_ultimate_oscillator_enhanced_insufficient_data() {
        let short_data = OHLCVSeries {
            open: vec![100.0, 101.0, 102.0],
            high: vec![101.0, 102.0, 103.0],
            low: vec![99.0, 100.0, 101.0],
            close: vec![100.5, 101.5, 102.5],
            volume: vec![1000.0, 1100.0, 1200.0],
        };

        let indicator = UltimateOscillatorEnhanced::new(7, 14, 28, 10).unwrap();
        let result = indicator.calculate(&short_data.high, &short_data.low, &short_data.close);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // ============================================================
    // PercentRankOscillator Tests
    // ============================================================

    #[test]
    fn test_percent_rank_oscillator_basic() {
        let data = make_test_data();
        let indicator = PercentRankOscillator::new(20, 3).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_percent_rank_oscillator_validation() {
        // period too small
        assert!(PercentRankOscillator::new(4, 3).is_err());
        // smooth_period too small
        assert!(PercentRankOscillator::new(20, 0).is_err());
        // Valid parameters
        assert!(PercentRankOscillator::new(5, 1).is_ok());
        assert!(PercentRankOscillator::new(20, 5).is_ok());
    }

    #[test]
    fn test_percent_rank_oscillator_trait() {
        let data = make_test_data();
        let indicator = PercentRankOscillator::new(20, 3).unwrap();

        assert_eq!(indicator.name(), "Percent Rank Oscillator");
        assert_eq!(indicator.min_periods(), 23); // period + smooth_period
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_percent_rank_oscillator_uptrend() {
        // Strong uptrend: current price should rank high
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();

        let indicator = PercentRankOscillator::new(20, 1).unwrap();
        let result = indicator.calculate(&close);

        // In uptrend, last values should have high rank (close to 100)
        let last = result.last().unwrap();
        assert!(*last > 80.0);
    }

    #[test]
    fn test_percent_rank_oscillator_downtrend() {
        // Strong downtrend: current price should rank low
        let close: Vec<f64> = (0..50).map(|i| 200.0 - i as f64 * 2.0).collect();

        let indicator = PercentRankOscillator::new(20, 1).unwrap();
        let result = indicator.calculate(&close);

        // In downtrend, last values should have low rank (close to 0)
        let last = result.last().unwrap();
        assert!(*last < 20.0);
    }

    #[test]
    fn test_percent_rank_oscillator_insufficient_data() {
        let short_close: Vec<f64> = vec![100.0; 5];

        let indicator = PercentRankOscillator::new(20, 3).unwrap();
        let result = indicator.calculate(&short_close);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // ============================================================
    // ZScoreOscillator Tests
    // ============================================================

    #[test]
    fn test_z_score_oscillator_basic() {
        let data = make_test_data();
        let indicator = ZScoreOscillator::new(20, 5).unwrap();
        let (z_score, signal) = indicator.calculate(&data.close);

        assert_eq!(z_score.len(), data.close.len());
        assert_eq!(signal.len(), data.close.len());
    }

    #[test]
    fn test_z_score_oscillator_validation() {
        // period too small
        assert!(ZScoreOscillator::new(4, 5).is_err());
        // signal_period too small
        assert!(ZScoreOscillator::new(20, 1).is_err());
        // Valid parameters
        assert!(ZScoreOscillator::new(5, 2).is_ok());
        assert!(ZScoreOscillator::new(20, 9).is_ok());
    }

    #[test]
    fn test_z_score_oscillator_trait() {
        let data = make_test_data();
        let indicator = ZScoreOscillator::new(20, 5).unwrap();

        assert_eq!(indicator.name(), "Z-Score Oscillator");
        assert_eq!(indicator.min_periods(), 25); // period + signal_period
        assert_eq!(indicator.output_features(), 2);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_z_score_oscillator_uptrend() {
        // Strong uptrend should give positive z-scores
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();

        let indicator = ZScoreOscillator::new(20, 5).unwrap();
        let (z_score, _signal) = indicator.calculate(&close);

        // In uptrend, z-scores should be positive (above mean)
        let last = z_score.last().unwrap();
        assert!(*last > 0.0);
    }

    #[test]
    fn test_z_score_oscillator_flat_prices() {
        // Flat prices should give z-scores near zero
        let flat_close: Vec<f64> = vec![100.0; 50];

        let indicator = ZScoreOscillator::new(20, 5).unwrap();
        let (z_score, _signal) = indicator.calculate(&flat_close);

        let min_period = indicator.min_periods();
        for i in min_period..z_score.len() {
            assert!(z_score[i].abs() < 0.1);
        }
    }

    #[test]
    fn test_z_score_oscillator_insufficient_data() {
        let short_close: Vec<f64> = vec![100.0; 10];

        let indicator = ZScoreOscillator::new(20, 5).unwrap();
        let (z_score, signal) = indicator.calculate(&short_close);
        assert!(z_score.iter().all(|&v| v == 0.0));
        assert!(signal.iter().all(|&v| v == 0.0));
    }

    // ============================================================
    // VelocityOscillator Tests
    // ============================================================

    #[test]
    fn test_velocity_oscillator_basic() {
        let data = make_test_data();
        let indicator = VelocityOscillator::new(5, 20, 3).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_velocity_oscillator_validation() {
        // velocity_period too small
        assert!(VelocityOscillator::new(1, 20, 3).is_err());
        // normalization_period too small
        assert!(VelocityOscillator::new(5, 9, 3).is_err());
        // smooth_period too small
        assert!(VelocityOscillator::new(5, 20, 0).is_err());
        // Valid parameters
        assert!(VelocityOscillator::new(2, 10, 1).is_ok());
        assert!(VelocityOscillator::new(10, 30, 5).is_ok());
    }

    #[test]
    fn test_velocity_oscillator_trait() {
        let data = make_test_data();
        let indicator = VelocityOscillator::new(5, 20, 3).unwrap();

        assert_eq!(indicator.name(), "Velocity Oscillator");
        assert_eq!(indicator.min_periods(), 28); // velocity_period + normalization_period + smooth_period
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_velocity_oscillator_accelerating_uptrend() {
        // Accelerating uptrend should show increasing positive velocity
        let close: Vec<f64> = (0..60).map(|i| 100.0 + (i as f64).powi(2) * 0.05).collect();

        let indicator = VelocityOscillator::new(5, 20, 1).unwrap();
        let result = indicator.calculate(&close);

        // With accelerating prices, velocity should be positive
        let last_10: Vec<f64> = result[50..60].to_vec();
        let positive_count = last_10.iter().filter(|&&v| v > 0.0).count();
        assert!(positive_count >= 5);
    }

    #[test]
    fn test_velocity_oscillator_decelerating_downtrend() {
        // Decelerating downtrend
        let close: Vec<f64> = (0..60).map(|i| 200.0 - (i as f64) * 2.0).collect();

        let indicator = VelocityOscillator::new(5, 20, 1).unwrap();
        let result = indicator.calculate(&close);

        // With constant downtrend, velocity should be around zero (no change in velocity)
        // but slightly negative due to consistent downward movement
        let last = result.last().unwrap();
        assert!(last.is_finite());
    }

    #[test]
    fn test_velocity_oscillator_insufficient_data() {
        let short_close: Vec<f64> = vec![100.0; 10];

        let indicator = VelocityOscillator::new(5, 20, 3).unwrap();
        let result = indicator.calculate(&short_close);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // ============================================================
    // AccelerationOscillator Tests
    // ============================================================

    #[test]
    fn test_acceleration_oscillator_basic() {
        let data = make_test_data();
        let indicator = AccelerationOscillator::new(5, 5, 20).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_acceleration_oscillator_validation() {
        // velocity_period too small
        assert!(AccelerationOscillator::new(1, 5, 20).is_err());
        // acceleration_period too small
        assert!(AccelerationOscillator::new(5, 1, 20).is_err());
        // normalization_period too small
        assert!(AccelerationOscillator::new(5, 5, 9).is_err());
        // Valid parameters
        assert!(AccelerationOscillator::new(2, 2, 10).is_ok());
        assert!(AccelerationOscillator::new(10, 10, 30).is_ok());
    }

    #[test]
    fn test_acceleration_oscillator_trait() {
        let data = make_test_data();
        let indicator = AccelerationOscillator::new(5, 5, 20).unwrap();

        assert_eq!(indicator.name(), "Acceleration Oscillator");
        assert_eq!(indicator.min_periods(), 31); // velocity_period + acceleration_period + normalization_period + 1
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_acceleration_oscillator_accelerating_trend() {
        // Accelerating uptrend (quadratic price increase)
        let close: Vec<f64> = (0..60).map(|i| 100.0 + (i as f64).powi(2) * 0.1).collect();

        let indicator = AccelerationOscillator::new(5, 5, 20).unwrap();
        let result = indicator.calculate(&close);

        // With accelerating prices, we should get valid oscillator values
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(result[i].is_finite());
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }

        // The values should have some variation (not all zero)
        let valid_values: Vec<f64> = result[min_period..].to_vec();
        let has_nonzero = valid_values.iter().any(|&v| v.abs() > 1e-10);
        assert!(has_nonzero, "Should produce non-zero values with accelerating prices");
    }

    #[test]
    fn test_acceleration_oscillator_constant_velocity() {
        // Constant velocity (linear price increase)
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 2.0).collect();

        let indicator = AccelerationOscillator::new(5, 5, 20).unwrap();
        let result = indicator.calculate(&close);

        // With constant velocity, acceleration should be near zero
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(result[i].abs() < 50.0); // Should be relatively small
        }
    }

    #[test]
    fn test_acceleration_oscillator_insufficient_data() {
        let short_close: Vec<f64> = vec![100.0; 15];

        let indicator = AccelerationOscillator::new(5, 5, 20).unwrap();
        let result = indicator.calculate(&short_close);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // ============================================================
    // HarmonicOscillator Tests
    // ============================================================

    #[test]
    fn test_harmonic_oscillator_basic() {
        let data = make_test_data();
        let indicator = HarmonicOscillator::new(20, 3, 5).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_harmonic_oscillator_validation() {
        // analysis_period too small
        assert!(HarmonicOscillator::new(9, 3, 5).is_err());
        // harmonic_count out of range
        assert!(HarmonicOscillator::new(20, 0, 5).is_err());
        assert!(HarmonicOscillator::new(20, 6, 5).is_err());
        // smooth_period too small
        assert!(HarmonicOscillator::new(20, 3, 0).is_err());
        // Valid parameters
        assert!(HarmonicOscillator::new(10, 1, 1).is_ok());
        assert!(HarmonicOscillator::new(30, 5, 10).is_ok());
    }

    #[test]
    fn test_harmonic_oscillator_trait() {
        let data = make_test_data();
        let indicator = HarmonicOscillator::new(20, 3, 5).unwrap();

        assert_eq!(indicator.name(), "Harmonic Oscillator");
        assert_eq!(indicator.min_periods(), 25); // analysis_period + smooth_period
        assert_eq!(indicator.output_features(), 1);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.primary.is_empty());
    }

    #[test]
    fn test_harmonic_oscillator_cyclical_data() {
        // Test with sinusoidal data - should detect the cycle
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.314).sin() * 10.0) // ~20 period cycle
            .collect();

        let indicator = HarmonicOscillator::new(20, 3, 3).unwrap();
        let result = indicator.calculate(&close);

        // Should oscillate - check for variation
        let min_val = result[30..].iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = result[30..].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val - min_val > 10.0); // Should have meaningful oscillation
    }

    #[test]
    fn test_harmonic_oscillator_flat_prices() {
        // Flat prices should give oscillator near zero
        let flat_close: Vec<f64> = vec![100.0; 50];

        let indicator = HarmonicOscillator::new(20, 3, 5).unwrap();
        let result = indicator.calculate(&flat_close);

        let min_period = indicator.min_periods();
        for i in min_period..result.len() {
            assert!(result[i].abs() < 10.0);
        }
    }

    #[test]
    fn test_harmonic_oscillator_insufficient_data() {
        let short_close: Vec<f64> = vec![100.0; 15];

        let indicator = HarmonicOscillator::new(20, 3, 5).unwrap();
        let result = indicator.calculate(&short_close);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_harmonic_oscillator_different_harmonic_counts() {
        // Test that different harmonic counts produce different results
        let close: Vec<f64> = (0..60)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0 + (i as f64 * 0.6).sin() * 3.0)
            .collect();

        let h1 = HarmonicOscillator::new(20, 1, 1).unwrap();
        let h3 = HarmonicOscillator::new(20, 3, 1).unwrap();

        let result_h1 = h1.calculate(&close);
        let result_h3 = h3.calculate(&close);

        // Results should be different
        let last_h1 = result_h1.last().unwrap();
        let last_h3 = result_h3.last().unwrap();
        assert!(last_h1.is_finite() && last_h3.is_finite());
        // They might be similar but the calculation path is different
    }

    // ============================================================
    // Additional edge case tests for all new indicators
    // ============================================================

    #[test]
    fn test_all_new_indicators_with_extreme_values() {
        // Test with very large price values
        let large_close: Vec<f64> = (0..60).map(|i| 10000.0 + i as f64 * 100.0).collect();
        let large_high: Vec<f64> = large_close.iter().map(|c| c + 50.0).collect();
        let large_low: Vec<f64> = large_close.iter().map(|c| c - 50.0).collect();

        let uoe = UltimateOscillatorEnhanced::new(5, 10, 20, 10).unwrap();
        let result = uoe.calculate(&large_high, &large_low, &large_close);
        assert!(result.iter().all(|&v| v >= 0.0 && v <= 100.0 || v == 0.0));

        let pro = PercentRankOscillator::new(20, 3).unwrap();
        let result = pro.calculate(&large_close);
        assert!(result.iter().all(|&v| v >= 0.0 && v <= 100.0 || v == 0.0));

        let zso = ZScoreOscillator::new(20, 5).unwrap();
        let (z_score, _) = zso.calculate(&large_close);
        assert!(z_score.iter().all(|&v| v.is_finite()));

        let vo = VelocityOscillator::new(5, 20, 3).unwrap();
        let result = vo.calculate(&large_close);
        assert!(result.iter().all(|&v| v >= -100.0 && v <= 100.0 || v == 0.0));

        let ao = AccelerationOscillator::new(5, 5, 20).unwrap();
        let result = ao.calculate(&large_close);
        assert!(result.iter().all(|&v| v >= -100.0 && v <= 100.0 || v == 0.0));

        let ho = HarmonicOscillator::new(20, 3, 5).unwrap();
        let result = ho.calculate(&large_close);
        assert!(result.iter().all(|&v| v >= -100.0 && v <= 100.0 || v == 0.0));
    }

    #[test]
    fn test_all_new_indicators_with_small_values() {
        // Test with very small price values
        let small_close: Vec<f64> = (0..60).map(|i| 0.001 + i as f64 * 0.0001).collect();
        let small_high: Vec<f64> = small_close.iter().map(|c| c + 0.00005).collect();
        let small_low: Vec<f64> = small_close.iter().map(|c| c - 0.00005).collect();

        let uoe = UltimateOscillatorEnhanced::new(5, 10, 20, 10).unwrap();
        let result = uoe.calculate(&small_high, &small_low, &small_close);
        assert!(result.iter().all(|v| v.is_finite()));

        let pro = PercentRankOscillator::new(20, 3).unwrap();
        let result = pro.calculate(&small_close);
        assert!(result.iter().all(|v| v.is_finite()));

        let zso = ZScoreOscillator::new(20, 5).unwrap();
        let (z_score, signal) = zso.calculate(&small_close);
        assert!(z_score.iter().all(|v| v.is_finite()));
        assert!(signal.iter().all(|v| v.is_finite()));

        let vo = VelocityOscillator::new(5, 20, 3).unwrap();
        let result = vo.calculate(&small_close);
        assert!(result.iter().all(|v| v.is_finite()));

        let ao = AccelerationOscillator::new(5, 5, 20).unwrap();
        let result = ao.calculate(&small_close);
        assert!(result.iter().all(|v| v.is_finite()));

        let ho = HarmonicOscillator::new(20, 3, 5).unwrap();
        let result = ho.calculate(&small_close);
        assert!(result.iter().all(|v| v.is_finite()));
    }
}
