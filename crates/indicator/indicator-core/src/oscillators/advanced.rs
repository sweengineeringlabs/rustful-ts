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
}
