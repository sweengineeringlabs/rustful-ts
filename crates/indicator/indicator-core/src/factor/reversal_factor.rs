//! Reversal Factor (IND-263)
//!
//! Short-term mean reversion factor that identifies overbought/oversold conditions
//! and potential reversal opportunities based on price deviations from moving averages.
//!
//! # Concept
//! The reversal factor measures how far price has deviated from its recent average,
//! normalized by volatility. Extreme deviations suggest potential mean reversion.
//!
//! # Interpretation
//! - High positive values: Price significantly above average, potential short opportunity
//! - High negative values: Price significantly below average, potential long opportunity
//! - Near zero: Price near equilibrium, no strong reversal signal

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Output from the Reversal Factor calculation.
#[derive(Debug, Clone)]
pub struct ReversalFactorOutput {
    /// Raw reversal factor values (z-score style).
    pub factor: Vec<f64>,
    /// Smoothed reversal factor for signal generation.
    pub signal: Vec<f64>,
    /// Reversal strength (0-100 scale).
    pub strength: Vec<f64>,
}

/// Signal interpretation for reversal factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReversalSignal {
    /// Strong bullish reversal expected (oversold).
    StrongBullish,
    /// Moderate bullish reversal expected.
    Bullish,
    /// No significant reversal signal.
    Neutral,
    /// Moderate bearish reversal expected.
    Bearish,
    /// Strong bearish reversal expected (overbought).
    StrongBearish,
}

/// Reversal Factor (IND-263)
///
/// Measures short-term mean reversion potential based on price deviation
/// from moving average, normalized by volatility.
///
/// # Formula
/// ```text
/// Deviation = Close - SMA(Close, period)
/// StdDev = StandardDeviation(Close, period)
/// ReversalFactor = Deviation / StdDev
/// Signal = EMA(ReversalFactor, signal_period)
/// Strength = |ReversalFactor| / threshold * 100, capped at 100
/// ```
///
/// # Example
/// ```
/// use indicator_core::factor::ReversalFactor;
///
/// let rf = ReversalFactor::new(20, 5, 2.0).unwrap();
/// let close = vec![100.0, 101.0, 99.0, 98.0, 102.0];
/// let output = rf.calculate(&close);
/// ```
#[derive(Debug, Clone)]
pub struct ReversalFactor {
    /// Lookback period for mean and deviation calculation.
    period: usize,
    /// Period for signal smoothing (EMA).
    signal_period: usize,
    /// Threshold for strong reversal signals (in standard deviations).
    threshold: f64,
}

impl ReversalFactor {
    /// Create a new Reversal Factor indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for mean calculation (minimum 5)
    /// * `signal_period` - Period for EMA signal smoothing (minimum 2)
    /// * `threshold` - Threshold in standard deviations for strong signals (typically 2.0)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid.
    pub fn new(period: usize, signal_period: usize, threshold: f64) -> Result<Self> {
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
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            period,
            signal_period,
            threshold,
        })
    }

    /// Create with default parameters (20, 5, 2.0).
    pub fn default_params() -> Result<Self> {
        Self::new(20, 5, 2.0)
    }

    /// Calculate the reversal factor values.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// ReversalFactorOutput containing factor, signal, and strength values.
    pub fn calculate(&self, close: &[f64]) -> ReversalFactorOutput {
        let n = close.len();
        let mut factor = vec![0.0; n];
        let mut signal = vec![0.0; n];
        let mut strength = vec![0.0; n];

        if n < self.period {
            return ReversalFactorOutput { factor, signal, strength };
        }

        // Calculate rolling mean and standard deviation
        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &close[start..=i];

            // Calculate mean
            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;

            // Calculate standard deviation
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            // Calculate reversal factor (z-score)
            if std_dev > 1e-10 {
                factor[i] = (close[i] - mean) / std_dev;
            }

            // Calculate strength (0-100 scale)
            strength[i] = (factor[i].abs() / self.threshold * 100.0).min(100.0);
        }

        // Calculate EMA signal
        let alpha = 2.0 / (self.signal_period as f64 + 1.0);
        signal[self.period - 1] = factor[self.period - 1];
        for i in self.period..n {
            signal[i] = alpha * factor[i] + (1.0 - alpha) * signal[i - 1];
        }

        ReversalFactorOutput { factor, signal, strength }
    }

    /// Calculate reversal factor with volume weighting.
    ///
    /// Volume-weighted version gives more importance to deviations on high-volume days.
    pub fn calculate_volume_weighted(&self, close: &[f64], volume: &[f64]) -> ReversalFactorOutput {
        let n = close.len().min(volume.len());
        let mut factor = vec![0.0; n];
        let mut signal = vec![0.0; n];
        let mut strength = vec![0.0; n];

        if n < self.period {
            return ReversalFactorOutput { factor, signal, strength };
        }

        // Calculate volume-weighted rolling mean and standard deviation
        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;

            // Volume-weighted mean
            let mut vol_sum = 0.0;
            let mut vwap = 0.0;
            for j in start..=i {
                vwap += close[j] * volume[j];
                vol_sum += volume[j];
            }
            let mean = if vol_sum > 1e-10 { vwap / vol_sum } else { 0.0 };

            // Volume-weighted standard deviation
            let mut weighted_var = 0.0;
            for j in start..=i {
                weighted_var += volume[j] * (close[j] - mean).powi(2);
            }
            let std_dev = if vol_sum > 1e-10 {
                (weighted_var / vol_sum).sqrt()
            } else {
                0.0
            };

            // Calculate volume-weighted reversal factor
            if std_dev > 1e-10 {
                // Weight current deviation by relative volume
                let avg_vol = vol_sum / self.period as f64;
                let vol_weight = if avg_vol > 1e-10 {
                    (volume[i] / avg_vol).sqrt()
                } else {
                    1.0
                };
                factor[i] = (close[i] - mean) / std_dev * vol_weight;
            }

            strength[i] = (factor[i].abs() / self.threshold * 100.0).min(100.0);
        }

        // Calculate EMA signal
        let alpha = 2.0 / (self.signal_period as f64 + 1.0);
        signal[self.period - 1] = factor[self.period - 1];
        for i in self.period..n {
            signal[i] = alpha * factor[i] + (1.0 - alpha) * signal[i - 1];
        }

        ReversalFactorOutput { factor, signal, strength }
    }

    /// Get signal interpretation for a reversal factor value.
    pub fn interpret(&self, factor_value: f64) -> ReversalSignal {
        let half_threshold = self.threshold / 2.0;

        if factor_value <= -self.threshold {
            ReversalSignal::StrongBullish
        } else if factor_value <= -half_threshold {
            ReversalSignal::Bullish
        } else if factor_value >= self.threshold {
            ReversalSignal::StrongBearish
        } else if factor_value >= half_threshold {
            ReversalSignal::Bearish
        } else {
            ReversalSignal::Neutral
        }
    }

    /// Get signals for all reversal factor values.
    pub fn signals(&self, output: &ReversalFactorOutput) -> Vec<ReversalSignal> {
        output.signal.iter().map(|&v| self.interpret(v)).collect()
    }

    /// Get the lookback period.
    pub fn period(&self) -> usize {
        self.period
    }

    /// Get the signal period.
    pub fn signal_period(&self) -> usize {
        self.signal_period
    }

    /// Get the threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

impl Default for ReversalFactor {
    fn default() -> Self {
        Self {
            period: 20,
            signal_period: 5,
            threshold: 2.0,
        }
    }
}

impl TechnicalIndicator for ReversalFactor {
    fn name(&self) -> &str {
        "Reversal Factor"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(output.factor, output.signal, output.strength))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        // Create price series with mean reversion characteristics
        let mut prices = Vec::with_capacity(50);
        let base = 100.0;
        for i in 0..50 {
            // Oscillating around base with some trend
            let deviation = (i as f64 * 0.3).sin() * 5.0;
            let trend = i as f64 * 0.1;
            prices.push(base + trend + deviation);
        }
        prices
    }

    fn make_ohlcv_data() -> OHLCVSeries {
        let close = make_test_data();
        let n = close.len();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open = close.clone();
        let volume: Vec<f64> = (0..n).map(|i| 1000.0 + (i as f64 * 0.5).sin() * 500.0).collect();

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_reversal_factor_basic() {
        let close = make_test_data();
        let rf = ReversalFactor::new(10, 3, 2.0).unwrap();
        let output = rf.calculate(&close);

        assert_eq!(output.factor.len(), close.len());
        assert_eq!(output.signal.len(), close.len());
        assert_eq!(output.strength.len(), close.len());

        // First period-1 values should be zero
        for i in 0..9 {
            assert_eq!(output.factor[i], 0.0);
        }

        // Values after warmup should be non-zero
        assert!(output.factor[20].abs() > 0.0);
    }

    #[test]
    fn test_reversal_factor_extreme_deviation() {
        // Create data with extreme deviation
        let mut close = vec![100.0; 30];
        // Add spike at the end
        close[29] = 120.0;

        let rf = ReversalFactor::new(20, 5, 2.0).unwrap();
        let output = rf.calculate(&close);

        // The spike should produce a high positive factor (overbought)
        assert!(output.factor[29] > 2.0);
        assert_eq!(output.strength[29], 100.0); // Capped at 100
    }

    #[test]
    fn test_reversal_factor_signal_smoothing() {
        let close = make_test_data();
        let rf = ReversalFactor::new(10, 5, 2.0).unwrap();
        let output = rf.calculate(&close);

        // Signal should be smoother than raw factor
        let mut factor_changes = 0.0;
        let mut signal_changes = 0.0;

        for i in 15..close.len() {
            factor_changes += (output.factor[i] - output.factor[i - 1]).abs();
            signal_changes += (output.signal[i] - output.signal[i - 1]).abs();
        }

        assert!(signal_changes < factor_changes);
    }

    #[test]
    fn test_reversal_factor_volume_weighted() {
        let close = make_test_data();
        let volume: Vec<f64> = (0..close.len())
            .map(|i| 1000.0 + (i % 5) as f64 * 200.0)
            .collect();

        let rf = ReversalFactor::new(10, 3, 2.0).unwrap();
        let regular = rf.calculate(&close);
        let vol_weighted = rf.calculate_volume_weighted(&close, &volume);

        // Volume weighted should produce different results
        let mut diff_count = 0;
        for i in 15..close.len() {
            if (regular.factor[i] - vol_weighted.factor[i]).abs() > 0.01 {
                diff_count += 1;
            }
        }
        assert!(diff_count > 0);
    }

    #[test]
    fn test_reversal_factor_interpretation() {
        let rf = ReversalFactor::new(20, 5, 2.0).unwrap();

        assert_eq!(rf.interpret(-2.5), ReversalSignal::StrongBullish);
        assert_eq!(rf.interpret(-1.5), ReversalSignal::Bullish);
        assert_eq!(rf.interpret(0.0), ReversalSignal::Neutral);
        assert_eq!(rf.interpret(1.5), ReversalSignal::Bearish);
        assert_eq!(rf.interpret(2.5), ReversalSignal::StrongBearish);
    }

    #[test]
    fn test_reversal_factor_strength_bounded() {
        let close = make_test_data();
        let rf = ReversalFactor::new(10, 3, 2.0).unwrap();
        let output = rf.calculate(&close);

        for strength in output.strength.iter() {
            assert!(*strength >= 0.0 && *strength <= 100.0);
        }
    }

    #[test]
    fn test_reversal_factor_technical_indicator() {
        let data = make_ohlcv_data();
        let rf = ReversalFactor::new(10, 3, 2.0).unwrap();

        assert_eq!(rf.name(), "Reversal Factor");
        assert_eq!(rf.min_periods(), 10);

        let output = rf.compute(&data).unwrap();
        assert!(output.values.contains_key("factor"));
        assert!(output.values.contains_key("signal"));
        assert!(output.values.contains_key("strength"));
    }

    #[test]
    fn test_reversal_factor_validation() {
        assert!(ReversalFactor::new(4, 3, 2.0).is_err()); // period too small
        assert!(ReversalFactor::new(10, 1, 2.0).is_err()); // signal_period too small
        assert!(ReversalFactor::new(10, 3, 0.0).is_err()); // threshold not positive
        assert!(ReversalFactor::new(10, 3, -1.0).is_err()); // threshold negative
    }

    #[test]
    fn test_reversal_factor_empty_input() {
        let rf = ReversalFactor::default();
        let output = rf.calculate(&[]);

        assert!(output.factor.is_empty());
        assert!(output.signal.is_empty());
        assert!(output.strength.is_empty());
    }

    #[test]
    fn test_reversal_factor_insufficient_data() {
        let rf = ReversalFactor::new(20, 5, 2.0).unwrap();
        let close = vec![100.0; 10]; // Less than period

        let output = rf.calculate(&close);

        // All values should be zero due to insufficient data
        for v in output.factor.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_reversal_factor_default() {
        let rf = ReversalFactor::default();
        assert_eq!(rf.period(), 20);
        assert_eq!(rf.signal_period(), 5);
        assert_eq!(rf.threshold(), 2.0);
    }
}
