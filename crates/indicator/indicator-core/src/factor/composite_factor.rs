//! Composite Factor Score (IND-264)
//!
//! Multi-factor combination indicator that combines momentum, value, quality,
//! and volatility factors into a single composite score.
//!
//! # Concept
//! Combines multiple investment factors commonly used in quantitative investing:
//! - Momentum: Price trend strength and persistence
//! - Value: Mean reversion/relative value
//! - Quality: Trend consistency and smoothness
//! - Low Volatility: Inverse of price volatility
//!
//! # Interpretation
//! - High positive scores indicate strong bullish factor alignment
//! - High negative scores indicate strong bearish factor alignment
//! - Near zero indicates mixed or neutral factor signals

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Weights for each factor component.
#[derive(Debug, Clone)]
pub struct FactorWeights {
    /// Weight for momentum factor (0.0-1.0).
    pub momentum: f64,
    /// Weight for value/mean-reversion factor (0.0-1.0).
    pub value: f64,
    /// Weight for quality factor (0.0-1.0).
    pub quality: f64,
    /// Weight for low-volatility factor (0.0-1.0).
    pub low_volatility: f64,
}

impl FactorWeights {
    /// Create new factor weights.
    ///
    /// Weights will be normalized to sum to 1.0.
    pub fn new(momentum: f64, value: f64, quality: f64, low_volatility: f64) -> Self {
        let total = momentum + value + quality + low_volatility;
        if total > 0.0 {
            Self {
                momentum: momentum / total,
                value: value / total,
                quality: quality / total,
                low_volatility: low_volatility / total,
            }
        } else {
            Self::default()
        }
    }

    /// Create equal-weighted factors.
    pub fn equal_weighted() -> Self {
        Self {
            momentum: 0.25,
            value: 0.25,
            quality: 0.25,
            low_volatility: 0.25,
        }
    }

    /// Create momentum-tilted weights.
    pub fn momentum_tilt() -> Self {
        Self::new(0.4, 0.2, 0.2, 0.2)
    }

    /// Create value-tilted weights.
    pub fn value_tilt() -> Self {
        Self::new(0.2, 0.4, 0.2, 0.2)
    }

    /// Create quality-tilted weights.
    pub fn quality_tilt() -> Self {
        Self::new(0.2, 0.2, 0.4, 0.2)
    }

    /// Create defensive/low-volatility tilted weights.
    pub fn defensive_tilt() -> Self {
        Self::new(0.15, 0.25, 0.25, 0.35)
    }
}

impl Default for FactorWeights {
    fn default() -> Self {
        Self::equal_weighted()
    }
}

/// Output from the Composite Factor Score calculation.
#[derive(Debug, Clone)]
pub struct CompositeFactorOutput {
    /// Composite factor score (weighted combination of all factors).
    pub composite: Vec<f64>,
    /// Individual momentum factor values.
    pub momentum: Vec<f64>,
    /// Individual value factor values.
    pub value: Vec<f64>,
    /// Individual quality factor values.
    pub quality: Vec<f64>,
    /// Individual low-volatility factor values.
    pub low_volatility: Vec<f64>,
    /// Signal strength (0-100).
    pub signal_strength: Vec<f64>,
}

/// Composite Factor Score (IND-264)
///
/// Multi-factor combination indicator combining momentum, value, quality,
/// and volatility factors.
///
/// # Calculation
/// ```text
/// Momentum = (Close - Close[momentum_period]) / Close[momentum_period] * 100, normalized
/// Value = (SMA - Close) / StdDev (inverted z-score, buying dips)
/// Quality = R-squared of linear regression over quality_period
/// LowVol = -StdDev / historical_avg_stddev + 1 (higher when vol is low)
///
/// Composite = w_m * Momentum + w_v * Value + w_q * Quality + w_lv * LowVol
/// ```
///
/// # Example
/// ```
/// use indicator_core::factor::{CompositeFactorScore, FactorWeights};
///
/// let cfs = CompositeFactorScore::new(20, 10, 20, FactorWeights::equal_weighted()).unwrap();
/// let close = vec![100.0, 101.0, 99.0, 102.0, 103.0];
/// let output = cfs.calculate(&close);
/// ```
#[derive(Debug, Clone)]
pub struct CompositeFactorScore {
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Period for value/mean-reversion calculation.
    value_period: usize,
    /// Period for quality calculation.
    quality_period: usize,
    /// Factor weights.
    weights: FactorWeights,
}

impl CompositeFactorScore {
    /// Create a new Composite Factor Score indicator.
    ///
    /// # Arguments
    /// * `momentum_period` - Lookback for momentum calculation (minimum 5)
    /// * `value_period` - Lookback for value calculation (minimum 5)
    /// * `quality_period` - Lookback for quality calculation (minimum 10)
    /// * `weights` - Factor weights
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid.
    pub fn new(
        momentum_period: usize,
        value_period: usize,
        quality_period: usize,
        weights: FactorWeights,
    ) -> Result<Self> {
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if value_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "value_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if quality_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "quality_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            value_period,
            quality_period,
            weights,
        })
    }

    /// Create with default parameters.
    pub fn default_params() -> Result<Self> {
        Self::new(20, 10, 20, FactorWeights::equal_weighted())
    }

    /// Get the maximum lookback period required.
    fn max_period(&self) -> usize {
        self.momentum_period
            .max(self.value_period)
            .max(self.quality_period)
    }

    /// Calculate the composite factor score.
    ///
    /// # Arguments
    /// * `close` - Slice of closing prices
    ///
    /// # Returns
    /// CompositeFactorOutput containing all factor values.
    pub fn calculate(&self, close: &[f64]) -> CompositeFactorOutput {
        let n = close.len();
        let max_period = self.max_period();

        let mut composite = vec![0.0; n];
        let mut momentum = vec![0.0; n];
        let mut value = vec![0.0; n];
        let mut quality = vec![0.0; n];
        let mut low_volatility = vec![0.0; n];
        let mut signal_strength = vec![0.0; n];

        if n < max_period {
            return CompositeFactorOutput {
                composite,
                momentum,
                value,
                quality,
                low_volatility,
                signal_strength,
            };
        }

        // Calculate each factor
        self.calculate_momentum_factor(close, &mut momentum);
        self.calculate_value_factor(close, &mut value);
        self.calculate_quality_factor(close, &mut quality);
        self.calculate_low_volatility_factor(close, &mut low_volatility);

        // Combine factors with weights
        for i in max_period..n {
            composite[i] = self.weights.momentum * momentum[i]
                + self.weights.value * value[i]
                + self.weights.quality * quality[i]
                + self.weights.low_volatility * low_volatility[i];

            // Calculate signal strength based on factor alignment
            let factors = [momentum[i], value[i], quality[i], low_volatility[i]];
            let positive_count = factors.iter().filter(|&&x| x > 0.0).count();
            let negative_count = factors.iter().filter(|&&x| x < 0.0).count();

            // Higher strength when factors agree
            let alignment = (positive_count.max(negative_count) as f64) / 4.0;
            signal_strength[i] = alignment * composite[i].abs().min(1.0) * 100.0;
        }

        CompositeFactorOutput {
            composite,
            momentum,
            value,
            quality,
            low_volatility,
            signal_strength,
        }
    }

    /// Calculate momentum factor.
    fn calculate_momentum_factor(&self, close: &[f64], momentum: &mut [f64]) {
        let n = close.len();

        // Calculate raw returns
        let mut returns = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period].abs() > 1e-10 {
                returns[i] = (close[i] / close[i - self.momentum_period] - 1.0) * 100.0;
            }
        }

        // Z-score normalize the returns
        if n > self.momentum_period {
            let valid_returns: Vec<f64> = returns[self.momentum_period..].to_vec();
            if !valid_returns.is_empty() {
                let mean: f64 = valid_returns.iter().sum::<f64>() / valid_returns.len() as f64;
                let variance: f64 = valid_returns.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / valid_returns.len() as f64;
                let std_dev = variance.sqrt();

                for i in self.momentum_period..n {
                    if std_dev > 1e-10 {
                        momentum[i] = (returns[i] - mean) / std_dev;
                    }
                }
            }
        }
    }

    /// Calculate value (mean-reversion) factor.
    fn calculate_value_factor(&self, close: &[f64], value: &mut [f64]) {
        let n = close.len();

        for i in (self.value_period - 1)..n {
            let start = i + 1 - self.value_period;
            let window = &close[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / self.value_period as f64;
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.value_period as f64;
            let std_dev = variance.sqrt();

            // Value factor is inverted - we want to buy when price is below mean
            if std_dev > 1e-10 {
                value[i] = (mean - close[i]) / std_dev;
            }
        }
    }

    /// Calculate quality factor (trend consistency via R-squared).
    fn calculate_quality_factor(&self, close: &[f64], quality: &mut [f64]) {
        let n = close.len();

        for i in (self.quality_period - 1)..n {
            let start = i + 1 - self.quality_period;
            let window = &close[start..=i];

            // Calculate R-squared of linear regression
            let r_squared = self.calculate_r_squared(window);

            // Determine trend direction
            let trend_direction = if close[i] > close[start] { 1.0 } else { -1.0 };

            // Quality factor: R-squared * trend direction
            // Positive when trending up with high R², negative when trending down with high R²
            quality[i] = r_squared * trend_direction;
        }
    }

    /// Calculate R-squared for a price series.
    fn calculate_r_squared(&self, prices: &[f64]) -> f64 {
        let n = prices.len();
        if n < 2 {
            return 0.0;
        }

        // Calculate means
        let x_mean = (n - 1) as f64 / 2.0;
        let y_mean: f64 = prices.iter().sum::<f64>() / n as f64;

        // Calculate sums for regression
        let mut ss_xy = 0.0;
        let mut ss_xx = 0.0;
        let mut ss_yy = 0.0;

        for (i, &y) in prices.iter().enumerate() {
            let x = i as f64;
            ss_xy += (x - x_mean) * (y - y_mean);
            ss_xx += (x - x_mean).powi(2);
            ss_yy += (y - y_mean).powi(2);
        }

        // Calculate R-squared
        if ss_xx.abs() > 1e-10 && ss_yy.abs() > 1e-10 {
            let r = ss_xy / (ss_xx.sqrt() * ss_yy.sqrt());
            r.powi(2)
        } else {
            0.0
        }
    }

    /// Calculate low-volatility factor.
    fn calculate_low_volatility_factor(&self, close: &[f64], low_volatility: &mut [f64]) {
        let n = close.len();
        let vol_period = self.value_period; // Use same period as value

        // Calculate rolling volatility
        let mut volatility = vec![0.0; n];
        for i in vol_period..n {
            let start = i + 1 - vol_period;
            let window = &close[start..=i];

            // Calculate standard deviation of returns
            let mut returns = Vec::with_capacity(vol_period - 1);
            for j in 1..window.len() {
                if window[j - 1].abs() > 1e-10 {
                    returns.push((window[j] / window[j - 1] - 1.0) * 100.0);
                }
            }

            if !returns.is_empty() {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance: f64 = returns.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                volatility[i] = variance.sqrt();
            }
        }

        // Calculate average volatility for normalization
        let valid_vol: Vec<f64> = volatility[vol_period..].iter().filter(|&&v| v > 0.0).copied().collect();
        let avg_vol = if !valid_vol.is_empty() {
            valid_vol.iter().sum::<f64>() / valid_vol.len() as f64
        } else {
            1.0
        };

        // Low volatility factor: higher when volatility is below average
        for i in vol_period..n {
            if avg_vol > 1e-10 {
                low_volatility[i] = 1.0 - volatility[i] / avg_vol;
            }
        }
    }

    /// Calculate with custom volume weighting for momentum.
    pub fn calculate_with_volume(&self, close: &[f64], volume: &[f64]) -> CompositeFactorOutput {
        let n = close.len().min(volume.len());
        let mut output = self.calculate(&close[..n]);

        // Adjust momentum factor by volume confirmation
        for i in self.momentum_period..n {
            let start = i.saturating_sub(self.momentum_period);
            let avg_vol: f64 = volume[start..=i].iter().sum::<f64>() / self.momentum_period as f64;

            if avg_vol > 1e-10 {
                let vol_ratio = volume[i] / avg_vol;
                // Boost momentum when confirmed by high volume
                output.momentum[i] *= 1.0 + (vol_ratio - 1.0) * 0.5;
            }
        }

        // Recalculate composite
        let max_period = self.max_period();
        for i in max_period..n {
            output.composite[i] = self.weights.momentum * output.momentum[i]
                + self.weights.value * output.value[i]
                + self.weights.quality * output.quality[i]
                + self.weights.low_volatility * output.low_volatility[i];
        }

        output
    }

    /// Get the momentum period.
    pub fn momentum_period(&self) -> usize {
        self.momentum_period
    }

    /// Get the value period.
    pub fn value_period(&self) -> usize {
        self.value_period
    }

    /// Get the quality period.
    pub fn quality_period(&self) -> usize {
        self.quality_period
    }

    /// Get the factor weights.
    pub fn weights(&self) -> &FactorWeights {
        &self.weights
    }
}

impl Default for CompositeFactorScore {
    fn default() -> Self {
        Self {
            momentum_period: 20,
            value_period: 10,
            quality_period: 20,
            weights: FactorWeights::default(),
        }
    }
}

impl TechnicalIndicator for CompositeFactorScore {
    fn name(&self) -> &str {
        "Composite Factor Score"
    }

    fn min_periods(&self) -> usize {
        self.max_period()
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate_with_volume(&data.close, &data.volume);
        Ok(IndicatorOutput::triple(output.composite, output.momentum, output.value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        // Create trending price series with some noise
        (0..50)
            .map(|i| 100.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin() * 2.0)
            .collect()
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
    fn test_composite_factor_basic() {
        let close = make_test_data();
        let cfs = CompositeFactorScore::default_params().unwrap();
        let output = cfs.calculate(&close);

        assert_eq!(output.composite.len(), close.len());
        assert_eq!(output.momentum.len(), close.len());
        assert_eq!(output.value.len(), close.len());
        assert_eq!(output.quality.len(), close.len());
        assert_eq!(output.low_volatility.len(), close.len());
    }

    #[test]
    fn test_composite_factor_trending() {
        // Strong uptrend
        let close: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 2.0).collect();
        let cfs = CompositeFactorScore::new(10, 5, 15, FactorWeights::equal_weighted()).unwrap();
        let output = cfs.calculate(&close);

        // In uptrend, momentum and quality should be positive
        let last = output.composite.len() - 1;
        assert!(output.momentum[last] > 0.0);
        assert!(output.quality[last] > 0.0);
    }

    #[test]
    fn test_composite_factor_mean_reverting() {
        // Price below its mean
        let mut close = vec![100.0; 30];
        close[29] = 90.0; // Drop at the end

        let cfs = CompositeFactorScore::new(10, 10, 15, FactorWeights::equal_weighted()).unwrap();
        let output = cfs.calculate(&close);

        // Value factor should be positive (buying opportunity)
        assert!(output.value[29] > 0.0);
    }

    #[test]
    fn test_factor_weights_normalization() {
        let weights = FactorWeights::new(2.0, 2.0, 2.0, 2.0);

        // Should be normalized to equal weights
        assert!((weights.momentum - 0.25).abs() < 1e-10);
        assert!((weights.value - 0.25).abs() < 1e-10);
        assert!((weights.quality - 0.25).abs() < 1e-10);
        assert!((weights.low_volatility - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_factor_weights_presets() {
        let momentum_tilt = FactorWeights::momentum_tilt();
        assert!(momentum_tilt.momentum > momentum_tilt.value);

        let value_tilt = FactorWeights::value_tilt();
        assert!(value_tilt.value > value_tilt.momentum);

        let quality_tilt = FactorWeights::quality_tilt();
        assert!(quality_tilt.quality > quality_tilt.momentum);

        let defensive = FactorWeights::defensive_tilt();
        assert!(defensive.low_volatility > defensive.momentum);
    }

    #[test]
    fn test_composite_factor_with_volume() {
        let close = make_test_data();
        let volume: Vec<f64> = (0..close.len())
            .map(|i| 1000.0 + (i % 5) as f64 * 200.0)
            .collect();

        let cfs = CompositeFactorScore::default_params().unwrap();
        let with_vol = cfs.calculate_with_volume(&close, &volume);
        let without_vol = cfs.calculate(&close);

        // Volume adjustment should cause some differences
        let mut diff_count = 0;
        for i in 25..close.len() {
            if (with_vol.momentum[i] - without_vol.momentum[i]).abs() > 0.001 {
                diff_count += 1;
            }
        }
        assert!(diff_count > 0);
    }

    #[test]
    fn test_composite_factor_signal_strength() {
        let close = make_test_data();
        let cfs = CompositeFactorScore::default_params().unwrap();
        let output = cfs.calculate(&close);

        // Signal strength should be bounded 0-100
        for strength in output.signal_strength.iter() {
            assert!(*strength >= 0.0 && *strength <= 100.0);
        }
    }

    #[test]
    fn test_composite_factor_technical_indicator() {
        let data = make_ohlcv_data();
        let cfs = CompositeFactorScore::default_params().unwrap();

        assert_eq!(cfs.name(), "Composite Factor Score");
        assert_eq!(cfs.min_periods(), 20); // max of all periods

        let output = cfs.compute(&data).unwrap();
        assert!(output.values.contains_key("composite"));
        assert!(output.values.contains_key("momentum"));
        assert!(output.values.contains_key("value"));
        assert!(output.values.contains_key("quality"));
        assert!(output.values.contains_key("low_volatility"));
    }

    #[test]
    fn test_composite_factor_validation() {
        assert!(CompositeFactorScore::new(4, 10, 20, FactorWeights::default()).is_err());
        assert!(CompositeFactorScore::new(10, 4, 20, FactorWeights::default()).is_err());
        assert!(CompositeFactorScore::new(10, 10, 9, FactorWeights::default()).is_err());
    }

    #[test]
    fn test_composite_factor_empty_input() {
        let cfs = CompositeFactorScore::default();
        let output = cfs.calculate(&[]);

        assert!(output.composite.is_empty());
        assert!(output.momentum.is_empty());
    }

    #[test]
    fn test_composite_factor_insufficient_data() {
        let cfs = CompositeFactorScore::new(20, 10, 20, FactorWeights::default()).unwrap();
        let close = vec![100.0; 15]; // Less than max period

        let output = cfs.calculate(&close);

        // All composite values should be zero
        for v in output.composite.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_r_squared_calculation() {
        let cfs = CompositeFactorScore::default();

        // Perfect linear trend should have R² = 1
        let perfect_trend: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let r_sq = cfs.calculate_r_squared(&perfect_trend);
        assert!((r_sq - 1.0).abs() < 0.01);

        // Random/noisy data should have lower R²
        let noisy: Vec<f64> = vec![1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0, 7.0];
        let r_sq_noisy = cfs.calculate_r_squared(&noisy);
        assert!(r_sq_noisy < 0.5);
    }
}
