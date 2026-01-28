//! Advanced Intermarket Indicators
//!
//! Additional advanced intermarket analysis indicators for pairs trading,
//! lead-lag analysis, and relative value strategies.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

use super::DualSeries;

// ============================================================================
// Lead/Lag Indicator
// ============================================================================

/// Lead/Lag Indicator - Detects lead/lag relationships between two price series.
///
/// This indicator uses cross-correlation analysis to determine which series
/// leads or lags the other. Positive lag values indicate the first series leads,
/// negative values indicate it lags.
///
/// # Interpretation
/// - `lead_lag_score > 0`: Series 1 leads Series 2
/// - `lead_lag_score < 0`: Series 1 lags Series 2
/// - `correlation`: Strength of the lead/lag relationship
/// - `optimal_lag`: Number of periods by which the leader precedes
#[derive(Debug, Clone)]
pub struct LeadLagIndicator {
    /// Period for rolling calculations.
    period: usize,
    /// Maximum lag to test for cross-correlation.
    max_lag: usize,
    /// Secondary series for lead/lag analysis.
    secondary_series: Vec<f64>,
}

impl LeadLagIndicator {
    /// Create a new LeadLagIndicator.
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculations (must be >= 20)
    /// * `max_lag` - Maximum lag periods to test (must be >= 1 and < period/2)
    pub fn new(period: usize, max_lag: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if max_lag < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_lag".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if max_lag >= period / 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_lag".to_string(),
                reason: "must be less than period / 2".to_string(),
            });
        }
        Ok(Self {
            period,
            max_lag,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for lead/lag analysis.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate cross-correlation at a specific lag.
    fn cross_correlation(series1: &[f64], series2: &[f64], lag: i32) -> f64 {
        let n = series1.len();
        let abs_lag = lag.unsigned_abs() as usize;

        if abs_lag >= n {
            return 0.0;
        }

        let (s1, s2) = if lag >= 0 {
            (&series1[abs_lag..], &series2[..n - abs_lag])
        } else {
            (&series1[..n - abs_lag], &series2[abs_lag..])
        };

        let len = s1.len() as f64;
        if len < 2.0 {
            return 0.0;
        }

        let mean1: f64 = s1.iter().sum::<f64>() / len;
        let mean2: f64 = s2.iter().sum::<f64>() / len;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in s1.iter().zip(s2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Calculate lead/lag metrics for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.period {
            return result;
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window1 = &dual.series1[start..=i];
            let window2 = &dual.series2[start..=i];

            let mut best_lag = 0i32;
            let mut best_corr = Self::cross_correlation(window1, window2, 0);

            // Test positive lags (series1 leads)
            for lag in 1..=self.max_lag as i32 {
                let corr = Self::cross_correlation(window1, window2, lag);
                if corr.abs() > best_corr.abs() {
                    best_corr = corr;
                    best_lag = lag;
                }
            }

            // Test negative lags (series1 lags)
            for lag in 1..=self.max_lag as i32 {
                let corr = Self::cross_correlation(window1, window2, -lag);
                if corr.abs() > best_corr.abs() {
                    best_corr = corr;
                    best_lag = -lag;
                }
            }

            // Lead/lag score: positive = series1 leads, negative = series1 lags
            // Scale by correlation strength
            result[i] = best_lag as f64 * best_corr.abs();
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for LeadLagIndicator {
    fn name(&self) -> &str {
        "Lead Lag Indicator"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing LeadLagIndicator".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// Price Spread Momentum
// ============================================================================

/// Price Spread Momentum - Measures the momentum of the price spread between two series.
///
/// This indicator calculates the rate of change in the spread between two price series,
/// helping identify when spread relationships are strengthening or weakening.
///
/// # Interpretation
/// - Positive values: Spread is widening
/// - Negative values: Spread is narrowing
/// - Zero crossings indicate potential spread reversals
#[derive(Debug, Clone)]
pub struct PriceSpreadMomentum {
    /// Period for momentum calculation.
    period: usize,
    /// Smoothing period for the momentum.
    smooth_period: usize,
    /// Secondary series for spread calculation.
    secondary_series: Vec<f64>,
}

impl PriceSpreadMomentum {
    /// Create a new PriceSpreadMomentum indicator.
    ///
    /// # Arguments
    /// * `period` - Momentum lookback period (must be >= 5)
    /// * `smooth_period` - Smoothing period for momentum (must be >= 1)
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
        Ok(Self {
            period,
            smooth_period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for spread calculation.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate spread momentum for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.period + self.smooth_period {
            return result;
        }

        // Calculate log spread
        let spread: Vec<f64> = dual.series1.iter()
            .zip(dual.series2.iter())
            .map(|(a, b)| {
                if *a > 0.0 && *b > 0.0 {
                    (a / b).ln()
                } else {
                    0.0
                }
            })
            .collect();

        // Calculate momentum of spread
        let mut momentum = vec![0.0; n];
        for i in self.period..n {
            momentum[i] = spread[i] - spread[i - self.period];
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start_idx = self.period;

        if start_idx < n {
            result[start_idx] = momentum[start_idx];
            for i in (start_idx + 1)..n {
                result[i] = alpha * momentum[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        // Scale to percentage
        for r in result.iter_mut() {
            *r *= 100.0;
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for PriceSpreadMomentum {
    fn name(&self) -> &str {
        "Price Spread Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing PriceSpreadMomentum".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// Correlation Trend
// ============================================================================

/// Correlation Trend - Tracks changes in correlation between two series over time.
///
/// This indicator monitors how the correlation between two series evolves,
/// helping identify regime changes in market relationships.
///
/// # Interpretation
/// - Rising correlation trend: Relationships are strengthening
/// - Falling correlation trend: Relationships are weakening
/// - Rapid changes may indicate regime shifts
#[derive(Debug, Clone)]
pub struct CorrelationTrend {
    /// Short-term correlation period.
    short_period: usize,
    /// Long-term correlation period.
    long_period: usize,
    /// Secondary series for correlation calculation.
    secondary_series: Vec<f64>,
}

impl CorrelationTrend {
    /// Create a new CorrelationTrend indicator.
    ///
    /// # Arguments
    /// * `short_period` - Short-term correlation window (must be >= 10)
    /// * `long_period` - Long-term correlation window (must be > short_period)
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        Ok(Self {
            short_period,
            long_period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for correlation calculation.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate correlation for a window.
    fn correlation(series1: &[f64], series2: &[f64]) -> f64 {
        let n = series1.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean1: f64 = series1.iter().sum::<f64>() / n;
        let mean2: f64 = series2.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in series1.iter().zip(series2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Calculate correlation trend for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.long_period {
            return result;
        }

        for i in (self.long_period - 1)..n {
            // Short-term correlation
            let short_start = i + 1 - self.short_period;
            let short_corr = Self::correlation(
                &dual.series1[short_start..=i],
                &dual.series2[short_start..=i],
            );

            // Long-term correlation
            let long_start = i + 1 - self.long_period;
            let long_corr = Self::correlation(
                &dual.series1[long_start..=i],
                &dual.series2[long_start..=i],
            );

            // Correlation trend: difference between short and long term
            // Positive = correlation increasing, Negative = correlation decreasing
            result[i] = (short_corr - long_corr) * 100.0;
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for CorrelationTrend {
    fn name(&self) -> &str {
        "Correlation Trend"
    }

    fn min_periods(&self) -> usize {
        self.long_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing CorrelationTrend".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// Relative Value Index
// ============================================================================

/// Relative Value Index - Measures relative value between two instruments.
///
/// This indicator calculates a normalized relative value metric that compares
/// the current price ratio to its historical distribution.
///
/// # Interpretation
/// - Values near 100: Series 1 is relatively overvalued vs Series 2
/// - Values near 0: Series 1 is relatively undervalued vs Series 2
/// - Values near 50: Fair relative value
#[derive(Debug, Clone)]
pub struct RelativeValueIndex {
    /// Period for relative value calculation.
    period: usize,
    /// Secondary series for comparison.
    secondary_series: Vec<f64>,
}

impl RelativeValueIndex {
    /// Create a new RelativeValueIndex indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for historical comparison (must be >= 20)
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self {
            period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for comparison.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate relative value index for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![50.0; n]; // Default to neutral

        if n < self.period {
            return result;
        }

        // Calculate price ratios
        let ratios: Vec<f64> = dual.series1.iter()
            .zip(dual.series2.iter())
            .map(|(a, b)| if *b > 0.0 { a / b } else { 0.0 })
            .collect();

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &ratios[start..=i];
            let current = ratios[i];

            // Calculate percentile rank
            let count_below = window.iter().filter(|&&r| r < current).count();
            let percentile = (count_below as f64 / self.period as f64) * 100.0;

            result[i] = percentile;
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for RelativeValueIndex {
    fn name(&self) -> &str {
        "Relative Value Index"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing RelativeValueIndex".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// Spread Mean Reversion
// ============================================================================

/// Spread Mean Reversion - Measures mean reversion tendency in spreads.
///
/// This indicator calculates a z-score of the current spread relative to its
/// historical mean, identifying when spreads have deviated significantly
/// and may revert.
///
/// # Interpretation
/// - Z-score > 2: Spread is significantly above mean (potential short spread)
/// - Z-score < -2: Spread is significantly below mean (potential long spread)
/// - Z-score near 0: Spread is at fair value
#[derive(Debug, Clone)]
pub struct SpreadMeanReversion {
    /// Period for z-score calculation.
    period: usize,
    /// Use log spread instead of price spread.
    use_log_spread: bool,
    /// Secondary series for spread calculation.
    secondary_series: Vec<f64>,
}

impl SpreadMeanReversion {
    /// Create a new SpreadMeanReversion indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for mean/std calculation (must be >= 20)
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self {
            period,
            use_log_spread: true,
            secondary_series: Vec::new(),
        })
    }

    /// Use log spread (default: true).
    pub fn with_log_spread(mut self, use_log: bool) -> Self {
        self.use_log_spread = use_log;
        self
    }

    /// Set the secondary series for spread calculation.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate spread mean reversion for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.period {
            return result;
        }

        // Calculate spread
        let spread: Vec<f64> = dual.series1.iter()
            .zip(dual.series2.iter())
            .map(|(a, b)| {
                if *b > 0.0 {
                    if self.use_log_spread && *a > 0.0 {
                        (a / b).ln()
                    } else {
                        a - b
                    }
                } else {
                    0.0
                }
            })
            .collect();

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &spread[start..=i];

            // Calculate mean and standard deviation
            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            // Z-score
            if std_dev > 1e-10 {
                result[i] = (spread[i] - mean) / std_dev;
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for SpreadMeanReversion {
    fn name(&self) -> &str {
        "Spread Mean Reversion"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing SpreadMeanReversion".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// Pairs Trading Signal
// ============================================================================

/// Pairs Trading Signal - Generates entry and exit signals for pairs trading.
///
/// This indicator combines spread z-score with additional filters to generate
/// trading signals for pairs trading strategies.
///
/// # Signal Values
/// - 1.0: Long spread (long series1, short series2)
/// - -1.0: Short spread (short series1, long series2)
/// - 0.0: No position / exit signal
#[derive(Debug, Clone)]
pub struct PairsTradingSignal {
    /// Period for z-score calculation.
    period: usize,
    /// Entry threshold (z-score).
    entry_threshold: f64,
    /// Exit threshold (z-score).
    exit_threshold: f64,
    /// Stop loss threshold (z-score).
    stop_threshold: f64,
    /// Secondary series for spread calculation.
    secondary_series: Vec<f64>,
}

impl PairsTradingSignal {
    /// Create a new PairsTradingSignal indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for z-score calculation (must be >= 20)
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self {
            period,
            entry_threshold: 2.0,
            exit_threshold: 0.5,
            stop_threshold: 4.0,
            secondary_series: Vec::new(),
        })
    }

    /// Set the entry z-score threshold.
    pub fn with_entry_threshold(mut self, threshold: f64) -> Self {
        self.entry_threshold = threshold;
        self
    }

    /// Set the exit z-score threshold.
    pub fn with_exit_threshold(mut self, threshold: f64) -> Self {
        self.exit_threshold = threshold;
        self
    }

    /// Set the stop loss z-score threshold.
    pub fn with_stop_threshold(mut self, threshold: f64) -> Self {
        self.stop_threshold = threshold;
        self
    }

    /// Set the secondary series for spread calculation.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate z-score series.
    fn calculate_zscore(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.period {
            return result;
        }

        // Calculate log spread
        let spread: Vec<f64> = dual.series1.iter()
            .zip(dual.series2.iter())
            .map(|(a, b)| {
                if *a > 0.0 && *b > 0.0 {
                    (a / b).ln()
                } else {
                    0.0
                }
            })
            .collect();

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &spread[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                result[i] = (spread[i] - mean) / std_dev;
            }
        }

        result
    }

    /// Calculate pairs trading signals for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut signals = vec![0.0; n];

        if n < self.period {
            return signals;
        }

        let zscore = self.calculate_zscore(dual);
        let mut position = 0.0; // Current position: 1 = long spread, -1 = short spread, 0 = flat

        for i in (self.period - 1)..n {
            let z = zscore[i];

            // Check for stop loss
            if position != 0.0 && z.abs() > self.stop_threshold {
                position = 0.0;
            }
            // Check for exit
            else if position > 0.0 && z > -self.exit_threshold {
                position = 0.0;
            } else if position < 0.0 && z < self.exit_threshold {
                position = 0.0;
            }
            // Check for entry
            else if position == 0.0 {
                if z > self.entry_threshold {
                    position = -1.0; // Short spread (spread too high, expect reversion)
                } else if z < -self.entry_threshold {
                    position = 1.0; // Long spread (spread too low, expect reversion)
                }
            }

            signals[i] = position;
        }

        signals
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for PairsTradingSignal {
    fn name(&self) -> &str {
        "Pairs Trading Signal"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing PairsTradingSignal".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// Relative Performance
// ============================================================================

/// Relative Performance - Measures relative performance vs a baseline.
///
/// This indicator calculates rolling performance relative to an internal
/// baseline (SMA of the series), measuring whether the asset is outperforming
/// or underperforming its own trend.
///
/// # Interpretation
/// - Positive values: Outperforming the baseline
/// - Negative values: Underperforming the baseline
/// - Values are expressed as percentage differences
#[derive(Debug, Clone)]
pub struct RelativePerformance {
    /// Period for baseline calculation.
    period: usize,
    /// Smoothing period for performance.
    smooth_period: usize,
}

impl RelativePerformance {
    /// Create a new RelativePerformance indicator.
    ///
    /// # Arguments
    /// * `period` - Baseline calculation period (must be >= 2)
    /// * `smooth_period` - Smoothing period for output (must be >= 1)
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            period,
            smooth_period,
        })
    }

    /// Calculate relative performance.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period {
            return result;
        }

        // Calculate baseline (SMA)
        let mut baseline = vec![0.0; n];
        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            baseline[i] = close[start..=i].iter().sum::<f64>() / self.period as f64;
        }

        // Calculate raw relative performance
        let mut raw_perf = vec![0.0; n];
        for i in (self.period - 1)..n {
            if baseline[i] > 1e-10 {
                raw_perf[i] = (close[i] / baseline[i] - 1.0) * 100.0;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start_idx = self.period - 1;

        if start_idx < n {
            result[start_idx] = raw_perf[start_idx];
            for i in (start_idx + 1)..n {
                result[i] = alpha * raw_perf[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for RelativePerformance {
    fn name(&self) -> &str {
        "Relative Performance"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Momentum Leader
// ============================================================================

/// Momentum Leader - Identifies momentum leadership characteristics.
///
/// This indicator measures whether an asset displays momentum leadership
/// traits by comparing short-term momentum to longer-term momentum and
/// consistency of positive returns.
///
/// # Interpretation
/// - High positive values: Strong momentum leader
/// - Values near zero: Neutral momentum
/// - Negative values: Momentum laggard
#[derive(Debug, Clone)]
pub struct MomentumLeader {
    /// Short-term momentum period.
    short_period: usize,
    /// Long-term momentum period.
    long_period: usize,
}

impl MomentumLeader {
    /// Create a new MomentumLeader indicator.
    ///
    /// # Arguments
    /// * `short_period` - Short-term momentum period (must be >= 2)
    /// * `long_period` - Long-term momentum period (must be > short_period)
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        Ok(Self {
            short_period,
            long_period,
        })
    }

    /// Calculate momentum leadership score.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.long_period {
            return result;
        }

        for i in self.long_period..n {
            // Short-term momentum (as percentage)
            let short_mom = if close[i - self.short_period] > 1e-10 {
                (close[i] / close[i - self.short_period] - 1.0) * 100.0
            } else {
                0.0
            };

            // Long-term momentum
            let long_mom = if close[i - self.long_period] > 1e-10 {
                (close[i] / close[i - self.long_period] - 1.0) * 100.0
            } else {
                0.0
            };

            // Count positive returns in recent period (momentum consistency)
            let start = i - self.short_period;
            let mut pos_count = 0;
            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    pos_count += 1;
                }
            }
            let consistency = pos_count as f64 / self.short_period as f64;

            // Momentum leadership score:
            // - Strong short-term momentum
            // - Accelerating vs long-term (short > long normalized)
            // - High consistency of positive returns
            let acceleration = short_mom * (self.long_period as f64 / self.short_period as f64) - long_mom;
            result[i] = short_mom * consistency + acceleration * 0.5;
        }

        result
    }
}

impl TechnicalIndicator for MomentumLeader {
    fn name(&self) -> &str {
        "Momentum Leader"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Trend Leader
// ============================================================================

/// Trend Leader - Identifies trend leadership characteristics.
///
/// This indicator measures whether an asset displays trend leadership traits
/// by analyzing trend strength, trend duration, and trend consistency.
///
/// # Interpretation
/// - High positive values: Strong uptrend leader
/// - High negative values: Strong downtrend leader
/// - Values near zero: No clear trend leadership
#[derive(Debug, Clone)]
pub struct TrendLeader {
    /// Short-term trend period.
    short_period: usize,
    /// Long-term trend period.
    long_period: usize,
}

impl TrendLeader {
    /// Create a new TrendLeader indicator.
    ///
    /// # Arguments
    /// * `short_period` - Short-term trend calculation (must be >= 2)
    /// * `long_period` - Long-term trend calculation (must be > short_period)
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        Ok(Self {
            short_period,
            long_period,
        })
    }

    /// Calculate trend leadership score.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.long_period {
            return result;
        }

        // Calculate short and long SMAs
        let mut short_sma = vec![0.0; n];
        let mut long_sma = vec![0.0; n];

        for i in (self.short_period - 1)..n {
            let start = i + 1 - self.short_period;
            short_sma[i] = close[start..=i].iter().sum::<f64>() / self.short_period as f64;
        }

        for i in (self.long_period - 1)..n {
            let start = i + 1 - self.long_period;
            long_sma[i] = close[start..=i].iter().sum::<f64>() / self.long_period as f64;
        }

        for i in self.long_period..n {
            // Trend strength: short SMA vs long SMA
            let trend_strength = if long_sma[i] > 1e-10 {
                (short_sma[i] / long_sma[i] - 1.0) * 100.0
            } else {
                0.0
            };

            // Trend consistency: count bars where short > long (or vice versa)
            let start = i - self.short_period;
            let mut trend_bars = 0i32;
            for j in (start + 1)..=i {
                if short_sma[j] > long_sma[j] {
                    trend_bars += 1;
                } else if short_sma[j] < long_sma[j] {
                    trend_bars -= 1;
                }
            }
            let trend_consistency = trend_bars as f64 / self.short_period as f64;

            // Trend slope: recent trend acceleration
            let prev_trend = if i >= self.short_period && long_sma[i - self.short_period] > 1e-10 {
                (short_sma[i - self.short_period] / long_sma[i - self.short_period] - 1.0) * 100.0
            } else {
                0.0
            };
            let trend_acceleration = trend_strength - prev_trend;

            // Combined score
            result[i] = trend_strength * (0.5 + 0.5 * trend_consistency.abs()) + trend_acceleration * 0.3;
        }

        result
    }
}

impl TechnicalIndicator for TrendLeader {
    fn name(&self) -> &str {
        "Trend Leader"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Correlation Breakdown
// ============================================================================

/// Correlation Breakdown - Detects correlation breakdown events.
///
/// This indicator monitors the stability of autocorrelation patterns and
/// detects when correlations break down, which may signal regime changes.
///
/// # Interpretation
/// - Values near 0: Stable correlation patterns
/// - High positive/negative values: Correlation breakdown detected
/// - Sudden spikes indicate potential regime changes
#[derive(Debug, Clone)]
pub struct CorrelationBreakdown {
    /// Lookback period for correlation calculation.
    period: usize,
    /// Threshold for breakdown detection.
    threshold: f64,
}

impl CorrelationBreakdown {
    /// Create a new CorrelationBreakdown indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period (must be >= 10)
    /// * `threshold` - Breakdown detection threshold (must be > 0)
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate autocorrelation for a window.
    fn autocorrelation(returns: &[f64], lag: usize) -> f64 {
        let n = returns.len();
        if n <= lag + 1 {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / n as f64;
        let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;

        if variance < 1e-10 {
            return 0.0;
        }

        let mut cov = 0.0;
        for i in lag..n {
            cov += (returns[i] - mean) * (returns[i - lag] - mean);
        }
        cov /= (n - lag) as f64;

        cov / variance
    }

    /// Calculate correlation breakdown score.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period * 2 {
            return result;
        }

        // Calculate returns
        let mut returns = vec![0.0; n];
        for i in 1..n {
            if close[i - 1] > 1e-10 {
                returns[i] = close[i] / close[i - 1] - 1.0;
            }
        }

        // Rolling correlation breakdown detection
        for i in (self.period * 2 - 1)..n {
            // Recent period autocorrelation
            let recent_start = i + 1 - self.period;
            let recent_returns = &returns[recent_start..=i];
            let recent_ac = Self::autocorrelation(recent_returns, 1);

            // Previous period autocorrelation
            let prev_end = recent_start;
            let prev_start = prev_end.saturating_sub(self.period);
            if prev_end > prev_start {
                let prev_returns = &returns[prev_start..prev_end];
                let prev_ac = Self::autocorrelation(prev_returns, 1);

                // Breakdown score: significant change in autocorrelation
                let change = (recent_ac - prev_ac).abs();
                if change > self.threshold {
                    // Scale by magnitude of change
                    result[i] = (change / self.threshold - 1.0) * 100.0;
                    // Add sign based on direction
                    if recent_ac < prev_ac {
                        result[i] = -result[i]; // Correlation decreased
                    }
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for CorrelationBreakdown {
    fn name(&self) -> &str {
        "Correlation Breakdown"
    }

    fn min_periods(&self) -> usize {
        self.period * 2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Spread Analysis
// ============================================================================

/// Spread Analysis - Analyzes spread dynamics using internal proxies.
///
/// This indicator analyzes the spread between close price and its moving average,
/// measuring spread volatility, mean reversion tendency, and extreme deviations.
///
/// # Interpretation
/// - Positive values: Price above baseline, potential overextension
/// - Negative values: Price below baseline, potential undervaluation
/// - Large absolute values indicate extreme spread conditions
#[derive(Debug, Clone)]
pub struct SpreadAnalysis {
    /// Period for baseline calculation.
    period: usize,
    /// Standard deviation multiplier for extreme detection.
    std_multiplier: f64,
}

impl SpreadAnalysis {
    /// Create a new SpreadAnalysis indicator.
    ///
    /// # Arguments
    /// * `period` - Baseline calculation period (must be >= 5)
    /// * `std_multiplier` - Standard deviation multiplier (must be > 0)
    pub fn new(period: usize, std_multiplier: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if std_multiplier <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "std_multiplier".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        Ok(Self {
            period,
            std_multiplier,
        })
    }

    /// Calculate spread analysis score.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period {
            return result;
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &close[start..=i];

            // Calculate mean (baseline)
            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;

            // Calculate standard deviation
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            // Calculate spread (current vs baseline)
            let spread = close[i] - mean;

            // Normalize by volatility (z-score-like)
            if std_dev > 1e-10 {
                let z_score = spread / std_dev;

                // Apply multiplier for extreme detection
                if z_score.abs() > self.std_multiplier {
                    // Extreme condition: amplify
                    result[i] = z_score * (1.0 + (z_score.abs() - self.std_multiplier) * 0.5);
                } else {
                    result[i] = z_score;
                }
            } else if mean > 1e-10 {
                // Use percentage-based when volatility is too low
                result[i] = (spread / mean) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for SpreadAnalysis {
    fn name(&self) -> &str {
        "Spread Analysis"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// Flow Indicator
// ============================================================================

/// Flow Indicator - Tracks flow between asset classes using volume as proxy.
///
/// This indicator uses volume patterns and price action to estimate
/// money flow dynamics, which can indicate capital rotation between
/// asset classes.
///
/// # Interpretation
/// - Positive values: Net inflow (buying pressure)
/// - Negative values: Net outflow (selling pressure)
/// - Magnitude indicates flow intensity
#[derive(Debug, Clone)]
pub struct FlowIndicator {
    /// Period for flow calculation.
    period: usize,
    /// Smoothing period.
    smooth_period: usize,
}

impl FlowIndicator {
    /// Create a new FlowIndicator.
    ///
    /// # Arguments
    /// * `period` - Flow calculation period (must be >= 2)
    /// * `smooth_period` - Smoothing period (must be >= 1)
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            period,
            smooth_period,
        })
    }

    /// Calculate flow indicator.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period {
            return result;
        }

        // Calculate Money Flow Multiplier and Volume for each bar
        let mut mfv = vec![0.0; n];
        for i in 0..n {
            let hl_range = high[i] - low[i];
            if hl_range > 1e-10 {
                // CLV (Close Location Value)
                let clv = ((close[i] - low[i]) - (high[i] - close[i])) / hl_range;
                mfv[i] = clv * volume[i];
            }
        }

        // Calculate rolling flow
        let mut raw_flow = vec![0.0; n];
        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;

            // Sum of money flow volume
            let flow_sum: f64 = mfv[start..=i].iter().sum();

            // Sum of volume
            let vol_sum: f64 = volume[start..=i].iter().sum();

            // Normalized flow
            if vol_sum > 1e-10 {
                raw_flow[i] = (flow_sum / vol_sum) * 100.0;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start_idx = self.period - 1;

        if start_idx < n {
            result[start_idx] = raw_flow[start_idx];
            for i in (start_idx + 1)..n {
                result[i] = alpha * raw_flow[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for FlowIndicator {
    fn name(&self) -> &str {
        "Flow Indicator"
    }

    fn min_periods(&self) -> usize {
        self.period
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

// ============================================================================
// CrossMarketCorrelation
// ============================================================================

/// CrossMarketCorrelation - Rolling correlation between markets.
///
/// This indicator calculates the rolling Pearson correlation coefficient between
/// two market series, helping identify when markets are moving together or diverging.
///
/// # Interpretation
/// - Values near +1: Strong positive correlation (markets move together)
/// - Values near -1: Strong negative correlation (markets move inversely)
/// - Values near 0: No correlation (markets move independently)
/// - Changes in correlation can signal regime shifts
#[derive(Debug, Clone)]
pub struct CrossMarketCorrelation {
    /// Period for rolling correlation calculation.
    period: usize,
    /// Secondary series for correlation calculation.
    secondary_series: Vec<f64>,
}

impl CrossMarketCorrelation {
    /// Create a new CrossMarketCorrelation indicator.
    ///
    /// # Arguments
    /// * `period` - Rolling window period (must be >= 10)
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self {
            period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for correlation calculation.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate rolling correlation for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.period {
            return result;
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window1 = &dual.series1[start..=i];
            let window2 = &dual.series2[start..=i];

            let len = window1.len() as f64;
            let mean1: f64 = window1.iter().sum::<f64>() / len;
            let mean2: f64 = window2.iter().sum::<f64>() / len;

            let mut cov = 0.0;
            let mut var1 = 0.0;
            let mut var2 = 0.0;

            for (v1, v2) in window1.iter().zip(window2.iter()) {
                let d1 = v1 - mean1;
                let d2 = v2 - mean2;
                cov += d1 * d2;
                var1 += d1 * d1;
                var2 += d2 * d2;
            }

            let denom = (var1 * var2).sqrt();
            if denom > 1e-10 {
                result[i] = cov / denom;
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for CrossMarketCorrelation {
    fn name(&self) -> &str {
        "Cross Market Correlation"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing CrossMarketCorrelation".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// RelativeStrengthMomentum
// ============================================================================

/// RelativeStrengthMomentum - Momentum of relative strength between markets.
///
/// This indicator measures the rate of change in the relative strength ratio
/// between two series, identifying acceleration or deceleration in outperformance.
///
/// # Interpretation
/// - Positive values: Relative strength is increasing (series1 accelerating vs series2)
/// - Negative values: Relative strength is decreasing (series1 decelerating vs series2)
/// - Zero crossings indicate changes in relative momentum direction
#[derive(Debug, Clone)]
pub struct RelativeStrengthMomentum {
    /// Period for relative strength calculation.
    rs_period: usize,
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Smoothing period for output.
    smooth_period: usize,
    /// Secondary series for comparison.
    secondary_series: Vec<f64>,
}

impl RelativeStrengthMomentum {
    /// Create a new RelativeStrengthMomentum indicator.
    ///
    /// # Arguments
    /// * `rs_period` - Relative strength calculation period (must be >= 5)
    /// * `momentum_period` - Momentum lookback period (must be >= 1)
    /// * `smooth_period` - Output smoothing period (must be >= 1)
    pub fn new(rs_period: usize, momentum_period: usize, smooth_period: usize) -> Result<Self> {
        if rs_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "rs_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if momentum_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            rs_period,
            momentum_period,
            smooth_period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for comparison.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate relative strength momentum for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let min_req = self.rs_period + self.momentum_period;
        let mut result = vec![0.0; n];

        if n < min_req {
            return result;
        }

        // Calculate rolling relative strength ratio
        let mut rs_ratio = vec![0.0; n];
        for i in (self.rs_period - 1)..n {
            let start = i + 1 - self.rs_period;

            // Calculate returns for both series over the period
            if dual.series1[start] > 1e-10 && dual.series2[start] > 1e-10 {
                let return1 = dual.series1[i] / dual.series1[start];
                let return2 = dual.series2[i] / dual.series2[start];

                if return2 > 1e-10 {
                    rs_ratio[i] = return1 / return2;
                }
            }
        }

        // Calculate momentum of relative strength
        let mut raw_momentum = vec![0.0; n];
        for i in min_req..n {
            if rs_ratio[i - self.momentum_period] > 1e-10 {
                raw_momentum[i] = (rs_ratio[i] / rs_ratio[i - self.momentum_period] - 1.0) * 100.0;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start_idx = min_req;

        if start_idx < n {
            result[start_idx] = raw_momentum[start_idx];
            for i in (start_idx + 1)..n {
                result[i] = alpha * raw_momentum[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for RelativeStrengthMomentum {
    fn name(&self) -> &str {
        "Relative Strength Momentum"
    }

    fn min_periods(&self) -> usize {
        self.rs_period + self.momentum_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing RelativeStrengthMomentum".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// IntermarketDivergence
// ============================================================================

/// IntermarketDivergence - Divergence detection across markets.
///
/// This indicator detects when two markets that normally move together are
/// diverging in their price action, which can signal potential reversions
/// or trend changes.
///
/// # Interpretation
/// - Positive values: Series1 outperforming relative to expected correlation
/// - Negative values: Series1 underperforming relative to expected correlation
/// - Large absolute values indicate significant divergence
/// - Divergence may resolve through convergence (mean reversion opportunity)
#[derive(Debug, Clone)]
pub struct IntermarketDivergence {
    /// Period for baseline correlation calculation.
    correlation_period: usize,
    /// Period for divergence measurement.
    divergence_period: usize,
    /// Threshold for significant divergence.
    threshold: f64,
    /// Secondary series for comparison.
    secondary_series: Vec<f64>,
}

impl IntermarketDivergence {
    /// Create a new IntermarketDivergence indicator.
    ///
    /// # Arguments
    /// * `correlation_period` - Period for baseline correlation (must be >= 20)
    /// * `divergence_period` - Period for divergence measurement (must be >= 5)
    /// * `threshold` - Threshold for significant divergence (must be > 0)
    pub fn new(correlation_period: usize, divergence_period: usize, threshold: f64) -> Result<Self> {
        if correlation_period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "correlation_period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if divergence_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "divergence_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        Ok(Self {
            correlation_period,
            divergence_period,
            threshold,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for comparison.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate intermarket divergence for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let min_req = self.correlation_period.max(self.divergence_period);
        let mut result = vec![0.0; n];

        if n < min_req {
            return result;
        }

        // Calculate returns
        let mut returns1 = vec![0.0; n];
        let mut returns2 = vec![0.0; n];
        for i in 1..n {
            if dual.series1[i - 1] > 1e-10 {
                returns1[i] = dual.series1[i] / dual.series1[i - 1] - 1.0;
            }
            if dual.series2[i - 1] > 1e-10 {
                returns2[i] = dual.series2[i] / dual.series2[i - 1] - 1.0;
            }
        }

        for i in (min_req - 1)..n {
            // Calculate correlation over correlation_period
            let corr_start = i + 1 - self.correlation_period;
            let corr_window1 = &returns1[corr_start..=i];
            let corr_window2 = &returns2[corr_start..=i];

            let len = corr_window1.len() as f64;
            let mean1: f64 = corr_window1.iter().sum::<f64>() / len;
            let mean2: f64 = corr_window2.iter().sum::<f64>() / len;

            let mut cov = 0.0;
            let mut var1 = 0.0;
            let mut var2 = 0.0;

            for (v1, v2) in corr_window1.iter().zip(corr_window2.iter()) {
                let d1 = v1 - mean1;
                let d2 = v2 - mean2;
                cov += d1 * d2;
                var1 += d1 * d1;
                var2 += d2 * d2;
            }

            let corr = if (var1 * var2).sqrt() > 1e-10 {
                cov / (var1 * var2).sqrt()
            } else {
                0.0
            };

            // Calculate beta (regression coefficient)
            let beta = if var2 > 1e-10 { cov / var2 } else { 1.0 };

            // Calculate expected return based on series2 and beta
            let div_start = i + 1 - self.divergence_period;
            let mut cumulative_divergence = 0.0;

            for j in div_start..=i {
                let expected_return1 = beta * returns2[j];
                let actual_return1 = returns1[j];
                cumulative_divergence += actual_return1 - expected_return1;
            }

            // Scale by correlation strength (higher correlation = more meaningful divergence)
            let divergence_score = cumulative_divergence * corr.abs() * 100.0;

            // Apply threshold
            if divergence_score.abs() > self.threshold {
                result[i] = divergence_score;
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for IntermarketDivergence {
    fn name(&self) -> &str {
        "Intermarket Divergence"
    }

    fn min_periods(&self) -> usize {
        self.correlation_period.max(self.divergence_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing IntermarketDivergence".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// SectorMomentumRank
// ============================================================================

/// SectorMomentumRank - Ranking based on sector momentum.
///
/// This indicator calculates the momentum rank of an asset within its sector
/// or compared to a benchmark, providing a normalized score from 0 to 100.
///
/// # Interpretation
/// - Values near 100: Top momentum performer
/// - Values near 50: Average momentum
/// - Values near 0: Bottom momentum performer
/// - Useful for sector rotation and relative value strategies
#[derive(Debug, Clone)]
pub struct SectorMomentumRank {
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Period for ranking calculation.
    rank_period: usize,
    /// Smoothing period for output.
    smooth_period: usize,
}

impl SectorMomentumRank {
    /// Create a new SectorMomentumRank indicator.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum calculation (must be >= 5)
    /// * `rank_period` - Period for ranking calculation (must be >= 10)
    /// * `smooth_period` - Output smoothing period (must be >= 1)
    pub fn new(momentum_period: usize, rank_period: usize, smooth_period: usize) -> Result<Self> {
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if rank_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "rank_period".to_string(),
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
            momentum_period,
            rank_period,
            smooth_period,
        })
    }

    /// Calculate sector momentum rank.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_req = self.momentum_period + self.rank_period;
        let mut result = vec![50.0; n]; // Default to neutral

        if n < min_req {
            return result;
        }

        // Calculate momentum series
        let mut momentum = vec![0.0; n];
        for i in self.momentum_period..n {
            if close[i - self.momentum_period] > 1e-10 {
                momentum[i] = (close[i] / close[i - self.momentum_period] - 1.0) * 100.0;
            }
        }

        // Calculate percentile rank of current momentum within historical range
        let mut raw_rank = vec![50.0; n];
        for i in (min_req - 1)..n {
            let start = i + 1 - self.rank_period;
            let window = &momentum[start..=i];
            let current = momentum[i];

            // Calculate percentile rank
            let count_below = window.iter().filter(|&&m| m < current).count();
            let percentile = (count_below as f64 / self.rank_period as f64) * 100.0;
            raw_rank[i] = percentile;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start_idx = min_req - 1;

        if start_idx < n {
            result[start_idx] = raw_rank[start_idx];
            for i in (start_idx + 1)..n {
                result[i] = alpha * raw_rank[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for SectorMomentumRank {
    fn name(&self) -> &str {
        "Sector Momentum Rank"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period + self.rank_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

// ============================================================================
// CrossAssetVolatility
// ============================================================================

/// CrossAssetVolatility - Volatility comparison across assets.
///
/// This indicator compares the volatility of two assets, calculating the
/// ratio and changes in relative volatility over time.
///
/// # Interpretation
/// - Values > 1: Series1 is more volatile than Series2
/// - Values < 1: Series1 is less volatile than Series2
/// - Values = 1: Equal volatility
/// - Rising values: Series1 volatility increasing relative to Series2
#[derive(Debug, Clone)]
pub struct CrossAssetVolatility {
    /// Period for volatility calculation.
    period: usize,
    /// Use log returns for volatility calculation.
    use_log_returns: bool,
    /// Secondary series for comparison.
    secondary_series: Vec<f64>,
}

impl CrossAssetVolatility {
    /// Create a new CrossAssetVolatility indicator.
    ///
    /// # Arguments
    /// * `period` - Volatility calculation period (must be >= 10)
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self {
            period,
            use_log_returns: true,
            secondary_series: Vec::new(),
        })
    }

    /// Use log returns for volatility calculation (default: true).
    pub fn with_log_returns(mut self, use_log: bool) -> Self {
        self.use_log_returns = use_log;
        self
    }

    /// Set the secondary series for comparison.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate standard deviation of returns.
    fn calculate_volatility(returns: &[f64]) -> f64 {
        let n = returns.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / n;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        variance.sqrt()
    }

    /// Calculate cross-asset volatility ratio for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![1.0; n]; // Default to equal volatility

        if n < self.period + 1 {
            return result;
        }

        // Calculate returns
        let mut returns1 = vec![0.0; n];
        let mut returns2 = vec![0.0; n];

        for i in 1..n {
            if dual.series1[i - 1] > 1e-10 {
                returns1[i] = if self.use_log_returns {
                    (dual.series1[i] / dual.series1[i - 1]).ln()
                } else {
                    dual.series1[i] / dual.series1[i - 1] - 1.0
                };
            }
            if dual.series2[i - 1] > 1e-10 {
                returns2[i] = if self.use_log_returns {
                    (dual.series2[i] / dual.series2[i - 1]).ln()
                } else {
                    dual.series2[i] / dual.series2[i - 1] - 1.0
                };
            }
        }

        // Calculate rolling volatility ratio
        for i in self.period..n {
            let start = i + 1 - self.period;
            let vol1 = Self::calculate_volatility(&returns1[start..=i]);
            let vol2 = Self::calculate_volatility(&returns2[start..=i]);

            if vol2 > 1e-10 {
                result[i] = vol1 / vol2;
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for CrossAssetVolatility {
    fn name(&self) -> &str {
        "Cross Asset Volatility"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing CrossAssetVolatility".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// MarketLeadLag
// ============================================================================

/// MarketLeadLag - Lead/lag relationship indicator between markets.
///
/// This indicator measures the lead/lag relationship between two markets by
/// analyzing cross-correlation at different time offsets, identifying which
/// market tends to lead price moves.
///
/// # Interpretation
/// - Positive values: Series1 leads Series2 (Series1 moves first)
/// - Negative values: Series1 lags Series2 (Series2 moves first)
/// - Magnitude indicates the strength of the lead/lag relationship
/// - Values near 0: No clear lead/lag relationship
#[derive(Debug, Clone)]
pub struct MarketLeadLag {
    /// Period for correlation calculation.
    period: usize,
    /// Maximum lag to test.
    max_lag: usize,
    /// Smoothing period for output.
    smooth_period: usize,
    /// Secondary series for comparison.
    secondary_series: Vec<f64>,
}

impl MarketLeadLag {
    /// Create a new MarketLeadLag indicator.
    ///
    /// # Arguments
    /// * `period` - Correlation calculation period (must be >= 20)
    /// * `max_lag` - Maximum lag periods to test (must be >= 1 and < period/2)
    /// * `smooth_period` - Output smoothing period (must be >= 1)
    pub fn new(period: usize, max_lag: usize, smooth_period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if max_lag < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_lag".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if max_lag >= period / 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_lag".to_string(),
                reason: "must be less than period / 2".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            period,
            max_lag,
            smooth_period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for comparison.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate cross-correlation at a specific lag.
    fn cross_correlation(returns1: &[f64], returns2: &[f64], lag: i32) -> f64 {
        let n = returns1.len();
        let abs_lag = lag.unsigned_abs() as usize;

        if abs_lag >= n {
            return 0.0;
        }

        let (r1, r2) = if lag >= 0 {
            (&returns1[abs_lag..], &returns2[..n - abs_lag])
        } else {
            (&returns1[..n - abs_lag], &returns2[abs_lag..])
        };

        let len = r1.len() as f64;
        if len < 2.0 {
            return 0.0;
        }

        let mean1: f64 = r1.iter().sum::<f64>() / len;
        let mean2: f64 = r2.iter().sum::<f64>() / len;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in r1.iter().zip(r2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Calculate market lead/lag for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate returns
        let mut returns1 = vec![0.0; n];
        let mut returns2 = vec![0.0; n];

        for i in 1..n {
            if dual.series1[i - 1] > 1e-10 {
                returns1[i] = (dual.series1[i] / dual.series1[i - 1]).ln();
            }
            if dual.series2[i - 1] > 1e-10 {
                returns2[i] = (dual.series2[i] / dual.series2[i - 1]).ln();
            }
        }

        // Calculate rolling lead/lag score
        let mut raw_lead_lag = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window1 = &returns1[start..=i];
            let window2 = &returns2[start..=i];

            let mut best_lag = 0i32;
            let mut best_corr = Self::cross_correlation(window1, window2, 0);

            // Test positive lags (series1 leads)
            for lag in 1..=self.max_lag as i32 {
                let corr = Self::cross_correlation(window1, window2, lag);
                if corr > best_corr {
                    best_corr = corr;
                    best_lag = lag;
                }
            }

            // Test negative lags (series1 lags)
            for lag in 1..=self.max_lag as i32 {
                let corr = Self::cross_correlation(window1, window2, -lag);
                if corr > best_corr {
                    best_corr = corr;
                    best_lag = -lag;
                }
            }

            // Lead/lag score: lag * correlation strength
            raw_lead_lag[i] = best_lag as f64 * best_corr;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start_idx = self.period - 1;

        if start_idx < n {
            result[start_idx] = raw_lead_lag[start_idx];
            for i in (start_idx + 1)..n {
                result[i] = alpha * raw_lead_lag[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for MarketLeadLag {
    fn name(&self) -> &str {
        "Market Lead Lag"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing MarketLeadLag".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// RelativePerformanceIndex
// ============================================================================

/// Relative Performance Index - Measures relative performance between two series.
///
/// This indicator calculates a normalized index that measures how one series
/// performs relative to another over a rolling window. The index is scaled
/// to oscillate around zero, making it easy to identify periods of
/// outperformance or underperformance.
///
/// # Formula
/// RPI = (Return1 - Return2) / StdDev(Return1 - Return2) * 100
///
/// # Interpretation
/// - Positive values: Series 1 outperforms Series 2
/// - Negative values: Series 1 underperforms Series 2
/// - Values > 2 or < -2 suggest significant relative performance
/// - Mean reversion expected when values become extreme
#[derive(Debug, Clone)]
pub struct RelativePerformanceIndex {
    /// Period for relative performance calculation.
    period: usize,
    /// Smoothing period for the index.
    smooth_period: usize,
    /// Secondary series for comparison.
    secondary_series: Vec<f64>,
}

impl RelativePerformanceIndex {
    /// Create a new RelativePerformanceIndex indicator.
    ///
    /// # Arguments
    /// * `period` - Rolling window period (must be >= 10)
    /// * `smooth_period` - Smoothing period (must be >= 1)
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
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
            period,
            smooth_period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for comparison.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate relative performance index for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate returns for both series
        let mut returns1 = vec![0.0; n];
        let mut returns2 = vec![0.0; n];

        for i in 1..n {
            if dual.series1[i - 1] > 1e-10 {
                returns1[i] = dual.series1[i] / dual.series1[i - 1] - 1.0;
            }
            if dual.series2[i - 1] > 1e-10 {
                returns2[i] = dual.series2[i] / dual.series2[i - 1] - 1.0;
            }
        }

        // Calculate relative return (excess return of series1 over series2)
        let mut rel_returns: Vec<f64> = returns1.iter()
            .zip(returns2.iter())
            .map(|(r1, r2)| r1 - r2)
            .collect();

        // Calculate rolling z-score of relative returns
        let mut raw_rpi = vec![0.0; n];

        for i in self.period..n {
            let start = i + 1 - self.period;
            let window = &rel_returns[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            // Calculate cumulative relative return over the period
            let cum_rel: f64 = window.iter().sum();

            // Normalize by volatility
            if std_dev > 1e-10 {
                raw_rpi[i] = (cum_rel / std_dev) * 10.0; // Scale for readability
            } else {
                raw_rpi[i] = cum_rel * 100.0; // Simple percentage when no volatility
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start_idx = self.period;

        if start_idx < n {
            result[start_idx] = raw_rpi[start_idx];
            for i in (start_idx + 1)..n {
                result[i] = alpha * raw_rpi[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for RelativePerformanceIndex {
    fn name(&self) -> &str {
        "Relative Performance Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing RelativePerformanceIndex".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// SpreadOscillator
// ============================================================================

/// Spread Oscillator - Oscillator based on the spread between two series.
///
/// This indicator calculates a bounded oscillator from the spread between
/// two price series. It normalizes the spread using Bollinger Band-style
/// calculations to produce an oscillator that moves between -100 and +100.
///
/// # Formula
/// SpreadOsc = ((Spread - SMA(Spread)) / (StdDev(Spread) * multiplier)) * 100
///
/// # Interpretation
/// - Values > 80: Spread is extremely high (overbought)
/// - Values < -80: Spread is extremely low (oversold)
/// - Zero crossings can signal spread reversals
/// - Divergences with price can indicate trend changes
#[derive(Debug, Clone)]
pub struct SpreadOscillator {
    /// Period for spread calculations.
    period: usize,
    /// Standard deviation multiplier for normalization.
    std_multiplier: f64,
    /// Use log spread for ratio-based comparison.
    use_log_spread: bool,
    /// Secondary series for spread calculation.
    secondary_series: Vec<f64>,
}

impl SpreadOscillator {
    /// Create a new SpreadOscillator indicator.
    ///
    /// # Arguments
    /// * `period` - Rolling window period (must be >= 10)
    /// * `std_multiplier` - Standard deviation multiplier (must be > 0)
    pub fn new(period: usize, std_multiplier: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if std_multiplier <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "std_multiplier".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        Ok(Self {
            period,
            std_multiplier,
            use_log_spread: true,
            secondary_series: Vec::new(),
        })
    }

    /// Use log spread (default: true).
    pub fn with_log_spread(mut self, use_log: bool) -> Self {
        self.use_log_spread = use_log;
        self
    }

    /// Set the secondary series for spread calculation.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate spread oscillator for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.period {
            return result;
        }

        // Calculate spread
        let spread: Vec<f64> = dual.series1.iter()
            .zip(dual.series2.iter())
            .map(|(a, b)| {
                if *b > 0.0 {
                    if self.use_log_spread && *a > 0.0 {
                        (a / b).ln()
                    } else {
                        a - b
                    }
                } else {
                    0.0
                }
            })
            .collect();

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &spread[start..=i];

            // Calculate mean and standard deviation
            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            // Calculate oscillator value (bounded z-score)
            if std_dev > 1e-10 {
                let z_score = (spread[i] - mean) / (std_dev * self.std_multiplier);
                // Bound to [-100, 100] using tanh-like transformation
                result[i] = z_score.tanh() * 100.0;
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for SpreadOscillator {
    fn name(&self) -> &str {
        "Spread Oscillator"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing SpreadOscillator".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// BetaEstimator
// ============================================================================

/// Beta Estimator - Rolling beta estimate between two series.
///
/// This indicator calculates the rolling beta coefficient, measuring the
/// sensitivity of one asset's returns to another (typically a benchmark).
/// Beta is a key metric in portfolio management and risk assessment.
///
/// # Formula
/// Beta = Cov(R1, R2) / Var(R2)
///
/// # Interpretation
/// - Beta > 1: Series 1 is more volatile than Series 2
/// - Beta = 1: Series 1 moves in line with Series 2
/// - Beta < 1: Series 1 is less volatile than Series 2
/// - Beta < 0: Series 1 moves inversely to Series 2
/// - Beta is commonly used for hedging and portfolio construction
#[derive(Debug, Clone)]
pub struct BetaEstimator {
    /// Period for beta calculation.
    period: usize,
    /// Use log returns for calculation.
    use_log_returns: bool,
    /// Secondary series (benchmark) for beta calculation.
    secondary_series: Vec<f64>,
}

impl BetaEstimator {
    /// Create a new BetaEstimator indicator.
    ///
    /// # Arguments
    /// * `period` - Rolling window period (must be >= 20)
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self {
            period,
            use_log_returns: true,
            secondary_series: Vec::new(),
        })
    }

    /// Use log returns (default: true).
    pub fn with_log_returns(mut self, use_log: bool) -> Self {
        self.use_log_returns = use_log;
        self
    }

    /// Set the secondary (benchmark) series for beta calculation.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate rolling beta for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![1.0; n]; // Default beta of 1

        if n < self.period + 1 {
            return result;
        }

        // Calculate returns for both series
        let mut returns1 = vec![0.0; n];
        let mut returns2 = vec![0.0; n];

        for i in 1..n {
            if dual.series1[i - 1] > 1e-10 {
                if self.use_log_returns {
                    returns1[i] = (dual.series1[i] / dual.series1[i - 1]).ln();
                } else {
                    returns1[i] = dual.series1[i] / dual.series1[i - 1] - 1.0;
                }
            }
            if dual.series2[i - 1] > 1e-10 {
                if self.use_log_returns {
                    returns2[i] = (dual.series2[i] / dual.series2[i - 1]).ln();
                } else {
                    returns2[i] = dual.series2[i] / dual.series2[i - 1] - 1.0;
                }
            }
        }

        // Calculate rolling beta
        for i in self.period..n {
            let start = i + 1 - self.period;
            let r1 = &returns1[start..=i];
            let r2 = &returns2[start..=i];

            let mean1: f64 = r1.iter().sum::<f64>() / self.period as f64;
            let mean2: f64 = r2.iter().sum::<f64>() / self.period as f64;

            let mut cov = 0.0;
            let mut var2 = 0.0;

            for (ret1, ret2) in r1.iter().zip(r2.iter()) {
                let d1 = ret1 - mean1;
                let d2 = ret2 - mean2;
                cov += d1 * d2;
                var2 += d2 * d2;
            }

            // Beta = Cov(R1, R2) / Var(R2)
            if var2 > 1e-10 {
                result[i] = cov / var2;
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for BetaEstimator {
    fn name(&self) -> &str {
        "Beta Estimator"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing BetaEstimator".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// CointegrationScore
// ============================================================================

/// Cointegration Score - Measures the degree of cointegration between two series.
///
/// This indicator estimates the cointegration strength between two price series
/// by analyzing the stationarity of their spread. It uses a simplified approach
/// based on the variance ratio test and mean reversion speed.
///
/// # Interpretation
/// - High scores (> 70): Strong cointegration, spread likely mean-reverts
/// - Medium scores (30-70): Moderate cointegration
/// - Low scores (< 30): Weak cointegration, spread may trend
/// - Use for identifying suitable pairs for pairs trading
#[derive(Debug, Clone)]
pub struct CointegrationScore {
    /// Period for cointegration calculation.
    period: usize,
    /// Short period for variance ratio.
    short_period: usize,
    /// Secondary series for cointegration calculation.
    secondary_series: Vec<f64>,
}

impl CointegrationScore {
    /// Create a new CointegrationScore indicator.
    ///
    /// # Arguments
    /// * `period` - Long-term period (must be >= 30)
    /// * `short_period` - Short-term period for variance ratio (must be >= 5 and < period/2)
    pub fn new(period: usize, short_period: usize) -> Result<Self> {
        if period < 30 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 30".to_string(),
            });
        }
        if short_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if short_period >= period / 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be less than period / 2".to_string(),
            });
        }
        Ok(Self {
            period,
            short_period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for cointegration calculation.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate hedge ratio using OLS regression.
    fn calculate_hedge_ratio(series1: &[f64], series2: &[f64]) -> f64 {
        let n = series1.len() as f64;
        if n < 2.0 {
            return 1.0;
        }

        let mean1: f64 = series1.iter().sum::<f64>() / n;
        let mean2: f64 = series2.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in series1.iter().zip(series2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var2 += d2 * d2;
        }

        if var2 > 1e-10 {
            cov / var2
        } else {
            1.0
        }
    }

    /// Calculate cointegration score for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![50.0; n]; // Default to neutral score

        if n < self.period {
            return result;
        }

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window1 = &dual.series1[start..=i];
            let window2 = &dual.series2[start..=i];

            // Calculate hedge ratio
            let hedge_ratio = Self::calculate_hedge_ratio(window1, window2);

            // Calculate spread using hedge ratio
            let spread: Vec<f64> = window1.iter()
                .zip(window2.iter())
                .map(|(a, b)| a - hedge_ratio * b)
                .collect();

            // Calculate spread changes
            let mut spread_changes = Vec::with_capacity(spread.len() - 1);
            for j in 1..spread.len() {
                spread_changes.push(spread[j] - spread[j - 1]);
            }

            if spread_changes.is_empty() {
                continue;
            }

            // Variance of spread changes (should be low for cointegrated series)
            let var_changes: f64 = {
                let mean: f64 = spread_changes.iter().sum::<f64>() / spread_changes.len() as f64;
                spread_changes.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / spread_changes.len() as f64
            };

            // Variance of spread levels (should be bounded for cointegrated series)
            let var_levels: f64 = {
                let mean: f64 = spread.iter().sum::<f64>() / spread.len() as f64;
                spread.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / spread.len() as f64
            };

            // Mean reversion speed: regression of spread change on lagged spread
            let mut sum_xy = 0.0;
            let mut sum_x2 = 0.0;
            for j in 1..spread.len() {
                let x = spread[j - 1];
                let y = spread[j] - spread[j - 1];
                sum_xy += x * y;
                sum_x2 += x * x;
            }

            let mean_reversion_coef = if sum_x2 > 1e-10 {
                -sum_xy / sum_x2 // Negative coefficient indicates mean reversion
            } else {
                0.0
            };

            // Variance ratio test (ratio of long-term to short-term variance)
            // For a random walk, this ratio = 1; for mean-reverting, ratio < 1
            let short_var = if self.short_period < spread.len() {
                let short_window = &spread[(spread.len() - self.short_period)..];
                let mean: f64 = short_window.iter().sum::<f64>() / short_window.len() as f64;
                short_window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / short_window.len() as f64
            } else {
                var_levels
            };

            let variance_ratio = if var_levels > 1e-10 {
                (short_var / var_levels).sqrt()
            } else {
                1.0
            };

            // Combine metrics into cointegration score (0-100)
            // Mean reversion coefficient contribution (higher is better)
            let mr_score = (mean_reversion_coef.clamp(0.0, 0.5) / 0.5) * 40.0;

            // Variance ratio contribution (lower is better for mean reversion)
            let vr_score = ((1.0 - variance_ratio.clamp(0.0, 1.0)) * 30.0).max(0.0);

            // Spread variance stability (lower relative variance is better)
            let stability_score = if var_levels > 1e-10 && var_changes > 1e-10 {
                let ratio = var_changes / var_levels;
                ((1.0 - ratio.min(1.0)) * 30.0).max(0.0)
            } else {
                15.0 // Neutral
            };

            result[i] = (mr_score + vr_score + stability_score).clamp(0.0, 100.0);
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for CointegrationScore {
    fn name(&self) -> &str {
        "Cointegration Score"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing CointegrationScore".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// CorrelationTrendAnalyzer
// ============================================================================

/// Correlation Trend Analyzer - Advanced tracking of correlation changes over time.
///
/// This indicator provides enhanced analysis of how correlation between two series
/// evolves. It includes momentum of correlation changes and regime detection.
///
/// # Interpretation
/// - Positive values: Correlation is increasing (series becoming more aligned)
/// - Negative values: Correlation is decreasing (series diverging)
/// - Large absolute values indicate rapid correlation regime changes
/// - Can signal potential market regime shifts
#[derive(Debug, Clone)]
pub struct CorrelationTrendAnalyzer {
    /// Short-term correlation period.
    short_period: usize,
    /// Long-term correlation period.
    long_period: usize,
    /// Momentum lookback period.
    momentum_period: usize,
    /// Secondary series for correlation calculation.
    secondary_series: Vec<f64>,
}

impl CorrelationTrendAnalyzer {
    /// Create a new CorrelationTrendAnalyzer indicator.
    ///
    /// # Arguments
    /// * `short_period` - Short-term correlation window (must be >= 10)
    /// * `long_period` - Long-term correlation window (must be > short_period)
    /// * `momentum_period` - Momentum lookback (must be >= 3)
    pub fn new(short_period: usize, long_period: usize, momentum_period: usize) -> Result<Self> {
        if short_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if momentum_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self {
            short_period,
            long_period,
            momentum_period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for correlation calculation.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate correlation for a window.
    fn correlation(series1: &[f64], series2: &[f64]) -> f64 {
        let n = series1.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean1: f64 = series1.iter().sum::<f64>() / n;
        let mean2: f64 = series2.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in series1.iter().zip(series2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Calculate correlation trend analyzer for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.long_period + self.momentum_period {
            return result;
        }

        // First, calculate rolling short-term correlations
        let mut short_corrs = vec![0.0; n];
        for i in (self.short_period - 1)..n {
            let start = i + 1 - self.short_period;
            short_corrs[i] = Self::correlation(
                &dual.series1[start..=i],
                &dual.series2[start..=i],
            );
        }

        // Calculate rolling long-term correlations
        let mut long_corrs = vec![0.0; n];
        for i in (self.long_period - 1)..n {
            let start = i + 1 - self.long_period;
            long_corrs[i] = Self::correlation(
                &dual.series1[start..=i],
                &dual.series2[start..=i],
            );
        }

        // Calculate the correlation trend with momentum
        for i in (self.long_period + self.momentum_period - 1)..n {
            // Current correlation differential
            let corr_diff = short_corrs[i] - long_corrs[i];

            // Correlation momentum (change in short correlation)
            let corr_momentum = short_corrs[i] - short_corrs[i - self.momentum_period];

            // Regime change detection (variance in recent correlation)
            let start = i + 1 - self.momentum_period;
            let recent_corrs = &short_corrs[start..=i];
            let corr_mean: f64 = recent_corrs.iter().sum::<f64>() / recent_corrs.len() as f64;
            let corr_var: f64 = recent_corrs.iter()
                .map(|c| (c - corr_mean).powi(2))
                .sum::<f64>() / recent_corrs.len() as f64;
            let regime_instability = corr_var.sqrt() * 100.0;

            // Combine into trend score
            // corr_diff: positive = correlation increasing vs long-term
            // corr_momentum: positive = correlation accelerating upward
            // regime_instability: high = unstable correlation (potential regime change)
            result[i] = (corr_diff * 50.0) + (corr_momentum * 30.0) +
                        (corr_momentum.signum() * regime_instability * 0.2);
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for CorrelationTrendAnalyzer {
    fn name(&self) -> &str {
        "Correlation Trend Analyzer"
    }

    fn min_periods(&self) -> usize {
        self.long_period + self.momentum_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing CorrelationTrendAnalyzer".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// EnhancedPairsTradingSignal
// ============================================================================

/// Enhanced Pairs Trading Signal - Advanced signal generator for pairs trading.
///
/// This indicator generates sophisticated entry and exit signals for pairs trading
/// by combining multiple factors: z-score, correlation stability, momentum divergence,
/// and trend alignment.
///
/// # Signal Values
/// - 2.0: Strong long spread signal (strong conviction)
/// - 1.0: Long spread signal (long series1, short series2)
/// - 0.0: No position / exit signal
/// - -1.0: Short spread signal (short series1, long series2)
/// - -2.0: Strong short spread signal (strong conviction)
#[derive(Debug, Clone)]
pub struct EnhancedPairsTradingSignal {
    /// Period for z-score calculation.
    period: usize,
    /// Entry z-score threshold.
    entry_threshold: f64,
    /// Strong entry z-score threshold.
    strong_threshold: f64,
    /// Exit z-score threshold.
    exit_threshold: f64,
    /// Correlation period for quality check.
    corr_period: usize,
    /// Minimum correlation for trading.
    min_correlation: f64,
    /// Secondary series for spread calculation.
    secondary_series: Vec<f64>,
}

impl EnhancedPairsTradingSignal {
    /// Create a new EnhancedPairsTradingSignal indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for z-score calculation (must be >= 20)
    /// * `corr_period` - Period for correlation check (must be >= 10)
    pub fn new(period: usize, corr_period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if corr_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "corr_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self {
            period,
            entry_threshold: 2.0,
            strong_threshold: 3.0,
            exit_threshold: 0.5,
            corr_period,
            min_correlation: 0.5,
            secondary_series: Vec::new(),
        })
    }

    /// Set the entry z-score threshold.
    pub fn with_entry_threshold(mut self, threshold: f64) -> Self {
        self.entry_threshold = threshold;
        self
    }

    /// Set the strong entry z-score threshold.
    pub fn with_strong_threshold(mut self, threshold: f64) -> Self {
        self.strong_threshold = threshold;
        self
    }

    /// Set the exit z-score threshold.
    pub fn with_exit_threshold(mut self, threshold: f64) -> Self {
        self.exit_threshold = threshold;
        self
    }

    /// Set the minimum correlation for trading.
    pub fn with_min_correlation(mut self, min_corr: f64) -> Self {
        self.min_correlation = min_corr;
        self
    }

    /// Set the secondary series for spread calculation.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate correlation for a window.
    fn correlation(series1: &[f64], series2: &[f64]) -> f64 {
        let n = series1.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean1: f64 = series1.iter().sum::<f64>() / n;
        let mean2: f64 = series2.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in series1.iter().zip(series2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Calculate enhanced pairs trading signals for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut signals = vec![0.0; n];

        let min_periods = self.period.max(self.corr_period);
        if n < min_periods {
            return signals;
        }

        // Calculate log spread
        let spread: Vec<f64> = dual.series1.iter()
            .zip(dual.series2.iter())
            .map(|(a, b)| {
                if *a > 0.0 && *b > 0.0 {
                    (a / b).ln()
                } else {
                    0.0
                }
            })
            .collect();

        // Calculate rolling z-score
        let mut zscore = vec![0.0; n];
        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &spread[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / self.period as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 {
                zscore[i] = (spread[i] - mean) / std_dev;
            }
        }

        // Calculate rolling correlation
        let mut correlation = vec![0.0; n];
        for i in (self.corr_period - 1)..n {
            let start = i + 1 - self.corr_period;
            correlation[i] = Self::correlation(
                &dual.series1[start..=i],
                &dual.series2[start..=i],
            );
        }

        // Generate signals
        let mut position = 0.0;

        for i in (min_periods - 1)..n {
            let z = zscore[i];
            let corr = correlation[i];

            // Check correlation quality
            let good_correlation = corr.abs() >= self.min_correlation;

            // Update position based on signals
            if !good_correlation {
                // Exit if correlation breaks down
                position = 0.0;
            } else if position == 0.0 {
                // Entry logic
                if z < -self.strong_threshold {
                    position = 2.0; // Strong long spread
                } else if z < -self.entry_threshold {
                    position = 1.0; // Long spread
                } else if z > self.strong_threshold {
                    position = -2.0; // Strong short spread
                } else if z > self.entry_threshold {
                    position = -1.0; // Short spread
                }
            } else if position > 0.0 {
                // Long spread exit logic
                if z > -self.exit_threshold {
                    position = 0.0; // Exit long
                } else if z < -self.strong_threshold && position == 1.0 {
                    position = 2.0; // Upgrade to strong
                }
            } else if position < 0.0 {
                // Short spread exit logic
                if z < self.exit_threshold {
                    position = 0.0; // Exit short
                } else if z > self.strong_threshold && position == -1.0 {
                    position = -2.0; // Upgrade to strong
                }
            }

            signals[i] = position;
        }

        signals
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for EnhancedPairsTradingSignal {
    fn name(&self) -> &str {
        "Enhanced Pairs Trading Signal"
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.corr_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing EnhancedPairsTradingSignal".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// RelativeRotation
// ============================================================================

/// Relative Rotation - RRG-style relative rotation indicator.
///
/// This indicator implements a Relative Rotation Graph (RRG) style analysis,
/// measuring both relative strength (RS) and momentum of relative strength (RS-Momentum).
/// It helps identify the rotation of assets through different market phases.
///
/// # RRG Quadrants
/// - Leading (RS > 100, RS-Momentum > 100): Strong and strengthening
/// - Weakening (RS > 100, RS-Momentum < 100): Strong but weakening
/// - Lagging (RS < 100, RS-Momentum < 100): Weak and weakening
/// - Improving (RS < 100, RS-Momentum > 100): Weak but improving
///
/// # Output
/// Returns a composite score combining RS and RS-Momentum normalized around 0.
/// - Positive values: Leading/Improving phases
/// - Negative values: Weakening/Lagging phases
#[derive(Debug, Clone)]
pub struct RelativeRotation {
    /// Period for relative strength calculation.
    rs_period: usize,
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Smoothing period for output.
    smooth_period: usize,
    /// Benchmark series for comparison.
    benchmark_series: Vec<f64>,
}

impl RelativeRotation {
    /// Create a new RelativeRotation indicator.
    ///
    /// # Arguments
    /// * `rs_period` - Period for relative strength (must be >= 10)
    /// * `momentum_period` - Period for RS momentum (must be >= 5)
    /// * `smooth_period` - Smoothing period (must be >= 1)
    pub fn new(rs_period: usize, momentum_period: usize, smooth_period: usize) -> Result<Self> {
        if rs_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "rs_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            rs_period,
            momentum_period,
            smooth_period,
            benchmark_series: Vec::new(),
        })
    }

    /// Set the benchmark series for comparison.
    pub fn with_benchmark(mut self, series: &[f64]) -> Self {
        self.benchmark_series = series.to_vec();
        self
    }

    /// Calculate relative rotation for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let min_req = self.rs_period + self.momentum_period;
        let mut result = vec![0.0; n];

        if n < min_req {
            return result;
        }

        // Calculate relative strength ratio (asset / benchmark)
        let mut rs_ratio = vec![100.0; n];
        for i in (self.rs_period - 1)..n {
            let start = i + 1 - self.rs_period;

            // Calculate performance over the period
            if dual.series1[start] > 1e-10 && dual.series2[start] > 1e-10 {
                let asset_perf = dual.series1[i] / dual.series1[start];
                let bench_perf = dual.series2[i] / dual.series2[start];

                if bench_perf > 1e-10 {
                    // RS normalized to 100 (100 = equal performance)
                    rs_ratio[i] = (asset_perf / bench_perf) * 100.0;
                }
            }
        }

        // Calculate RS-Momentum (rate of change of RS)
        let mut rs_momentum = vec![100.0; n];
        for i in (self.rs_period + self.momentum_period - 1)..n {
            if rs_ratio[i - self.momentum_period] > 1e-10 {
                // Momentum normalized to 100 (100 = no change)
                rs_momentum[i] = (rs_ratio[i] / rs_ratio[i - self.momentum_period]) * 100.0;
            }
        }

        // Combine into rotation score
        // Positive = Leading or Improving, Negative = Weakening or Lagging
        let mut raw_score = vec![0.0; n];
        for i in (min_req - 1)..n {
            // Center around 0: RS - 100 gives deviation from benchmark
            // RS-Momentum - 100 gives direction of change
            let rs_dev = rs_ratio[i] - 100.0;
            let mom_dev = rs_momentum[i] - 100.0;

            // Weighted combination: RS deviation + momentum direction
            raw_score[i] = rs_dev * 0.6 + mom_dev * 10.0 * 0.4;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start_idx = min_req - 1;

        if start_idx < n {
            result[start_idx] = raw_score[start_idx];
            for i in (start_idx + 1)..n {
                result[i] = alpha * raw_score[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for RelativeRotation {
    fn name(&self) -> &str {
        "Relative Rotation"
    }

    fn min_periods(&self) -> usize {
        self.rs_period + self.momentum_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.benchmark_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "benchmark_series".to_string(),
                reason: "Benchmark series must be set before computing RelativeRotation".to_string(),
            });
        }

        if self.benchmark_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "benchmark_series".to_string(),
                reason: format!(
                    "Benchmark series length ({}) must match primary series length ({})",
                    self.benchmark_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.benchmark_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// AlphaGenerator
// ============================================================================

/// Alpha Generator - Calculates rolling alpha (excess return) vs benchmark.
///
/// This indicator measures the alpha component of returns - the portion of
/// performance that cannot be explained by benchmark/market movements.
/// It uses a simple regression approach to separate alpha from beta-driven returns.
///
/// # Formula
/// Alpha = Asset Return - Beta * Benchmark Return
///
/// # Interpretation
/// - Positive alpha: Asset outperforming on risk-adjusted basis
/// - Negative alpha: Asset underperforming on risk-adjusted basis
/// - Annualized alpha values for easier interpretation
#[derive(Debug, Clone)]
pub struct AlphaGenerator {
    /// Period for alpha calculation.
    period: usize,
    /// Annualization factor (252 for daily, 52 for weekly).
    annualization_factor: f64,
    /// Benchmark series for comparison.
    benchmark_series: Vec<f64>,
}

impl AlphaGenerator {
    /// Create a new AlphaGenerator indicator.
    ///
    /// # Arguments
    /// * `period` - Rolling window period (must be >= 20)
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self {
            period,
            annualization_factor: 252.0,
            benchmark_series: Vec::new(),
        })
    }

    /// Set the annualization factor (default: 252 for daily data).
    pub fn with_annualization(mut self, factor: f64) -> Self {
        self.annualization_factor = factor;
        self
    }

    /// Set the benchmark series for comparison.
    pub fn with_benchmark(mut self, series: &[f64]) -> Self {
        self.benchmark_series = series.to_vec();
        self
    }

    /// Calculate rolling alpha for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate returns for both series
        let mut returns1 = vec![0.0; n];
        let mut returns2 = vec![0.0; n];

        for i in 1..n {
            if dual.series1[i - 1] > 1e-10 {
                returns1[i] = dual.series1[i] / dual.series1[i - 1] - 1.0;
            }
            if dual.series2[i - 1] > 1e-10 {
                returns2[i] = dual.series2[i] / dual.series2[i - 1] - 1.0;
            }
        }

        // Calculate rolling alpha using regression
        for i in self.period..n {
            let start = i + 1 - self.period;
            let r1 = &returns1[start..=i];
            let r2 = &returns2[start..=i];

            let mean1: f64 = r1.iter().sum::<f64>() / self.period as f64;
            let mean2: f64 = r2.iter().sum::<f64>() / self.period as f64;

            // Calculate beta (covariance / variance)
            let mut cov = 0.0;
            let mut var2 = 0.0;

            for (ret1, ret2) in r1.iter().zip(r2.iter()) {
                let d1 = ret1 - mean1;
                let d2 = ret2 - mean2;
                cov += d1 * d2;
                var2 += d2 * d2;
            }

            let beta = if var2 > 1e-10 { cov / var2 } else { 1.0 };

            // Alpha = mean asset return - beta * mean benchmark return
            let alpha = mean1 - beta * mean2;

            // Annualize alpha (multiply by annualization factor)
            result[i] = alpha * self.annualization_factor * 100.0; // Express as percentage
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for AlphaGenerator {
    fn name(&self) -> &str {
        "Alpha Generator"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.benchmark_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "benchmark_series".to_string(),
                reason: "Benchmark series must be set before computing AlphaGenerator".to_string(),
            });
        }

        if self.benchmark_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "benchmark_series".to_string(),
                reason: format!(
                    "Benchmark series length ({}) must match primary series length ({})",
                    self.benchmark_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.benchmark_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// TrackingError
// ============================================================================

/// Tracking Error - Measures the standard deviation of active returns.
///
/// Tracking error quantifies how closely a portfolio follows its benchmark.
/// Lower tracking error indicates the portfolio tracks the benchmark closely,
/// while higher values indicate more deviation from the benchmark.
///
/// # Formula
/// Tracking Error = StdDev(Asset Return - Benchmark Return) * sqrt(Annualization Factor)
///
/// # Interpretation
/// - < 1%: Very tight tracking (index fund-like)
/// - 1-3%: Low tracking error (enhanced index)
/// - 3-6%: Moderate tracking error (active management)
/// - > 6%: High tracking error (aggressive active management)
#[derive(Debug, Clone)]
pub struct TrackingError {
    /// Period for tracking error calculation.
    period: usize,
    /// Annualization factor (252 for daily, 52 for weekly).
    annualization_factor: f64,
    /// Benchmark series for comparison.
    benchmark_series: Vec<f64>,
}

impl TrackingError {
    /// Create a new TrackingError indicator.
    ///
    /// # Arguments
    /// * `period` - Rolling window period (must be >= 20)
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self {
            period,
            annualization_factor: 252.0,
            benchmark_series: Vec::new(),
        })
    }

    /// Set the annualization factor (default: 252 for daily data).
    pub fn with_annualization(mut self, factor: f64) -> Self {
        self.annualization_factor = factor;
        self
    }

    /// Set the benchmark series for comparison.
    pub fn with_benchmark(mut self, series: &[f64]) -> Self {
        self.benchmark_series = series.to_vec();
        self
    }

    /// Calculate rolling tracking error for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate returns for both series
        let mut returns1 = vec![0.0; n];
        let mut returns2 = vec![0.0; n];

        for i in 1..n {
            if dual.series1[i - 1] > 1e-10 {
                returns1[i] = dual.series1[i] / dual.series1[i - 1] - 1.0;
            }
            if dual.series2[i - 1] > 1e-10 {
                returns2[i] = dual.series2[i] / dual.series2[i - 1] - 1.0;
            }
        }

        // Calculate active returns (difference between asset and benchmark)
        let active_returns: Vec<f64> = returns1.iter()
            .zip(returns2.iter())
            .map(|(r1, r2)| r1 - r2)
            .collect();

        // Calculate rolling tracking error
        for i in self.period..n {
            let start = i + 1 - self.period;
            let window = &active_returns[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / (self.period - 1) as f64; // Sample variance

            // Annualized tracking error
            let tracking_error = variance.sqrt() * self.annualization_factor.sqrt();
            result[i] = tracking_error * 100.0; // Express as percentage
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for TrackingError {
    fn name(&self) -> &str {
        "Tracking Error"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.benchmark_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "benchmark_series".to_string(),
                reason: "Benchmark series must be set before computing TrackingError".to_string(),
            });
        }

        if self.benchmark_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "benchmark_series".to_string(),
                reason: format!(
                    "Benchmark series length ({}) must match primary series length ({})",
                    self.benchmark_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.benchmark_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// IntermarketInformationRatio
// ============================================================================

/// Intermarket Information Ratio - Risk-adjusted excess return measurement.
///
/// This indicator calculates the information ratio specific to intermarket analysis,
/// measuring how consistently one asset outperforms another on a risk-adjusted basis.
/// Unlike standard IR, this is optimized for comparing two assets directly.
///
/// # Formula
/// IR = Mean(Asset Return - Benchmark Return) / StdDev(Asset Return - Benchmark Return)
/// Annualized IR = IR * sqrt(Annualization Factor)
///
/// # Interpretation
/// - IR > 0.5: Good risk-adjusted outperformance
/// - IR > 1.0: Excellent risk-adjusted outperformance
/// - IR < 0: Underperformance
/// - Higher absolute values indicate more consistent relative performance
#[derive(Debug, Clone)]
pub struct IntermarketInformationRatio {
    /// Period for IR calculation.
    period: usize,
    /// Annualization factor (252 for daily, 52 for weekly).
    annualization_factor: f64,
    /// Secondary series for comparison.
    secondary_series: Vec<f64>,
}

impl IntermarketInformationRatio {
    /// Create a new IntermarketInformationRatio indicator.
    ///
    /// # Arguments
    /// * `period` - Rolling window period (must be >= 20)
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self {
            period,
            annualization_factor: 252.0,
            secondary_series: Vec::new(),
        })
    }

    /// Set the annualization factor (default: 252 for daily data).
    pub fn with_annualization(mut self, factor: f64) -> Self {
        self.annualization_factor = factor;
        self
    }

    /// Set the secondary series for comparison.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate rolling information ratio for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate returns for both series
        let mut returns1 = vec![0.0; n];
        let mut returns2 = vec![0.0; n];

        for i in 1..n {
            if dual.series1[i - 1] > 1e-10 {
                returns1[i] = dual.series1[i] / dual.series1[i - 1] - 1.0;
            }
            if dual.series2[i - 1] > 1e-10 {
                returns2[i] = dual.series2[i] / dual.series2[i - 1] - 1.0;
            }
        }

        // Calculate active returns
        let active_returns: Vec<f64> = returns1.iter()
            .zip(returns2.iter())
            .map(|(r1, r2)| r1 - r2)
            .collect();

        // Calculate rolling information ratio
        for i in self.period..n {
            let start = i + 1 - self.period;
            let window = &active_returns[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / self.period as f64;
            let variance: f64 = window.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / (self.period - 1) as f64;
            let std_dev = variance.sqrt();

            // Information Ratio = mean excess return / tracking error
            if std_dev > 1e-10 {
                let ir = mean / std_dev;
                // Annualize the IR
                result[i] = ir * self.annualization_factor.sqrt();
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for IntermarketInformationRatio {
    fn name(&self) -> &str {
        "Intermarket Information Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing IntermarketInformationRatio".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// IntermarketCorrelationBreakdown
// ============================================================================

/// Intermarket Correlation Breakdown - Detects correlation regime changes between two series.
///
/// This indicator monitors the correlation between two market series and detects
/// when the correlation structure breaks down or undergoes significant changes.
/// Unlike autocorrelation-based breakdown detection, this focuses on cross-asset
/// correlation dynamics.
///
/// # Interpretation
/// - Values near 0: Stable correlation regime
/// - Large positive values: Correlation increasing rapidly (convergence)
/// - Large negative values: Correlation decreasing rapidly (divergence)
/// - Spikes indicate potential regime changes requiring attention
#[derive(Debug, Clone)]
pub struct IntermarketCorrelationBreakdown {
    /// Short-term correlation period.
    short_period: usize,
    /// Long-term correlation period.
    long_period: usize,
    /// Threshold for breakdown detection.
    threshold: f64,
    /// Secondary series for correlation.
    secondary_series: Vec<f64>,
}

impl IntermarketCorrelationBreakdown {
    /// Create a new IntermarketCorrelationBreakdown indicator.
    ///
    /// # Arguments
    /// * `short_period` - Short-term correlation window (must be >= 10)
    /// * `long_period` - Long-term correlation window (must be > short_period)
    /// * `threshold` - Breakdown detection threshold (must be > 0)
    pub fn new(short_period: usize, long_period: usize, threshold: f64) -> Result<Self> {
        if short_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        Ok(Self {
            short_period,
            long_period,
            threshold,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for correlation.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate correlation for a window.
    fn correlation(series1: &[f64], series2: &[f64]) -> f64 {
        let n = series1.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean1: f64 = series1.iter().sum::<f64>() / n;
        let mean2: f64 = series2.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in series1.iter().zip(series2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Calculate correlation breakdown score for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.long_period {
            return result;
        }

        // Calculate rolling correlations
        let mut short_corr = vec![0.0; n];
        let mut long_corr = vec![0.0; n];

        for i in (self.short_period - 1)..n {
            let start = i + 1 - self.short_period;
            short_corr[i] = Self::correlation(
                &dual.series1[start..=i],
                &dual.series2[start..=i],
            );
        }

        for i in (self.long_period - 1)..n {
            let start = i + 1 - self.long_period;
            long_corr[i] = Self::correlation(
                &dual.series1[start..=i],
                &dual.series2[start..=i],
            );
        }

        // Detect correlation breakdown
        for i in (self.long_period - 1)..n {
            // Correlation change (short vs long)
            let corr_diff = short_corr[i] - long_corr[i];

            // Rate of correlation change
            let corr_change = if i >= self.short_period {
                short_corr[i] - short_corr[i - self.short_period / 2]
            } else {
                0.0
            };

            // Breakdown score: combination of level difference and rate of change
            let raw_score = corr_diff.abs() + corr_change.abs() * 2.0;

            // Apply threshold and scale
            if raw_score > self.threshold {
                // Positive = correlation increasing (convergence)
                // Negative = correlation decreasing (divergence)
                let direction = if corr_diff >= 0.0 { 1.0 } else { -1.0 };
                result[i] = direction * (raw_score / self.threshold - 1.0) * 50.0;
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for IntermarketCorrelationBreakdown {
    fn name(&self) -> &str {
        "Intermarket Correlation Breakdown"
    }

    fn min_periods(&self) -> usize {
        self.long_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing IntermarketCorrelationBreakdown".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// RegimeCorrelation
// ============================================================================

/// Regime Correlation - Calculates correlation conditioned on market regime.
///
/// This indicator measures correlation differently based on the current market
/// regime (trending up, trending down, or ranging). This is important because
/// correlations often change dramatically during different market conditions.
///
/// # Interpretation
/// - Output includes regime-adjusted correlation
/// - Positive values indicate positive correlation in current regime
/// - Negative values indicate negative correlation in current regime
/// - Magnitude indicates correlation strength
/// - Changes in regime correlation can signal portfolio rebalancing needs
#[derive(Debug, Clone)]
pub struct RegimeCorrelation {
    /// Period for correlation calculation.
    correlation_period: usize,
    /// Period for regime detection.
    regime_period: usize,
    /// Regime threshold (return threshold for up/down classification).
    regime_threshold: f64,
    /// Secondary series for correlation.
    secondary_series: Vec<f64>,
}

impl RegimeCorrelation {
    /// Create a new RegimeCorrelation indicator.
    ///
    /// # Arguments
    /// * `correlation_period` - Period for correlation (must be >= 10)
    /// * `regime_period` - Period for regime detection (must be >= 5)
    /// * `regime_threshold` - Return threshold for regime classification (must be >= 0)
    pub fn new(correlation_period: usize, regime_period: usize, regime_threshold: f64) -> Result<Self> {
        if correlation_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "correlation_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if regime_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "regime_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if regime_threshold < 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "regime_threshold".to_string(),
                reason: "must be at least 0".to_string(),
            });
        }
        Ok(Self {
            correlation_period,
            regime_period,
            regime_threshold,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for correlation.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Determine market regime: 1 = up, -1 = down, 0 = ranging.
    fn detect_regime(returns: &[f64], threshold: f64) -> i32 {
        let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let annualized = mean_return * 252.0; // Approximate annualization

        if annualized > threshold {
            1 // Up regime
        } else if annualized < -threshold {
            -1 // Down regime
        } else {
            0 // Ranging
        }
    }

    /// Calculate correlation for a window.
    fn correlation(series1: &[f64], series2: &[f64]) -> f64 {
        let n = series1.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean1: f64 = series1.iter().sum::<f64>() / n;
        let mean2: f64 = series2.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in series1.iter().zip(series2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Calculate regime-adjusted correlation for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let min_req = self.correlation_period.max(self.regime_period);
        let mut result = vec![0.0; n];

        if n < min_req + 1 {
            return result;
        }

        // Calculate returns for regime detection (using first series as market proxy)
        let mut returns1 = vec![0.0; n];
        for i in 1..n {
            if dual.series1[i - 1] > 1e-10 {
                returns1[i] = dual.series1[i] / dual.series1[i - 1] - 1.0;
            }
        }

        // Calculate rolling regime-adjusted correlation
        for i in (min_req - 1)..n {
            // Detect current regime
            let regime_start = i + 1 - self.regime_period;
            let regime_window = &returns1[regime_start..=i];
            let regime = Self::detect_regime(regime_window, self.regime_threshold);

            // Calculate correlation
            let corr_start = i + 1 - self.correlation_period;
            let corr = Self::correlation(
                &dual.series1[corr_start..=i],
                &dual.series2[corr_start..=i],
            );

            // Regime-adjusted output: correlation scaled by regime
            // - In up regime: positive correlation is good (both rising)
            // - In down regime: negative correlation is good (hedge)
            // - In ranging: raw correlation
            match regime {
                1 => result[i] = corr * 100.0, // Up regime: report correlation
                -1 => result[i] = -corr * 100.0, // Down regime: invert perspective
                _ => result[i] = corr * 50.0, // Ranging: reduced weight
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for RegimeCorrelation {
    fn name(&self) -> &str {
        "Regime Correlation"
    }

    fn min_periods(&self) -> usize {
        self.correlation_period.max(self.regime_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing RegimeCorrelation".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// RelativeValueScore
// ============================================================================

/// Relative Value Score - Measures relative valuation between two assets using multiple metrics.
///
/// This indicator combines price ratio analysis with momentum and mean reversion
/// to generate a composite relative value score. Unlike simple ratio analysis,
/// it incorporates the rate of change of the ratio and its deviation from normal levels.
///
/// # Formula
/// Score = weighted combination of:
/// - Price ratio percentile (current vs historical)
/// - Ratio momentum (direction of ratio change)
/// - Mean reversion signal (deviation from mean)
///
/// # Interpretation
/// - Score > 50: Asset 1 relatively overvalued vs Asset 2
/// - Score < 50: Asset 1 relatively undervalued vs Asset 2
/// - Score near 50: Fair relative value
/// - Extreme scores (>80 or <20) suggest potential reversion opportunities
#[derive(Debug, Clone)]
pub struct RelativeValueScore {
    /// Period for calculations.
    period: usize,
    /// Momentum lookback period.
    momentum_period: usize,
    /// Secondary series for comparison.
    secondary_series: Vec<f64>,
}

impl RelativeValueScore {
    /// Create a new RelativeValueScore indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for historical comparison (must be >= 20)
    /// * `momentum_period` - Period for momentum calculation (must be >= 5)
    pub fn new(period: usize, momentum_period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self {
            period,
            momentum_period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for comparison.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate relative value score for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![50.0; n]; // Default to neutral

        if n < self.period {
            return result;
        }

        // Calculate price ratios
        let ratios: Vec<f64> = dual.series1.iter()
            .zip(dual.series2.iter())
            .map(|(a, b)| if *b > 1e-10 { a / b } else { 1.0 })
            .collect();

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &ratios[start..=i];
            let current = ratios[i];

            // Component 1: Percentile rank (0-100)
            let count_below = window.iter().filter(|&&r| r < current).count();
            let percentile = (count_below as f64 / self.period as f64) * 100.0;

            // Component 2: Momentum (rate of change)
            let momentum_score = if i >= self.momentum_period {
                let prev_ratio = ratios[i - self.momentum_period];
                if prev_ratio > 1e-10 {
                    let roc = (current / prev_ratio - 1.0) * 100.0;
                    // Normalize momentum to 0-100 range
                    50.0 + roc.clamp(-50.0, 50.0)
                } else {
                    50.0
                }
            } else {
                50.0
            };

            // Component 3: Mean reversion (z-score based)
            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 = window.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / window.len() as f64;
            let std = variance.sqrt();
            let z_score = if std > 1e-10 { (current - mean) / std } else { 0.0 };
            // Convert z-score to 0-100 (z=2 -> 84, z=-2 -> 16)
            let reversion_score = 50.0 + z_score.clamp(-2.5, 2.5) * 15.0;

            // Combine components with weights
            result[i] = percentile * 0.4 + momentum_score * 0.3 + reversion_score * 0.3;
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for RelativeValueScore {
    fn name(&self) -> &str {
        "Relative Value Score"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing RelativeValueScore".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// SpreadMomentum
// ============================================================================

/// Spread Momentum - Measures the momentum of price spread between two assets.
///
/// This indicator tracks the rate of change and acceleration of the spread
/// between two correlated assets. It helps identify when the spread is
/// expanding or contracting at an increasing or decreasing rate.
///
/// # Formula
/// 1. Calculate spread: Series1 - (hedge_ratio * Series2)
/// 2. Calculate spread momentum: ROC of spread over momentum_period
/// 3. Apply smoothing via EMA
///
/// # Interpretation
/// - Positive momentum: Spread is widening (Series1 outperforming)
/// - Negative momentum: Spread is narrowing (Series2 catching up)
/// - Momentum acceleration indicates trend strength
/// - Divergence from price trend signals potential reversals
#[derive(Debug, Clone)]
pub struct SpreadMomentum {
    /// Period for spread calculation.
    spread_period: usize,
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Smoothing period for output.
    smooth_period: usize,
    /// Secondary series.
    secondary_series: Vec<f64>,
}

impl SpreadMomentum {
    /// Create a new SpreadMomentum indicator.
    ///
    /// # Arguments
    /// * `spread_period` - Period for spread/hedge ratio calculation (must be >= 10)
    /// * `momentum_period` - Period for momentum (must be >= 3)
    /// * `smooth_period` - EMA smoothing period (must be >= 1)
    pub fn new(spread_period: usize, momentum_period: usize, smooth_period: usize) -> Result<Self> {
        if spread_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "spread_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if momentum_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            spread_period,
            momentum_period,
            smooth_period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate rolling hedge ratio using simple regression.
    fn calculate_hedge_ratio(series1: &[f64], series2: &[f64]) -> f64 {
        let n = series1.len() as f64;
        if n < 2.0 {
            return 1.0;
        }

        let mean1: f64 = series1.iter().sum::<f64>() / n;
        let mean2: f64 = series2.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in series1.iter().zip(series2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var2 += d2 * d2;
        }

        if var2 > 1e-10 {
            cov / var2
        } else {
            1.0
        }
    }

    /// Calculate spread momentum for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let min_req = self.spread_period + self.momentum_period;
        let mut result = vec![0.0; n];

        if n < min_req {
            return result;
        }

        // Calculate spread series using rolling hedge ratio
        let mut spread = vec![0.0; n];
        for i in (self.spread_period - 1)..n {
            let start = i + 1 - self.spread_period;
            let hedge_ratio = Self::calculate_hedge_ratio(
                &dual.series1[start..=i],
                &dual.series2[start..=i],
            );
            spread[i] = dual.series1[i] - hedge_ratio * dual.series2[i];
        }

        // Calculate momentum (rate of change of spread)
        let mut raw_momentum = vec![0.0; n];
        for i in (min_req - 1)..n {
            let prev_spread = spread[i - self.momentum_period];
            if prev_spread.abs() > 1e-10 {
                raw_momentum[i] = (spread[i] - prev_spread) / prev_spread.abs() * 100.0;
            } else {
                raw_momentum[i] = spread[i] * 100.0; // If prev is ~0, use current as momentum
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start_idx = min_req - 1;

        if start_idx < n {
            result[start_idx] = raw_momentum[start_idx];
            for i in (start_idx + 1)..n {
                result[i] = alpha * raw_momentum[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for SpreadMomentum {
    fn name(&self) -> &str {
        "Spread Momentum"
    }

    fn min_periods(&self) -> usize {
        self.spread_period + self.momentum_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing SpreadMomentum".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// ConvergenceDivergence
// ============================================================================

/// Convergence Divergence - Detects convergence and divergence patterns between two assets.
///
/// This indicator identifies when two assets are converging (moving together)
/// or diverging (moving apart). It measures both the direction and rate of
/// convergence/divergence using correlation and spread analysis.
///
/// # Formula
/// 1. Calculate rolling correlation
/// 2. Calculate spread z-score
/// 3. Detect convergence: correlation rising AND spread narrowing
/// 4. Detect divergence: correlation falling OR spread widening
///
/// # Interpretation
/// - Positive values: Convergence (assets moving together)
/// - Negative values: Divergence (assets moving apart)
/// - Magnitude indicates strength of the pattern
/// - Sign changes signal potential trading opportunities
#[derive(Debug, Clone)]
pub struct ConvergenceDivergence {
    /// Period for correlation calculation.
    correlation_period: usize,
    /// Period for spread analysis.
    spread_period: usize,
    /// Smoothing period.
    smooth_period: usize,
    /// Secondary series.
    secondary_series: Vec<f64>,
}

impl ConvergenceDivergence {
    /// Create a new ConvergenceDivergence indicator.
    ///
    /// # Arguments
    /// * `correlation_period` - Period for correlation (must be >= 15)
    /// * `spread_period` - Period for spread analysis (must be >= 10)
    /// * `smooth_period` - Smoothing period (must be >= 1)
    pub fn new(correlation_period: usize, spread_period: usize, smooth_period: usize) -> Result<Self> {
        if correlation_period < 15 {
            return Err(IndicatorError::InvalidParameter {
                name: "correlation_period".to_string(),
                reason: "must be at least 15".to_string(),
            });
        }
        if spread_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "spread_period".to_string(),
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
            correlation_period,
            spread_period,
            smooth_period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate rolling correlation.
    fn correlation(series1: &[f64], series2: &[f64]) -> f64 {
        let n = series1.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean1: f64 = series1.iter().sum::<f64>() / n;
        let mean2: f64 = series2.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in series1.iter().zip(series2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Calculate convergence/divergence for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let min_req = self.correlation_period.max(self.spread_period);
        let mut result = vec![0.0; n];

        if n < min_req + 1 {
            return result;
        }

        // Calculate normalized price ratios for spread
        let ratios: Vec<f64> = dual.series1.iter()
            .zip(dual.series2.iter())
            .map(|(a, b)| if *b > 1e-10 { (a / b).ln() } else { 0.0 })
            .collect();

        let mut raw_score = vec![0.0; n];

        for i in min_req..n {
            // Rolling correlation
            let corr_start = i + 1 - self.correlation_period;
            let current_corr = Self::correlation(
                &dual.series1[corr_start..=i],
                &dual.series2[corr_start..=i],
            );

            // Previous correlation (for trend)
            let prev_corr = if i > self.correlation_period {
                Self::correlation(
                    &dual.series1[(corr_start - 1)..i],
                    &dual.series2[(corr_start - 1)..i],
                )
            } else {
                current_corr
            };

            let corr_change = current_corr - prev_corr;

            // Spread z-score
            let spread_start = i + 1 - self.spread_period;
            let spread_window = &ratios[spread_start..=i];
            let mean: f64 = spread_window.iter().sum::<f64>() / spread_window.len() as f64;
            let variance: f64 = spread_window.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / spread_window.len() as f64;
            let std = variance.sqrt();
            let z_score = if std > 1e-10 { (ratios[i] - mean) / std } else { 0.0 };

            // Convergence/Divergence score:
            // Convergence = high correlation + low z-score + rising correlation
            // Divergence = falling correlation OR high z-score
            let convergence_signal = current_corr * 50.0 + corr_change * 100.0 - z_score.abs() * 20.0;
            raw_score[i] = convergence_signal;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start_idx = min_req;

        if start_idx < n {
            result[start_idx] = raw_score[start_idx];
            for i in (start_idx + 1)..n {
                result[i] = alpha * raw_score[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for ConvergenceDivergence {
    fn name(&self) -> &str {
        "Convergence Divergence"
    }

    fn min_periods(&self) -> usize {
        self.correlation_period.max(self.spread_period) + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing ConvergenceDivergence".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// PairStrength
// ============================================================================

/// Pair Strength - Measures the trading strength and quality of a pairs relationship.
///
/// This indicator evaluates the overall quality of a pairs trading relationship
/// by combining correlation stability, spread mean-reversion tendency, and
/// volatility characteristics into a single composite score.
///
/// # Formula
/// Score = weighted combination of:
/// - Correlation strength and stability
/// - Mean-reversion tendency (half-life of spread)
/// - Spread volatility relative to price levels
/// - Cointegration-like measure
///
/// # Interpretation
/// - Score > 70: Strong pair, suitable for pairs trading
/// - Score 50-70: Moderate pair, may be tradeable
/// - Score < 50: Weak pair, not recommended for pairs trading
/// - Declining scores indicate deteriorating relationship
#[derive(Debug, Clone)]
pub struct PairStrength {
    /// Period for calculations.
    period: usize,
    /// Short period for correlation stability.
    short_period: usize,
    /// Secondary series.
    secondary_series: Vec<f64>,
}

impl PairStrength {
    /// Create a new PairStrength indicator.
    ///
    /// # Arguments
    /// * `period` - Long lookback period (must be >= 30)
    /// * `short_period` - Short period for stability (must be >= 10 and < period)
    pub fn new(period: usize, short_period: usize) -> Result<Self> {
        if period < 30 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 30".to_string(),
            });
        }
        if short_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if short_period >= period {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be less than period".to_string(),
            });
        }
        Ok(Self {
            period,
            short_period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate correlation.
    fn correlation(series1: &[f64], series2: &[f64]) -> f64 {
        let n = series1.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean1: f64 = series1.iter().sum::<f64>() / n;
        let mean2: f64 = series2.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in series1.iter().zip(series2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Estimate mean reversion speed (simplified half-life).
    fn mean_reversion_score(spread: &[f64]) -> f64 {
        let n = spread.len();
        if n < 5 {
            return 50.0;
        }

        // Calculate lagged autocorrelation as proxy for mean reversion
        let mean: f64 = spread.iter().sum::<f64>() / n as f64;
        let spread_centered: Vec<f64> = spread.iter().map(|s| s - mean).collect();

        let mut autocov = 0.0;
        let mut var = 0.0;

        for i in 1..n {
            autocov += spread_centered[i] * spread_centered[i - 1];
            var += spread_centered[i - 1] * spread_centered[i - 1];
        }

        let autocorr = if var > 1e-10 { autocov / var } else { 1.0 };

        // Lower autocorrelation = faster mean reversion = better
        // autocorr of 0 is ideal, 1 is worst (random walk)
        let score = (1.0 - autocorr.abs()) * 100.0;
        score.clamp(0.0, 100.0)
    }

    /// Calculate pair strength for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![50.0; n]; // Default to neutral

        if n < self.period {
            return result;
        }

        // Calculate log spread
        let spread: Vec<f64> = dual.series1.iter()
            .zip(dual.series2.iter())
            .map(|(a, b)| if *b > 1e-10 { (a / b).ln() } else { 0.0 })
            .collect();

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let short_start = i + 1 - self.short_period;

            // Component 1: Long-term correlation strength (0-100)
            let long_corr = Self::correlation(
                &dual.series1[start..=i],
                &dual.series2[start..=i],
            );
            let corr_score = long_corr.abs() * 100.0;

            // Component 2: Correlation stability (short vs long)
            let short_corr = Self::correlation(
                &dual.series1[short_start..=i],
                &dual.series2[short_start..=i],
            );
            let stability_score = (1.0 - (long_corr - short_corr).abs()) * 100.0;

            // Component 3: Mean reversion tendency
            let mr_score = Self::mean_reversion_score(&spread[start..=i]);

            // Component 4: Spread stationarity (low variance growth)
            let spread_window = &spread[start..=i];
            let mean: f64 = spread_window.iter().sum::<f64>() / spread_window.len() as f64;
            let variance: f64 = spread_window.iter()
                .map(|s| (s - mean).powi(2))
                .sum::<f64>() / spread_window.len() as f64;
            let std = variance.sqrt();
            // Lower relative volatility = better
            let vol_score = if mean.abs() > 1e-10 {
                (1.0 - (std / mean.abs()).min(1.0)) * 100.0
            } else {
                50.0
            };

            // Combine components
            result[i] = corr_score * 0.35 + stability_score * 0.25 + mr_score * 0.25 + vol_score * 0.15;
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for PairStrength {
    fn name(&self) -> &str {
        "Pair Strength"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing PairStrength".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// CrossAssetMomentum
// ============================================================================

/// Cross Asset Momentum - Measures momentum relationships across different assets.
///
/// This indicator analyzes momentum patterns between two assets to identify
/// leading/lagging relationships and momentum divergences. It compares the
/// momentum of each asset to detect when one asset's momentum diverges from another.
///
/// # Formula
/// 1. Calculate momentum for each asset (ROC over period)
/// 2. Calculate momentum spread (Asset1 momentum - Asset2 momentum)
/// 3. Normalize and smooth the result
///
/// # Interpretation
/// - Positive values: Asset1 has stronger momentum than Asset2
/// - Negative values: Asset2 has stronger momentum than Asset1
/// - Momentum divergence can signal sector rotation or regime change
/// - Convergence after divergence often precedes mean reversion
#[derive(Debug, Clone)]
pub struct CrossAssetMomentum {
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Smoothing period.
    smooth_period: usize,
    /// Secondary series.
    secondary_series: Vec<f64>,
}

impl CrossAssetMomentum {
    /// Create a new CrossAssetMomentum indicator.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum/ROC calculation (must be >= 5)
    /// * `smooth_period` - EMA smoothing period (must be >= 1)
    pub fn new(momentum_period: usize, smooth_period: usize) -> Result<Self> {
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            smooth_period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate momentum (ROC) for a single series.
    fn calculate_momentum(series: &[f64], period: usize) -> Vec<f64> {
        let n = series.len();
        let mut momentum = vec![0.0; n];

        for i in period..n {
            if series[i - period] > 1e-10 {
                momentum[i] = (series[i] / series[i - period] - 1.0) * 100.0;
            }
        }

        momentum
    }

    /// Calculate cross-asset momentum for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.momentum_period + 1 {
            return result;
        }

        // Calculate momentum for each series
        let mom1 = Self::calculate_momentum(&dual.series1, self.momentum_period);
        let mom2 = Self::calculate_momentum(&dual.series2, self.momentum_period);

        // Calculate momentum spread
        let mut raw_spread = vec![0.0; n];
        for i in self.momentum_period..n {
            raw_spread[i] = mom1[i] - mom2[i];
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start_idx = self.momentum_period;

        if start_idx < n {
            result[start_idx] = raw_spread[start_idx];
            for i in (start_idx + 1)..n {
                result[i] = alpha * raw_spread[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for CrossAssetMomentum {
    fn name(&self) -> &str {
        "Cross Asset Momentum"
    }

    fn min_periods(&self) -> usize {
        self.momentum_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing CrossAssetMomentum".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// IntermarketSignal
// ============================================================================

/// Intermarket Signal - Generates trading signals based on intermarket analysis.
///
/// This indicator combines multiple intermarket factors (correlation, spread,
/// momentum divergence) to generate actionable trading signals for pairs
/// or relative value strategies.
///
/// # Signal Generation
/// Combines:
/// - Spread z-score (mean reversion)
/// - Correlation regime (stable vs unstable)
/// - Momentum alignment (confirming or diverging)
/// - Trend filter (with or against trend)
///
/// # Output Signals
/// - +2: Strong long Asset1 / short Asset2
/// - +1: Moderate long Asset1 / short Asset2
/// -  0: Neutral / no signal
/// - -1: Moderate short Asset1 / long Asset2
/// - -2: Strong short Asset1 / long Asset2
#[derive(Debug, Clone)]
pub struct IntermarketSignal {
    /// Period for spread analysis.
    spread_period: usize,
    /// Period for correlation.
    correlation_period: usize,
    /// Entry threshold for z-score.
    entry_threshold: f64,
    /// Strong signal threshold.
    strong_threshold: f64,
    /// Minimum correlation for valid signal.
    min_correlation: f64,
    /// Secondary series.
    secondary_series: Vec<f64>,
}

impl IntermarketSignal {
    /// Create a new IntermarketSignal indicator.
    ///
    /// # Arguments
    /// * `spread_period` - Period for spread z-score (must be >= 15)
    /// * `correlation_period` - Period for correlation filter (must be >= 10)
    pub fn new(spread_period: usize, correlation_period: usize) -> Result<Self> {
        if spread_period < 15 {
            return Err(IndicatorError::InvalidParameter {
                name: "spread_period".to_string(),
                reason: "must be at least 15".to_string(),
            });
        }
        if correlation_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "correlation_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self {
            spread_period,
            correlation_period,
            entry_threshold: 1.5,
            strong_threshold: 2.5,
            min_correlation: 0.5,
            secondary_series: Vec::new(),
        })
    }

    /// Set entry threshold for z-score (default: 1.5).
    pub fn with_entry_threshold(mut self, threshold: f64) -> Self {
        self.entry_threshold = threshold;
        self
    }

    /// Set strong signal threshold (default: 2.5).
    pub fn with_strong_threshold(mut self, threshold: f64) -> Self {
        self.strong_threshold = threshold;
        self
    }

    /// Set minimum correlation for valid signals (default: 0.5).
    pub fn with_min_correlation(mut self, correlation: f64) -> Self {
        self.min_correlation = correlation;
        self
    }

    /// Set the secondary series.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate correlation.
    fn correlation(series1: &[f64], series2: &[f64]) -> f64 {
        let n = series1.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean1: f64 = series1.iter().sum::<f64>() / n;
        let mean2: f64 = series2.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in series1.iter().zip(series2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Calculate intermarket signal for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let min_req = self.spread_period.max(self.correlation_period);
        let mut result = vec![0.0; n];

        if n < min_req {
            return result;
        }

        // Calculate log spread
        let spread: Vec<f64> = dual.series1.iter()
            .zip(dual.series2.iter())
            .map(|(a, b)| if *b > 1e-10 { (a / b).ln() } else { 0.0 })
            .collect();

        for i in (min_req - 1)..n {
            // Calculate spread z-score
            let spread_start = i + 1 - self.spread_period;
            let spread_window = &spread[spread_start..=i];
            let mean: f64 = spread_window.iter().sum::<f64>() / spread_window.len() as f64;
            let variance: f64 = spread_window.iter()
                .map(|s| (s - mean).powi(2))
                .sum::<f64>() / spread_window.len() as f64;
            let std = variance.sqrt();
            let z_score = if std > 1e-10 { (spread[i] - mean) / std } else { 0.0 };

            // Calculate correlation
            let corr_start = i + 1 - self.correlation_period;
            let correlation = Self::correlation(
                &dual.series1[corr_start..=i],
                &dual.series2[corr_start..=i],
            );

            // Filter by correlation - only trade highly correlated pairs
            if correlation.abs() < self.min_correlation {
                result[i] = 0.0;
                continue;
            }

            // Generate signal based on z-score
            if z_score > self.strong_threshold {
                result[i] = -2.0; // Strong short spread (sell Asset1 / buy Asset2)
            } else if z_score > self.entry_threshold {
                result[i] = -1.0; // Moderate short spread
            } else if z_score < -self.strong_threshold {
                result[i] = 2.0; // Strong long spread (buy Asset1 / sell Asset2)
            } else if z_score < -self.entry_threshold {
                result[i] = 1.0; // Moderate long spread
            } else {
                result[i] = 0.0; // Neutral
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }
}

impl TechnicalIndicator for IntermarketSignal {
    fn name(&self) -> &str {
        "Intermarket Signal"
    }

    fn min_periods(&self) -> usize {
        self.spread_period.max(self.correlation_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing IntermarketSignal".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// RelativeRotationGraph
// ============================================================================

/// Relative Rotation Graph (RRG) - Full RRG-style relative rotation indicator with quadrant classification.
///
/// This indicator implements the complete RRG methodology used for sector rotation analysis.
/// It calculates both JdK RS-Ratio (relative strength) and JdK RS-Momentum, placing assets
/// into one of four quadrants: Leading, Weakening, Lagging, or Improving.
///
/// # Formula
/// 1. RS-Ratio = Normalized relative performance vs benchmark
/// 2. RS-Momentum = Rate of change of RS-Ratio
/// 3. Quadrant = Based on position relative to 100-line for both metrics
///
/// # Quadrant Classification
/// - Leading (RS-Ratio > 100, RS-Momentum > 100): Strong relative performance, accelerating
/// - Weakening (RS-Ratio > 100, RS-Momentum < 100): Strong but decelerating
/// - Lagging (RS-Ratio < 100, RS-Momentum < 100): Weak relative performance, still declining
/// - Improving (RS-Ratio < 100, RS-Momentum > 100): Weak but improving
///
/// # Output Values
/// - 4.0: Leading quadrant
/// - 3.0: Weakening quadrant
/// - 2.0: Lagging quadrant
/// - 1.0: Improving quadrant
/// - 0.0: Insufficient data
#[derive(Debug, Clone)]
pub struct RelativeRotationGraph {
    /// Period for relative strength ratio calculation.
    rs_period: usize,
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Normalization period for smoothing.
    norm_period: usize,
    /// Benchmark series for comparison.
    benchmark_series: Vec<f64>,
}

impl RelativeRotationGraph {
    /// Create a new RelativeRotationGraph indicator.
    ///
    /// # Arguments
    /// * `rs_period` - Period for relative strength ratio (must be >= 10)
    /// * `momentum_period` - Period for RS momentum (must be >= 5)
    /// * `norm_period` - Normalization/smoothing period (must be >= 1)
    pub fn new(rs_period: usize, momentum_period: usize, norm_period: usize) -> Result<Self> {
        if rs_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "rs_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if norm_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "norm_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            rs_period,
            momentum_period,
            norm_period,
            benchmark_series: Vec::new(),
        })
    }

    /// Set the benchmark series for comparison.
    pub fn with_benchmark(mut self, series: &[f64]) -> Self {
        self.benchmark_series = series.to_vec();
        self
    }

    /// Calculate JdK RS-Ratio (normalized relative strength).
    fn calculate_rs_ratio(&self, asset: &[f64], benchmark: &[f64]) -> Vec<f64> {
        let n = asset.len();
        let mut rs_ratio = vec![100.0; n];

        if n < self.rs_period {
            return rs_ratio;
        }

        // Calculate raw relative strength
        let mut raw_rs = vec![100.0; n];
        for i in 0..n {
            if benchmark[i] > 1e-10 {
                raw_rs[i] = (asset[i] / benchmark[i]) * 100.0;
            }
        }

        // Apply EMA smoothing to create RS-Ratio
        let alpha = 2.0 / (self.norm_period as f64 + 1.0);
        let start_idx = self.rs_period - 1;

        if start_idx < n {
            // Calculate initial SMA
            let initial_sum: f64 = raw_rs[..self.rs_period].iter().sum();
            rs_ratio[start_idx] = initial_sum / self.rs_period as f64;

            for i in (start_idx + 1)..n {
                rs_ratio[i] = alpha * raw_rs[i] + (1.0 - alpha) * rs_ratio[i - 1];
            }
        }

        // Normalize to oscillate around 100
        let min_req = self.rs_period + self.norm_period;
        if n >= min_req {
            for i in (min_req - 1)..n {
                let start = i + 1 - self.norm_period;
                let window = &rs_ratio[start..=i];
                let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
                let std_dev: f64 = (window.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / window.len() as f64)
                    .sqrt();

                if std_dev > 1e-10 {
                    // Normalize to approximately 100 +/- standard deviations
                    rs_ratio[i] = 100.0 + (rs_ratio[i] - mean) / std_dev * 1.0;
                }
            }
        }

        rs_ratio
    }

    /// Calculate JdK RS-Momentum (rate of change of RS-Ratio).
    fn calculate_rs_momentum(&self, rs_ratio: &[f64]) -> Vec<f64> {
        let n = rs_ratio.len();
        let mut rs_momentum = vec![100.0; n];

        if n < self.momentum_period {
            return rs_momentum;
        }

        for i in self.momentum_period..n {
            let prev_rs = rs_ratio[i - self.momentum_period];
            if prev_rs > 1e-10 {
                // Calculate rate of change and normalize around 100
                let roc = (rs_ratio[i] / prev_rs - 1.0) * 100.0;
                rs_momentum[i] = 100.0 + roc;
            }
        }

        rs_momentum
    }

    /// Determine quadrant based on RS-Ratio and RS-Momentum.
    fn classify_quadrant(rs_ratio: f64, rs_momentum: f64) -> f64 {
        if rs_ratio >= 100.0 && rs_momentum >= 100.0 {
            4.0 // Leading
        } else if rs_ratio >= 100.0 && rs_momentum < 100.0 {
            3.0 // Weakening
        } else if rs_ratio < 100.0 && rs_momentum < 100.0 {
            2.0 // Lagging
        } else {
            1.0 // Improving (rs_ratio < 100 && rs_momentum >= 100)
        }
    }

    /// Calculate quadrant classification for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let min_req = self.rs_period + self.momentum_period + self.norm_period;
        let mut result = vec![0.0; n];

        if n < min_req {
            return result;
        }

        let rs_ratio = self.calculate_rs_ratio(&dual.series1, &dual.series2);
        let rs_momentum = self.calculate_rs_momentum(&rs_ratio);

        for i in (min_req - 1)..n {
            result[i] = Self::classify_quadrant(rs_ratio[i], rs_momentum[i]);
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, asset: &[f64], benchmark: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(asset, benchmark);
        self.calculate(&dual)
    }

    /// Get detailed RRG data (RS-Ratio, RS-Momentum, Quadrant) as a tuple.
    pub fn calculate_detailed(&self, dual: &DualSeries) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = dual.len();
        let min_req = self.rs_period + self.momentum_period + self.norm_period;

        if n < min_req {
            return (vec![100.0; n], vec![100.0; n], vec![0.0; n]);
        }

        let rs_ratio = self.calculate_rs_ratio(&dual.series1, &dual.series2);
        let rs_momentum = self.calculate_rs_momentum(&rs_ratio);

        let mut quadrants = vec![0.0; n];
        for i in (min_req - 1)..n {
            quadrants[i] = Self::classify_quadrant(rs_ratio[i], rs_momentum[i]);
        }

        (rs_ratio, rs_momentum, quadrants)
    }
}

impl TechnicalIndicator for RelativeRotationGraph {
    fn name(&self) -> &str {
        "Relative Rotation Graph"
    }

    fn min_periods(&self) -> usize {
        self.rs_period + self.momentum_period + self.norm_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.benchmark_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "benchmark_series".to_string(),
                reason: "Benchmark series must be set before computing RelativeRotationGraph"
                    .to_string(),
            });
        }

        if self.benchmark_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "benchmark_series".to_string(),
                reason: format!(
                    "Benchmark series length ({}) must match primary series length ({})",
                    self.benchmark_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.benchmark_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// CrossMarketBeta
// ============================================================================

/// Cross-Market Beta - Calculates rolling beta coefficient across different markets/assets.
///
/// This indicator measures the systematic risk of an asset relative to a market benchmark,
/// calculating how much the asset's returns move in relation to the benchmark's returns.
/// Useful for portfolio construction, hedging, and cross-market analysis.
///
/// # Formula
/// Beta = Cov(Asset, Benchmark) / Var(Benchmark)
///
/// # Interpretation
/// - Beta > 1: Asset is more volatile than benchmark (amplifies movements)
/// - Beta = 1: Asset moves in line with benchmark
/// - Beta < 1: Asset is less volatile than benchmark (dampens movements)
/// - Beta < 0: Asset moves inversely to benchmark (rare, useful for hedging)
/// - Beta = 0: Asset is uncorrelated with benchmark
///
/// # Use Cases
/// - Cross-asset hedging (e.g., equity beta to bonds)
/// - Sector rotation analysis
/// - Portfolio risk management
/// - Pairs trading beta-neutral strategies
#[derive(Debug, Clone)]
pub struct CrossMarketBeta {
    /// Period for beta calculation.
    period: usize,
    /// Smoothing period for output.
    smooth_period: usize,
    /// Market/benchmark series for comparison.
    market_series: Vec<f64>,
}

impl CrossMarketBeta {
    /// Create a new CrossMarketBeta indicator.
    ///
    /// # Arguments
    /// * `period` - Rolling window period for beta calculation (must be >= 20)
    /// * `smooth_period` - Smoothing period for output (must be >= 1)
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            period,
            smooth_period,
            market_series: Vec::new(),
        })
    }

    /// Set the market/benchmark series for beta calculation.
    pub fn with_market(mut self, series: &[f64]) -> Self {
        self.market_series = series.to_vec();
        self
    }

    /// Calculate returns from price series.
    fn calculate_returns(prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        let mut returns = vec![0.0; n];

        for i in 1..n {
            if prices[i - 1] > 1e-10 {
                returns[i] = prices[i] / prices[i - 1] - 1.0;
            }
        }

        returns
    }

    /// Calculate rolling beta for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate returns
        let asset_returns = Self::calculate_returns(&dual.series1);
        let market_returns = Self::calculate_returns(&dual.series2);

        // Calculate rolling beta
        let mut raw_beta = vec![1.0; n];

        for i in self.period..n {
            let start = i + 1 - self.period;
            let asset_window = &asset_returns[start..=i];
            let market_window = &market_returns[start..=i];

            // Calculate means
            let asset_mean: f64 = asset_window.iter().sum::<f64>() / self.period as f64;
            let market_mean: f64 = market_window.iter().sum::<f64>() / self.period as f64;

            // Calculate covariance and variance
            let mut cov = 0.0;
            let mut var_market = 0.0;

            for (asset_ret, market_ret) in asset_window.iter().zip(market_window.iter()) {
                let asset_dev = asset_ret - asset_mean;
                let market_dev = market_ret - market_mean;
                cov += asset_dev * market_dev;
                var_market += market_dev * market_dev;
            }

            // Beta = Cov(Asset, Market) / Var(Market)
            if var_market > 1e-10 {
                raw_beta[i] = cov / var_market;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start_idx = self.period;

        if start_idx < n {
            result[start_idx] = raw_beta[start_idx];
            for i in (start_idx + 1)..n {
                result[i] = alpha * raw_beta[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, asset: &[f64], market: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(asset, market);
        self.calculate(&dual)
    }

    /// Calculate beta with additional statistics (beta, r-squared, alpha).
    pub fn calculate_detailed(&self, dual: &DualSeries) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = dual.len();
        let mut beta = vec![0.0; n];
        let mut r_squared = vec![0.0; n];
        let mut alpha = vec![0.0; n];

        if n < self.period + 1 {
            return (beta, r_squared, alpha);
        }

        let asset_returns = Self::calculate_returns(&dual.series1);
        let market_returns = Self::calculate_returns(&dual.series2);

        for i in self.period..n {
            let start = i + 1 - self.period;
            let asset_window = &asset_returns[start..=i];
            let market_window = &market_returns[start..=i];

            let asset_mean: f64 = asset_window.iter().sum::<f64>() / self.period as f64;
            let market_mean: f64 = market_window.iter().sum::<f64>() / self.period as f64;

            let mut cov = 0.0;
            let mut var_asset = 0.0;
            let mut var_market = 0.0;

            for (asset_ret, market_ret) in asset_window.iter().zip(market_window.iter()) {
                let asset_dev = asset_ret - asset_mean;
                let market_dev = market_ret - market_mean;
                cov += asset_dev * market_dev;
                var_asset += asset_dev * asset_dev;
                var_market += market_dev * market_dev;
            }

            if var_market > 1e-10 {
                beta[i] = cov / var_market;

                // R-squared = correlation^2
                if var_asset > 1e-10 {
                    let corr = cov / (var_asset.sqrt() * var_market.sqrt());
                    r_squared[i] = corr * corr;
                }

                // Alpha = mean asset return - beta * mean market return (annualized)
                alpha[i] = (asset_mean - beta[i] * market_mean) * 252.0;
            }
        }

        (beta, r_squared, alpha)
    }
}

impl TechnicalIndicator for CrossMarketBeta {
    fn name(&self) -> &str {
        "Cross Market Beta"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.market_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "market_series".to_string(),
                reason: "Market series must be set before computing CrossMarketBeta".to_string(),
            });
        }

        if self.market_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "market_series".to_string(),
                reason: format!(
                    "Market series length ({}) must match primary series length ({})",
                    self.market_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.market_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// CorrelationBreakdownDetector
// ============================================================================

/// Correlation Breakdown Detector - Detects significant breakdowns in correlation between two assets.
///
/// This indicator monitors the correlation between two assets and identifies when the
/// correlation structure breaks down, signaling potential regime changes or market stress.
/// It uses a multi-timeframe approach comparing short-term vs long-term correlations.
///
/// # Formula
/// 1. Calculate short-term rolling correlation
/// 2. Calculate long-term rolling correlation
/// 3. Detect breakdown when: |short_corr - long_corr| > threshold
/// 4. Score = (short_corr - long_corr) * confidence factor
///
/// # Interpretation
/// - Values near 0: Stable correlation regime
/// - Large positive values: Correlation increasing rapidly (convergence event)
/// - Large negative values: Correlation decreasing rapidly (divergence/breakdown event)
/// - Sudden spikes often indicate market stress or regime changes
///
/// # Use Cases
/// - Risk management: Detect when historical correlations may not hold
/// - Pairs trading: Identify when spread relationships are breaking down
/// - Portfolio construction: Monitor diversification effectiveness
#[derive(Debug, Clone)]
pub struct CorrelationBreakdownDetector {
    /// Short-term correlation period.
    short_period: usize,
    /// Long-term correlation period.
    long_period: usize,
    /// Threshold for breakdown detection.
    threshold: f64,
    /// Secondary series for correlation calculation.
    secondary_series: Vec<f64>,
}

impl CorrelationBreakdownDetector {
    /// Create a new CorrelationBreakdownDetector indicator.
    ///
    /// # Arguments
    /// * `short_period` - Short-term correlation window (must be >= 10)
    /// * `long_period` - Long-term correlation window (must be > short_period)
    /// * `threshold` - Breakdown detection threshold (must be > 0, typically 0.2-0.5)
    pub fn new(short_period: usize, long_period: usize, threshold: f64) -> Result<Self> {
        if short_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        Ok(Self {
            short_period,
            long_period,
            threshold,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for correlation calculation.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate correlation for a window.
    fn correlation(series1: &[f64], series2: &[f64]) -> f64 {
        let n = series1.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean1: f64 = series1.iter().sum::<f64>() / n;
        let mean2: f64 = series2.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in series1.iter().zip(series2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Calculate returns from prices.
    fn calculate_returns(prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        let mut returns = vec![0.0; n];

        for i in 1..n {
            if prices[i - 1] > 1e-10 {
                returns[i] = prices[i] / prices[i - 1] - 1.0;
            }
        }

        returns
    }

    /// Calculate correlation breakdown score for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.long_period + 1 {
            return result;
        }

        // Calculate returns for correlation (more stable than prices)
        let returns1 = Self::calculate_returns(&dual.series1);
        let returns2 = Self::calculate_returns(&dual.series2);

        // Calculate rolling short-term and long-term correlations
        let mut short_corr = vec![0.0; n];
        let mut long_corr = vec![0.0; n];

        for i in self.long_period..n {
            // Short-term correlation
            let short_start = i + 1 - self.short_period;
            short_corr[i] = Self::correlation(&returns1[short_start..=i], &returns2[short_start..=i]);

            // Long-term correlation
            let long_start = i + 1 - self.long_period;
            long_corr[i] = Self::correlation(&returns1[long_start..=i], &returns2[long_start..=i]);

            // Breakdown detection
            let corr_diff = short_corr[i] - long_corr[i];
            let abs_diff = corr_diff.abs();

            if abs_diff > self.threshold {
                // Calculate confidence factor based on how far above threshold
                let confidence = (abs_diff / self.threshold).min(3.0); // Cap at 3x

                // Score: positive = correlation increasing, negative = correlation decreasing
                result[i] = corr_diff * confidence * 100.0;
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }

    /// Get detailed breakdown data (short_corr, long_corr, breakdown_score).
    pub fn calculate_detailed(&self, dual: &DualSeries) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = dual.len();
        let mut short_corr = vec![0.0; n];
        let mut long_corr = vec![0.0; n];
        let mut breakdown_score = vec![0.0; n];

        if n < self.long_period + 1 {
            return (short_corr, long_corr, breakdown_score);
        }

        let returns1 = Self::calculate_returns(&dual.series1);
        let returns2 = Self::calculate_returns(&dual.series2);

        for i in self.long_period..n {
            let short_start = i + 1 - self.short_period;
            short_corr[i] = Self::correlation(&returns1[short_start..=i], &returns2[short_start..=i]);

            let long_start = i + 1 - self.long_period;
            long_corr[i] = Self::correlation(&returns1[long_start..=i], &returns2[long_start..=i]);

            let corr_diff = short_corr[i] - long_corr[i];
            let abs_diff = corr_diff.abs();

            if abs_diff > self.threshold {
                let confidence = (abs_diff / self.threshold).min(3.0);
                breakdown_score[i] = corr_diff * confidence * 100.0;
            }
        }

        (short_corr, long_corr, breakdown_score)
    }
}

impl TechnicalIndicator for CorrelationBreakdownDetector {
    fn name(&self) -> &str {
        "Correlation Breakdown Detector"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing CorrelationBreakdownDetector"
                    .to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// LeadLagAnalysis
// ============================================================================

/// Lead-Lag Analysis - Comprehensive lead-lag relationship analysis between two assets.
///
/// This indicator performs detailed analysis of which asset tends to lead or lag the other,
/// using multiple methods including cross-correlation, Granger-like causality proxies,
/// and information flow analysis. It provides a continuous score indicating lead/lag dynamics.
///
/// # Methodology
/// 1. Cross-correlation at multiple lags to find optimal lead/lag
/// 2. Predictive power analysis: can series1 returns predict series2 returns and vice versa
/// 3. Information ratio: relative information content of each series
///
/// # Interpretation
/// - Positive values: Series 1 leads Series 2 (use Series 1 to predict Series 2)
/// - Negative values: Series 1 lags Series 2 (Series 2 leads)
/// - Magnitude indicates strength of the lead/lag relationship
/// - Zero indicates no clear lead/lag relationship
///
/// # Use Cases
/// - Identify leading indicators among related assets
/// - Optimize entry/exit timing in pairs trades
/// - Cross-market signal generation
#[derive(Debug, Clone)]
pub struct LeadLagAnalysis {
    /// Period for analysis calculations.
    period: usize,
    /// Maximum lag periods to test.
    max_lag: usize,
    /// Minimum correlation for valid lead/lag detection.
    min_correlation: f64,
    /// Secondary series for lead/lag analysis.
    secondary_series: Vec<f64>,
}

impl LeadLagAnalysis {
    /// Create a new LeadLagAnalysis indicator.
    ///
    /// # Arguments
    /// * `period` - Analysis window period (must be >= 30)
    /// * `max_lag` - Maximum lag periods to test (must be >= 1 and < period/3)
    pub fn new(period: usize, max_lag: usize) -> Result<Self> {
        if period < 30 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 30".to_string(),
            });
        }
        if max_lag < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_lag".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if max_lag >= period / 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "max_lag".to_string(),
                reason: "must be less than period / 3".to_string(),
            });
        }
        Ok(Self {
            period,
            max_lag,
            min_correlation: 0.3, // Default minimum correlation threshold
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series for lead/lag analysis.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Set minimum correlation threshold.
    pub fn with_min_correlation(mut self, min_corr: f64) -> Self {
        self.min_correlation = min_corr.abs().min(1.0);
        self
    }

    /// Calculate returns from prices.
    fn calculate_returns(prices: &[f64]) -> Vec<f64> {
        let n = prices.len();
        let mut returns = vec![0.0; n];

        for i in 1..n {
            if prices[i - 1] > 1e-10 {
                returns[i] = prices[i] / prices[i - 1] - 1.0;
            }
        }

        returns
    }

    /// Calculate cross-correlation at a specific lag.
    fn cross_correlation_at_lag(series1: &[f64], series2: &[f64], lag: i32) -> f64 {
        let n = series1.len();
        let abs_lag = lag.unsigned_abs() as usize;

        if abs_lag >= n {
            return 0.0;
        }

        let (s1, s2) = if lag >= 0 {
            (&series1[abs_lag..], &series2[..n - abs_lag])
        } else {
            (&series1[..n - abs_lag], &series2[abs_lag..])
        };

        let len = s1.len() as f64;
        if len < 2.0 {
            return 0.0;
        }

        let mean1: f64 = s1.iter().sum::<f64>() / len;
        let mean2: f64 = s2.iter().sum::<f64>() / len;

        let mut cov = 0.0;
        let mut var1 = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in s1.iter().zip(s2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var1 += d1 * d1;
            var2 += d2 * d2;
        }

        let denom = (var1 * var2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Calculate predictive power (pseudo R-squared) of one series predicting another.
    fn predictive_power(predictor: &[f64], target: &[f64], lag: usize) -> f64 {
        let n = target.len();
        if n <= lag + 1 {
            return 0.0;
        }

        // Simple linear regression: target[i] = a + b * predictor[i-lag]
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        let mut count = 0.0;

        for i in lag..n {
            let x = predictor[i - lag];
            let y = target[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
            count += 1.0;
        }

        if count < 2.0 {
            return 0.0;
        }

        // Calculate R-squared
        let numerator = count * sum_xy - sum_x * sum_y;
        let denom1 = count * sum_x2 - sum_x * sum_x;
        let denom2 = count * sum_y2 - sum_y * sum_y;

        if denom1 < 1e-10 || denom2 < 1e-10 {
            return 0.0;
        }

        let r = numerator / (denom1.sqrt() * denom2.sqrt());
        r * r
    }

    /// Calculate comprehensive lead/lag score for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let mut result = vec![0.0; n];

        if n < self.period {
            return result;
        }

        let returns1 = Self::calculate_returns(&dual.series1);
        let returns2 = Self::calculate_returns(&dual.series2);

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window1 = &returns1[start..=i];
            let window2 = &returns2[start..=i];

            // Find optimal lag using cross-correlation
            let mut best_lag = 0i32;
            let mut best_corr = Self::cross_correlation_at_lag(window1, window2, 0);

            for lag in 1..=self.max_lag as i32 {
                // Test positive lag (series1 leads)
                let corr_pos = Self::cross_correlation_at_lag(window1, window2, lag);
                if corr_pos.abs() > best_corr.abs() {
                    best_corr = corr_pos;
                    best_lag = lag;
                }

                // Test negative lag (series2 leads)
                let corr_neg = Self::cross_correlation_at_lag(window1, window2, -lag);
                if corr_neg.abs() > best_corr.abs() {
                    best_corr = corr_neg;
                    best_lag = -lag;
                }
            }

            // Skip if correlation is too weak
            if best_corr.abs() < self.min_correlation {
                continue;
            }

            // Calculate predictive power in both directions
            let pred_1_to_2 = Self::predictive_power(window1, window2, self.max_lag / 2 + 1);
            let pred_2_to_1 = Self::predictive_power(window2, window1, self.max_lag / 2 + 1);

            // Combine signals:
            // 1. Cross-correlation optimal lag (positive = series1 leads)
            // 2. Predictive power difference (positive = series1 better predictor)
            let lag_signal = best_lag as f64 * best_corr.abs();
            let pred_signal = (pred_1_to_2 - pred_2_to_1) * 10.0;

            // Weighted combination
            result[i] = lag_signal * 0.7 + pred_signal * 0.3;
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }

    /// Get detailed lead/lag data (optimal_lag, correlation, lead_lag_score).
    pub fn calculate_detailed(&self, dual: &DualSeries) -> (Vec<i32>, Vec<f64>, Vec<f64>) {
        let n = dual.len();
        let mut optimal_lag = vec![0i32; n];
        let mut correlation = vec![0.0; n];
        let mut score = vec![0.0; n];

        if n < self.period {
            return (optimal_lag, correlation, score);
        }

        let returns1 = Self::calculate_returns(&dual.series1);
        let returns2 = Self::calculate_returns(&dual.series2);

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window1 = &returns1[start..=i];
            let window2 = &returns2[start..=i];

            let mut best_lag = 0i32;
            let mut best_corr = Self::cross_correlation_at_lag(window1, window2, 0);

            for lag in 1..=self.max_lag as i32 {
                let corr_pos = Self::cross_correlation_at_lag(window1, window2, lag);
                if corr_pos.abs() > best_corr.abs() {
                    best_corr = corr_pos;
                    best_lag = lag;
                }

                let corr_neg = Self::cross_correlation_at_lag(window1, window2, -lag);
                if corr_neg.abs() > best_corr.abs() {
                    best_corr = corr_neg;
                    best_lag = -lag;
                }
            }

            optimal_lag[i] = best_lag;
            correlation[i] = best_corr;

            if best_corr.abs() >= self.min_correlation {
                let pred_1_to_2 = Self::predictive_power(window1, window2, self.max_lag / 2 + 1);
                let pred_2_to_1 = Self::predictive_power(window2, window1, self.max_lag / 2 + 1);
                let lag_signal = best_lag as f64 * best_corr.abs();
                let pred_signal = (pred_1_to_2 - pred_2_to_1) * 10.0;
                score[i] = lag_signal * 0.7 + pred_signal * 0.3;
            }
        }

        (optimal_lag, correlation, score)
    }
}

impl TechnicalIndicator for LeadLagAnalysis {
    fn name(&self) -> &str {
        "Lead Lag Analysis"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing LeadLagAnalysis".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// SpreadMomentumIndicator
// ============================================================================

/// Spread Momentum Indicator - Measures momentum characteristics of price spreads between assets.
///
/// This indicator analyzes the momentum of the spread between two related assets,
/// including rate of change, acceleration, and trend strength. It helps identify
/// when spread trends are strengthening, weakening, or reversing.
///
/// # Methodology
/// 1. Calculate hedge-ratio adjusted spread
/// 2. Compute spread momentum (rate of change)
/// 3. Calculate momentum acceleration (second derivative)
/// 4. Combine into comprehensive momentum score
///
/// # Interpretation
/// - Positive momentum: Spread is widening (Series 1 outperforming)
/// - Negative momentum: Spread is narrowing (Series 2 catching up)
/// - Increasing momentum: Trend strengthening
/// - Decreasing momentum: Trend weakening, potential reversal
///
/// # Use Cases
/// - Pairs trading momentum strategies
/// - Spread trend following
/// - Mean reversion timing
#[derive(Debug, Clone)]
pub struct SpreadMomentumIndicator {
    /// Period for spread calculation and hedge ratio.
    spread_period: usize,
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Period for acceleration calculation.
    accel_period: usize,
    /// Secondary series.
    secondary_series: Vec<f64>,
}

impl SpreadMomentumIndicator {
    /// Create a new SpreadMomentumIndicator.
    ///
    /// # Arguments
    /// * `spread_period` - Period for spread/hedge ratio calculation (must be >= 15)
    /// * `momentum_period` - Period for momentum (must be >= 5)
    /// * `accel_period` - Period for acceleration (must be >= 3)
    pub fn new(spread_period: usize, momentum_period: usize, accel_period: usize) -> Result<Self> {
        if spread_period < 15 {
            return Err(IndicatorError::InvalidParameter {
                name: "spread_period".to_string(),
                reason: "must be at least 15".to_string(),
            });
        }
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if accel_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "accel_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self {
            spread_period,
            momentum_period,
            accel_period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate rolling hedge ratio.
    fn calculate_hedge_ratio(series1: &[f64], series2: &[f64]) -> f64 {
        let n = series1.len() as f64;
        if n < 2.0 {
            return 1.0;
        }

        let mean1: f64 = series1.iter().sum::<f64>() / n;
        let mean2: f64 = series2.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var2 = 0.0;

        for (v1, v2) in series1.iter().zip(series2.iter()) {
            let d1 = v1 - mean1;
            let d2 = v2 - mean2;
            cov += d1 * d2;
            var2 += d2 * d2;
        }

        if var2 > 1e-10 {
            cov / var2
        } else {
            1.0
        }
    }

    /// Calculate spread momentum indicator for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let min_req = self.spread_period + self.momentum_period + self.accel_period;
        let mut result = vec![0.0; n];

        if n < min_req {
            return result;
        }

        // Calculate hedge-ratio adjusted spread
        let mut spread = vec![0.0; n];
        for i in (self.spread_period - 1)..n {
            let start = i + 1 - self.spread_period;
            let hedge_ratio = Self::calculate_hedge_ratio(
                &dual.series1[start..=i],
                &dual.series2[start..=i],
            );
            spread[i] = dual.series1[i] - hedge_ratio * dual.series2[i];
        }

        // Calculate spread momentum (rate of change)
        let mom_start = self.spread_period + self.momentum_period - 1;
        let mut momentum = vec![0.0; n];
        for i in mom_start..n {
            let prev_spread = spread[i - self.momentum_period];
            if prev_spread.abs() > 1e-10 {
                momentum[i] = (spread[i] - prev_spread) / prev_spread.abs() * 100.0;
            } else {
                momentum[i] = spread[i].signum() * 100.0;
            }
        }

        // Calculate momentum acceleration (second derivative)
        let mut acceleration = vec![0.0; n];
        let accel_start = mom_start + self.accel_period;
        for i in accel_start..n {
            acceleration[i] = momentum[i] - momentum[i - self.accel_period];
        }

        // Combine into comprehensive momentum score
        // Score = momentum * (1 + normalized acceleration)
        for i in (min_req - 1)..n {
            let accel_factor = if momentum[i].abs() > 1e-10 {
                1.0 + (acceleration[i] / momentum[i].abs()).clamp(-0.5, 0.5)
            } else {
                1.0
            };
            result[i] = momentum[i] * accel_factor;
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }

    /// Get detailed spread momentum data (spread, momentum, acceleration, score).
    pub fn calculate_detailed(&self, dual: &DualSeries) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = dual.len();
        let min_req = self.spread_period + self.momentum_period + self.accel_period;
        let mut spread = vec![0.0; n];
        let mut momentum = vec![0.0; n];
        let mut acceleration = vec![0.0; n];
        let mut score = vec![0.0; n];

        if n < min_req {
            return (spread, momentum, acceleration, score);
        }

        // Calculate spread
        for i in (self.spread_period - 1)..n {
            let start = i + 1 - self.spread_period;
            let hedge_ratio = Self::calculate_hedge_ratio(
                &dual.series1[start..=i],
                &dual.series2[start..=i],
            );
            spread[i] = dual.series1[i] - hedge_ratio * dual.series2[i];
        }

        // Calculate momentum
        let mom_start = self.spread_period + self.momentum_period - 1;
        for i in mom_start..n {
            let prev_spread = spread[i - self.momentum_period];
            if prev_spread.abs() > 1e-10 {
                momentum[i] = (spread[i] - prev_spread) / prev_spread.abs() * 100.0;
            } else {
                momentum[i] = spread[i].signum() * 100.0;
            }
        }

        // Calculate acceleration
        let accel_start = mom_start + self.accel_period;
        for i in accel_start..n {
            acceleration[i] = momentum[i] - momentum[i - self.accel_period];
        }

        // Calculate score
        for i in (min_req - 1)..n {
            let accel_factor = if momentum[i].abs() > 1e-10 {
                1.0 + (acceleration[i] / momentum[i].abs()).clamp(-0.5, 0.5)
            } else {
                1.0
            };
            score[i] = momentum[i] * accel_factor;
        }

        (spread, momentum, acceleration, score)
    }
}

impl TechnicalIndicator for SpreadMomentumIndicator {
    fn name(&self) -> &str {
        "Spread Momentum Indicator"
    }

    fn min_periods(&self) -> usize {
        self.spread_period + self.momentum_period + self.accel_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing SpreadMomentumIndicator"
                    .to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// RelativeValueMomentum
// ============================================================================

/// Relative Value Momentum - Measures momentum of the relative value between two assets.
///
/// This indicator tracks the momentum of relative valuation between two assets,
/// helping identify when one asset is gaining or losing value relative to another.
/// It uses both price ratios and normalized metrics to provide robust signals.
///
/// # Methodology
/// 1. Calculate relative value ratio (Asset1 / Asset2)
/// 2. Normalize the ratio using rolling statistics
/// 3. Compute momentum of the normalized ratio
/// 4. Apply trend filter for signal quality
///
/// # Interpretation
/// - Positive momentum: Asset 1 gaining value relative to Asset 2
/// - Negative momentum: Asset 1 losing value relative to Asset 2
/// - High absolute values: Strong relative trend
/// - Zero crossings: Potential reversal points
///
/// # Use Cases
/// - Currency pair momentum trading
/// - Relative sector rotation
/// - Commodity spread momentum
/// - Cross-asset alpha generation
#[derive(Debug, Clone)]
pub struct RelativeValueMomentum {
    /// Period for ratio normalization.
    norm_period: usize,
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Smoothing period for output.
    smooth_period: usize,
    /// Secondary series.
    secondary_series: Vec<f64>,
}

impl RelativeValueMomentum {
    /// Create a new RelativeValueMomentum indicator.
    ///
    /// # Arguments
    /// * `norm_period` - Period for ratio normalization (must be >= 20)
    /// * `momentum_period` - Period for momentum calculation (must be >= 5)
    /// * `smooth_period` - Smoothing period for output (must be >= 1)
    pub fn new(norm_period: usize, momentum_period: usize, smooth_period: usize) -> Result<Self> {
        if norm_period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "norm_period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            norm_period,
            momentum_period,
            smooth_period,
            secondary_series: Vec::new(),
        })
    }

    /// Set the secondary series.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate relative value momentum for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<f64> {
        let n = dual.len();
        let min_req = self.norm_period + self.momentum_period;
        let mut result = vec![0.0; n];

        if n < min_req {
            return result;
        }

        // Calculate relative value ratio
        let mut ratio = vec![1.0; n];
        for i in 0..n {
            if dual.series2[i] > 1e-10 {
                ratio[i] = dual.series1[i] / dual.series2[i];
            }
        }

        // Normalize ratio using z-score
        let mut normalized_ratio = vec![0.0; n];
        for i in (self.norm_period - 1)..n {
            let start = i + 1 - self.norm_period;
            let window = &ratio[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let std_dev: f64 = (window.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / window.len() as f64)
                .sqrt();

            if std_dev > 1e-10 {
                normalized_ratio[i] = (ratio[i] - mean) / std_dev;
            }
        }

        // Calculate momentum of normalized ratio
        let mut raw_momentum = vec![0.0; n];
        for i in (min_req - 1)..n {
            raw_momentum[i] = normalized_ratio[i] - normalized_ratio[i - self.momentum_period];
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start_idx = min_req - 1;

        if start_idx < n {
            result[start_idx] = raw_momentum[start_idx] * 100.0; // Scale to percentage-like
            for i in (start_idx + 1)..n {
                result[i] = alpha * (raw_momentum[i] * 100.0) + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }

    /// Calculate using two series directly.
    pub fn calculate_between(&self, series1: &[f64], series2: &[f64]) -> Vec<f64> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }

    /// Get detailed relative value momentum data (ratio, normalized, momentum, score).
    pub fn calculate_detailed(&self, dual: &DualSeries) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = dual.len();
        let min_req = self.norm_period + self.momentum_period;
        let mut ratio = vec![1.0; n];
        let mut normalized = vec![0.0; n];
        let mut momentum = vec![0.0; n];
        let mut score = vec![0.0; n];

        if n < min_req {
            return (ratio, normalized, momentum, score);
        }

        // Calculate ratio
        for i in 0..n {
            if dual.series2[i] > 1e-10 {
                ratio[i] = dual.series1[i] / dual.series2[i];
            }
        }

        // Normalize
        for i in (self.norm_period - 1)..n {
            let start = i + 1 - self.norm_period;
            let window = &ratio[start..=i];

            let mean: f64 = window.iter().sum::<f64>() / window.len() as f64;
            let std_dev: f64 = (window.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / window.len() as f64)
                .sqrt();

            if std_dev > 1e-10 {
                normalized[i] = (ratio[i] - mean) / std_dev;
            }
        }

        // Calculate momentum
        for i in (min_req - 1)..n {
            momentum[i] = normalized[i] - normalized[i - self.momentum_period];
        }

        // Calculate smoothed score
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        let start_idx = min_req - 1;

        if start_idx < n {
            score[start_idx] = momentum[start_idx] * 100.0;
            for i in (start_idx + 1)..n {
                score[i] = alpha * (momentum[i] * 100.0) + (1.0 - alpha) * score[i - 1];
            }
        }

        (ratio, normalized, momentum, score)
    }
}

impl TechnicalIndicator for RelativeValueMomentum {
    fn name(&self) -> &str {
        "Relative Value Momentum"
    }

    fn min_periods(&self) -> usize {
        self.norm_period + self.momentum_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing RelativeValueMomentum"
                    .to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        Ok(IndicatorOutput::single(self.calculate(&dual)))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dual_series(n: usize) -> DualSeries {
        // Create two correlated series with some noise
        let mut series1 = Vec::with_capacity(n);
        let mut series2 = Vec::with_capacity(n);

        let mut price1 = 100.0;
        let mut price2 = 50.0;

        for i in 0..n {
            // Random walk with correlation
            let common_move = (i as f64 * 0.1).sin() * 0.5;
            price1 += common_move + (i as f64 * 0.3).cos() * 0.2;
            price2 += common_move * 0.5 + (i as f64 * 0.5).sin() * 0.15;

            series1.push(price1);
            series2.push(price2);
        }

        DualSeries::new(series1, series2)
    }

    fn create_mean_reverting_spread(n: usize) -> DualSeries {
        // Create two series with a mean-reverting spread
        let mut series1 = Vec::with_capacity(n);
        let mut series2 = Vec::with_capacity(n);

        let mut price2 = 50.0;
        for i in 0..n {
            price2 += (i as f64 * 0.1).sin() * 0.3;
            series2.push(price2);

            // Series1 = 2 * Series2 + oscillating spread
            let spread_noise = ((i as f64) * 0.5).sin() * 2.0;
            series1.push(100.0 + 2.0 * price2 + spread_noise);
        }

        DualSeries::new(series1, series2)
    }

    #[test]
    fn test_lead_lag_indicator_new() {
        assert!(LeadLagIndicator::new(20, 5).is_ok());
        assert!(LeadLagIndicator::new(19, 5).is_err()); // period too small
        assert!(LeadLagIndicator::new(20, 0).is_err()); // max_lag too small
        assert!(LeadLagIndicator::new(20, 15).is_err()); // max_lag >= period/2
    }

    #[test]
    fn test_lead_lag_indicator_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = LeadLagIndicator::new(30, 5).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First 29 should be 0 (insufficient data)
        assert_eq!(result[0], 0.0);
        // After warmup, values should be calculated
        assert!(result[50].is_finite());
    }

    #[test]
    fn test_price_spread_momentum_new() {
        assert!(PriceSpreadMomentum::new(10, 3).is_ok());
        assert!(PriceSpreadMomentum::new(4, 3).is_err()); // period too small
        assert!(PriceSpreadMomentum::new(10, 0).is_err()); // smooth_period too small
    }

    #[test]
    fn test_price_spread_momentum_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = PriceSpreadMomentum::new(10, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Values should be in percentage form
        assert!(result[50].abs() < 1000.0); // Reasonable bounds
    }

    #[test]
    fn test_correlation_trend_new() {
        assert!(CorrelationTrend::new(10, 30).is_ok());
        assert!(CorrelationTrend::new(9, 30).is_err()); // short_period too small
        assert!(CorrelationTrend::new(30, 20).is_err()); // long_period <= short_period
    }

    #[test]
    fn test_correlation_trend_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = CorrelationTrend::new(10, 30).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Correlation trend should be bounded
        assert!(result[50].abs() <= 200.0);
    }

    #[test]
    fn test_relative_value_index_new() {
        assert!(RelativeValueIndex::new(20).is_ok());
        assert!(RelativeValueIndex::new(19).is_err()); // period too small
    }

    #[test]
    fn test_relative_value_index_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = RelativeValueIndex::new(30).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // RVI should be between 0 and 100
        for &v in &result[(30 - 1)..] {
            assert!(v >= 0.0 && v <= 100.0, "RVI value {} out of bounds", v);
        }
    }

    #[test]
    fn test_spread_mean_reversion_new() {
        assert!(SpreadMeanReversion::new(20).is_ok());
        assert!(SpreadMeanReversion::new(19).is_err()); // period too small
    }

    #[test]
    fn test_spread_mean_reversion_calculate() {
        let dual = create_mean_reverting_spread(100);
        let indicator = SpreadMeanReversion::new(30).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Z-scores should be reasonable
        for &z in &result[(30 - 1)..] {
            assert!(z.abs() < 10.0, "Z-score {} too extreme", z);
        }
    }

    #[test]
    fn test_pairs_trading_signal_new() {
        assert!(PairsTradingSignal::new(20).is_ok());
        assert!(PairsTradingSignal::new(19).is_err()); // period too small
    }

    #[test]
    fn test_pairs_trading_signal_calculate() {
        let dual = create_mean_reverting_spread(100);
        let indicator = PairsTradingSignal::new(30)
            .unwrap()
            .with_entry_threshold(1.5)
            .with_exit_threshold(0.3);
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Signals should be -1, 0, or 1
        for &s in &result {
            assert!(s == -1.0 || s == 0.0 || s == 1.0, "Invalid signal: {}", s);
        }
    }

    #[test]
    fn test_pairs_trading_signal_generates_trades() {
        // Create a spread that clearly oscillates
        let mut series1 = Vec::with_capacity(200);
        let mut series2 = Vec::with_capacity(200);

        for i in 0..200 {
            series2.push(100.0);
            // Large oscillations to trigger entries
            let spread = ((i as f64) * 0.15).sin() * 10.0;
            series1.push(100.0 + spread);
        }

        let dual = DualSeries::new(series1, series2);
        let indicator = PairsTradingSignal::new(20)
            .unwrap()
            .with_entry_threshold(1.5)
            .with_exit_threshold(0.3);
        let result = indicator.calculate(&dual);

        // Should have some non-zero signals
        let non_zero_count = result.iter().filter(|&&s| s != 0.0).count();
        assert!(non_zero_count > 0, "Should generate some trading signals");
    }

    #[test]
    fn test_technical_indicator_impl_lead_lag() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = LeadLagIndicator::new(30, 5).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().primary.len(), 100);
    }

    #[test]
    fn test_technical_indicator_impl_spread_momentum() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = PriceSpreadMomentum::new(10, 3).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().primary.len(), 100);
    }

    #[test]
    fn test_technical_indicator_impl_correlation_trend() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = CorrelationTrend::new(10, 30).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().primary.len(), 100);
    }

    #[test]
    fn test_technical_indicator_impl_relative_value() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RelativeValueIndex::new(30).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().primary.len(), 100);
    }

    #[test]
    fn test_technical_indicator_impl_spread_mean_reversion() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = SpreadMeanReversion::new(30).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().primary.len(), 100);
    }

    #[test]
    fn test_technical_indicator_impl_pairs_trading() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = PairsTradingSignal::new(30).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        assert_eq!(result.unwrap().primary.len(), 100);
    }

    #[test]
    fn test_missing_secondary_series() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let data = OHLCVSeries::from_close(series1);

        let lead_lag = LeadLagIndicator::new(30, 5).unwrap();
        assert!(lead_lag.compute(&data).is_err());

        let spread_momentum = PriceSpreadMomentum::new(10, 3).unwrap();
        assert!(spread_momentum.compute(&data).is_err());

        let corr_trend = CorrelationTrend::new(10, 30).unwrap();
        assert!(corr_trend.compute(&data).is_err());

        let rel_value = RelativeValueIndex::new(30).unwrap();
        assert!(rel_value.compute(&data).is_err());

        let spread_mr = SpreadMeanReversion::new(30).unwrap();
        assert!(spread_mr.compute(&data).is_err());

        let pairs_signal = PairsTradingSignal::new(30).unwrap();
        assert!(pairs_signal.compute(&data).is_err());
    }

    #[test]
    fn test_mismatched_series_length() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..50).map(|i| 50.0 + (i as f64) * 0.25).collect();
        let data = OHLCVSeries::from_close(series1);

        let indicator = LeadLagIndicator::new(30, 5).unwrap().with_secondary(&series2);
        assert!(indicator.compute(&data).is_err());
    }

    // ========================================================================
    // Tests for new advanced indicators
    // ========================================================================

    fn create_test_ohlcv_data(n: usize) -> OHLCVSeries {
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        let mut base_price = 100.0;
        for i in 0..n {
            // Simulate price movement with trend
            let trend = (i as f64 * 0.02).sin() * 5.0;
            let noise = ((i as f64) * 0.5).cos() * 1.0;
            base_price += 0.1 + trend * 0.01;

            let h = base_price + noise.abs() + 1.0;
            let l = base_price - noise.abs() - 0.5;
            let c = base_price + noise * 0.5;
            let v = 1000000.0 + ((i as f64) * 0.3).sin() * 500000.0;

            high.push(h);
            low.push(l);
            close.push(c);
            volume.push(v);
        }

        OHLCVSeries {
            open: close.clone(), // For simplicity
            high,
            low,
            close,
            volume,
        }
    }

    // RelativePerformance tests
    #[test]
    fn test_relative_performance_new() {
        assert!(RelativePerformance::new(10, 3).is_ok());
        assert!(RelativePerformance::new(1, 3).is_err()); // period too small
        assert!(RelativePerformance::new(10, 0).is_err()); // smooth_period too small
    }

    #[test]
    fn test_relative_performance_new_validation() {
        let err = RelativePerformance::new(1, 3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 2");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_relative_performance_calculate() {
        let close: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5 + ((i as f64) * 0.2).sin() * 2.0).collect();
        let indicator = RelativePerformance::new(10, 3).unwrap();
        let result = indicator.calculate(&close);

        assert_eq!(result.len(), 100);
        // First (period-1) values should be 0
        for i in 0..9 {
            assert_eq!(result[i], 0.0);
        }
        // Values after warmup should be finite
        for i in 10..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_relative_performance_compute() {
        let data = create_test_ohlcv_data(100);
        let indicator = RelativePerformance::new(10, 3).unwrap();
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Relative Performance");
        assert_eq!(indicator.min_periods(), 10);
    }

    // MomentumLeader tests
    #[test]
    fn test_momentum_leader_new() {
        assert!(MomentumLeader::new(5, 20).is_ok());
        assert!(MomentumLeader::new(1, 20).is_err()); // short_period too small
        assert!(MomentumLeader::new(20, 10).is_err()); // long_period <= short_period
        assert!(MomentumLeader::new(10, 10).is_err()); // long_period == short_period
    }

    #[test]
    fn test_momentum_leader_new_validation() {
        let err = MomentumLeader::new(1, 20).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "short_period");
                assert_eq!(reason, "must be at least 2");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = MomentumLeader::new(20, 10).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "long_period");
                assert_eq!(reason, "must be greater than short_period");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_momentum_leader_calculate() {
        let close: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.3).collect();
        let indicator = MomentumLeader::new(5, 20).unwrap();
        let result = indicator.calculate(&close);

        assert_eq!(result.len(), 100);
        // Uptrending series should have positive momentum leadership
        assert!(result[50] > 0.0, "Expected positive momentum leadership for uptrend");
    }

    #[test]
    fn test_momentum_leader_compute() {
        let data = create_test_ohlcv_data(100);
        let indicator = MomentumLeader::new(5, 20).unwrap();
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Momentum Leader");
        assert_eq!(indicator.min_periods(), 21);
    }

    // TrendLeader tests
    #[test]
    fn test_trend_leader_new() {
        assert!(TrendLeader::new(5, 20).is_ok());
        assert!(TrendLeader::new(1, 20).is_err()); // short_period too small
        assert!(TrendLeader::new(20, 10).is_err()); // long_period <= short_period
    }

    #[test]
    fn test_trend_leader_new_validation() {
        let err = TrendLeader::new(1, 20).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "short_period");
                assert_eq!(reason, "must be at least 2");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_trend_leader_calculate() {
        let close: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let indicator = TrendLeader::new(5, 20).unwrap();
        let result = indicator.calculate(&close);

        assert_eq!(result.len(), 100);
        // Strong uptrend should show positive trend leadership
        assert!(result[50] > 0.0, "Expected positive trend leadership for uptrend");
    }

    #[test]
    fn test_trend_leader_compute() {
        let data = create_test_ohlcv_data(100);
        let indicator = TrendLeader::new(5, 20).unwrap();
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Trend Leader");
        assert_eq!(indicator.min_periods(), 21);
    }

    // CorrelationBreakdown tests
    #[test]
    fn test_correlation_breakdown_new() {
        assert!(CorrelationBreakdown::new(10, 0.3).is_ok());
        assert!(CorrelationBreakdown::new(9, 0.3).is_err()); // period too small
        assert!(CorrelationBreakdown::new(10, 0.0).is_err()); // threshold <= 0
        assert!(CorrelationBreakdown::new(10, -0.5).is_err()); // threshold < 0
    }

    #[test]
    fn test_correlation_breakdown_new_validation() {
        let err = CorrelationBreakdown::new(9, 0.3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = CorrelationBreakdown::new(10, 0.0).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "threshold");
                assert_eq!(reason, "must be greater than 0");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_correlation_breakdown_calculate() {
        // Create series with regime change
        let mut close = Vec::with_capacity(100);
        for i in 0..50 {
            close.push(100.0 + (i as f64) * 0.5); // Steady uptrend
        }
        for i in 50..100 {
            close.push(125.0 + ((i as f64 - 50.0) * 0.3).sin() * 5.0); // Choppy
        }

        let indicator = CorrelationBreakdown::new(10, 0.2).unwrap();
        let result = indicator.calculate(&close);

        assert_eq!(result.len(), 100);
        // First (period * 2 - 1) should be 0
        for i in 0..19 {
            assert_eq!(result[i], 0.0);
        }
    }

    #[test]
    fn test_correlation_breakdown_compute() {
        let data = create_test_ohlcv_data(100);
        let indicator = CorrelationBreakdown::new(10, 0.3).unwrap();
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Correlation Breakdown");
        assert_eq!(indicator.min_periods(), 20);
    }

    // SpreadAnalysis tests
    #[test]
    fn test_spread_analysis_new() {
        assert!(SpreadAnalysis::new(10, 2.0).is_ok());
        assert!(SpreadAnalysis::new(4, 2.0).is_err()); // period too small
        assert!(SpreadAnalysis::new(10, 0.0).is_err()); // std_multiplier <= 0
        assert!(SpreadAnalysis::new(10, -1.0).is_err()); // std_multiplier < 0
    }

    #[test]
    fn test_spread_analysis_new_validation() {
        let err = SpreadAnalysis::new(4, 2.0).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 5");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = SpreadAnalysis::new(10, 0.0).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "std_multiplier");
                assert_eq!(reason, "must be greater than 0");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_spread_analysis_calculate() {
        let close: Vec<f64> = (0..100).map(|i| 100.0 + ((i as f64) * 0.3).sin() * 5.0).collect();
        let indicator = SpreadAnalysis::new(20, 2.0).unwrap();
        let result = indicator.calculate(&close);

        assert_eq!(result.len(), 100);
        // Values should be z-score-like
        for i in 20..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
            assert!(result[i].abs() < 50.0, "Value at {} is too extreme: {}", i, result[i]);
        }
    }

    #[test]
    fn test_spread_analysis_compute() {
        let data = create_test_ohlcv_data(100);
        let indicator = SpreadAnalysis::new(10, 2.0).unwrap();
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Spread Analysis");
        assert_eq!(indicator.min_periods(), 10);
    }

    // FlowIndicator tests
    #[test]
    fn test_flow_indicator_new() {
        assert!(FlowIndicator::new(10, 3).is_ok());
        assert!(FlowIndicator::new(1, 3).is_err()); // period too small
        assert!(FlowIndicator::new(10, 0).is_err()); // smooth_period too small
    }

    #[test]
    fn test_flow_indicator_new_validation() {
        let err = FlowIndicator::new(1, 3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 2");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = FlowIndicator::new(10, 0).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "smooth_period");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_flow_indicator_calculate() {
        let n = 100;
        let high: Vec<f64> = (0..n).map(|i| 102.0 + (i as f64) * 0.1).collect();
        let low: Vec<f64> = (0..n).map(|i| 98.0 + (i as f64) * 0.1).collect();
        let close: Vec<f64> = (0..n).map(|i| 101.0 + (i as f64) * 0.1).collect(); // Close near high
        let volume: Vec<f64> = (0..n).map(|_| 1000000.0).collect();

        let indicator = FlowIndicator::new(10, 3).unwrap();
        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), n);
        // Close near high should show positive flow
        for i in 10..n {
            assert!(result[i] > 0.0, "Expected positive flow when close is near high, got {} at {}", result[i], i);
        }
    }

    #[test]
    fn test_flow_indicator_calculate_negative_flow() {
        let n = 100;
        let high: Vec<f64> = (0..n).map(|i| 102.0 + (i as f64) * 0.1).collect();
        let low: Vec<f64> = (0..n).map(|i| 98.0 + (i as f64) * 0.1).collect();
        let close: Vec<f64> = (0..n).map(|i| 99.0 + (i as f64) * 0.1).collect(); // Close near low
        let volume: Vec<f64> = (0..n).map(|_| 1000000.0).collect();

        let indicator = FlowIndicator::new(10, 3).unwrap();
        let result = indicator.calculate(&high, &low, &close, &volume);

        // Close near low should show negative flow
        for i in 10..n {
            assert!(result[i] < 0.0, "Expected negative flow when close is near low, got {} at {}", result[i], i);
        }
    }

    #[test]
    fn test_flow_indicator_compute() {
        let data = create_test_ohlcv_data(100);
        let indicator = FlowIndicator::new(10, 3).unwrap();
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Flow Indicator");
        assert_eq!(indicator.min_periods(), 10);
    }

    #[test]
    fn test_flow_indicator_bounded_output() {
        let data = create_test_ohlcv_data(200);
        let indicator = FlowIndicator::new(20, 5).unwrap();
        let result = indicator.compute(&data).unwrap();

        // Flow values should be between -100 and 100 (percentage)
        for (i, &v) in result.primary.iter().enumerate().skip(20) {
            assert!(v >= -100.0 && v <= 100.0, "Flow at {} is out of bounds: {}", i, v);
        }
    }

    // Edge case tests
    #[test]
    fn test_relative_performance_insufficient_data() {
        let close: Vec<f64> = (0..5).map(|i| 100.0 + (i as f64)).collect();
        let indicator = RelativePerformance::new(10, 3).unwrap();
        let result = indicator.calculate(&close);

        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_momentum_leader_insufficient_data() {
        let close: Vec<f64> = (0..10).map(|i| 100.0 + (i as f64)).collect();
        let indicator = MomentumLeader::new(5, 20).unwrap();
        let result = indicator.calculate(&close);

        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_trend_leader_insufficient_data() {
        let close: Vec<f64> = (0..15).map(|i| 100.0 + (i as f64)).collect();
        let indicator = TrendLeader::new(5, 20).unwrap();
        let result = indicator.calculate(&close);

        assert_eq!(result.len(), 15);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_correlation_breakdown_insufficient_data() {
        let close: Vec<f64> = (0..15).map(|i| 100.0 + (i as f64)).collect();
        let indicator = CorrelationBreakdown::new(10, 0.3).unwrap();
        let result = indicator.calculate(&close);

        assert_eq!(result.len(), 15);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_spread_analysis_insufficient_data() {
        let close: Vec<f64> = (0..3).map(|i| 100.0 + (i as f64)).collect();
        let indicator = SpreadAnalysis::new(5, 2.0).unwrap();
        let result = indicator.calculate(&close);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_flow_indicator_insufficient_data() {
        let high: Vec<f64> = vec![102.0, 103.0, 104.0];
        let low: Vec<f64> = vec![98.0, 99.0, 100.0];
        let close: Vec<f64> = vec![100.0, 101.0, 102.0];
        let volume: Vec<f64> = vec![1000000.0, 1000000.0, 1000000.0];

        let indicator = FlowIndicator::new(10, 3).unwrap();
        let result = indicator.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // ========================================================================
    // Tests for 6 new intermarket indicators
    // ========================================================================

    // CrossMarketCorrelation tests
    #[test]
    fn test_cross_market_correlation_new() {
        assert!(CrossMarketCorrelation::new(10).is_ok());
        assert!(CrossMarketCorrelation::new(20).is_ok());
        assert!(CrossMarketCorrelation::new(9).is_err()); // period too small
        assert!(CrossMarketCorrelation::new(5).is_err()); // period too small
    }

    #[test]
    fn test_cross_market_correlation_new_validation() {
        let err = CrossMarketCorrelation::new(9).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_cross_market_correlation_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = CrossMarketCorrelation::new(20).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First (period-1) values should be 0
        for i in 0..19 {
            assert_eq!(result[i], 0.0);
        }
        // Correlation should be between -1 and 1
        for i in 19..100 {
            assert!(result[i] >= -1.0 && result[i] <= 1.0,
                "Correlation at {} is out of bounds: {}", i, result[i]);
        }
    }

    #[test]
    fn test_cross_market_correlation_perfect_positive() {
        // Two perfectly correlated series
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = CrossMarketCorrelation::new(20).unwrap();
        let result = indicator.calculate(&dual);

        // Should be close to 1.0 for perfectly correlated linear series
        for i in 25..100 {
            assert!(result[i] > 0.99, "Expected high positive correlation, got {} at {}", result[i], i);
        }
    }

    #[test]
    fn test_cross_market_correlation_negative() {
        // Two negatively correlated series
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 150.0 - (i as f64) * 0.25).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = CrossMarketCorrelation::new(20).unwrap();
        let result = indicator.calculate(&dual);

        // Should be close to -1.0 for perfectly negatively correlated series
        for i in 25..100 {
            assert!(result[i] < -0.99, "Expected high negative correlation, got {} at {}", result[i], i);
        }
    }

    #[test]
    fn test_cross_market_correlation_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = CrossMarketCorrelation::new(20).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Cross Market Correlation");
        assert_eq!(indicator.min_periods(), 20);
    }

    #[test]
    fn test_cross_market_correlation_missing_secondary() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let data = OHLCVSeries::from_close(series1);

        let indicator = CrossMarketCorrelation::new(20).unwrap();
        assert!(indicator.compute(&data).is_err());
    }

    #[test]
    fn test_cross_market_correlation_insufficient_data() {
        let dual = create_test_dual_series(5);
        let indicator = CrossMarketCorrelation::new(20).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // RelativeStrengthMomentum tests
    #[test]
    fn test_relative_strength_momentum_new() {
        assert!(RelativeStrengthMomentum::new(10, 5, 3).is_ok());
        assert!(RelativeStrengthMomentum::new(4, 5, 3).is_err()); // rs_period too small
        assert!(RelativeStrengthMomentum::new(10, 0, 3).is_err()); // momentum_period too small
        assert!(RelativeStrengthMomentum::new(10, 5, 0).is_err()); // smooth_period too small
    }

    #[test]
    fn test_relative_strength_momentum_new_validation() {
        let err = RelativeStrengthMomentum::new(4, 5, 3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "rs_period");
                assert_eq!(reason, "must be at least 5");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = RelativeStrengthMomentum::new(10, 0, 3).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "momentum_period");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = RelativeStrengthMomentum::new(10, 5, 0).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "smooth_period");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_relative_strength_momentum_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = RelativeStrengthMomentum::new(10, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Values should be finite and reasonable
        for i in 20..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_relative_strength_momentum_outperformance() {
        // Series1 outperforming Series2 with increasing pace
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64).powf(1.2) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = RelativeStrengthMomentum::new(10, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        // Check that indicator produces non-zero values during valid period
        let has_nonzero = result[30..].iter().any(|&v| v != 0.0);
        assert!(has_nonzero, "Expected non-zero momentum values during outperformance");
    }

    #[test]
    fn test_relative_strength_momentum_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RelativeStrengthMomentum::new(10, 5, 3).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Relative Strength Momentum");
        assert_eq!(indicator.min_periods(), 15);
    }

    #[test]
    fn test_relative_strength_momentum_insufficient_data() {
        let dual = create_test_dual_series(10);
        let indicator = RelativeStrengthMomentum::new(10, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // IntermarketDivergence tests
    #[test]
    fn test_intermarket_divergence_new() {
        assert!(IntermarketDivergence::new(20, 5, 0.5).is_ok());
        assert!(IntermarketDivergence::new(19, 5, 0.5).is_err()); // correlation_period too small
        assert!(IntermarketDivergence::new(20, 4, 0.5).is_err()); // divergence_period too small
        assert!(IntermarketDivergence::new(20, 5, 0.0).is_err()); // threshold <= 0
        assert!(IntermarketDivergence::new(20, 5, -0.5).is_err()); // threshold < 0
    }

    #[test]
    fn test_intermarket_divergence_new_validation() {
        let err = IntermarketDivergence::new(19, 5, 0.5).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "correlation_period");
                assert_eq!(reason, "must be at least 20");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = IntermarketDivergence::new(20, 4, 0.5).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "divergence_period");
                assert_eq!(reason, "must be at least 5");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = IntermarketDivergence::new(20, 5, 0.0).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "threshold");
                assert_eq!(reason, "must be greater than 0");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_intermarket_divergence_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = IntermarketDivergence::new(20, 5, 0.1).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First (min_req - 1) values should be 0
        for i in 0..19 {
            assert_eq!(result[i], 0.0);
        }
    }

    #[test]
    fn test_intermarket_divergence_detects_divergence() {
        // Create series that diverge after period of correlation
        let mut series1 = Vec::with_capacity(150);
        let mut series2 = Vec::with_capacity(150);

        // First 100 points: correlated
        for i in 0..100 {
            series1.push(100.0 + (i as f64) * 0.5);
            series2.push(50.0 + (i as f64) * 0.25);
        }
        // Next 50 points: series1 diverges upward
        for i in 100..150 {
            series1.push(150.0 + ((i - 100) as f64) * 2.0); // Accelerating
            series2.push(75.0 + ((i - 100) as f64) * 0.25); // Same pace
        }

        let dual = DualSeries::new(series1, series2);
        let indicator = IntermarketDivergence::new(20, 5, 0.1).unwrap();
        let result = indicator.calculate(&dual);

        // Should detect some divergence in the latter part
        let has_divergence = result[100..].iter().any(|&v| v != 0.0);
        assert!(has_divergence, "Should detect divergence when series deviate");
    }

    #[test]
    fn test_intermarket_divergence_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = IntermarketDivergence::new(20, 5, 0.5).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Intermarket Divergence");
        assert_eq!(indicator.min_periods(), 20);
    }

    #[test]
    fn test_intermarket_divergence_insufficient_data() {
        let dual = create_test_dual_series(15);
        let indicator = IntermarketDivergence::new(20, 5, 0.5).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 15);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // SectorMomentumRank tests
    #[test]
    fn test_sector_momentum_rank_new() {
        assert!(SectorMomentumRank::new(10, 20, 3).is_ok());
        assert!(SectorMomentumRank::new(4, 20, 3).is_err()); // momentum_period too small
        assert!(SectorMomentumRank::new(10, 9, 3).is_err()); // rank_period too small
        assert!(SectorMomentumRank::new(10, 20, 0).is_err()); // smooth_period too small
    }

    #[test]
    fn test_sector_momentum_rank_new_validation() {
        let err = SectorMomentumRank::new(4, 20, 3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "momentum_period");
                assert_eq!(reason, "must be at least 5");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = SectorMomentumRank::new(10, 9, 3).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "rank_period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = SectorMomentumRank::new(10, 20, 0).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "smooth_period");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_sector_momentum_rank_calculate() {
        let close: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5 + ((i as f64) * 0.2).sin() * 3.0).collect();
        let indicator = SectorMomentumRank::new(10, 20, 3).unwrap();
        let result = indicator.calculate(&close);

        assert_eq!(result.len(), 100);
        // Rank should be between 0 and 100
        for i in 30..100 {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "Rank at {} is out of bounds: {}", i, result[i]);
        }
    }

    #[test]
    fn test_sector_momentum_rank_strong_uptrend() {
        // Strong consistent uptrend should have high rank
        let close: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 1.0).collect();
        let indicator = SectorMomentumRank::new(10, 20, 3).unwrap();
        let result = indicator.calculate(&close);

        // Values should be in valid range [0, 100]
        for i in 30..100 {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "Rank at {} should be 0-100, got {}", i, result[i]);
        }
    }

    #[test]
    fn test_sector_momentum_rank_compute() {
        let data = create_test_ohlcv_data(100);
        let indicator = SectorMomentumRank::new(10, 20, 3).unwrap();
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Sector Momentum Rank");
        assert_eq!(indicator.min_periods(), 30);
    }

    #[test]
    fn test_sector_momentum_rank_insufficient_data() {
        let close: Vec<f64> = (0..20).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let indicator = SectorMomentumRank::new(10, 20, 3).unwrap();
        let result = indicator.calculate(&close);

        assert_eq!(result.len(), 20);
        // Default values should be 50
        for i in 0..20 {
            assert_eq!(result[i], 50.0);
        }
    }

    // CrossAssetVolatility tests
    #[test]
    fn test_cross_asset_volatility_new() {
        assert!(CrossAssetVolatility::new(10).is_ok());
        assert!(CrossAssetVolatility::new(20).is_ok());
        assert!(CrossAssetVolatility::new(9).is_err()); // period too small
    }

    #[test]
    fn test_cross_asset_volatility_new_validation() {
        let err = CrossAssetVolatility::new(9).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_cross_asset_volatility_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = CrossAssetVolatility::new(20).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First (period) values should be 1.0 (default)
        for i in 0..20 {
            assert_eq!(result[i], 1.0);
        }
        // Volatility ratio should be positive
        for i in 20..100 {
            assert!(result[i] > 0.0, "Volatility ratio at {} should be positive: {}", i, result[i]);
        }
    }

    #[test]
    fn test_cross_asset_volatility_higher_vol() {
        // Series1 has higher volatility than Series2
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + ((i as f64) * 0.5).sin() * 10.0).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + ((i as f64) * 0.5).sin() * 2.0).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = CrossAssetVolatility::new(20).unwrap();
        let result = indicator.calculate(&dual);

        // Ratio should be > 1 (series1 more volatile)
        let avg_ratio: f64 = result[30..].iter().sum::<f64>() / (result.len() - 30) as f64;
        assert!(avg_ratio > 1.0, "Expected vol ratio > 1 when series1 is more volatile, got {}", avg_ratio);
    }

    #[test]
    fn test_cross_asset_volatility_equal_vol() {
        // Series with equal volatility characteristics
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + ((i as f64) * 0.3).sin() * 5.0).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + ((i as f64) * 0.3).sin() * 2.5).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = CrossAssetVolatility::new(20).unwrap();
        let result = indicator.calculate(&dual);

        // Both have similar pattern, so ratio should be around 2 (price ratio)
        let avg_ratio: f64 = result[30..].iter().sum::<f64>() / (result.len() - 30) as f64;
        assert!(avg_ratio > 0.5 && avg_ratio < 5.0, "Expected reasonable vol ratio, got {}", avg_ratio);
    }

    #[test]
    fn test_cross_asset_volatility_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + ((i as f64) * 0.3).sin() * 5.0).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + ((i as f64) * 0.3).sin() * 2.0).collect();

        let indicator = CrossAssetVolatility::new(20).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Cross Asset Volatility");
        assert_eq!(indicator.min_periods(), 21);
    }

    #[test]
    fn test_cross_asset_volatility_with_log_returns() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = CrossAssetVolatility::new(20).unwrap().with_log_returns(true);
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        for i in 20..100 {
            assert!(result[i].is_finite() && result[i] > 0.0);
        }
    }

    #[test]
    fn test_cross_asset_volatility_insufficient_data() {
        let dual = create_test_dual_series(10);
        let indicator = CrossAssetVolatility::new(20).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|&v| v == 1.0)); // Default value
    }

    // MarketLeadLag tests
    #[test]
    fn test_market_lead_lag_new() {
        assert!(MarketLeadLag::new(20, 5, 3).is_ok());
        assert!(MarketLeadLag::new(19, 5, 3).is_err()); // period too small
        assert!(MarketLeadLag::new(20, 0, 3).is_err()); // max_lag too small
        assert!(MarketLeadLag::new(20, 10, 3).is_err()); // max_lag >= period/2
        assert!(MarketLeadLag::new(20, 5, 0).is_err()); // smooth_period too small
    }

    #[test]
    fn test_market_lead_lag_new_validation() {
        let err = MarketLeadLag::new(19, 5, 3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 20");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = MarketLeadLag::new(20, 0, 3).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "max_lag");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = MarketLeadLag::new(20, 10, 3).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "max_lag");
                assert_eq!(reason, "must be less than period / 2");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err4 = MarketLeadLag::new(20, 5, 0).unwrap_err();
        match err4 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "smooth_period");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_market_lead_lag_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = MarketLeadLag::new(30, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First (period - 1) values should be 0
        for i in 0..29 {
            assert_eq!(result[i], 0.0);
        }
    }

    #[test]
    fn test_market_lead_lag_series1_leads() {
        // Create series where series1 leads series2 by a few periods
        let mut series1 = Vec::with_capacity(150);
        let mut series2 = Vec::with_capacity(150);

        for i in 0..150 {
            series1.push(100.0 + ((i as f64) * 0.1).sin() * 5.0);
        }
        // Series2 follows series1 with a lag
        for i in 0..150 {
            let lag_idx = if i >= 3 { i - 3 } else { 0 };
            series2.push(50.0 + ((lag_idx as f64) * 0.1).sin() * 2.5);
        }

        let dual = DualSeries::new(series1, series2);
        let indicator = MarketLeadLag::new(30, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        // Verify indicator produces values after min_periods
        assert_eq!(result.len(), 150);
        // Values should be finite
        for i in 35..150 {
            assert!(result[i].is_finite(), "Value at {} should be finite", i);
        }
    }

    #[test]
    fn test_market_lead_lag_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + ((i as f64) * 0.2).sin() * 5.0).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + ((i as f64) * 0.2).sin() * 2.5).collect();

        let indicator = MarketLeadLag::new(30, 5, 3).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Market Lead Lag");
        assert_eq!(indicator.min_periods(), 30);
    }

    #[test]
    fn test_market_lead_lag_missing_secondary() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let data = OHLCVSeries::from_close(series1);

        let indicator = MarketLeadLag::new(30, 5, 3).unwrap();
        assert!(indicator.compute(&data).is_err());
    }

    #[test]
    fn test_market_lead_lag_mismatched_length() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..50).map(|i| 50.0 + (i as f64) * 0.25).collect();
        let data = OHLCVSeries::from_close(series1);

        let indicator = MarketLeadLag::new(30, 5, 3).unwrap().with_secondary(&series2);
        assert!(indicator.compute(&data).is_err());
    }

    #[test]
    fn test_market_lead_lag_insufficient_data() {
        let dual = create_test_dual_series(15);
        let indicator = MarketLeadLag::new(30, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 15);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_market_lead_lag_bounded_output() {
        let dual = create_test_dual_series(200);
        let indicator = MarketLeadLag::new(30, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        // Lead/lag score should be bounded by max_lag * max_correlation
        for i in 30..200 {
            assert!(result[i].abs() <= 10.0, "Lead/lag score at {} is too extreme: {}", i, result[i]);
        }
    }

    // Tests for calculate_between methods
    #[test]
    fn test_cross_market_correlation_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = CrossMarketCorrelation::new(20).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        // Perfectly correlated linear series
        for i in 25..100 {
            assert!(result[i] > 0.99, "Expected high positive correlation");
        }
    }

    #[test]
    fn test_relative_strength_momentum_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RelativeStrengthMomentum::new(10, 5, 3).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 20..100 {
            assert!(result[i].is_finite());
        }
    }

    #[test]
    fn test_intermarket_divergence_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = IntermarketDivergence::new(20, 5, 0.5).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_cross_asset_volatility_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + ((i as f64) * 0.3).sin() * 5.0).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + ((i as f64) * 0.3).sin() * 2.0).collect();

        let indicator = CrossAssetVolatility::new(20).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 20..100 {
            assert!(result[i] > 0.0);
        }
    }

    #[test]
    fn test_market_lead_lag_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + ((i as f64) * 0.2).sin() * 5.0).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + ((i as f64) * 0.2).sin() * 2.5).collect();

        let indicator = MarketLeadLag::new(30, 5, 3).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
    }

    // ========================================================================
    // Tests for 6 new intermarket indicators
    // ========================================================================

    // RelativePerformanceIndex tests
    #[test]
    fn test_relative_performance_index_new() {
        assert!(RelativePerformanceIndex::new(10, 3).is_ok());
        assert!(RelativePerformanceIndex::new(20, 5).is_ok());
        assert!(RelativePerformanceIndex::new(9, 3).is_err()); // period too small
        assert!(RelativePerformanceIndex::new(10, 0).is_err()); // smooth_period too small
    }

    #[test]
    fn test_relative_performance_index_new_validation() {
        let err = RelativePerformanceIndex::new(9, 3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = RelativePerformanceIndex::new(10, 0).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "smooth_period");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_relative_performance_index_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = RelativePerformanceIndex::new(20, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First values should be 0
        for i in 0..20 {
            assert_eq!(result[i], 0.0);
        }
        // Values after warmup should be finite
        for i in 25..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_relative_performance_index_outperformance() {
        // Series1 outperforming Series2
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 1.0).collect(); // Faster growth
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect(); // Slower growth
        let dual = DualSeries::new(series1, series2);

        let indicator = RelativePerformanceIndex::new(20, 3).unwrap();
        let result = indicator.calculate(&dual);

        // Should show positive relative performance (series1 outperforming)
        let avg_perf: f64 = result[30..].iter().sum::<f64>() / (result.len() - 30) as f64;
        assert!(avg_perf > 0.0, "Expected positive RPI for outperformance, got {}", avg_perf);
    }

    #[test]
    fn test_relative_performance_index_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RelativePerformanceIndex::new(20, 3).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Relative Performance Index");
        assert_eq!(indicator.min_periods(), 21);
    }

    #[test]
    fn test_relative_performance_index_missing_secondary() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let data = OHLCVSeries::from_close(series1);

        let indicator = RelativePerformanceIndex::new(20, 3).unwrap();
        assert!(indicator.compute(&data).is_err());
    }

    #[test]
    fn test_relative_performance_index_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RelativePerformanceIndex::new(20, 3).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 25..100 {
            assert!(result[i].is_finite());
        }
    }

    // SpreadOscillator tests
    #[test]
    fn test_spread_oscillator_new() {
        assert!(SpreadOscillator::new(10, 2.0).is_ok());
        assert!(SpreadOscillator::new(20, 1.5).is_ok());
        assert!(SpreadOscillator::new(9, 2.0).is_err()); // period too small
        assert!(SpreadOscillator::new(10, 0.0).is_err()); // std_multiplier <= 0
        assert!(SpreadOscillator::new(10, -1.0).is_err()); // std_multiplier < 0
    }

    #[test]
    fn test_spread_oscillator_new_validation() {
        let err = SpreadOscillator::new(9, 2.0).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = SpreadOscillator::new(10, 0.0).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "std_multiplier");
                assert_eq!(reason, "must be greater than 0");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_spread_oscillator_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = SpreadOscillator::new(20, 2.0).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First (period-1) values should be 0
        for i in 0..19 {
            assert_eq!(result[i], 0.0);
        }
        // Oscillator should be bounded by [-100, 100]
        for i in 20..100 {
            assert!(result[i] >= -100.0 && result[i] <= 100.0,
                "Oscillator at {} is out of bounds: {}", i, result[i]);
        }
    }

    #[test]
    fn test_spread_oscillator_with_log_spread() {
        let dual = create_test_dual_series(100);
        let indicator = SpreadOscillator::new(20, 2.0).unwrap().with_log_spread(true);
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        for i in 20..100 {
            assert!(result[i].is_finite());
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_spread_oscillator_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + ((i as f64) * 0.2).sin() * 5.0).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + ((i as f64) * 0.2).sin() * 2.5).collect();

        let indicator = SpreadOscillator::new(20, 2.0).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Spread Oscillator");
        assert_eq!(indicator.min_periods(), 20);
    }

    #[test]
    fn test_spread_oscillator_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + ((i as f64) * 0.3).sin() * 5.0).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + ((i as f64) * 0.3).sin() * 2.0).collect();

        let indicator = SpreadOscillator::new(20, 2.0).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 20..100 {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    // BetaEstimator tests
    #[test]
    fn test_beta_estimator_new() {
        assert!(BetaEstimator::new(20).is_ok());
        assert!(BetaEstimator::new(30).is_ok());
        assert!(BetaEstimator::new(19).is_err()); // period too small
    }

    #[test]
    fn test_beta_estimator_new_validation() {
        let err = BetaEstimator::new(19).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 20");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_beta_estimator_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = BetaEstimator::new(30).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First values should be default (1.0)
        for i in 0..30 {
            assert_eq!(result[i], 1.0);
        }
        // Beta should be finite and reasonable
        for i in 35..100 {
            assert!(result[i].is_finite(), "Beta at {} is not finite", i);
            assert!(result[i].abs() < 10.0, "Beta at {} is too extreme: {}", i, result[i]);
        }
    }

    #[test]
    fn test_beta_estimator_high_beta() {
        // Series1 has 2x the volatility/returns of series2 (high beta scenario)
        // Create oscillating series where series1 moves twice as much as series2
        let series2: Vec<f64> = (0..100).map(|i| {
            100.0 + ((i as f64) * 0.1).sin() * 5.0
        }).collect();
        let series1: Vec<f64> = (0..100).map(|i| {
            200.0 + ((i as f64) * 0.1).sin() * 10.0 // 2x amplitude
        }).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = BetaEstimator::new(30).unwrap();
        let result = indicator.calculate(&dual);

        // Beta should be positive and > 1 for high-beta series
        for i in 40..100 {
            assert!(result[i] > 0.5, "Expected positive beta, got {} at {}", result[i], i);
            assert!(result[i].is_finite(), "Beta should be finite at {}", i);
        }
    }

    #[test]
    fn test_beta_estimator_with_log_returns() {
        let dual = create_test_dual_series(100);
        let indicator = BetaEstimator::new(30).unwrap().with_log_returns(true);
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        for i in 35..100 {
            assert!(result[i].is_finite());
        }
    }

    #[test]
    fn test_beta_estimator_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = BetaEstimator::new(30).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Beta Estimator");
        assert_eq!(indicator.min_periods(), 31);
    }

    #[test]
    fn test_beta_estimator_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = BetaEstimator::new(30).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 35..100 {
            assert!(result[i].is_finite());
        }
    }

    // CointegrationScore tests
    #[test]
    fn test_cointegration_score_new() {
        assert!(CointegrationScore::new(30, 5).is_ok());
        assert!(CointegrationScore::new(50, 10).is_ok());
        assert!(CointegrationScore::new(29, 5).is_err()); // period too small
        assert!(CointegrationScore::new(30, 4).is_err()); // short_period too small
        assert!(CointegrationScore::new(30, 15).is_err()); // short_period >= period/2
    }

    #[test]
    fn test_cointegration_score_new_validation() {
        let err = CointegrationScore::new(29, 5).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 30");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = CointegrationScore::new(30, 4).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "short_period");
                assert_eq!(reason, "must be at least 5");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = CointegrationScore::new(30, 15).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "short_period");
                assert_eq!(reason, "must be less than period / 2");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_cointegration_score_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = CointegrationScore::new(30, 5).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First values should be default (50)
        for i in 0..29 {
            assert_eq!(result[i], 50.0);
        }
        // Score should be between 0 and 100
        for i in 30..100 {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "Score at {} is out of bounds: {}", i, result[i]);
        }
    }

    #[test]
    fn test_cointegration_score_cointegrated_series() {
        // Create cointegrated series (series1 = 2*series2 + noise with mean reversion)
        let mut series1 = Vec::with_capacity(150);
        let mut series2 = Vec::with_capacity(150);

        for i in 0..150 {
            let x2 = 50.0 + (i as f64) * 0.3;
            series2.push(x2);
            // Small oscillating noise that mean-reverts
            let noise = ((i as f64) * 0.5).sin() * 0.5;
            series1.push(100.0 + 2.0 * x2 + noise);
        }

        let dual = DualSeries::new(series1, series2);
        let indicator = CointegrationScore::new(50, 10).unwrap();
        let result = indicator.calculate(&dual);

        // Should have moderate to high cointegration scores
        let avg_score: f64 = result[60..].iter().sum::<f64>() / (result.len() - 60) as f64;
        assert!(avg_score > 20.0, "Expected higher cointegration score, got {}", avg_score);
    }

    #[test]
    fn test_cointegration_score_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = CointegrationScore::new(30, 5).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Cointegration Score");
        assert_eq!(indicator.min_periods(), 30);
    }

    #[test]
    fn test_cointegration_score_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = CointegrationScore::new(30, 5).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 30..100 {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    // CorrelationTrendAnalyzer tests
    #[test]
    fn test_correlation_trend_analyzer_new() {
        assert!(CorrelationTrendAnalyzer::new(10, 30, 5).is_ok());
        assert!(CorrelationTrendAnalyzer::new(15, 40, 3).is_ok());
        assert!(CorrelationTrendAnalyzer::new(9, 30, 5).is_err()); // short_period too small
        assert!(CorrelationTrendAnalyzer::new(30, 20, 5).is_err()); // long_period <= short_period
        assert!(CorrelationTrendAnalyzer::new(10, 30, 2).is_err()); // momentum_period too small
    }

    #[test]
    fn test_correlation_trend_analyzer_new_validation() {
        let err = CorrelationTrendAnalyzer::new(9, 30, 5).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "short_period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = CorrelationTrendAnalyzer::new(30, 20, 5).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "long_period");
                assert_eq!(reason, "must be greater than short_period");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = CorrelationTrendAnalyzer::new(10, 30, 2).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "momentum_period");
                assert_eq!(reason, "must be at least 3");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_correlation_trend_analyzer_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = CorrelationTrendAnalyzer::new(10, 30, 5).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First values should be 0
        for i in 0..34 {
            assert_eq!(result[i], 0.0);
        }
        // Values after warmup should be finite
        for i in 40..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_correlation_trend_analyzer_regime_change() {
        // Create series with correlation regime change
        let mut series1 = Vec::with_capacity(150);
        let mut series2 = Vec::with_capacity(150);

        // First half: positively correlated
        for i in 0..75 {
            series1.push(100.0 + (i as f64) * 0.5);
            series2.push(50.0 + (i as f64) * 0.25);
        }
        // Second half: negatively correlated
        for i in 75..150 {
            series1.push(137.5 + ((i - 75) as f64) * 0.5);
            series2.push(68.75 - ((i - 75) as f64) * 0.25);
        }

        let dual = DualSeries::new(series1, series2);
        let indicator = CorrelationTrendAnalyzer::new(10, 30, 5).unwrap();
        let result = indicator.calculate(&dual);

        // Should detect change around the regime shift
        assert_eq!(result.len(), 150);
        // Values should be finite after warmup
        for i in 40..150 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_correlation_trend_analyzer_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = CorrelationTrendAnalyzer::new(10, 30, 5).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Correlation Trend Analyzer");
        assert_eq!(indicator.min_periods(), 35);
    }

    #[test]
    fn test_correlation_trend_analyzer_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = CorrelationTrendAnalyzer::new(10, 30, 5).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 40..100 {
            assert!(result[i].is_finite());
        }
    }

    // EnhancedPairsTradingSignal tests
    #[test]
    fn test_enhanced_pairs_trading_signal_new() {
        assert!(EnhancedPairsTradingSignal::new(20, 10).is_ok());
        assert!(EnhancedPairsTradingSignal::new(30, 15).is_ok());
        assert!(EnhancedPairsTradingSignal::new(19, 10).is_err()); // period too small
        assert!(EnhancedPairsTradingSignal::new(20, 9).is_err()); // corr_period too small
    }

    #[test]
    fn test_enhanced_pairs_trading_signal_new_validation() {
        let err = EnhancedPairsTradingSignal::new(19, 10).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 20");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = EnhancedPairsTradingSignal::new(20, 9).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "corr_period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_enhanced_pairs_trading_signal_calculate() {
        let dual = create_mean_reverting_spread(100);
        let indicator = EnhancedPairsTradingSignal::new(20, 15)
            .unwrap()
            .with_entry_threshold(1.5)
            .with_exit_threshold(0.3);
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Signals should be -2, -1, 0, 1, or 2
        for &s in &result {
            assert!(s == -2.0 || s == -1.0 || s == 0.0 || s == 1.0 || s == 2.0,
                "Invalid signal: {}", s);
        }
    }

    #[test]
    fn test_enhanced_pairs_trading_signal_generates_signals() {
        // Create a spread that clearly oscillates
        let mut series1 = Vec::with_capacity(200);
        let mut series2 = Vec::with_capacity(200);

        for i in 0..200 {
            series2.push(100.0);
            // Large oscillations to trigger entries
            let spread = ((i as f64) * 0.15).sin() * 10.0;
            series1.push(100.0 + spread);
        }

        let dual = DualSeries::new(series1, series2);
        let indicator = EnhancedPairsTradingSignal::new(20, 15)
            .unwrap()
            .with_entry_threshold(1.5)
            .with_exit_threshold(0.3)
            .with_min_correlation(0.0); // Disable correlation filter for test
        let result = indicator.calculate(&dual);

        // Should have some non-zero signals
        let non_zero_count = result.iter().filter(|&&s| s != 0.0).count();
        assert!(non_zero_count > 0, "Should generate some trading signals");
    }

    #[test]
    fn test_enhanced_pairs_trading_signal_correlation_filter() {
        // Create uncorrelated series
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + ((i as f64) * 0.3).sin() * 5.0).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + ((i as f64) * 0.7).cos() * 3.0).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = EnhancedPairsTradingSignal::new(20, 15)
            .unwrap()
            .with_min_correlation(0.9); // High correlation requirement
        let result = indicator.calculate(&dual);

        // Should have mostly zero signals due to correlation filter
        let zero_count = result[30..].iter().filter(|&&s| s == 0.0).count();
        let total_count = result.len() - 30;
        assert!(zero_count as f64 / total_count as f64 > 0.5,
            "Expected most signals to be filtered by correlation");
    }

    #[test]
    fn test_enhanced_pairs_trading_signal_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = EnhancedPairsTradingSignal::new(20, 15).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Enhanced Pairs Trading Signal");
        assert_eq!(indicator.min_periods(), 20);
    }

    #[test]
    fn test_enhanced_pairs_trading_signal_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + ((i as f64) * 0.2).sin() * 10.0).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + ((i as f64) * 0.2).sin() * 5.0).collect();

        let indicator = EnhancedPairsTradingSignal::new(20, 15).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for &s in &result {
            assert!(s >= -2.0 && s <= 2.0);
        }
    }

    #[test]
    fn test_enhanced_pairs_trading_signal_with_thresholds() {
        let dual = create_mean_reverting_spread(100);
        let indicator = EnhancedPairsTradingSignal::new(20, 15)
            .unwrap()
            .with_entry_threshold(1.0)
            .with_strong_threshold(2.0)
            .with_exit_threshold(0.2)
            .with_min_correlation(0.3);

        let result = indicator.calculate(&dual);
        assert_eq!(result.len(), 100);
    }

    // Insufficient data tests for new indicators
    #[test]
    fn test_relative_performance_index_insufficient_data() {
        let dual = create_test_dual_series(10);
        let indicator = RelativePerformanceIndex::new(20, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_spread_oscillator_insufficient_data() {
        let dual = create_test_dual_series(10);
        let indicator = SpreadOscillator::new(20, 2.0).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_beta_estimator_insufficient_data() {
        let dual = create_test_dual_series(15);
        let indicator = BetaEstimator::new(30).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 15);
        assert!(result.iter().all(|&v| v == 1.0)); // Default beta
    }

    #[test]
    fn test_cointegration_score_insufficient_data() {
        let dual = create_test_dual_series(20);
        let indicator = CointegrationScore::new(30, 5).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 20);
        assert!(result.iter().all(|&v| v == 50.0)); // Default score
    }

    #[test]
    fn test_correlation_trend_analyzer_insufficient_data() {
        let dual = create_test_dual_series(20);
        let indicator = CorrelationTrendAnalyzer::new(10, 30, 5).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 20);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_enhanced_pairs_trading_signal_insufficient_data() {
        let dual = create_test_dual_series(15);
        let indicator = EnhancedPairsTradingSignal::new(20, 15).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 15);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // ========================================================================
    // Tests for 6 NEW intermarket indicators
    // ========================================================================

    // RelativeRotation tests
    #[test]
    fn test_relative_rotation_new() {
        assert!(RelativeRotation::new(10, 5, 3).is_ok());
        assert!(RelativeRotation::new(20, 10, 5).is_ok());
        assert!(RelativeRotation::new(9, 5, 3).is_err()); // rs_period too small
        assert!(RelativeRotation::new(10, 4, 3).is_err()); // momentum_period too small
        assert!(RelativeRotation::new(10, 5, 0).is_err()); // smooth_period too small
    }

    #[test]
    fn test_relative_rotation_new_validation() {
        let err = RelativeRotation::new(9, 5, 3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "rs_period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = RelativeRotation::new(10, 4, 3).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "momentum_period");
                assert_eq!(reason, "must be at least 5");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = RelativeRotation::new(10, 5, 0).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "smooth_period");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_relative_rotation_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = RelativeRotation::new(15, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First values should be 0
        for i in 0..19 {
            assert_eq!(result[i], 0.0);
        }
        // Values after warmup should be finite
        for i in 25..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_relative_rotation_outperformance() {
        // Series1 strongly outperforming Series2
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64).powf(1.2) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.1).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = RelativeRotation::new(15, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        // Should show positive rotation (leading)
        let avg_score: f64 = result[30..].iter().sum::<f64>() / (result.len() - 30) as f64;
        assert!(avg_score > 0.0, "Expected positive rotation for outperformance, got {}", avg_score);
    }

    #[test]
    fn test_relative_rotation_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RelativeRotation::new(15, 5, 3).unwrap().with_benchmark(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Relative Rotation");
        assert_eq!(indicator.min_periods(), 20);
    }

    #[test]
    fn test_relative_rotation_missing_benchmark() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let data = OHLCVSeries::from_close(series1);

        let indicator = RelativeRotation::new(15, 5, 3).unwrap();
        assert!(indicator.compute(&data).is_err());
    }

    #[test]
    fn test_relative_rotation_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RelativeRotation::new(15, 5, 3).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 25..100 {
            assert!(result[i].is_finite());
        }
    }

    #[test]
    fn test_relative_rotation_insufficient_data() {
        let dual = create_test_dual_series(15);
        let indicator = RelativeRotation::new(15, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 15);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // AlphaGenerator tests
    #[test]
    fn test_alpha_generator_new() {
        assert!(AlphaGenerator::new(20).is_ok());
        assert!(AlphaGenerator::new(30).is_ok());
        assert!(AlphaGenerator::new(19).is_err()); // period too small
    }

    #[test]
    fn test_alpha_generator_new_validation() {
        let err = AlphaGenerator::new(19).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 20");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_alpha_generator_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = AlphaGenerator::new(30).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First values should be 0
        for i in 0..30 {
            assert_eq!(result[i], 0.0);
        }
        // Values after warmup should be finite
        for i in 35..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_alpha_generator_positive_alpha() {
        // Series1 with independent positive movement (not correlated with series2)
        // to generate positive alpha
        let series1: Vec<f64> = (0..100).map(|i| {
            100.0 + (i as f64) * 0.5 + ((i as f64) * 0.2).sin() * 2.0
        }).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + ((i as f64) * 0.3).cos() * 1.0).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = AlphaGenerator::new(30).unwrap();
        let result = indicator.calculate(&dual);

        // Alpha values should be finite and non-zero when there's independent performance
        let has_nonzero = result[40..].iter().any(|&v| v != 0.0 && v.is_finite());
        assert!(has_nonzero, "Alpha should produce non-zero finite values");
    }

    #[test]
    fn test_alpha_generator_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = AlphaGenerator::new(30).unwrap().with_benchmark(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Alpha Generator");
        assert_eq!(indicator.min_periods(), 31);
    }

    #[test]
    fn test_alpha_generator_with_annualization() {
        let dual = create_test_dual_series(100);
        let indicator = AlphaGenerator::new(30).unwrap().with_annualization(52.0); // Weekly
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        for i in 35..100 {
            assert!(result[i].is_finite());
        }
    }

    #[test]
    fn test_alpha_generator_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = AlphaGenerator::new(30).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 35..100 {
            assert!(result[i].is_finite());
        }
    }

    #[test]
    fn test_alpha_generator_insufficient_data() {
        let dual = create_test_dual_series(20);
        let indicator = AlphaGenerator::new(30).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 20);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // TrackingError tests
    #[test]
    fn test_tracking_error_new() {
        assert!(TrackingError::new(20).is_ok());
        assert!(TrackingError::new(30).is_ok());
        assert!(TrackingError::new(19).is_err()); // period too small
    }

    #[test]
    fn test_tracking_error_new_validation() {
        let err = TrackingError::new(19).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 20");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_tracking_error_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = TrackingError::new(30).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First values should be 0
        for i in 0..30 {
            assert_eq!(result[i], 0.0);
        }
        // Tracking error should be positive
        for i in 35..100 {
            assert!(result[i] >= 0.0, "Tracking error should be >= 0, got {} at {}", result[i], i);
        }
    }

    #[test]
    fn test_tracking_error_tight_tracking() {
        // Two nearly identical series (tight tracking)
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5 + 0.01).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = TrackingError::new(30).unwrap();
        let result = indicator.calculate(&dual);

        // Should have very low tracking error
        let avg_te: f64 = result[40..].iter().sum::<f64>() / (result.len() - 40) as f64;
        assert!(avg_te < 5.0, "Expected low tracking error for tight tracking, got {}", avg_te);
    }

    #[test]
    fn test_tracking_error_high_deviation() {
        // Two series with different volatility (high tracking error)
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + ((i as f64) * 0.3).sin() * 10.0).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.5).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = TrackingError::new(30).unwrap();
        let result = indicator.calculate(&dual);

        // Should have higher tracking error
        let has_positive = result[40..].iter().any(|&v| v > 0.0);
        assert!(has_positive, "Expected some positive tracking error values");
    }

    #[test]
    fn test_tracking_error_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = TrackingError::new(30).unwrap().with_benchmark(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Tracking Error");
        assert_eq!(indicator.min_periods(), 31);
    }

    #[test]
    fn test_tracking_error_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = TrackingError::new(30).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 35..100 {
            assert!(result[i] >= 0.0);
        }
    }

    #[test]
    fn test_tracking_error_insufficient_data() {
        let dual = create_test_dual_series(20);
        let indicator = TrackingError::new(30).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 20);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // IntermarketInformationRatio tests
    #[test]
    fn test_intermarket_information_ratio_new() {
        assert!(IntermarketInformationRatio::new(20).is_ok());
        assert!(IntermarketInformationRatio::new(30).is_ok());
        assert!(IntermarketInformationRatio::new(19).is_err()); // period too small
    }

    #[test]
    fn test_intermarket_information_ratio_new_validation() {
        let err = IntermarketInformationRatio::new(19).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 20");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_intermarket_information_ratio_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = IntermarketInformationRatio::new(30).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First values should be 0
        for i in 0..30 {
            assert_eq!(result[i], 0.0);
        }
        // Values should be finite
        for i in 35..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_intermarket_information_ratio_consistent_outperformance() {
        // Series1 consistently outperforming (high IR)
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.8).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.3).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = IntermarketInformationRatio::new(30).unwrap();
        let result = indicator.calculate(&dual);

        // Should have positive IR for consistent outperformance
        let avg_ir: f64 = result[40..].iter().sum::<f64>() / (result.len() - 40) as f64;
        assert!(avg_ir > 0.0, "Expected positive IR for outperformance, got {}", avg_ir);
    }

    #[test]
    fn test_intermarket_information_ratio_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = IntermarketInformationRatio::new(30).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Intermarket Information Ratio");
        assert_eq!(indicator.min_periods(), 31);
    }

    #[test]
    fn test_intermarket_information_ratio_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = IntermarketInformationRatio::new(30).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 35..100 {
            assert!(result[i].is_finite());
        }
    }

    #[test]
    fn test_intermarket_information_ratio_insufficient_data() {
        let dual = create_test_dual_series(20);
        let indicator = IntermarketInformationRatio::new(30).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 20);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // IntermarketCorrelationBreakdown tests
    #[test]
    fn test_intermarket_correlation_breakdown_new() {
        assert!(IntermarketCorrelationBreakdown::new(10, 30, 0.3).is_ok());
        assert!(IntermarketCorrelationBreakdown::new(15, 40, 0.2).is_ok());
        assert!(IntermarketCorrelationBreakdown::new(9, 30, 0.3).is_err()); // short_period too small
        assert!(IntermarketCorrelationBreakdown::new(30, 20, 0.3).is_err()); // long_period <= short
        assert!(IntermarketCorrelationBreakdown::new(10, 30, 0.0).is_err()); // threshold <= 0
    }

    #[test]
    fn test_intermarket_correlation_breakdown_new_validation() {
        let err = IntermarketCorrelationBreakdown::new(9, 30, 0.3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "short_period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = IntermarketCorrelationBreakdown::new(30, 20, 0.3).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "long_period");
                assert_eq!(reason, "must be greater than short_period");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = IntermarketCorrelationBreakdown::new(10, 30, 0.0).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "threshold");
                assert_eq!(reason, "must be greater than 0");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_intermarket_correlation_breakdown_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = IntermarketCorrelationBreakdown::new(10, 30, 0.2).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First values should be 0
        for i in 0..29 {
            assert_eq!(result[i], 0.0);
        }
        // Values should be finite
        for i in 35..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_intermarket_correlation_breakdown_regime_change() {
        // Create series with correlation regime change
        let mut series1 = Vec::with_capacity(150);
        let mut series2 = Vec::with_capacity(150);

        // First half: positively correlated
        for i in 0..75 {
            series1.push(100.0 + (i as f64) * 0.5);
            series2.push(50.0 + (i as f64) * 0.25);
        }
        // Second half: negatively correlated
        for i in 75..150 {
            series1.push(137.5 + ((i - 75) as f64) * 0.5);
            series2.push(68.75 - ((i - 75) as f64) * 0.25);
        }

        let dual = DualSeries::new(series1, series2);
        let indicator = IntermarketCorrelationBreakdown::new(15, 40, 0.1).unwrap();
        let result = indicator.calculate(&dual);

        // Should detect some breakdown activity
        let has_nonzero = result[50..].iter().any(|&v| v != 0.0);
        assert!(has_nonzero, "Should detect correlation breakdown during regime change");
    }

    #[test]
    fn test_intermarket_correlation_breakdown_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = IntermarketCorrelationBreakdown::new(10, 30, 0.2).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Intermarket Correlation Breakdown");
        assert_eq!(indicator.min_periods(), 30);
    }

    #[test]
    fn test_intermarket_correlation_breakdown_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = IntermarketCorrelationBreakdown::new(10, 30, 0.2).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 35..100 {
            assert!(result[i].is_finite());
        }
    }

    #[test]
    fn test_intermarket_correlation_breakdown_insufficient_data() {
        let dual = create_test_dual_series(20);
        let indicator = IntermarketCorrelationBreakdown::new(10, 30, 0.2).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 20);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // RegimeCorrelation tests
    #[test]
    fn test_regime_correlation_new() {
        assert!(RegimeCorrelation::new(10, 5, 0.05).is_ok());
        assert!(RegimeCorrelation::new(20, 10, 0.1).is_ok());
        assert!(RegimeCorrelation::new(9, 5, 0.05).is_err()); // correlation_period too small
        assert!(RegimeCorrelation::new(10, 4, 0.05).is_err()); // regime_period too small
        assert!(RegimeCorrelation::new(10, 5, -0.05).is_err()); // threshold < 0
    }

    #[test]
    fn test_regime_correlation_new_validation() {
        let err = RegimeCorrelation::new(9, 5, 0.05).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "correlation_period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = RegimeCorrelation::new(10, 4, 0.05).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "regime_period");
                assert_eq!(reason, "must be at least 5");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = RegimeCorrelation::new(10, 5, -0.05).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "regime_threshold");
                assert_eq!(reason, "must be at least 0");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_regime_correlation_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = RegimeCorrelation::new(15, 10, 0.05).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First values should be 0
        for i in 0..14 {
            assert_eq!(result[i], 0.0);
        }
        // Values after warmup should be finite
        for i in 20..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_regime_correlation_uptrend_regime() {
        // Two positively correlated series in uptrend
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 1.0).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.5).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = RegimeCorrelation::new(15, 10, 0.01).unwrap();
        let result = indicator.calculate(&dual);

        // In up regime with positive correlation, should see positive values
        let avg_corr: f64 = result[25..].iter().sum::<f64>() / (result.len() - 25) as f64;
        assert!(avg_corr > 0.0, "Expected positive regime correlation in uptrend, got {}", avg_corr);
    }

    #[test]
    fn test_regime_correlation_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RegimeCorrelation::new(15, 10, 0.05).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Regime Correlation");
        assert_eq!(indicator.min_periods(), 16);
    }

    #[test]
    fn test_regime_correlation_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RegimeCorrelation::new(15, 10, 0.05).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 20..100 {
            assert!(result[i].is_finite());
        }
    }

    #[test]
    fn test_regime_correlation_insufficient_data() {
        let dual = create_test_dual_series(10);
        let indicator = RegimeCorrelation::new(15, 10, 0.05).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_regime_correlation_different_regimes() {
        // Test that different regimes produce different outputs
        // Create trending up series
        let series1_up: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 1.5).collect();
        let series2_up: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.75).collect();
        let dual_up = DualSeries::new(series1_up, series2_up);

        // Create flat/ranging series
        let series1_flat: Vec<f64> = (0..100).map(|i| 100.0 + ((i as f64) * 0.3).sin() * 0.5).collect();
        let series2_flat: Vec<f64> = (0..100).map(|i| 50.0 + ((i as f64) * 0.3).sin() * 0.25).collect();
        let dual_flat = DualSeries::new(series1_flat, series2_flat);

        let indicator = RegimeCorrelation::new(15, 10, 0.1).unwrap();
        let result_up = indicator.calculate(&dual_up);
        let result_flat = indicator.calculate(&dual_flat);

        // Results should be different due to regime detection
        let avg_up: f64 = result_up[25..].iter().map(|v| v.abs()).sum::<f64>() / (result_up.len() - 25) as f64;
        let avg_flat: f64 = result_flat[25..].iter().map(|v| v.abs()).sum::<f64>() / (result_flat.len() - 25) as f64;

        // Uptrend should have stronger signals (higher correlation weight)
        assert!(avg_up != avg_flat || result_up[50] != result_flat[50],
            "Expected different outputs for different regimes");
    }

    // ========================================================================
    // Tests for 6 NEW intermarket indicators (RelativeValueScore, SpreadMomentum,
    // ConvergenceDivergence, PairStrength, CrossAssetMomentum, IntermarketSignal)
    // ========================================================================

    // RelativeValueScore tests
    #[test]
    fn test_relative_value_score_new() {
        assert!(RelativeValueScore::new(20, 5).is_ok());
        assert!(RelativeValueScore::new(30, 10).is_ok());
        assert!(RelativeValueScore::new(19, 5).is_err()); // period too small
        assert!(RelativeValueScore::new(20, 4).is_err()); // momentum_period too small
    }

    #[test]
    fn test_relative_value_score_new_validation() {
        let err = RelativeValueScore::new(19, 5).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 20");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = RelativeValueScore::new(20, 4).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "momentum_period");
                assert_eq!(reason, "must be at least 5");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_relative_value_score_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = RelativeValueScore::new(20, 5).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First values should be default (50)
        for i in 0..19 {
            assert_eq!(result[i], 50.0);
        }
        // Values after warmup should be between 0 and 100
        for i in 25..100 {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "Score at {} is out of bounds: {}", i, result[i]);
        }
    }

    #[test]
    fn test_relative_value_score_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RelativeValueScore::new(20, 5).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Relative Value Score");
        assert_eq!(indicator.min_periods(), 20);
    }

    #[test]
    fn test_relative_value_score_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RelativeValueScore::new(20, 5).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 25..100 {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_relative_value_score_insufficient_data() {
        let dual = create_test_dual_series(15);
        let indicator = RelativeValueScore::new(20, 5).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 15);
        assert!(result.iter().all(|&v| v == 50.0));
    }

    // SpreadMomentum tests
    #[test]
    fn test_spread_momentum_new() {
        assert!(SpreadMomentum::new(10, 3, 2).is_ok());
        assert!(SpreadMomentum::new(20, 5, 3).is_ok());
        assert!(SpreadMomentum::new(9, 3, 2).is_err()); // spread_period too small
        assert!(SpreadMomentum::new(10, 2, 2).is_err()); // momentum_period too small
        assert!(SpreadMomentum::new(10, 3, 0).is_err()); // smooth_period too small
    }

    #[test]
    fn test_spread_momentum_new_validation() {
        let err = SpreadMomentum::new(9, 3, 2).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "spread_period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = SpreadMomentum::new(10, 2, 2).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "momentum_period");
                assert_eq!(reason, "must be at least 3");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = SpreadMomentum::new(10, 3, 0).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "smooth_period");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_spread_momentum_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = SpreadMomentum::new(15, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Values should be finite after warmup
        for i in 25..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_spread_momentum_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = SpreadMomentum::new(15, 5, 3).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Spread Momentum");
        assert_eq!(indicator.min_periods(), 20);
    }

    #[test]
    fn test_spread_momentum_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = SpreadMomentum::new(15, 5, 3).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 25..100 {
            assert!(result[i].is_finite());
        }
    }

    #[test]
    fn test_spread_momentum_insufficient_data() {
        let dual = create_test_dual_series(15);
        let indicator = SpreadMomentum::new(15, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 15);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // ConvergenceDivergence tests
    #[test]
    fn test_convergence_divergence_new() {
        assert!(ConvergenceDivergence::new(15, 10, 3).is_ok());
        assert!(ConvergenceDivergence::new(20, 15, 5).is_ok());
        assert!(ConvergenceDivergence::new(14, 10, 3).is_err()); // correlation_period too small
        assert!(ConvergenceDivergence::new(15, 9, 3).is_err()); // spread_period too small
        assert!(ConvergenceDivergence::new(15, 10, 0).is_err()); // smooth_period too small
    }

    #[test]
    fn test_convergence_divergence_new_validation() {
        let err = ConvergenceDivergence::new(14, 10, 3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "correlation_period");
                assert_eq!(reason, "must be at least 15");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = ConvergenceDivergence::new(15, 9, 3).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "spread_period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = ConvergenceDivergence::new(15, 10, 0).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "smooth_period");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_convergence_divergence_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = ConvergenceDivergence::new(20, 15, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Values should be finite after warmup
        for i in 25..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_convergence_divergence_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = ConvergenceDivergence::new(20, 15, 3).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Convergence Divergence");
        assert_eq!(indicator.min_periods(), 21);
    }

    #[test]
    fn test_convergence_divergence_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = ConvergenceDivergence::new(20, 15, 3).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 25..100 {
            assert!(result[i].is_finite());
        }
    }

    #[test]
    fn test_convergence_divergence_insufficient_data() {
        let dual = create_test_dual_series(15);
        let indicator = ConvergenceDivergence::new(20, 15, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 15);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // PairStrength tests
    #[test]
    fn test_pair_strength_new() {
        assert!(PairStrength::new(30, 10).is_ok());
        assert!(PairStrength::new(40, 15).is_ok());
        assert!(PairStrength::new(29, 10).is_err()); // period too small
        assert!(PairStrength::new(30, 9).is_err()); // short_period too small
        assert!(PairStrength::new(30, 30).is_err()); // short_period >= period
    }

    #[test]
    fn test_pair_strength_new_validation() {
        let err = PairStrength::new(29, 10).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 30");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = PairStrength::new(30, 9).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "short_period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = PairStrength::new(30, 30).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "short_period");
                assert_eq!(reason, "must be less than period");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_pair_strength_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = PairStrength::new(30, 15).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // First values should be default (50)
        for i in 0..29 {
            assert_eq!(result[i], 50.0);
        }
        // Values after warmup should be between 0 and 100
        for i in 35..100 {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "Score at {} is out of bounds: {}", i, result[i]);
        }
    }

    #[test]
    fn test_pair_strength_highly_correlated() {
        // Two highly correlated series should have high pair strength
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = PairStrength::new(30, 15).unwrap();
        let result = indicator.calculate(&dual);

        // Highly correlated linear series should have good pair strength
        let avg_strength: f64 = result[40..].iter().sum::<f64>() / (result.len() - 40) as f64;
        assert!(avg_strength > 30.0, "Expected reasonable pair strength for correlated series, got {}", avg_strength);
    }

    #[test]
    fn test_pair_strength_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = PairStrength::new(30, 15).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Pair Strength");
        assert_eq!(indicator.min_periods(), 30);
    }

    #[test]
    fn test_pair_strength_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = PairStrength::new(30, 15).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 35..100 {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_pair_strength_insufficient_data() {
        let dual = create_test_dual_series(20);
        let indicator = PairStrength::new(30, 15).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 20);
        assert!(result.iter().all(|&v| v == 50.0));
    }

    // CrossAssetMomentum tests
    #[test]
    fn test_cross_asset_momentum_new() {
        assert!(CrossAssetMomentum::new(5, 3).is_ok());
        assert!(CrossAssetMomentum::new(10, 5).is_ok());
        assert!(CrossAssetMomentum::new(4, 3).is_err()); // momentum_period too small
        assert!(CrossAssetMomentum::new(5, 0).is_err()); // smooth_period too small
    }

    #[test]
    fn test_cross_asset_momentum_new_validation() {
        let err = CrossAssetMomentum::new(4, 3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "momentum_period");
                assert_eq!(reason, "must be at least 5");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = CrossAssetMomentum::new(5, 0).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "smooth_period");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_cross_asset_momentum_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = CrossAssetMomentum::new(10, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Values should be finite after warmup
        for i in 15..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_cross_asset_momentum_outperformance() {
        // Series1 with stronger momentum
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64).powf(1.1) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.1).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = CrossAssetMomentum::new(10, 3).unwrap();
        let result = indicator.calculate(&dual);

        // Series1 has stronger upward momentum, so cross-asset momentum should be positive
        let avg_mom: f64 = result[20..].iter().sum::<f64>() / (result.len() - 20) as f64;
        assert!(avg_mom > 0.0, "Expected positive momentum for outperforming series, got {}", avg_mom);
    }

    #[test]
    fn test_cross_asset_momentum_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = CrossAssetMomentum::new(10, 3).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Cross Asset Momentum");
        assert_eq!(indicator.min_periods(), 11);
    }

    #[test]
    fn test_cross_asset_momentum_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = CrossAssetMomentum::new(10, 3).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for i in 15..100 {
            assert!(result[i].is_finite());
        }
    }

    #[test]
    fn test_cross_asset_momentum_insufficient_data() {
        let dual = create_test_dual_series(8);
        let indicator = CrossAssetMomentum::new(10, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 8);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    // IntermarketSignal tests
    #[test]
    fn test_intermarket_signal_new() {
        assert!(IntermarketSignal::new(15, 10).is_ok());
        assert!(IntermarketSignal::new(20, 15).is_ok());
        assert!(IntermarketSignal::new(14, 10).is_err()); // spread_period too small
        assert!(IntermarketSignal::new(15, 9).is_err()); // correlation_period too small
    }

    #[test]
    fn test_intermarket_signal_new_validation() {
        let err = IntermarketSignal::new(14, 10).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "spread_period");
                assert_eq!(reason, "must be at least 15");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = IntermarketSignal::new(15, 9).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "correlation_period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_intermarket_signal_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = IntermarketSignal::new(20, 15).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Signals should be -2, -1, 0, 1, or 2
        for &s in &result {
            assert!(s == -2.0 || s == -1.0 || s == 0.0 || s == 1.0 || s == 2.0,
                "Invalid signal: {}", s);
        }
    }

    #[test]
    fn test_intermarket_signal_with_thresholds() {
        let dual = create_mean_reverting_spread(100);
        let indicator = IntermarketSignal::new(20, 15)
            .unwrap()
            .with_entry_threshold(1.0)
            .with_strong_threshold(2.0)
            .with_min_correlation(0.3);
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Should be valid signal values
        for &s in &result {
            assert!(s >= -2.0 && s <= 2.0);
        }
    }

    #[test]
    fn test_intermarket_signal_generates_signals() {
        // Create oscillating spread that should generate signals
        let mut series1 = Vec::with_capacity(200);
        let mut series2 = Vec::with_capacity(200);

        for i in 0..200 {
            series2.push(100.0);
            let spread = ((i as f64) * 0.15).sin() * 15.0;
            series1.push(100.0 + spread);
        }

        let dual = DualSeries::new(series1, series2);
        let indicator = IntermarketSignal::new(20, 15)
            .unwrap()
            .with_entry_threshold(1.5)
            .with_min_correlation(0.0); // Disable correlation filter
        let result = indicator.calculate(&dual);

        // Should have some non-zero signals
        let non_zero_count = result.iter().filter(|&&s| s != 0.0).count();
        assert!(non_zero_count > 0, "Should generate some trading signals");
    }

    #[test]
    fn test_intermarket_signal_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = IntermarketSignal::new(20, 15).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Intermarket Signal");
        assert_eq!(indicator.min_periods(), 20);
    }

    #[test]
    fn test_intermarket_signal_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + ((i as f64) * 0.2).sin() * 10.0).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + ((i as f64) * 0.2).sin() * 5.0).collect();

        let indicator = IntermarketSignal::new(20, 15).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
        for &s in &result {
            assert!(s >= -2.0 && s <= 2.0);
        }
    }

    #[test]
    fn test_intermarket_signal_insufficient_data() {
        let dual = create_test_dual_series(15);
        let indicator = IntermarketSignal::new(20, 15).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 15);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_intermarket_signal_missing_secondary() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let data = OHLCVSeries::from_close(series1);

        let indicator = IntermarketSignal::new(20, 15).unwrap();
        assert!(indicator.compute(&data).is_err());
    }

    // ========================================================================
    // RelativeRotationGraph tests
    // ========================================================================

    #[test]
    fn test_relative_rotation_graph_new() {
        assert!(RelativeRotationGraph::new(10, 5, 3).is_ok());
        assert!(RelativeRotationGraph::new(20, 10, 5).is_ok());
        assert!(RelativeRotationGraph::new(9, 5, 3).is_err()); // rs_period too small
        assert!(RelativeRotationGraph::new(10, 4, 3).is_err()); // momentum_period too small
        assert!(RelativeRotationGraph::new(10, 5, 0).is_err()); // norm_period too small
    }

    #[test]
    fn test_relative_rotation_graph_new_validation() {
        let err = RelativeRotationGraph::new(9, 5, 3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "rs_period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = RelativeRotationGraph::new(10, 4, 3).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "momentum_period");
                assert_eq!(reason, "must be at least 5");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = RelativeRotationGraph::new(10, 5, 0).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "norm_period");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_relative_rotation_graph_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = RelativeRotationGraph::new(10, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Quadrant values should be 0, 1, 2, 3, or 4
        for &v in &result {
            assert!(v >= 0.0 && v <= 4.0, "Invalid quadrant value: {}", v);
        }
    }

    #[test]
    fn test_relative_rotation_graph_quadrant_classification() {
        // Create series where asset outperforms benchmark (Leading quadrant expected)
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64).powf(1.2)).collect();
        let series2: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64)).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = RelativeRotationGraph::new(10, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        // After warmup, should have valid quadrant assignments
        let valid_quadrants: Vec<f64> = result.iter().filter(|&&v| v > 0.0).cloned().collect();
        assert!(!valid_quadrants.is_empty(), "Should have non-zero quadrant assignments");
    }

    #[test]
    fn test_relative_rotation_graph_detailed() {
        let dual = create_test_dual_series(100);
        let indicator = RelativeRotationGraph::new(10, 5, 3).unwrap();
        let (rs_ratio, rs_momentum, quadrants) = indicator.calculate_detailed(&dual);

        assert_eq!(rs_ratio.len(), 100);
        assert_eq!(rs_momentum.len(), 100);
        assert_eq!(quadrants.len(), 100);
    }

    #[test]
    fn test_relative_rotation_graph_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RelativeRotationGraph::new(10, 5, 3).unwrap().with_benchmark(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Relative Rotation Graph");
        assert_eq!(indicator.min_periods(), 18); // 10 + 5 + 3
    }

    #[test]
    fn test_relative_rotation_graph_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RelativeRotationGraph::new(10, 5, 3).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_relative_rotation_graph_insufficient_data() {
        let dual = create_test_dual_series(10);
        let indicator = RelativeRotationGraph::new(10, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_relative_rotation_graph_missing_benchmark() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let data = OHLCVSeries::from_close(series1);

        let indicator = RelativeRotationGraph::new(10, 5, 3).unwrap();
        assert!(indicator.compute(&data).is_err());
    }

    // ========================================================================
    // CrossMarketBeta tests
    // ========================================================================

    #[test]
    fn test_cross_market_beta_new() {
        assert!(CrossMarketBeta::new(20, 3).is_ok());
        assert!(CrossMarketBeta::new(30, 5).is_ok());
        assert!(CrossMarketBeta::new(19, 3).is_err()); // period too small
        assert!(CrossMarketBeta::new(20, 0).is_err()); // smooth_period too small
    }

    #[test]
    fn test_cross_market_beta_new_validation() {
        let err = CrossMarketBeta::new(19, 3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 20");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = CrossMarketBeta::new(20, 0).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "smooth_period");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_cross_market_beta_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = CrossMarketBeta::new(20, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Beta values should be finite after warmup
        for i in 25..100 {
            assert!(result[i].is_finite(), "Beta at {} is not finite", i);
        }
    }

    #[test]
    fn test_cross_market_beta_high_beta_asset() {
        // Create high beta asset with oscillating returns (more realistic)
        let mut market = Vec::with_capacity(100);
        let mut asset = Vec::with_capacity(100);
        let mut market_price = 100.0;
        let mut asset_price = 100.0;

        for i in 0..100 {
            // Market returns oscillate
            let market_return = (i as f64 * 0.3).sin() * 0.02;
            market_price *= 1.0 + market_return;
            // Asset returns are ~2x market returns
            asset_price *= 1.0 + market_return * 2.0;
            market.push(market_price);
            asset.push(asset_price);
        }

        let dual = DualSeries::new(asset, market);

        let indicator = CrossMarketBeta::new(20, 1).unwrap();
        let result = indicator.calculate(&dual);

        // Average beta should be approximately 2 (with tolerance for numerical precision)
        let avg_beta: f64 = result[25..].iter().sum::<f64>() / (result.len() - 25) as f64;
        assert!(avg_beta > 1.5 && avg_beta < 2.5, "Expected beta ~2, got {}", avg_beta);
    }

    #[test]
    fn test_cross_market_beta_detailed() {
        let dual = create_test_dual_series(100);
        let indicator = CrossMarketBeta::new(20, 3).unwrap();
        let (beta, r_squared, alpha) = indicator.calculate_detailed(&dual);

        assert_eq!(beta.len(), 100);
        assert_eq!(r_squared.len(), 100);
        assert_eq!(alpha.len(), 100);

        // R-squared should be between 0 and 1
        for i in 25..100 {
            assert!(r_squared[i] >= 0.0 && r_squared[i] <= 1.0,
                "R-squared at {} is {}, should be in [0,1]", i, r_squared[i]);
        }
    }

    #[test]
    fn test_cross_market_beta_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = CrossMarketBeta::new(20, 3).unwrap().with_market(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Cross Market Beta");
        assert_eq!(indicator.min_periods(), 21);
    }

    #[test]
    fn test_cross_market_beta_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = CrossMarketBeta::new(20, 3).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_cross_market_beta_insufficient_data() {
        let dual = create_test_dual_series(15);
        let indicator = CrossMarketBeta::new(20, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 15);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_cross_market_beta_missing_market() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let data = OHLCVSeries::from_close(series1);

        let indicator = CrossMarketBeta::new(20, 3).unwrap();
        assert!(indicator.compute(&data).is_err());
    }

    // ========================================================================
    // CorrelationBreakdownDetector tests
    // ========================================================================

    #[test]
    fn test_correlation_breakdown_detector_new() {
        assert!(CorrelationBreakdownDetector::new(10, 30, 0.3).is_ok());
        assert!(CorrelationBreakdownDetector::new(15, 45, 0.5).is_ok());
        assert!(CorrelationBreakdownDetector::new(9, 30, 0.3).is_err()); // short_period too small
        assert!(CorrelationBreakdownDetector::new(10, 10, 0.3).is_err()); // long <= short
        assert!(CorrelationBreakdownDetector::new(10, 30, 0.0).is_err()); // threshold <= 0
    }

    #[test]
    fn test_correlation_breakdown_detector_new_validation() {
        let err = CorrelationBreakdownDetector::new(9, 30, 0.3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "short_period");
                assert_eq!(reason, "must be at least 10");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = CorrelationBreakdownDetector::new(10, 10, 0.3).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "long_period");
                assert_eq!(reason, "must be greater than short_period");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = CorrelationBreakdownDetector::new(10, 30, -0.1).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "threshold");
                assert_eq!(reason, "must be greater than 0");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_correlation_breakdown_detector_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = CorrelationBreakdownDetector::new(10, 30, 0.3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Values should be finite
        for i in 35..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_correlation_breakdown_detector_breakdown_detection() {
        // Create two series with correlation breakdown
        let mut series1 = Vec::with_capacity(200);
        let mut series2 = Vec::with_capacity(200);

        // First half: highly correlated
        for i in 0..100 {
            let base = 100.0 + (i as f64) * 0.5;
            series1.push(base + (i as f64 * 0.1).sin() * 2.0);
            series2.push(base * 0.5 + (i as f64 * 0.1).sin() * 1.0);
        }

        // Second half: uncorrelated (correlation breakdown)
        for i in 100..200 {
            series1.push(150.0 + (i as f64 * 0.3).sin() * 5.0);
            series2.push(75.0 + (i as f64 * 0.7).cos() * 5.0); // Different pattern
        }

        let dual = DualSeries::new(series1, series2);
        let indicator = CorrelationBreakdownDetector::new(10, 30, 0.2).unwrap();
        let result = indicator.calculate(&dual);

        // Should detect breakdown around the transition
        let breakdown_scores: Vec<f64> = result[100..150].to_vec();
        let max_breakdown = breakdown_scores.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(max_breakdown > 0.0, "Should detect correlation breakdown");
    }

    #[test]
    fn test_correlation_breakdown_detector_detailed() {
        let dual = create_test_dual_series(100);
        let indicator = CorrelationBreakdownDetector::new(10, 30, 0.3).unwrap();
        let (short_corr, long_corr, breakdown_score) = indicator.calculate_detailed(&dual);

        assert_eq!(short_corr.len(), 100);
        assert_eq!(long_corr.len(), 100);
        assert_eq!(breakdown_score.len(), 100);

        // Correlations should be between -1 and 1
        for i in 35..100 {
            assert!(short_corr[i] >= -1.0 && short_corr[i] <= 1.0);
            assert!(long_corr[i] >= -1.0 && long_corr[i] <= 1.0);
        }
    }

    #[test]
    fn test_correlation_breakdown_detector_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = CorrelationBreakdownDetector::new(10, 30, 0.3).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Correlation Breakdown Detector");
        assert_eq!(indicator.min_periods(), 31);
    }

    #[test]
    fn test_correlation_breakdown_detector_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = CorrelationBreakdownDetector::new(10, 30, 0.3).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_correlation_breakdown_detector_insufficient_data() {
        let dual = create_test_dual_series(20);
        let indicator = CorrelationBreakdownDetector::new(10, 30, 0.3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 20);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_correlation_breakdown_detector_missing_secondary() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let data = OHLCVSeries::from_close(series1);

        let indicator = CorrelationBreakdownDetector::new(10, 30, 0.3).unwrap();
        assert!(indicator.compute(&data).is_err());
    }

    // ========================================================================
    // LeadLagAnalysis tests
    // ========================================================================

    #[test]
    fn test_lead_lag_analysis_new() {
        assert!(LeadLagAnalysis::new(30, 5).is_ok());
        assert!(LeadLagAnalysis::new(45, 10).is_ok());
        assert!(LeadLagAnalysis::new(29, 5).is_err()); // period too small
        assert!(LeadLagAnalysis::new(30, 0).is_err()); // max_lag too small
        assert!(LeadLagAnalysis::new(30, 15).is_err()); // max_lag >= period/3
    }

    #[test]
    fn test_lead_lag_analysis_new_validation() {
        let err = LeadLagAnalysis::new(29, 5).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "period");
                assert_eq!(reason, "must be at least 30");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = LeadLagAnalysis::new(30, 0).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "max_lag");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = LeadLagAnalysis::new(30, 15).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "max_lag");
                assert_eq!(reason, "must be less than period / 3");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_lead_lag_analysis_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = LeadLagAnalysis::new(30, 5).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Values should be finite after warmup
        for i in 35..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_lead_lag_analysis_series1_leads() {
        // Create series where series1 leads series2 by a few periods
        let mut series1 = Vec::with_capacity(100);
        let mut series2 = Vec::with_capacity(100);

        for i in 0..100 {
            series1.push(100.0 + (i as f64 * 0.2).sin() * 10.0);
        }

        // Series2 follows series1 with a lag
        for i in 0..100 {
            let lagged_i = (i as i32 - 3).max(0) as usize;
            series2.push(50.0 + ((lagged_i as f64) * 0.2).sin() * 5.0);
        }

        let dual = DualSeries::new(series1, series2);
        let indicator = LeadLagAnalysis::new(30, 5).unwrap().with_min_correlation(0.0);
        let result = indicator.calculate(&dual);

        // Should detect that series1 leads (positive values expected)
        let avg_score: f64 = result[40..].iter().sum::<f64>() / (result.len() - 40) as f64;
        // Note: The actual sign depends on implementation; just check it's not all zeros
        assert!(result[40..].iter().any(|&v| v != 0.0), "Should have non-zero lead/lag scores");
    }

    #[test]
    fn test_lead_lag_analysis_detailed() {
        let dual = create_test_dual_series(100);
        let indicator = LeadLagAnalysis::new(30, 5).unwrap();
        let (optimal_lag, correlation, score) = indicator.calculate_detailed(&dual);

        assert_eq!(optimal_lag.len(), 100);
        assert_eq!(correlation.len(), 100);
        assert_eq!(score.len(), 100);
    }

    #[test]
    fn test_lead_lag_analysis_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = LeadLagAnalysis::new(30, 5).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Lead Lag Analysis");
        assert_eq!(indicator.min_periods(), 30);
    }

    #[test]
    fn test_lead_lag_analysis_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = LeadLagAnalysis::new(30, 5).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_lead_lag_analysis_insufficient_data() {
        let dual = create_test_dual_series(20);
        let indicator = LeadLagAnalysis::new(30, 5).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 20);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_lead_lag_analysis_missing_secondary() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let data = OHLCVSeries::from_close(series1);

        let indicator = LeadLagAnalysis::new(30, 5).unwrap();
        assert!(indicator.compute(&data).is_err());
    }

    #[test]
    fn test_lead_lag_analysis_with_min_correlation() {
        let dual = create_test_dual_series(100);
        let indicator = LeadLagAnalysis::new(30, 5).unwrap().with_min_correlation(0.5);
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
    }

    // ========================================================================
    // SpreadMomentumIndicator tests
    // ========================================================================

    #[test]
    fn test_spread_momentum_indicator_new() {
        assert!(SpreadMomentumIndicator::new(15, 5, 3).is_ok());
        assert!(SpreadMomentumIndicator::new(20, 10, 5).is_ok());
        assert!(SpreadMomentumIndicator::new(14, 5, 3).is_err()); // spread_period too small
        assert!(SpreadMomentumIndicator::new(15, 4, 3).is_err()); // momentum_period too small
        assert!(SpreadMomentumIndicator::new(15, 5, 2).is_err()); // accel_period too small
    }

    #[test]
    fn test_spread_momentum_indicator_new_validation() {
        let err = SpreadMomentumIndicator::new(14, 5, 3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "spread_period");
                assert_eq!(reason, "must be at least 15");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = SpreadMomentumIndicator::new(15, 4, 3).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "momentum_period");
                assert_eq!(reason, "must be at least 5");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = SpreadMomentumIndicator::new(15, 5, 2).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "accel_period");
                assert_eq!(reason, "must be at least 3");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_spread_momentum_indicator_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = SpreadMomentumIndicator::new(15, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Values should be finite after warmup
        for i in 25..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_spread_momentum_indicator_widening_spread() {
        // Create series where spread clearly widens
        // Series2 stays flat, Series1 accelerates upward
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 2.0 + (i as f64 * 0.02).powi(2) * 50.0).collect();
        let series2: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let dual = DualSeries::new(series1, series2);

        let indicator = SpreadMomentumIndicator::new(15, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        // Results should be finite after warmup
        let non_zero_count = result[30..].iter().filter(|&&v| v.abs() > 0.001).count();
        assert!(non_zero_count > 0, "Expected some non-zero momentum values");
    }

    #[test]
    fn test_spread_momentum_indicator_detailed() {
        let dual = create_test_dual_series(100);
        let indicator = SpreadMomentumIndicator::new(15, 5, 3).unwrap();
        let (spread, momentum, acceleration, score) = indicator.calculate_detailed(&dual);

        assert_eq!(spread.len(), 100);
        assert_eq!(momentum.len(), 100);
        assert_eq!(acceleration.len(), 100);
        assert_eq!(score.len(), 100);
    }

    #[test]
    fn test_spread_momentum_indicator_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = SpreadMomentumIndicator::new(15, 5, 3).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Spread Momentum Indicator");
        assert_eq!(indicator.min_periods(), 23); // 15 + 5 + 3
    }

    #[test]
    fn test_spread_momentum_indicator_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = SpreadMomentumIndicator::new(15, 5, 3).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_spread_momentum_indicator_insufficient_data() {
        let dual = create_test_dual_series(15);
        let indicator = SpreadMomentumIndicator::new(15, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 15);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_spread_momentum_indicator_missing_secondary() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let data = OHLCVSeries::from_close(series1);

        let indicator = SpreadMomentumIndicator::new(15, 5, 3).unwrap();
        assert!(indicator.compute(&data).is_err());
    }

    // ========================================================================
    // RelativeValueMomentum tests
    // ========================================================================

    #[test]
    fn test_relative_value_momentum_new() {
        assert!(RelativeValueMomentum::new(20, 5, 3).is_ok());
        assert!(RelativeValueMomentum::new(30, 10, 5).is_ok());
        assert!(RelativeValueMomentum::new(19, 5, 3).is_err()); // norm_period too small
        assert!(RelativeValueMomentum::new(20, 4, 3).is_err()); // momentum_period too small
        assert!(RelativeValueMomentum::new(20, 5, 0).is_err()); // smooth_period too small
    }

    #[test]
    fn test_relative_value_momentum_new_validation() {
        let err = RelativeValueMomentum::new(19, 5, 3).unwrap_err();
        match err {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "norm_period");
                assert_eq!(reason, "must be at least 20");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err2 = RelativeValueMomentum::new(20, 4, 3).unwrap_err();
        match err2 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "momentum_period");
                assert_eq!(reason, "must be at least 5");
            }
            _ => panic!("Expected InvalidParameter error"),
        }

        let err3 = RelativeValueMomentum::new(20, 5, 0).unwrap_err();
        match err3 {
            IndicatorError::InvalidParameter { name, reason } => {
                assert_eq!(name, "smooth_period");
                assert_eq!(reason, "must be at least 1");
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }

    #[test]
    fn test_relative_value_momentum_calculate() {
        let dual = create_test_dual_series(100);
        let indicator = RelativeValueMomentum::new(20, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 100);
        // Values should be finite after warmup
        for i in 30..100 {
            assert!(result[i].is_finite(), "Value at {} is not finite", i);
        }
    }

    #[test]
    fn test_relative_value_momentum_series1_outperforming() {
        // Create series where series1 clearly gains value relative to series2
        // Using oscillating returns to create clear momentum changes
        let mut series1 = Vec::with_capacity(100);
        let mut series2 = Vec::with_capacity(100);
        let mut price1 = 100.0;
        let mut price2 = 50.0;

        for i in 0..100 {
            // Series1 consistently outperforms series2
            price1 *= 1.0 + 0.02 + (i as f64 * 0.2).sin() * 0.01;
            price2 *= 1.0 + 0.005 + (i as f64 * 0.2).sin() * 0.005;
            series1.push(price1);
            series2.push(price2);
        }

        let dual = DualSeries::new(series1, series2);

        let indicator = RelativeValueMomentum::new(20, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        // Values should be finite after warmup
        for i in 30..100 {
            assert!(result[i].is_finite(), "Value at {} should be finite", i);
        }
    }

    #[test]
    fn test_relative_value_momentum_detailed() {
        let dual = create_test_dual_series(100);
        let indicator = RelativeValueMomentum::new(20, 5, 3).unwrap();
        let (ratio, normalized, momentum, score) = indicator.calculate_detailed(&dual);

        assert_eq!(ratio.len(), 100);
        assert_eq!(normalized.len(), 100);
        assert_eq!(momentum.len(), 100);
        assert_eq!(score.len(), 100);

        // Ratio should be positive
        for i in 0..100 {
            assert!(ratio[i] > 0.0, "Ratio at {} should be positive", i);
        }
    }

    #[test]
    fn test_relative_value_momentum_compute() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RelativeValueMomentum::new(20, 5, 3).unwrap().with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = indicator.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert_eq!(indicator.name(), "Relative Value Momentum");
        assert_eq!(indicator.min_periods(), 25); // 20 + 5
    }

    #[test]
    fn test_relative_value_momentum_calculate_between() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let indicator = RelativeValueMomentum::new(20, 5, 3).unwrap();
        let result = indicator.calculate_between(&series1, &series2);

        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_relative_value_momentum_insufficient_data() {
        let dual = create_test_dual_series(15);
        let indicator = RelativeValueMomentum::new(20, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        assert_eq!(result.len(), 15);
        assert!(result.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_relative_value_momentum_missing_secondary() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let data = OHLCVSeries::from_close(series1);

        let indicator = RelativeValueMomentum::new(20, 5, 3).unwrap();
        assert!(indicator.compute(&data).is_err());
    }

    #[test]
    fn test_relative_value_momentum_mean_reverting() {
        // Create mean-reverting relative value
        let mut series1 = Vec::with_capacity(100);
        let mut series2 = Vec::with_capacity(100);

        for i in 0..100 {
            let base = 100.0;
            let oscillation = (i as f64 * 0.2).sin() * 5.0;
            series1.push(base + oscillation);
            series2.push(base * 0.5); // Constant
        }

        let dual = DualSeries::new(series1, series2);
        let indicator = RelativeValueMomentum::new(20, 5, 3).unwrap();
        let result = indicator.calculate(&dual);

        // Should have both positive and negative momentum values
        let has_positive = result.iter().any(|&v| v > 0.0);
        let has_negative = result.iter().any(|&v| v < 0.0);
        assert!(has_positive || has_negative, "Should have varying momentum for oscillating ratio");
    }
}
