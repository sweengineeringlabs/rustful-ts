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
}
