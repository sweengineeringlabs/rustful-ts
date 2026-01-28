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
}
