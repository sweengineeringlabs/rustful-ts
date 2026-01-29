//! Rolling Seasonality Indicator (IND-244)
//!
//! Analyzes multi-year seasonal patterns in market behavior.
//! Identifies recurring patterns at specific times of year.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Seasonal pattern type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeasonalPattern {
    /// January Effect (early year outperformance)
    JanuaryEffect,
    /// Sell in May (summer weakness)
    SellInMay,
    /// Halloween Effect (Nov-Apr strength)
    HalloweenEffect,
    /// Year-end Rally (Santa Rally)
    YearEndRally,
    /// Tax Loss Selling (late year)
    TaxLossSelling,
    /// No clear pattern
    NoPattern,
}

/// Configuration for the Rolling Seasonality indicator.
#[derive(Debug, Clone)]
pub struct RollingSeasonalityConfig {
    /// Number of years for historical comparison
    pub lookback_years: usize,
    /// Trading days per year
    pub trading_days_per_year: usize,
    /// Window for pattern matching (days)
    pub pattern_window: usize,
    /// Smoothing period
    pub smoothing: usize,
}

impl Default for RollingSeasonalityConfig {
    fn default() -> Self {
        Self {
            lookback_years: 5,
            trading_days_per_year: 252,
            pattern_window: 21, // ~1 month
            smoothing: 5,
        }
    }
}

/// Rolling Seasonality indicator for multi-year pattern analysis.
///
/// This indicator identifies:
/// - Month-of-year effects
/// - Day-of-year patterns
/// - Multi-year recurring trends
/// - Holiday effects
#[derive(Debug, Clone)]
pub struct RollingSeasonality {
    config: RollingSeasonalityConfig,
}

impl RollingSeasonality {
    /// Create a new Rolling Seasonality indicator.
    pub fn new() -> Self {
        Self {
            config: RollingSeasonalityConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: RollingSeasonalityConfig) -> Self {
        Self { config }
    }

    /// Estimate day of year from index (0-251).
    fn estimate_day_of_year(&self, day_index: usize) -> usize {
        day_index % self.config.trading_days_per_year
    }

    /// Estimate month from day of year (1-12).
    fn estimate_month(&self, day_of_year: usize) -> usize {
        // Approximately 21 trading days per month
        (day_of_year / 21).min(11) + 1
    }

    /// Calculate average returns for each day of year.
    fn calculate_day_of_year_returns(&self, close: &[f64]) -> Vec<f64> {
        let days_per_year = self.config.trading_days_per_year;
        let mut day_returns = vec![0.0; days_per_year];
        let mut day_counts = vec![0usize; days_per_year];

        for i in 1..close.len() {
            let day_of_year = self.estimate_day_of_year(i);
            if close[i - 1] > 0.0 {
                let ret = (close[i] - close[i - 1]) / close[i - 1];
                day_returns[day_of_year] += ret;
                day_counts[day_of_year] += 1;
            }
        }

        // Average returns
        for i in 0..days_per_year {
            if day_counts[i] > 0 {
                day_returns[i] /= day_counts[i] as f64;
            }
        }

        day_returns
    }

    /// Calculate average returns for each month.
    fn calculate_monthly_returns(&self, close: &[f64]) -> [f64; 12] {
        let mut month_returns = [0.0; 12];
        let mut month_counts = [0usize; 12];
        let days_per_year = self.config.trading_days_per_year;

        // Calculate monthly returns (start of month to end of month)
        let days_per_month = days_per_year / 12;

        for year in 0..close.len() / days_per_year {
            for month in 0..12 {
                let month_start = year * days_per_year + month * days_per_month;
                let month_end = (month_start + days_per_month).min(close.len() - 1);

                if month_start < close.len() && month_end < close.len() && close[month_start] > 0.0 {
                    let ret = (close[month_end] - close[month_start]) / close[month_start];
                    month_returns[month] += ret;
                    month_counts[month] += 1;
                }
            }
        }

        // Average returns
        for i in 0..12 {
            if month_counts[i] > 0 {
                month_returns[i] /= month_counts[i] as f64;
            }
        }

        month_returns
    }

    /// Detect current seasonal pattern based on time of year and historical data.
    fn detect_pattern(&self, day_of_year: usize, monthly_returns: &[f64; 12]) -> SeasonalPattern {
        let month = self.estimate_month(day_of_year);

        // Check for known seasonal effects
        match month {
            1 => {
                if monthly_returns[0] > 0.02 {
                    SeasonalPattern::JanuaryEffect
                } else {
                    SeasonalPattern::NoPattern
                }
            }
            5 | 6 | 7 | 8 => {
                let summer_avg = (monthly_returns[4] + monthly_returns[5] + monthly_returns[6] + monthly_returns[7]) / 4.0;
                if summer_avg < 0.0 {
                    SeasonalPattern::SellInMay
                } else {
                    SeasonalPattern::NoPattern
                }
            }
            11 | 12 => {
                let winter_avg = (monthly_returns[10] + monthly_returns[11]) / 2.0;
                if winter_avg > 0.015 {
                    SeasonalPattern::HalloweenEffect
                } else if month == 12 && monthly_returns[11] > 0.01 {
                    SeasonalPattern::YearEndRally
                } else {
                    SeasonalPattern::NoPattern
                }
            }
            10 => {
                if monthly_returns[9] < -0.01 {
                    SeasonalPattern::TaxLossSelling
                } else {
                    SeasonalPattern::NoPattern
                }
            }
            _ => SeasonalPattern::NoPattern,
        }
    }

    /// Calculate rolling correlation with historical same-period data.
    fn calculate_seasonal_correlation(&self, close: &[f64], idx: usize) -> f64 {
        if idx < self.config.trading_days_per_year + self.config.pattern_window {
            return f64::NAN;
        }

        let window = self.config.pattern_window;
        let year_days = self.config.trading_days_per_year;

        // Current period returns
        let current_returns: Vec<f64> = (0..window)
            .filter_map(|j| {
                let i = idx - j;
                if i > 0 && close[i - 1] > 0.0 {
                    Some((close[i] - close[i - 1]) / close[i - 1])
                } else {
                    None
                }
            })
            .collect();

        if current_returns.len() < window / 2 {
            return f64::NAN;
        }

        // Historical same-period returns
        let mut correlations = Vec::new();

        for year_back in 1..=self.config.lookback_years {
            let hist_idx = idx.saturating_sub(year_back * year_days);
            if hist_idx < window {
                continue;
            }

            let historical_returns: Vec<f64> = (0..window)
                .filter_map(|j| {
                    let i = hist_idx - j;
                    if i > 0 && i < close.len() && close[i - 1] > 0.0 {
                        Some((close[i] - close[i - 1]) / close[i - 1])
                    } else {
                        None
                    }
                })
                .collect();

            if historical_returns.len() >= window / 2 {
                let corr = self.calculate_correlation(&current_returns, &historical_returns);
                if !corr.is_nan() {
                    correlations.push(corr);
                }
            }
        }

        if correlations.is_empty() {
            f64::NAN
        } else {
            correlations.iter().sum::<f64>() / correlations.len() as f64
        }
    }

    /// Calculate Pearson correlation between two series.
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        if n < 2 {
            return f64::NAN;
        }

        let x_mean: f64 = x[..n].iter().sum::<f64>() / n as f64;
        let y_mean: f64 = y[..n].iter().sum::<f64>() / n as f64;

        let mut num = 0.0;
        let mut x_var = 0.0;
        let mut y_var = 0.0;

        for i in 0..n {
            let x_diff = x[i] - x_mean;
            let y_diff = y[i] - y_mean;
            num += x_diff * y_diff;
            x_var += x_diff * x_diff;
            y_var += y_diff * y_diff;
        }

        let denom = (x_var * y_var).sqrt();
        if denom > 0.0 {
            num / denom
        } else {
            f64::NAN
        }
    }

    /// Calculate the rolling seasonality indicators.
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<SeasonalPattern>) {
        let n = close.len();
        let day_of_year_returns = self.calculate_day_of_year_returns(close);
        let monthly_returns = self.calculate_monthly_returns(close);

        let mut seasonal_bias = vec![f64::NAN; n];
        let mut seasonal_correlation = vec![f64::NAN; n];
        let mut pattern = vec![SeasonalPattern::NoPattern; n];

        for i in 0..n {
            let day_of_year = self.estimate_day_of_year(i);
            pattern[i] = self.detect_pattern(day_of_year, &monthly_returns);

            if i >= self.config.trading_days_per_year {
                // Seasonal bias is the average return for this day of year
                seasonal_bias[i] = day_of_year_returns[day_of_year] * 10000.0; // Basis points

                seasonal_correlation[i] = self.calculate_seasonal_correlation(close, i);
            }
        }

        // Apply smoothing
        if self.config.smoothing > 1 {
            seasonal_bias = self.smooth(&seasonal_bias);
            seasonal_correlation = self.smooth(&seasonal_correlation);
        }

        (seasonal_bias, seasonal_correlation, pattern)
    }

    /// Simple moving average smoothing.
    fn smooth(&self, values: &[f64]) -> Vec<f64> {
        let n = values.len();
        let period = self.config.smoothing;
        let mut smoothed = vec![f64::NAN; n];

        for i in period - 1..n {
            let sum: f64 = (0..period)
                .map(|j| values[i - j])
                .filter(|v| !v.is_nan())
                .sum();
            let count = (0..period).filter(|&j| !values[i - j].is_nan()).count();
            if count > 0 {
                smoothed[i] = sum / count as f64;
            }
        }

        smoothed
    }

    /// Get the primary seasonal signal (bias).
    pub fn calculate_signal(&self, close: &[f64]) -> Vec<f64> {
        self.calculate(close).0
    }
}

impl Default for RollingSeasonality {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for RollingSeasonality {
    fn name(&self) -> &str {
        "RollingSeasonality"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_periods = self.config.trading_days_per_year * 2;
        if data.close.len() < min_periods {
            return Err(IndicatorError::InsufficientData {
                required: min_periods,
                got: data.close.len(),
            });
        }

        let (seasonal_bias, seasonal_correlation, _) = self.calculate(&data.close);

        Ok(IndicatorOutput::dual(
            seasonal_bias,
            seasonal_correlation,
        ))
    }

    fn min_periods(&self) -> usize {
        self.config.trading_days_per_year * 2
    }
}

impl SignalIndicator for RollingSeasonality {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let min_periods = self.config.trading_days_per_year * 2;
        if data.close.len() < min_periods {
            return Ok(IndicatorSignal::Neutral);
        }

        let (seasonal_bias, seasonal_correlation, pattern) = self.calculate(&data.close);
        let n = seasonal_bias.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let bias = seasonal_bias[n - 1];
        let corr = seasonal_correlation[n - 1];
        let current_pattern = pattern[n - 1];

        // Strong signal when high correlation with historical pattern
        if !bias.is_nan() && !corr.is_nan() {
            // High correlation with historical same-period behavior
            if corr > 0.5 {
                if bias > 5.0 {
                    return Ok(IndicatorSignal::Bullish);
                } else if bias < -5.0 {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        // Signal based on known seasonal patterns
        match current_pattern {
            SeasonalPattern::JanuaryEffect | SeasonalPattern::HalloweenEffect | SeasonalPattern::YearEndRally => {
                Ok(IndicatorSignal::Bullish)
            }
            SeasonalPattern::SellInMay | SeasonalPattern::TaxLossSelling => {
                Ok(IndicatorSignal::Bearish)
            }
            SeasonalPattern::NoPattern => Ok(IndicatorSignal::Neutral),
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (seasonal_bias, seasonal_correlation, pattern) = self.calculate(&data.close);

        let signals = seasonal_bias
            .iter()
            .enumerate()
            .map(|(i, &bias)| {
                if !bias.is_nan() && !seasonal_correlation[i].is_nan() {
                    let corr = seasonal_correlation[i];
                    if corr > 0.5 {
                        if bias > 5.0 {
                            return IndicatorSignal::Bullish;
                        } else if bias < -5.0 {
                            return IndicatorSignal::Bearish;
                        }
                    }
                }

                match pattern[i] {
                    SeasonalPattern::JanuaryEffect
                    | SeasonalPattern::HalloweenEffect
                    | SeasonalPattern::YearEndRally => IndicatorSignal::Bullish,
                    SeasonalPattern::SellInMay | SeasonalPattern::TaxLossSelling => {
                        IndicatorSignal::Bearish
                    }
                    SeasonalPattern::NoPattern => IndicatorSignal::Neutral,
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(n: usize) -> Vec<f64> {
        // Create data with some seasonality (higher in winter months)
        (0..n)
            .map(|i| {
                let day_of_year = i % 252;
                let month = day_of_year / 21;
                let seasonal = if month < 3 || month > 9 { 0.002 } else { -0.001 };
                100.0 + i as f64 * 0.01 + seasonal * i as f64
            })
            .collect()
    }

    #[test]
    fn test_rolling_seasonality_basic() {
        let config = RollingSeasonalityConfig {
            lookback_years: 2,
            trading_days_per_year: 252,
            pattern_window: 10,
            smoothing: 3,
        };
        let indicator = RollingSeasonality::with_config(config);
        let close = create_test_data(600);

        let (seasonal_bias, seasonal_correlation, pattern) = indicator.calculate(&close);

        assert_eq!(seasonal_bias.len(), 600);
        assert_eq!(seasonal_correlation.len(), 600);
        assert_eq!(pattern.len(), 600);
    }

    #[test]
    fn test_month_estimation() {
        let indicator = RollingSeasonality::new();

        // January (trading days 0-20)
        assert_eq!(indicator.estimate_month(10), 1);

        // December (trading days ~231-251)
        assert_eq!(indicator.estimate_month(240), 12);

        // June (trading days ~105-125)
        assert_eq!(indicator.estimate_month(110), 6);
    }

    #[test]
    fn test_seasonal_patterns() {
        let indicator = RollingSeasonality::new();

        // Test pattern detection with strong January
        let strong_jan = [0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let pattern = indicator.detect_pattern(10, &strong_jan);
        assert_eq!(pattern, SeasonalPattern::JanuaryEffect);

        // Test summer weakness
        let weak_summer = [0.0, 0.0, 0.0, 0.0, -0.02, -0.02, -0.02, -0.02, 0.0, 0.0, 0.0, 0.0];
        let pattern = indicator.detect_pattern(110, &weak_summer);
        assert_eq!(pattern, SeasonalPattern::SellInMay);
    }

    #[test]
    fn test_monthly_returns() {
        let config = RollingSeasonalityConfig {
            lookback_years: 1,
            trading_days_per_year: 252,
            pattern_window: 10,
            smoothing: 1,
        };
        let indicator = RollingSeasonality::with_config(config);
        let close = create_test_data(300);

        let monthly = indicator.calculate_monthly_returns(&close);

        // Should have 12 months
        assert_eq!(monthly.len(), 12);
    }

    #[test]
    fn test_correlation_calculation() {
        let indicator = RollingSeasonality::new();

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let corr = indicator.calculate_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.001); // Perfect correlation

        let y_neg = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = indicator.calculate_correlation(&x, &y_neg);
        assert!((corr_neg - (-1.0)).abs() < 0.001); // Perfect negative correlation
    }

    #[test]
    fn test_rolling_seasonality_insufficient_data() {
        let indicator = RollingSeasonality::new();
        let data = OHLCVSeries {
            open: vec![100.0; 200],
            high: vec![101.0; 200],
            low: vec![99.0; 200],
            close: vec![100.0; 200],
            volume: vec![1000.0; 200],
        };

        let result = indicator.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_rolling_seasonality_technical_indicator() {
        let config = RollingSeasonalityConfig {
            lookback_years: 1,
            trading_days_per_year: 100,
            pattern_window: 10,
            smoothing: 3,
        };
        let indicator = RollingSeasonality::with_config(config);

        let close: Vec<f64> = (0..300).map(|i| 100.0 + i as f64 * 0.1).collect();
        let data = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|c| c + 1.0).collect(),
            low: close.iter().map(|c| c - 1.0).collect(),
            close,
            volume: vec![1000000.0; 300],
        };

        let result = indicator.compute(&data);
        assert!(result.is_ok());

        assert_eq!(indicator.name(), "RollingSeasonality");
    }
}
