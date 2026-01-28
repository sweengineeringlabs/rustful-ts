//! Volatility Cone implementation.
//!
//! Shows percentile bands of historical volatility over different lookback periods.
//! Used to compare current volatility to historical ranges.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use indicator_api::VolatilityConeConfig;

/// Volatility Cone output containing volatility data for all periods.
#[derive(Debug, Clone)]
pub struct VolatilityConeOutput {
    /// Current volatility for each lookback period.
    pub current_volatility: Vec<Vec<f64>>,
    /// Percentile values for each lookback period.
    /// Outer vec: periods, Inner vec: percentiles.
    pub percentile_values: Vec<Vec<Vec<f64>>>,
    /// Current volatility percentile rank (0-1) for each period.
    pub percentile_ranks: Vec<Vec<f64>>,
}

/// Volatility Cone.
///
/// The Volatility Cone calculates historical volatility for multiple lookback
/// periods and shows where the current volatility sits within the historical
/// distribution using configurable percentiles.
///
/// This is a key tool for:
/// - Options traders comparing implied vs realized volatility
/// - Risk managers assessing current market conditions
/// - Traders looking for mean reversion opportunities
///
/// # Algorithm
/// 1. Calculate historical volatility for multiple lookback periods (e.g., 20, 40, 60, 120, 252 days)
/// 2. For each period, compute percentiles (e.g., 10th, 25th, 50th, 75th, 90th)
/// 3. Output current volatility position relative to historical cone
///
/// # Signal Logic
/// - Current vol below 25th percentile (average across periods): Bullish (low vol precedes breakouts)
/// - Current vol above 75th percentile (average across periods): Bearish (high vol precedes reversion)
/// - Otherwise: Neutral
#[derive(Debug, Clone)]
pub struct VolatilityCone {
    /// Configuration for the volatility cone.
    config: VolatilityConeConfig,
    /// Number of trading days per year for annualization.
    trading_days: f64,
}

impl VolatilityCone {
    /// Create a new Volatility Cone indicator with config.
    pub fn new(config: VolatilityConeConfig) -> Self {
        Self {
            config,
            trading_days: 252.0,
        }
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self::new(VolatilityConeConfig::default())
    }

    /// Set custom trading days for annualization.
    pub fn with_trading_days(mut self, days: f64) -> Self {
        self.trading_days = days;
        self
    }

    /// Get the configuration.
    pub fn config(&self) -> &VolatilityConeConfig {
        &self.config
    }

    /// Calculate log returns from price data.
    fn log_returns(data: &[f64]) -> Vec<f64> {
        let mut returns = Vec::with_capacity(data.len().saturating_sub(1));
        for i in 1..data.len() {
            if data[i - 1] > 0.0 && data[i] > 0.0 {
                returns.push((data[i] / data[i - 1]).ln());
            } else {
                returns.push(f64::NAN);
            }
        }
        returns
    }

    /// Calculate annualized volatility for a window of log returns.
    fn calculate_volatility(&self, returns: &[f64], period: usize) -> f64 {
        let valid_returns: Vec<f64> = returns.iter().filter(|x| !x.is_nan()).copied().collect();

        if valid_returns.len() < period {
            return f64::NAN;
        }

        let mean: f64 = valid_returns.iter().sum::<f64>() / valid_returns.len() as f64;
        let variance: f64 = valid_returns
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / valid_returns.len() as f64;

        variance.sqrt() * self.trading_days.sqrt()
    }

    /// Calculate percentile from sorted data using linear interpolation.
    fn percentile(sorted_data: &[f64], p: f64) -> f64 {
        if sorted_data.is_empty() {
            return f64::NAN;
        }

        let n = sorted_data.len();
        if n == 1 {
            return sorted_data[0];
        }

        let pos = p * (n - 1) as f64;
        let lower = pos.floor() as usize;
        let upper = pos.ceil() as usize;

        if lower == upper || upper >= n {
            sorted_data[lower.min(n - 1)]
        } else {
            let frac = pos - lower as f64;
            sorted_data[lower] * (1.0 - frac) + sorted_data[upper] * frac
        }
    }

    /// Calculate the minimum required data length.
    pub fn min_data_length(&self) -> usize {
        // Need enough data for the longest period plus some history
        let max_period = self.config.periods.iter().max().copied().unwrap_or(252);
        // We need at least max_period + 1 for returns calculation
        // and some additional history for percentile distribution
        max_period + 1
    }

    /// Calculate Volatility Cone output for all periods and percentiles.
    pub fn calculate_full(&self, close: &[f64]) -> VolatilityConeOutput {
        let n = close.len();
        let returns = Self::log_returns(close);
        let num_periods = self.config.periods.len();
        let num_percentiles = self.config.percentiles.len();

        // Initialize output structures
        let mut current_volatility: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; num_periods];
        let mut percentile_values: Vec<Vec<Vec<f64>>> =
            vec![vec![vec![f64::NAN; n]; num_percentiles]; num_periods];
        let mut percentile_ranks: Vec<Vec<f64>> = vec![vec![f64::NAN; n]; num_periods];

        // Calculate for each period
        for (period_idx, &period) in self.config.periods.iter().enumerate() {
            if period == 0 || n < period + 1 {
                continue;
            }

            // Calculate rolling volatility for this period
            let warmup = period;
            for i in warmup..n {
                // Calculate current volatility
                let ret_end = i - 1; // returns array is 1 shorter than close
                if ret_end < period - 1 {
                    continue;
                }

                let ret_start = ret_end + 1 - period;
                let curr_vol = self.calculate_volatility(&returns[ret_start..=ret_end], period);
                current_volatility[period_idx][i] = curr_vol;

                // Build historical volatility distribution from all available history
                let mut vol_history = Vec::new();
                for j in (period - 1)..=ret_end {
                    let win_start = j + 1 - period;
                    let vol = self.calculate_volatility(&returns[win_start..=j], period);
                    if !vol.is_nan() {
                        vol_history.push(vol);
                    }
                }

                if vol_history.is_empty() {
                    continue;
                }

                // Sort for percentile calculation
                vol_history.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                // Calculate percentiles
                for (pct_idx, &pct) in self.config.percentiles.iter().enumerate() {
                    percentile_values[period_idx][pct_idx][i] =
                        Self::percentile(&vol_history, pct);
                }

                // Calculate percentile rank of current volatility
                if !curr_vol.is_nan() && vol_history.len() > 1 {
                    let min_vol = vol_history[0];
                    let max_vol = vol_history[vol_history.len() - 1];
                    let range = max_vol - min_vol;
                    if range > 0.0 {
                        percentile_ranks[period_idx][i] =
                            ((curr_vol - min_vol) / range).clamp(0.0, 1.0);
                    } else {
                        percentile_ranks[period_idx][i] = 0.5;
                    }
                }
            }
        }

        VolatilityConeOutput {
            current_volatility,
            percentile_values,
            percentile_ranks,
        }
    }

    /// Get the average percentile rank across all periods at the latest bar.
    pub fn average_percentile_rank(&self, close: &[f64]) -> f64 {
        let output = self.calculate_full(close);
        let n = close.len();
        if n == 0 {
            return f64::NAN;
        }

        let last_idx = n - 1;
        let mut sum = 0.0;
        let mut count = 0;

        for period_ranks in &output.percentile_ranks {
            let rank = period_ranks[last_idx];
            if !rank.is_nan() {
                sum += rank;
                count += 1;
            }
        }

        if count == 0 {
            f64::NAN
        } else {
            sum / count as f64
        }
    }

    /// Get current volatility for a specific period.
    pub fn current_volatility_for_period(&self, close: &[f64], period: usize) -> f64 {
        let output = self.calculate_full(close);
        let n = close.len();
        if n == 0 {
            return f64::NAN;
        }

        let last_idx = n - 1;
        if let Some(period_idx) = self.config.periods.iter().position(|&p| p == period) {
            output.current_volatility[period_idx][last_idx]
        } else {
            f64::NAN
        }
    }
}

impl TechnicalIndicator for VolatilityCone {
    fn name(&self) -> &str {
        "VolatilityCone"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_len = self.min_data_length();
        if data.close.len() < min_len {
            return Err(IndicatorError::InsufficientData {
                required: min_len,
                got: data.close.len(),
            });
        }

        let output = self.calculate_full(&data.close);
        let n = data.close.len();

        // Primary: average current volatility across all periods
        let mut avg_current_vol = vec![f64::NAN; n];
        for i in 0..n {
            let mut sum = 0.0;
            let mut count = 0;
            for period_vols in &output.current_volatility {
                let vol = period_vols[i];
                if !vol.is_nan() {
                    sum += vol;
                    count += 1;
                }
            }
            if count > 0 {
                avg_current_vol[i] = sum / count as f64;
            }
        }

        // Secondary: median percentile value (50th percentile if available, or middle percentile)
        let median_pct_idx = self.config.percentiles.len() / 2;
        let mut avg_median_vol = vec![f64::NAN; n];
        for i in 0..n {
            let mut sum = 0.0;
            let mut count = 0;
            for period_pcts in &output.percentile_values {
                if median_pct_idx < period_pcts.len() {
                    let vol = period_pcts[median_pct_idx][i];
                    if !vol.is_nan() {
                        sum += vol;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                avg_median_vol[i] = sum / count as f64;
            }
        }

        // Tertiary: average percentile rank across all periods
        let mut avg_pct_rank = vec![f64::NAN; n];
        for i in 0..n {
            let mut sum = 0.0;
            let mut count = 0;
            for period_ranks in &output.percentile_ranks {
                let rank = period_ranks[i];
                if !rank.is_nan() {
                    sum += rank;
                    count += 1;
                }
            }
            if count > 0 {
                avg_pct_rank[i] = sum / count as f64;
            }
        }

        Ok(IndicatorOutput::triple(
            avg_current_vol,
            avg_median_vol,
            avg_pct_rank,
        ))
    }

    fn min_periods(&self) -> usize {
        self.min_data_length()
    }

    fn output_features(&self) -> usize {
        3 // avg_current_vol, avg_median_vol, avg_percentile_rank
    }
}

impl SignalIndicator for VolatilityCone {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let rank = self.average_percentile_rank(&data.close);

        if rank.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Low volatility often precedes breakouts
        if rank < 0.25 {
            Ok(IndicatorSignal::Bullish)
        }
        // High volatility often precedes mean reversion
        else if rank > 0.75 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let output = self.calculate_full(&data.close);
        let n = data.close.len();

        let signals: Vec<IndicatorSignal> = (0..n)
            .map(|i| {
                // Calculate average percentile rank at this bar
                let mut sum = 0.0;
                let mut count = 0;
                for period_ranks in &output.percentile_ranks {
                    let rank = period_ranks[i];
                    if !rank.is_nan() {
                        sum += rank;
                        count += 1;
                    }
                }

                if count == 0 {
                    return IndicatorSignal::Neutral;
                }

                let avg_rank = sum / count as f64;

                if avg_rank < 0.25 {
                    IndicatorSignal::Bullish
                } else if avg_rank > 0.75 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(len: usize) -> Vec<f64> {
        (0..len)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0 + i as f64 * 0.05)
            .collect()
    }

    #[test]
    fn test_volatility_cone_basic() {
        let config = VolatilityConeConfig::new(vec![10, 20], vec![0.25, 0.5, 0.75]);
        let cone = VolatilityCone::new(config);

        let close = generate_test_data(100);
        let output = cone.calculate_full(&close);

        // Check dimensions
        assert_eq!(output.current_volatility.len(), 2); // 2 periods
        assert_eq!(output.percentile_values.len(), 2); // 2 periods
        assert_eq!(output.percentile_values[0].len(), 3); // 3 percentiles
        assert_eq!(output.percentile_ranks.len(), 2); // 2 periods

        // After warmup, should have valid values
        for period_vols in &output.current_volatility {
            let warmup = 20; // max period
            for i in warmup..100 {
                assert!(
                    !period_vols[i].is_nan(),
                    "Expected valid volatility at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_volatility_cone_percentile_ordering() {
        let config = VolatilityConeConfig::new(vec![15], vec![0.1, 0.25, 0.5, 0.75, 0.9]);
        let cone = VolatilityCone::new(config);

        let close = generate_test_data(100);
        let output = cone.calculate_full(&close);

        // Percentiles should be in order for each bar
        let warmup = 15;
        for i in warmup..100 {
            let p10 = output.percentile_values[0][0][i];
            let p25 = output.percentile_values[0][1][i];
            let p50 = output.percentile_values[0][2][i];
            let p75 = output.percentile_values[0][3][i];
            let p90 = output.percentile_values[0][4][i];

            if !p10.is_nan() && !p90.is_nan() {
                assert!(p10 <= p25 + 1e-10, "p10 should be <= p25 at index {}", i);
                assert!(p25 <= p50 + 1e-10, "p25 should be <= p50 at index {}", i);
                assert!(p50 <= p75 + 1e-10, "p50 should be <= p75 at index {}", i);
                assert!(p75 <= p90 + 1e-10, "p75 should be <= p90 at index {}", i);
            }
        }
    }

    #[test]
    fn test_volatility_cone_default_config() {
        let config = VolatilityConeConfig::default();
        assert_eq!(config.periods, vec![20, 40, 60, 120, 252]);
        assert_eq!(config.percentiles, vec![0.1, 0.25, 0.5, 0.75, 0.9]);
    }

    #[test]
    fn test_volatility_cone_percentile_rank_bounds() {
        let config = VolatilityConeConfig::new(vec![10, 20], vec![0.25, 0.5, 0.75]);
        let cone = VolatilityCone::new(config);

        let close = generate_test_data(80);
        let output = cone.calculate_full(&close);

        // Percentile ranks should be between 0 and 1
        for period_ranks in &output.percentile_ranks {
            for &rank in period_ranks {
                if !rank.is_nan() {
                    assert!(
                        rank >= 0.0 && rank <= 1.0,
                        "Percentile rank {} should be in [0, 1]",
                        rank
                    );
                }
            }
        }
    }

    #[test]
    fn test_volatility_cone_average_percentile_rank() {
        let config = VolatilityConeConfig::new(vec![10, 20], vec![0.25, 0.5, 0.75]);
        let cone = VolatilityCone::new(config);

        let close = generate_test_data(80);
        let rank = cone.average_percentile_rank(&close);

        if !rank.is_nan() {
            assert!(
                rank >= 0.0 && rank <= 1.0,
                "Average percentile rank should be in [0, 1]"
            );
        }
    }

    #[test]
    fn test_volatility_cone_technical_indicator() {
        let config = VolatilityConeConfig::new(vec![10, 20], vec![0.25, 0.5, 0.75]);
        let cone = VolatilityCone::new(config);

        assert_eq!(cone.name(), "VolatilityCone");
        assert_eq!(cone.min_periods(), 21); // max period + 1
        assert_eq!(cone.output_features(), 3);
    }

    #[test]
    fn test_volatility_cone_insufficient_data() {
        let config = VolatilityConeConfig::new(vec![10, 20], vec![0.25, 0.5, 0.75]);
        let cone = VolatilityCone::new(config);

        let series = OHLCVSeries {
            open: vec![100.0; 15],
            high: vec![102.0; 15],
            low: vec![98.0; 15],
            close: vec![100.0; 15],
            volume: vec![1000.0; 15],
        };

        let result = cone.compute(&series);
        assert!(result.is_err());
    }

    #[test]
    fn test_volatility_cone_compute_output() {
        let config = VolatilityConeConfig::new(vec![10, 20], vec![0.25, 0.5, 0.75]);
        let cone = VolatilityCone::new(config);

        let close = generate_test_data(100);
        let series = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|&c| c + 1.0).collect(),
            low: close.iter().map(|&c| c - 1.0).collect(),
            close: close.clone(),
            volume: vec![1000.0; close.len()],
        };

        let result = cone.compute(&series);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_volatility_cone_signal() {
        let config = VolatilityConeConfig::new(vec![10, 20], vec![0.25, 0.5, 0.75]);
        let cone = VolatilityCone::new(config);

        let close = generate_test_data(100);
        let series = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|&c| c + 1.0).collect(),
            low: close.iter().map(|&c| c - 1.0).collect(),
            close,
            volume: vec![1000.0; 100],
        };

        let signal = cone.signal(&series);
        assert!(signal.is_ok());

        let signals = cone.signals(&series);
        assert!(signals.is_ok());
        assert_eq!(signals.unwrap().len(), 100);
    }

    #[test]
    fn test_percentile_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((VolatilityCone::percentile(&data, 0.0) - 1.0).abs() < 1e-10);
        assert!((VolatilityCone::percentile(&data, 0.5) - 3.0).abs() < 1e-10);
        assert!((VolatilityCone::percentile(&data, 1.0) - 5.0).abs() < 1e-10);
        assert!((VolatilityCone::percentile(&data, 0.25) - 2.0).abs() < 1e-10);
        assert!((VolatilityCone::percentile(&data, 0.75) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_volatility_cone_multiple_periods() {
        let config =
            VolatilityConeConfig::new(vec![5, 10, 15, 20], vec![0.1, 0.25, 0.5, 0.75, 0.9]);
        let cone = VolatilityCone::new(config);

        let close = generate_test_data(100);
        let output = cone.calculate_full(&close);

        // Verify we have data for all 4 periods
        assert_eq!(output.current_volatility.len(), 4);

        // Shorter periods should have valid values earlier
        // Period 5: warmup ends at index 5 (needs 5 returns which needs 6 prices)
        // So index 4 and below should be NaN, index 5+ should be valid
        assert!(output.current_volatility[0][4].is_nan()); // warmup (before enough data)
        assert!(!output.current_volatility[0][5].is_nan()); // valid (first valid point)

        // Period 20: warmup ends at index 20 (needs 20 returns which needs 21 prices)
        // So index 19 and below should be NaN, index 20+ should be valid
        assert!(output.current_volatility[3][19].is_nan()); // warmup
        assert!(!output.current_volatility[3][20].is_nan()); // valid
    }

    #[test]
    fn test_volatility_cone_current_volatility_for_period() {
        let config = VolatilityConeConfig::new(vec![10, 20], vec![0.25, 0.5, 0.75]);
        let cone = VolatilityCone::new(config);

        let close = generate_test_data(100);

        let vol_10 = cone.current_volatility_for_period(&close, 10);
        let vol_20 = cone.current_volatility_for_period(&close, 20);
        let vol_30 = cone.current_volatility_for_period(&close, 30); // Not in config

        assert!(!vol_10.is_nan());
        assert!(!vol_20.is_nan());
        assert!(vol_30.is_nan()); // Period not configured
    }
}
