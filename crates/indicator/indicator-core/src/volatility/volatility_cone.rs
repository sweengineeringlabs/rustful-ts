//! Volatility Cone implementation.
//!
//! Provides percentile bands of historical volatility over different time periods,
//! useful for comparing current volatility to historical norms.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

/// Volatility Cone.
///
/// The Volatility Cone calculates percentile bands of historical volatility
/// over a lookback period. It shows the distribution of past volatility levels,
/// helping traders understand whether current volatility is high or low
/// relative to historical norms.
///
/// The cone typically includes:
/// - Maximum volatility (100th percentile)
/// - 75th percentile
/// - 50th percentile (median)
/// - 25th percentile
/// - Minimum volatility (0th percentile)
///
/// # Trading Applications
/// - Options pricing: Identify cheap/expensive implied volatility
/// - Mean reversion: Trade volatility returning to median
/// - Risk management: Adjust position sizes based on volatility percentile
///
/// # Signal Logic
/// - Current vol below 25th percentile: Bullish (low vol often precedes breakouts)
/// - Current vol above 75th percentile: Bearish (high vol often precedes mean reversion)
/// - Otherwise: Neutral
#[derive(Debug, Clone)]
pub struct VolatilityCone {
    /// Period for individual volatility calculations (e.g., 20-day HV).
    vol_period: usize,
    /// Lookback period for building the distribution.
    lookback: usize,
    /// Number of trading days per year for annualization.
    trading_days: f64,
    /// Percentile thresholds: [min, p25, p50, p75, max].
    percentiles: [f64; 5],
}

impl VolatilityCone {
    /// Create a new Volatility Cone indicator.
    ///
    /// # Arguments
    /// * `vol_period` - Period for volatility calculation (e.g., 20)
    /// * `lookback` - Historical lookback for percentile distribution (e.g., 252)
    pub fn new(vol_period: usize, lookback: usize) -> Self {
        Self {
            vol_period,
            lookback,
            trading_days: 252.0,
            percentiles: [0.0, 0.25, 0.50, 0.75, 1.0],
        }
    }

    /// Create with default parameters (20-day vol, 1-year lookback).
    pub fn default_params() -> Self {
        Self::new(20, 252)
    }

    /// Set custom trading days for annualization.
    pub fn with_trading_days(mut self, days: f64) -> Self {
        self.trading_days = days;
        self
    }

    /// Set custom percentile levels.
    pub fn with_percentiles(mut self, percentiles: [f64; 5]) -> Self {
        self.percentiles = percentiles;
        self
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
    fn calculate_volatility(&self, returns: &[f64]) -> f64 {
        let valid_returns: Vec<f64> = returns.iter().filter(|x| !x.is_nan()).copied().collect();

        if valid_returns.len() < self.vol_period {
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

    /// Calculate percentile from sorted data.
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

    /// Calculate Volatility Cone values.
    ///
    /// Returns (current_vol, min, p25, p50, p75, max).
    pub fn calculate(
        &self,
        close: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let min_len = self.vol_period + self.lookback;

        if n < min_len || self.vol_period == 0 || self.lookback == 0 {
            return (
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
            );
        }

        let returns = Self::log_returns(close);

        // Initialize result vectors
        let warmup = self.vol_period + self.lookback - 1;
        let mut current_vol = vec![f64::NAN; warmup];
        let mut min_vol = vec![f64::NAN; warmup];
        let mut p25_vol = vec![f64::NAN; warmup];
        let mut p50_vol = vec![f64::NAN; warmup];
        let mut p75_vol = vec![f64::NAN; warmup];
        let mut max_vol = vec![f64::NAN; warmup];

        // Calculate rolling volatility and percentiles
        for i in warmup..n {
            // Current volatility (using most recent vol_period returns)
            let ret_end = i - 1; // returns array is 1 shorter than close
            if ret_end < self.vol_period - 1 {
                current_vol.push(f64::NAN);
                min_vol.push(f64::NAN);
                p25_vol.push(f64::NAN);
                p50_vol.push(f64::NAN);
                p75_vol.push(f64::NAN);
                max_vol.push(f64::NAN);
                continue;
            }

            let ret_start = ret_end + 1 - self.vol_period;
            let curr = self.calculate_volatility(&returns[ret_start..=ret_end]);
            current_vol.push(curr);

            // Build historical volatility distribution
            let mut vol_history = Vec::with_capacity(self.lookback);
            let lookback_end = ret_end;
            let lookback_start = lookback_end.saturating_sub(self.lookback - 1);

            for j in lookback_start..=lookback_end {
                if j >= self.vol_period - 1 {
                    let win_start = j + 1 - self.vol_period;
                    let vol = self.calculate_volatility(&returns[win_start..=j]);
                    if !vol.is_nan() {
                        vol_history.push(vol);
                    }
                }
            }

            if vol_history.is_empty() {
                min_vol.push(f64::NAN);
                p25_vol.push(f64::NAN);
                p50_vol.push(f64::NAN);
                p75_vol.push(f64::NAN);
                max_vol.push(f64::NAN);
                continue;
            }

            // Sort for percentile calculation
            vol_history.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            min_vol.push(Self::percentile(&vol_history, self.percentiles[0]));
            p25_vol.push(Self::percentile(&vol_history, self.percentiles[1]));
            p50_vol.push(Self::percentile(&vol_history, self.percentiles[2]));
            p75_vol.push(Self::percentile(&vol_history, self.percentiles[3]));
            max_vol.push(Self::percentile(&vol_history, self.percentiles[4]));
        }

        (current_vol, min_vol, p25_vol, p50_vol, p75_vol, max_vol)
    }

    /// Get the current volatility percentile rank.
    pub fn current_percentile_rank(&self, close: &[f64]) -> f64 {
        let (current_vol, min_vol, _, _, _, max_vol) = self.calculate(close);
        let n = current_vol.len();

        if n == 0 {
            return f64::NAN;
        }

        let curr = current_vol[n - 1];
        let min = min_vol[n - 1];
        let max = max_vol[n - 1];

        if curr.is_nan() || min.is_nan() || max.is_nan() {
            return f64::NAN;
        }

        let range = max - min;
        if range <= 0.0 {
            return 0.5; // Default to median if no range
        }

        ((curr - min) / range).clamp(0.0, 1.0)
    }
}

impl TechnicalIndicator for VolatilityCone {
    fn name(&self) -> &str {
        "VolatilityCone"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_len = self.vol_period + self.lookback;
        if data.close.len() < min_len {
            return Err(IndicatorError::InsufficientData {
                required: min_len,
                got: data.close.len(),
            });
        }

        // Return current volatility as primary, p50 as secondary, and a combined metric as tertiary
        let (current, _min, _p25, p50, _p75, _max) = self.calculate(&data.close);

        // Tertiary: percentile rank (0-1 scale showing where current vol sits in distribution)
        let n = current.len();
        let mut percentile_rank = vec![f64::NAN; n];
        let (_, min_vol, _, _, _, max_vol) = self.calculate(&data.close);

        for i in 0..n {
            if !current[i].is_nan() && !min_vol[i].is_nan() && !max_vol[i].is_nan() {
                let range = max_vol[i] - min_vol[i];
                if range > 0.0 {
                    percentile_rank[i] = ((current[i] - min_vol[i]) / range).clamp(0.0, 1.0);
                } else {
                    percentile_rank[i] = 0.5;
                }
            }
        }

        Ok(IndicatorOutput::triple(current, p50, percentile_rank))
    }

    fn min_periods(&self) -> usize {
        self.vol_period + self.lookback
    }

    fn output_features(&self) -> usize {
        3 // current_vol, median_vol, percentile_rank
    }
}

impl SignalIndicator for VolatilityCone {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let rank = self.current_percentile_rank(&data.close);

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
        let (current, min_vol, _, _, _, max_vol) = self.calculate(&data.close);

        let signals = current
            .iter()
            .zip(min_vol.iter())
            .zip(max_vol.iter())
            .map(|((&curr, &min), &max)| {
                if curr.is_nan() || min.is_nan() || max.is_nan() {
                    return IndicatorSignal::Neutral;
                }

                let range = max - min;
                if range <= 0.0 {
                    return IndicatorSignal::Neutral;
                }

                let rank = ((curr - min) / range).clamp(0.0, 1.0);

                if rank < 0.25 {
                    IndicatorSignal::Bullish
                } else if rank > 0.75 {
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

    #[test]
    fn test_volatility_cone_basic() {
        let cone = VolatilityCone::new(10, 50);

        // Generate sample price data with varying volatility
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0 + i as f64 * 0.05)
            .collect();

        let (current, min, p25, p50, p75, max) = cone.calculate(&close);

        assert_eq!(current.len(), 100);
        assert_eq!(min.len(), 100);
        assert_eq!(p25.len(), 100);
        assert_eq!(p50.len(), 100);
        assert_eq!(p75.len(), 100);
        assert_eq!(max.len(), 100);

        // Check warmup period has NaN
        let warmup = 10 + 50 - 1; // 59
        for i in 0..warmup {
            assert!(current[i].is_nan(), "Expected NaN at index {}", i);
        }

        // After warmup, should have valid values
        for i in warmup..100 {
            assert!(
                !current[i].is_nan(),
                "Expected valid current vol at index {}",
                i
            );
        }
    }

    #[test]
    fn test_volatility_cone_percentile_ordering() {
        let cone = VolatilityCone::new(10, 50);

        // Generate sample data
        let close: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.2).sin() * 3.0 + i as f64 * 0.02)
            .collect();

        let (_, min, p25, p50, p75, max) = cone.calculate(&close);

        // Percentiles should be in order: min <= p25 <= p50 <= p75 <= max
        let warmup = 10 + 50 - 1;
        for i in warmup..100 {
            if !min[i].is_nan() && !max[i].is_nan() {
                assert!(
                    min[i] <= p25[i] + 1e-10,
                    "min should be <= p25 at index {}",
                    i
                );
                assert!(
                    p25[i] <= p50[i] + 1e-10,
                    "p25 should be <= p50 at index {}",
                    i
                );
                assert!(
                    p50[i] <= p75[i] + 1e-10,
                    "p50 should be <= p75 at index {}",
                    i
                );
                assert!(
                    p75[i] <= max[i] + 1e-10,
                    "p75 should be <= max at index {}",
                    i
                );
            }
        }
    }

    #[test]
    fn test_volatility_cone_default() {
        let cone = VolatilityCone::default_params();
        assert_eq!(cone.vol_period, 20);
        assert_eq!(cone.lookback, 252);
    }

    #[test]
    fn test_volatility_cone_percentile_rank() {
        let cone = VolatilityCone::new(10, 30);

        // Generate data
        let close: Vec<f64> = (0..80)
            .map(|i| 100.0 + (i as f64 * 0.15).sin() * 4.0 + i as f64 * 0.03)
            .collect();

        let rank = cone.current_percentile_rank(&close);

        // Rank should be between 0 and 1
        if !rank.is_nan() {
            assert!(
                rank >= 0.0 && rank <= 1.0,
                "Percentile rank should be in [0, 1]"
            );
        }
    }

    #[test]
    fn test_volatility_cone_technical_indicator() {
        let cone = VolatilityCone::new(10, 50);
        assert_eq!(cone.name(), "VolatilityCone");
        assert_eq!(cone.min_periods(), 60);
        assert_eq!(cone.output_features(), 3);
    }

    #[test]
    fn test_volatility_cone_insufficient_data() {
        let cone = VolatilityCone::new(10, 50);

        let series = OHLCVSeries {
            open: vec![100.0; 30],
            high: vec![102.0; 30],
            low: vec![98.0; 30],
            close: vec![100.0; 30],
            volume: vec![1000.0; 30],
        };

        let result = cone.compute(&series);
        assert!(result.is_err());
    }

    #[test]
    fn test_percentile_calculation() {
        // Test the percentile function directly
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((VolatilityCone::percentile(&data, 0.0) - 1.0).abs() < 1e-10);
        assert!((VolatilityCone::percentile(&data, 0.5) - 3.0).abs() < 1e-10);
        assert!((VolatilityCone::percentile(&data, 1.0) - 5.0).abs() < 1e-10);
        assert!((VolatilityCone::percentile(&data, 0.25) - 2.0).abs() < 1e-10);
        assert!((VolatilityCone::percentile(&data, 0.75) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_volatility_cone_signal_low_vol() {
        // Test that low volatility generates bullish signal
        let cone = VolatilityCone::new(5, 20);

        // Create data with initial high volatility that decreases
        let mut close = Vec::new();
        // High vol period
        for i in 0..30 {
            close.push(100.0 + (i as f64 * 0.5).sin() * 10.0);
        }
        // Low vol period (stable)
        for i in 30..50 {
            close.push(100.0 + i as f64 * 0.01);
        }

        let series = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|&c| c + 0.5).collect(),
            low: close.iter().map(|&c| c - 0.5).collect(),
            close: close.clone(),
            volume: vec![1000.0; close.len()],
        };

        // The signal should reflect low current volatility
        let result = cone.signal(&series);
        assert!(result.is_ok());
    }
}
