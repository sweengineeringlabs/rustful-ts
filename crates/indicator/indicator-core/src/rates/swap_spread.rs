//! Swap Spread indicator (IND-301).
//!
//! Measures the difference between interest rate swap rates and Treasury yields
//! of equivalent maturity. Reflects credit risk and liquidity conditions.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Swap Spread configuration.
#[derive(Debug, Clone)]
pub struct SwapSpreadConfig {
    /// Smoothing period for the spread (default: 1, no smoothing).
    pub period: usize,
    /// Normal range upper bound in basis points (default: 30).
    pub normal_upper: f64,
    /// Normal range lower bound in basis points (default: -10).
    pub normal_lower: f64,
}

impl SwapSpreadConfig {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            normal_upper: 30.0,
            normal_lower: -10.0,
        }
    }

    pub fn with_bounds(period: usize, lower: f64, upper: f64) -> Self {
        Self {
            period,
            normal_upper: upper,
            normal_lower: lower,
        }
    }
}

impl Default for SwapSpreadConfig {
    fn default() -> Self {
        Self {
            period: 1,
            normal_upper: 30.0,
            normal_lower: -10.0,
        }
    }
}

/// Swap Spread (IND-301).
///
/// Calculated as: Swap Rate - Treasury Yield (of same maturity)
///
/// The swap spread reflects:
/// - Credit risk premium of banks vs government
/// - Supply/demand dynamics in swap and Treasury markets
/// - Funding conditions and liquidity
///
/// Interpretation:
/// - Positive spread: Normal (banks riskier than government)
/// - Widening: Increased credit concerns or reduced liquidity
/// - Narrowing/negative: Flight to quality, special factors
/// - Negative spreads became common post-2008 due to regulatory changes
#[derive(Debug, Clone)]
pub struct SwapSpread {
    config: SwapSpreadConfig,
}

impl SwapSpread {
    pub fn new(period: usize) -> Self {
        Self {
            config: SwapSpreadConfig::new(period),
        }
    }

    pub fn from_config(config: SwapSpreadConfig) -> Self {
        Self { config }
    }

    /// Calculate swap spread from swap rates and Treasury yields.
    ///
    /// # Arguments
    /// * `swap_rates` - Interest rate swap rates (as percentages)
    /// * `treasury_yields` - Treasury yields (as percentages)
    ///
    /// # Returns
    /// Swap spread in basis points (Swap - Treasury) * 100
    pub fn calculate(&self, swap_rates: &[f64], treasury_yields: &[f64]) -> Vec<f64> {
        let n = swap_rates.len().min(treasury_yields.len());
        if n == 0 {
            return vec![];
        }

        // Calculate raw spread in basis points
        let mut spreads: Vec<f64> = (0..n)
            .map(|i| (swap_rates[i] - treasury_yields[i]) * 100.0)
            .collect();

        // Apply smoothing if period > 1
        if self.config.period > 1 && n >= self.config.period {
            spreads = self.smooth(&spreads);
        }

        spreads
    }

    /// Calculate spread from dual OHLCV series (swap as primary, Treasury as benchmark).
    pub fn calculate_from_series(&self, swap: &OHLCVSeries, treasury: &OHLCVSeries) -> Vec<f64> {
        self.calculate(&swap.close, &treasury.close)
    }

    /// Apply simple moving average smoothing.
    fn smooth(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let period = self.config.period;

        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];

        for i in (period - 1)..n {
            let sum: f64 = data[(i + 1 - period)..=i].iter().sum();
            result.push(sum / period as f64);
        }

        result
    }

    /// Classify current swap spread level.
    pub fn classify(&self, spread: f64) -> SwapSpreadCondition {
        if spread.is_nan() {
            SwapSpreadCondition::Unknown
        } else if spread < self.config.normal_lower {
            SwapSpreadCondition::Compressed
        } else if spread <= self.config.normal_upper {
            SwapSpreadCondition::Normal
        } else {
            SwapSpreadCondition::Wide
        }
    }

    /// Calculate term structure of swap spreads.
    /// Takes multiple maturity swap spreads and returns slope.
    pub fn term_structure_slope(&self, short_spread: f64, long_spread: f64) -> f64 {
        if short_spread.is_nan() || long_spread.is_nan() {
            f64::NAN
        } else {
            long_spread - short_spread
        }
    }

    /// Calculate percentile of current spread vs historical.
    pub fn percentile(&self, spreads: &[f64], lookback: usize) -> Vec<f64> {
        let n = spreads.len();
        if n < lookback {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; lookback - 1];

        for i in (lookback - 1)..n {
            let window = &spreads[(i + 1 - lookback)..=i];
            let current = spreads[i];

            if current.is_nan() {
                result.push(f64::NAN);
                continue;
            }

            let below_count = window.iter()
                .filter(|&&x| !x.is_nan() && x < current)
                .count();
            let valid_count = window.iter()
                .filter(|&&x| !x.is_nan())
                .count();

            if valid_count > 0 {
                result.push(100.0 * below_count as f64 / valid_count as f64);
            } else {
                result.push(f64::NAN);
            }
        }

        result
    }
}

/// Swap spread condition classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwapSpreadCondition {
    /// Spread is compressed (potentially negative)
    Compressed,
    /// Normal range
    Normal,
    /// Spread is wide (elevated)
    Wide,
    /// Cannot determine
    Unknown,
}

impl TechnicalIndicator for SwapSpread {
    fn name(&self) -> &str {
        "SwapSpread"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        // Convention: close contains swap rates, open contains Treasury yields
        if data.close.len() < self.config.period {
            return Err(IndicatorError::InsufficientData {
                required: self.config.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close, &data.open);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.config.period
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_spread_basic() {
        let indicator = SwapSpread::new(1);

        // Swap at 2.5%, Treasury at 2.3% = 20bp spread
        let swap = vec![2.5, 2.6, 2.7, 2.5, 2.4];
        let treasury = vec![2.3, 2.4, 2.5, 2.35, 2.25];

        let result = indicator.calculate(&swap, &treasury);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 20.0).abs() < 0.001); // 2.5 - 2.3 = 0.2% = 20bp
    }

    #[test]
    fn test_swap_spread_negative() {
        let indicator = SwapSpread::new(1);

        // Negative spread scenario (post-2008 phenomenon)
        let swap = vec![2.3, 2.4, 2.5];
        let treasury = vec![2.5, 2.6, 2.7];

        let result = indicator.calculate(&swap, &treasury);

        assert!(result[0] < 0.0); // -20bp
        assert!(result[1] < 0.0); // -20bp
    }

    #[test]
    fn test_swap_spread_smoothed() {
        let indicator = SwapSpread::new(3);

        let swap = vec![2.5, 2.6, 2.7, 2.5, 2.4];
        let treasury = vec![2.3, 2.4, 2.5, 2.35, 2.25];

        let result = indicator.calculate(&swap, &treasury);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Third value should be average of first 3 spreads
        let expected = (20.0 + 20.0 + 20.0) / 3.0;
        assert!((result[2] - expected).abs() < 0.001);
    }

    #[test]
    fn test_swap_spread_classification() {
        let indicator = SwapSpread::from_config(
            SwapSpreadConfig::with_bounds(1, -10.0, 30.0)
        );

        assert_eq!(indicator.classify(20.0), SwapSpreadCondition::Normal);
        assert_eq!(indicator.classify(50.0), SwapSpreadCondition::Wide);
        assert_eq!(indicator.classify(-20.0), SwapSpreadCondition::Compressed);
    }

    #[test]
    fn test_swap_spread_term_structure() {
        let indicator = SwapSpread::new(1);

        // 2-year spread: 15bp, 10-year spread: 25bp
        let slope = indicator.term_structure_slope(15.0, 25.0);

        assert!((slope - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_swap_spread_percentile() {
        let indicator = SwapSpread::new(1);

        // Current spread (30) is at 80th percentile
        let spreads = vec![10.0, 15.0, 20.0, 25.0, 30.0];
        let percentiles = indicator.percentile(&spreads, 5);

        assert_eq!(percentiles.len(), 5);
        assert!((percentiles[4] - 80.0).abs() < 0.001); // 4 out of 5 are below
    }
}
