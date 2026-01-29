//! Crack Spread Indicator (IND-417)
//!
//! Measures the refining margin by calculating the spread between crude oil
//! and refined products. A proxy for refinery profitability and energy sector health.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Output for CrackSpread indicator.
#[derive(Debug, Clone, Copy)]
pub struct CrackSpreadOutput {
    /// Raw crack spread value (percentage of input price).
    pub spread: f64,
    /// Z-score of spread relative to historical average.
    pub zscore: f64,
    /// Moving average of spread.
    pub ma: f64,
    /// Spread momentum (rate of change).
    pub momentum: f64,
}

/// Crack Spread Indicator (IND-417)
///
/// Estimates the refining margin using price range and volatility as proxies
/// for the crude-to-products spread. In practice, crack spread is calculated
/// as: Refined Products Revenue - Crude Oil Cost.
///
/// The 3-2-1 crack spread is common: 3 barrels crude = 2 barrels gasoline + 1 barrel diesel.
///
/// # Algorithm
/// 1. Use price range (high-low) as a proxy for spread dynamics
/// 2. Normalize by close price for percentage-based spread
/// 3. Calculate z-score relative to historical average
/// 4. Track momentum of spread changes
///
/// # Interpretation
/// - High spread: Strong refining margins, bullish for refiners
/// - Low spread: Weak margins, bearish for refiners
/// - Rising spread: Improving conditions
/// - Falling spread: Deteriorating conditions
///
/// # Example
/// ```ignore
/// let cs = CrackSpread::new(20, 10)?;
/// let output = cs.compute(&data)?;
/// ```
#[derive(Debug, Clone)]
pub struct CrackSpread {
    /// Period for moving average and z-score calculations.
    period: usize,
    /// Period for momentum calculation.
    momentum_period: usize,
}

impl CrackSpread {
    /// Create a new CrackSpread indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for statistics
    /// * `momentum_period` - Period for momentum calculation
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid.
    pub fn new(period: usize, momentum_period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if momentum_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            period,
            momentum_period,
        })
    }

    /// Create with default parameters (20, 10).
    pub fn default_params() -> Result<Self> {
        Self::new(20, 10)
    }

    /// Calculate crack spread values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Vec<Option<CrackSpreadOutput>> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![None; n];

        if n < self.period + self.momentum_period {
            return result;
        }

        // Calculate raw spreads (range as percentage of close)
        let mut spreads = vec![f64::NAN; n];
        for i in 0..n {
            if close[i] > 0.0 {
                spreads[i] = (high[i] - low[i]) / close[i] * 100.0;
            }
        }

        for i in (self.period - 1)..n {
            let start = i - self.period + 1;

            // Calculate MA and std of spreads
            let window: Vec<f64> = spreads[start..=i]
                .iter()
                .copied()
                .filter(|s| !s.is_nan())
                .collect();

            if window.is_empty() {
                continue;
            }

            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance = window.iter()
                .map(|s| (s - mean).powi(2))
                .sum::<f64>() / window.len() as f64;
            let std = variance.sqrt();

            let spread = spreads[i];
            if spread.is_nan() {
                continue;
            }

            // Z-score
            let zscore = if std > 0.0 {
                (spread - mean) / std
            } else {
                0.0
            };

            // Momentum
            let momentum = if i >= self.momentum_period && !spreads[i - self.momentum_period].is_nan() {
                spread - spreads[i - self.momentum_period]
            } else {
                f64::NAN
            };

            result[i] = Some(CrackSpreadOutput {
                spread,
                zscore,
                ma: mean,
                momentum,
            });
        }

        result
    }

    /// Calculate with two separate series (crude and product).
    pub fn calculate_from_series(
        &self,
        crude: &[f64],
        product: &[f64],
    ) -> Vec<Option<CrackSpreadOutput>> {
        let n = crude.len().min(product.len());
        let mut result = vec![None; n];

        if n < self.period + self.momentum_period {
            return result;
        }

        // Calculate raw spreads
        let mut spreads = vec![f64::NAN; n];
        for i in 0..n {
            if crude[i] > 0.0 {
                spreads[i] = (product[i] - crude[i]) / crude[i] * 100.0;
            }
        }

        for i in (self.period - 1)..n {
            let start = i - self.period + 1;

            // Calculate MA and std of spreads
            let window: Vec<f64> = spreads[start..=i]
                .iter()
                .copied()
                .filter(|s| !s.is_nan())
                .collect();

            if window.is_empty() {
                continue;
            }

            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance = window.iter()
                .map(|s| (s - mean).powi(2))
                .sum::<f64>() / window.len() as f64;
            let std = variance.sqrt();

            let spread = spreads[i];
            if spread.is_nan() {
                continue;
            }

            // Z-score
            let zscore = if std > 0.0 {
                (spread - mean) / std
            } else {
                0.0
            };

            // Momentum
            let momentum = if i >= self.momentum_period && !spreads[i - self.momentum_period].is_nan() {
                spread - spreads[i - self.momentum_period]
            } else {
                f64::NAN
            };

            result[i] = Some(CrackSpreadOutput {
                spread,
                zscore,
                ma: mean,
                momentum,
            });
        }

        result
    }

    /// Get the spread series.
    pub fn spread_series(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        self.calculate(high, low, close)
            .iter()
            .map(|o| o.map(|v| v.spread).unwrap_or(f64::NAN))
            .collect()
    }

    /// Get the z-score series.
    pub fn zscore_series(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        self.calculate(high, low, close)
            .iter()
            .map(|o| o.map(|v| v.zscore).unwrap_or(f64::NAN))
            .collect()
    }
}

impl Default for CrackSpread {
    fn default() -> Self {
        Self::default_params().unwrap()
    }
}

impl TechnicalIndicator for CrackSpread {
    fn name(&self) -> &str {
        "CrackSpread"
    }

    fn min_periods(&self) -> usize {
        self.period + self.momentum_period
    }

    fn output_features(&self) -> usize {
        2 // spread, zscore
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }

        let outputs = self.calculate(&data.high, &data.low, &data.close);

        let primary: Vec<f64> = outputs
            .iter()
            .map(|o| o.map(|v| v.spread).unwrap_or(f64::NAN))
            .collect();

        let secondary: Vec<f64> = outputs
            .iter()
            .map(|o| o.map(|v| v.zscore).unwrap_or(f64::NAN))
            .collect();

        Ok(IndicatorOutput::dual(primary, secondary))
    }
}

impl SignalIndicator for CrackSpread {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let outputs = self.calculate(&data.high, &data.low, &data.close);

        match outputs.last().and_then(|o| *o) {
            Some(out) => {
                // High z-score with positive momentum = bullish
                if out.zscore > 1.0 && out.momentum > 0.0 {
                    Ok(IndicatorSignal::Bullish)
                // Low z-score with negative momentum = bearish
                } else if out.zscore < -1.0 && out.momentum < 0.0 {
                    Ok(IndicatorSignal::Bearish)
                } else {
                    Ok(IndicatorSignal::Neutral)
                }
            }
            None => Ok(IndicatorSignal::Neutral),
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let outputs = self.calculate(&data.high, &data.low, &data.close);

        let signals = outputs
            .iter()
            .map(|o| match o {
                Some(out) => {
                    if out.zscore > 1.0 && out.momentum > 0.0 {
                        IndicatorSignal::Bullish
                    } else if out.zscore < -1.0 && out.momentum < 0.0 {
                        IndicatorSignal::Bearish
                    } else {
                        IndicatorSignal::Neutral
                    }
                }
                None => IndicatorSignal::Neutral,
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_wide_spread_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Wide high-low range (high spread)
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.1).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 5.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 5.0).collect();
        (high, low, close)
    }

    fn make_narrow_spread_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Narrow high-low range (low spread)
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.1).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.5).collect();
        (high, low, close)
    }

    fn make_expanding_spread_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Expanding spread over time
        let close: Vec<f64> = vec![100.0; n];
        let high: Vec<f64> = (0..n).map(|i| 100.0 + 1.0 + i as f64 * 0.2).collect();
        let low: Vec<f64> = (0..n).map(|i| 100.0 - 1.0 - i as f64 * 0.2).collect();
        (high, low, close)
    }

    #[test]
    fn test_new_valid_params() {
        let cs = CrackSpread::new(20, 10);
        assert!(cs.is_ok());
    }

    #[test]
    fn test_new_invalid_period() {
        let cs = CrackSpread::new(1, 10);
        assert!(cs.is_err());
    }

    #[test]
    fn test_new_invalid_momentum_period() {
        let cs = CrackSpread::new(20, 0);
        assert!(cs.is_err());
    }

    #[test]
    fn test_wide_spread_detection() {
        let (high, low, close) = make_wide_spread_data(50);
        let cs = CrackSpread::new(10, 5).unwrap();
        let outputs = cs.calculate(&high, &low, &close);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Wide spread should give higher values
        for out in &valid_outputs {
            assert!(out.spread > 5.0); // 10% range on 100 price
        }
    }

    #[test]
    fn test_narrow_spread_detection() {
        let (high, low, close) = make_narrow_spread_data(50);
        let cs = CrackSpread::new(10, 5).unwrap();
        let outputs = cs.calculate(&high, &low, &close);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Narrow spread should give lower values
        for out in &valid_outputs {
            assert!(out.spread < 2.0); // 1% range on 100 price
        }
    }

    #[test]
    fn test_expanding_spread_momentum() {
        let (high, low, close) = make_expanding_spread_data(50);
        let cs = CrackSpread::new(10, 5).unwrap();
        let outputs = cs.calculate(&high, &low, &close);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();

        // Expanding spread should have positive momentum
        let positive_momentum = valid_outputs
            .iter()
            .filter(|o| !o.momentum.is_nan() && o.momentum > 0.0)
            .count();
        assert!(positive_momentum > valid_outputs.len() / 2);
    }

    #[test]
    fn test_zscore_bounds() {
        let (high, low, close) = make_wide_spread_data(50);
        let cs = CrackSpread::default();
        let outputs = cs.calculate(&high, &low, &close);

        for out in outputs.iter().filter_map(|o| *o) {
            // Z-scores should be reasonable
            assert!(out.zscore.abs() < 5.0);
        }
    }

    #[test]
    fn test_calculate_from_series() {
        let crude: Vec<f64> = (0..50).map(|i| 80.0 + i as f64 * 0.1).collect();
        let product: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.15).collect();

        let cs = CrackSpread::new(10, 5).unwrap();
        let outputs = cs.calculate_from_series(&crude, &product);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Product > crude, so spread should be positive
        for out in valid_outputs {
            assert!(out.spread > 0.0);
        }
    }

    #[test]
    fn test_technical_indicator_impl() {
        let (high, low, close) = make_wide_spread_data(50);
        let cs = CrackSpread::default();

        let mut ohlcv = OHLCVSeries::from_close(close);
        ohlcv.high = high;
        ohlcv.low = low;

        let result = cs.compute(&ohlcv);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_signal_indicator_impl() {
        let (high, low, close) = make_wide_spread_data(50);
        let cs = CrackSpread::default();

        let mut ohlcv = OHLCVSeries::from_close(close);
        ohlcv.high = high;
        ohlcv.low = low;

        let signal = cs.signal(&ohlcv);
        assert!(signal.is_ok());
    }

    #[test]
    fn test_insufficient_data() {
        let (high, low, close) = make_wide_spread_data(10);
        let cs = CrackSpread::new(20, 10).unwrap();

        let mut ohlcv = OHLCVSeries::from_close(close);
        ohlcv.high = high;
        ohlcv.low = low;

        let result = cs.compute(&ohlcv);
        assert!(result.is_err());
    }

    #[test]
    fn test_spread_series() {
        let (high, low, close) = make_wide_spread_data(50);
        let cs = CrackSpread::new(10, 5).unwrap();
        let series = cs.spread_series(&high, &low, &close);

        assert_eq!(series.len(), 50);
    }

    #[test]
    fn test_zscore_series() {
        let (high, low, close) = make_wide_spread_data(50);
        let cs = CrackSpread::new(10, 5).unwrap();
        let series = cs.zscore_series(&high, &low, &close);

        assert_eq!(series.len(), 50);
    }

    #[test]
    fn test_default_impl() {
        let cs = CrackSpread::default();
        assert_eq!(cs.period, 20);
        assert_eq!(cs.momentum_period, 10);
    }
}
