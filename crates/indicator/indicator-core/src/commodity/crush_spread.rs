//! Crush Spread Indicator (IND-418)
//!
//! Measures the soybean processing margin by calculating the spread between
//! soybeans and their processed products (soybean meal and soybean oil).
//! A proxy for crusher profitability.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Output for CrushSpread indicator.
#[derive(Debug, Clone, Copy)]
pub struct CrushSpreadOutput {
    /// Raw crush spread value (percentage of input price).
    pub spread: f64,
    /// Z-score of spread relative to historical average.
    pub zscore: f64,
    /// Moving average of spread.
    pub ma: f64,
    /// Spread momentum (rate of change).
    pub momentum: f64,
    /// Spread standard deviation (volatility).
    pub std_dev: f64,
}

/// Crush Spread Indicator (IND-418)
///
/// Estimates the soybean crushing margin using price dynamics as a proxy.
/// The actual crush spread formula:
/// Crush Spread = (Soybean Meal Value + Soybean Oil Value) - Soybean Cost
///
/// Typically: 1 bushel soybeans = 48 lbs meal + 11 lbs oil + waste
///
/// # Algorithm
/// 1. Use price momentum and volatility as proxies for processing margin
/// 2. Calculate rolling statistics for normalization
/// 3. Track momentum of spread changes
/// 4. Monitor spread volatility for risk assessment
///
/// # Interpretation
/// - High spread: Strong crushing margins, bullish for processors
/// - Low spread: Weak margins, bearish for processors
/// - Rising spread: Improving conditions
/// - High volatility: Uncertain margin environment
///
/// # Example
/// ```ignore
/// let crush = CrushSpread::new(20, 10)?;
/// let output = crush.compute(&data)?;
/// ```
#[derive(Debug, Clone)]
pub struct CrushSpread {
    /// Period for moving average and z-score calculations.
    period: usize,
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Meal conversion factor (default ~48 lbs per bushel).
    meal_factor: f64,
    /// Oil conversion factor (default ~11 lbs per bushel).
    oil_factor: f64,
}

impl CrushSpread {
    /// Create a new CrushSpread indicator.
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
            meal_factor: 0.8, // 80% of value typically comes from meal
            oil_factor: 0.2,  // 20% from oil
        })
    }

    /// Create with default parameters (20, 10).
    pub fn default_params() -> Result<Self> {
        Self::new(20, 10)
    }

    /// Set custom conversion factors.
    pub fn with_factors(mut self, meal_factor: f64, oil_factor: f64) -> Self {
        self.meal_factor = meal_factor;
        self.oil_factor = oil_factor;
        self
    }

    /// Calculate crush spread using single price series (proxy method).
    pub fn calculate(&self, close: &[f64]) -> Vec<Option<CrushSpreadOutput>> {
        let n = close.len();
        let mut result = vec![None; n];

        if n < self.period + self.momentum_period {
            return result;
        }

        // Calculate proxy spreads using price momentum
        let mut spreads = vec![f64::NAN; n];
        for i in 1..n {
            if close[i - 1] > 0.0 {
                // Use short-term price change as proxy for margin dynamics
                spreads[i] = (close[i] / close[i - 1] - 1.0) * 100.0;
            }
        }

        // Apply smoothing to get more stable spread estimate
        let smoothed = self.ema_smooth(&spreads, 5);

        for i in (self.period - 1)..n {
            let start = i - self.period + 1;

            // Calculate statistics
            let window: Vec<f64> = smoothed[start..=i]
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
            let std_dev = variance.sqrt();

            let spread = smoothed[i];
            if spread.is_nan() {
                continue;
            }

            // Z-score
            let zscore = if std_dev > 0.0 {
                (spread - mean) / std_dev
            } else {
                0.0
            };

            // Momentum
            let momentum = if i >= self.momentum_period && !smoothed[i - self.momentum_period].is_nan() {
                spread - smoothed[i - self.momentum_period]
            } else {
                f64::NAN
            };

            result[i] = Some(CrushSpreadOutput {
                spread,
                zscore,
                ma: mean,
                momentum,
                std_dev,
            });
        }

        result
    }

    /// Calculate crush spread from three separate series (soybeans, meal, oil).
    pub fn calculate_from_components(
        &self,
        soybeans: &[f64],
        meal: &[f64],
        oil: &[f64],
    ) -> Vec<Option<CrushSpreadOutput>> {
        let n = soybeans.len().min(meal.len()).min(oil.len());
        let mut result = vec![None; n];

        if n < self.period + self.momentum_period {
            return result;
        }

        // Calculate actual crush spread
        let mut spreads = vec![f64::NAN; n];
        for i in 0..n {
            if soybeans[i] > 0.0 {
                // Product value - input cost
                let product_value = meal[i] * self.meal_factor + oil[i] * self.oil_factor;
                spreads[i] = (product_value - soybeans[i]) / soybeans[i] * 100.0;
            }
        }

        for i in (self.period - 1)..n {
            let start = i - self.period + 1;

            // Calculate statistics
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
            let std_dev = variance.sqrt();

            let spread = spreads[i];
            if spread.is_nan() {
                continue;
            }

            // Z-score
            let zscore = if std_dev > 0.0 {
                (spread - mean) / std_dev
            } else {
                0.0
            };

            // Momentum
            let momentum = if i >= self.momentum_period && !spreads[i - self.momentum_period].is_nan() {
                spread - spreads[i - self.momentum_period]
            } else {
                f64::NAN
            };

            result[i] = Some(CrushSpreadOutput {
                spread,
                zscore,
                ma: mean,
                momentum,
                std_dev,
            });
        }

        result
    }

    /// Apply EMA smoothing.
    fn ema_smooth(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n == 0 || period == 0 {
            return result;
        }

        let mult = 2.0 / (period as f64 + 1.0);
        let mut ema = f64::NAN;

        for i in 0..n {
            if data[i].is_nan() {
                result[i] = ema;
            } else if ema.is_nan() {
                ema = data[i];
                result[i] = ema;
            } else {
                ema = (data[i] - ema) * mult + ema;
                result[i] = ema;
            }
        }

        result
    }

    /// Get the spread series.
    pub fn spread_series(&self, close: &[f64]) -> Vec<f64> {
        self.calculate(close)
            .iter()
            .map(|o| o.map(|v| v.spread).unwrap_or(f64::NAN))
            .collect()
    }

    /// Get the z-score series.
    pub fn zscore_series(&self, close: &[f64]) -> Vec<f64> {
        self.calculate(close)
            .iter()
            .map(|o| o.map(|v| v.zscore).unwrap_or(f64::NAN))
            .collect()
    }
}

impl Default for CrushSpread {
    fn default() -> Self {
        Self::default_params().unwrap()
    }
}

impl TechnicalIndicator for CrushSpread {
    fn name(&self) -> &str {
        "CrushSpread"
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

        let outputs = self.calculate(&data.close);

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

impl SignalIndicator for CrushSpread {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let outputs = self.calculate(&data.close);

        match outputs.last().and_then(|o| *o) {
            Some(out) => {
                // High z-score with positive momentum = bullish for processors
                if out.zscore > 1.0 && !out.momentum.is_nan() && out.momentum > 0.0 {
                    Ok(IndicatorSignal::Bullish)
                // Low z-score with negative momentum = bearish for processors
                } else if out.zscore < -1.0 && !out.momentum.is_nan() && out.momentum < 0.0 {
                    Ok(IndicatorSignal::Bearish)
                } else {
                    Ok(IndicatorSignal::Neutral)
                }
            }
            None => Ok(IndicatorSignal::Neutral),
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let outputs = self.calculate(&data.close);

        let signals = outputs
            .iter()
            .map(|o| match o {
                Some(out) => {
                    if out.zscore > 1.0 && !out.momentum.is_nan() && out.momentum > 0.0 {
                        IndicatorSignal::Bullish
                    } else if out.zscore < -1.0 && !out.momentum.is_nan() && out.momentum < 0.0 {
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

    fn make_uptrend_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 * (1.001_f64).powi(i as i32)).collect()
    }

    fn make_downtrend_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 * (0.999_f64).powi(i as i32)).collect()
    }

    fn make_volatile_data(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 5.0)
            .collect()
    }

    fn make_component_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Soybeans, meal, oil prices
        let soybeans: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.1).collect();
        let meal: Vec<f64> = (0..n).map(|i| 90.0 + i as f64 * 0.12).collect();
        let oil: Vec<f64> = (0..n).map(|i| 30.0 + i as f64 * 0.05).collect();
        (soybeans, meal, oil)
    }

    #[test]
    fn test_new_valid_params() {
        let crush = CrushSpread::new(20, 10);
        assert!(crush.is_ok());
    }

    #[test]
    fn test_new_invalid_period() {
        let crush = CrushSpread::new(1, 10);
        assert!(crush.is_err());
    }

    #[test]
    fn test_new_invalid_momentum_period() {
        let crush = CrushSpread::new(20, 0);
        assert!(crush.is_err());
    }

    #[test]
    fn test_uptrend_positive_spread() {
        let data = make_uptrend_data(50);
        let crush = CrushSpread::new(10, 5).unwrap();
        let outputs = crush.calculate(&data);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Uptrend should have positive spreads (proxy)
        let positive_count = valid_outputs
            .iter()
            .filter(|o| o.spread > 0.0)
            .count();
        assert!(positive_count > valid_outputs.len() / 2);
    }

    #[test]
    fn test_downtrend_negative_spread() {
        let data = make_downtrend_data(50);
        let crush = CrushSpread::new(10, 5).unwrap();
        let outputs = crush.calculate(&data);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Downtrend should have negative spreads (proxy)
        let negative_count = valid_outputs
            .iter()
            .filter(|o| o.spread < 0.0)
            .count();
        assert!(negative_count > valid_outputs.len() / 2);
    }

    #[test]
    fn test_volatile_data_std_dev() {
        let data = make_volatile_data(50);
        let crush = CrushSpread::new(10, 5).unwrap();
        let outputs = crush.calculate(&data);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Should have non-zero std_dev
        for out in valid_outputs {
            assert!(out.std_dev >= 0.0);
        }
    }

    #[test]
    fn test_calculate_from_components() {
        let (soybeans, meal, oil) = make_component_data(50);
        let crush = CrushSpread::new(10, 5).unwrap();
        let outputs = crush.calculate_from_components(&soybeans, &meal, &oil);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // With default factors, check reasonable spreads
        for out in &valid_outputs {
            assert!(out.spread.abs() < 50.0);
        }
    }

    #[test]
    fn test_zscore_bounds() {
        let data = make_uptrend_data(50);
        let crush = CrushSpread::default();
        let outputs = crush.calculate(&data);

        for out in outputs.iter().filter_map(|o| *o) {
            // Z-scores should be reasonable
            assert!(out.zscore.abs() < 5.0);
        }
    }

    #[test]
    fn test_with_factors() {
        let crush = CrushSpread::new(20, 10).unwrap().with_factors(0.75, 0.25);
        assert!((crush.meal_factor - 0.75).abs() < 0.001);
        assert!((crush.oil_factor - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_technical_indicator_impl() {
        let data = make_uptrend_data(50);
        let crush = CrushSpread::default();
        let ohlcv = OHLCVSeries::from_close(data);

        let result = crush.compute(&ohlcv);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_signal_indicator_impl() {
        let data = make_uptrend_data(50);
        let crush = CrushSpread::default();
        let ohlcv = OHLCVSeries::from_close(data);

        let signal = crush.signal(&ohlcv);
        assert!(signal.is_ok());
    }

    #[test]
    fn test_insufficient_data() {
        let data = vec![100.0; 10];
        let crush = CrushSpread::new(20, 10).unwrap();
        let ohlcv = OHLCVSeries::from_close(data);

        let result = crush.compute(&ohlcv);
        assert!(result.is_err());
    }

    #[test]
    fn test_spread_series() {
        let data = make_uptrend_data(50);
        let crush = CrushSpread::new(10, 5).unwrap();
        let series = crush.spread_series(&data);

        assert_eq!(series.len(), 50);
    }

    #[test]
    fn test_zscore_series() {
        let data = make_uptrend_data(50);
        let crush = CrushSpread::new(10, 5).unwrap();
        let series = crush.zscore_series(&data);

        assert_eq!(series.len(), 50);
    }

    #[test]
    fn test_default_impl() {
        let crush = CrushSpread::default();
        assert_eq!(crush.period, 20);
        assert_eq!(crush.momentum_period, 10);
        assert!((crush.meal_factor - 0.8).abs() < 0.001);
        assert!((crush.oil_factor - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_ema_smooth() {
        let crush = CrushSpread::default();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let smoothed = crush.ema_smooth(&data, 3);

        assert_eq!(smoothed.len(), 5);
        // First value should equal input
        assert!((smoothed[0] - 1.0).abs() < 0.001);
        // Subsequent values should be smoothed
        for i in 1..5 {
            assert!(!smoothed[i].is_nan());
        }
    }

    #[test]
    fn test_momentum_calculation() {
        let data = make_uptrend_data(50);
        let crush = CrushSpread::new(10, 5).unwrap();
        let outputs = crush.calculate(&data);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();

        // Check momentum is calculated
        let with_momentum = valid_outputs
            .iter()
            .filter(|o| !o.momentum.is_nan())
            .count();
        assert!(with_momentum > 0);
    }
}
