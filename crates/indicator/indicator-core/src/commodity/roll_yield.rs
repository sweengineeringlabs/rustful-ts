//! Roll Yield Indicator (IND-414)
//!
//! Measures the calendar spread return from rolling futures positions.
//! Roll yield is the return earned (or lost) when rolling from an expiring
//! futures contract to a further-dated contract.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Output for RollYield indicator.
#[derive(Debug, Clone, Copy)]
pub struct RollYieldOutput {
    /// Annualized roll yield (percentage).
    pub annualized_yield: f64,
    /// Raw period return (percentage).
    pub period_return: f64,
    /// Smoothed roll yield (EMA).
    pub smoothed: f64,
    /// Cumulative roll yield over time.
    pub cumulative: f64,
}

/// Roll Yield Indicator (IND-414)
///
/// Calculates the return from rolling futures positions by analyzing price
/// momentum as a proxy for the calendar spread. In practice, roll yield
/// depends on the futures curve shape:
/// - Backwardation: Positive roll yield (rolling into cheaper contracts)
/// - Contango: Negative roll yield (rolling into more expensive contracts)
///
/// # Algorithm
/// 1. Calculate period-over-period returns
/// 2. Annualize the returns based on the roll period
/// 3. Apply EMA smoothing for trend identification
/// 4. Track cumulative roll yield
///
/// # Interpretation
/// - Positive roll yield suggests favorable rolling conditions
/// - Negative roll yield indicates headwinds from contango
/// - Smoothed values help identify persistent trends
///
/// # Example
/// ```ignore
/// let ry = RollYield::new(20, 10)?;
/// let output = ry.compute(&data)?;
/// ```
#[derive(Debug, Clone)]
pub struct RollYield {
    /// Period for roll yield calculation (simulates contract length).
    roll_period: usize,
    /// Smoothing period for EMA.
    smoothing_period: usize,
    /// Trading days per year for annualization.
    trading_days: f64,
}

impl RollYield {
    /// Create a new RollYield indicator.
    ///
    /// # Arguments
    /// * `roll_period` - Period representing contract roll frequency
    /// * `smoothing_period` - Period for EMA smoothing
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid.
    pub fn new(roll_period: usize, smoothing_period: usize) -> Result<Self> {
        if roll_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "roll_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if smoothing_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            roll_period,
            smoothing_period,
            trading_days: 252.0,
        })
    }

    /// Create with default parameters (20, 10).
    pub fn default_params() -> Result<Self> {
        Self::new(20, 10)
    }

    /// Set custom trading days for annualization.
    pub fn with_trading_days(mut self, days: f64) -> Self {
        self.trading_days = days;
        self
    }

    /// Calculate roll yield values.
    pub fn calculate(&self, close: &[f64]) -> Vec<Option<RollYieldOutput>> {
        let n = close.len();
        let mut result = vec![None; n];

        if n < self.roll_period + 1 {
            return result;
        }

        // Calculate period returns
        let mut period_returns = vec![f64::NAN; n];
        for i in self.roll_period..n {
            if close[i - self.roll_period] > 0.0 {
                let ret = (close[i] / close[i - self.roll_period]).ln();
                period_returns[i] = ret * 100.0;
            }
        }

        // Calculate EMA multiplier
        let ema_multiplier = 2.0 / (self.smoothing_period as f64 + 1.0);

        // Initialize EMA and cumulative
        let mut ema = f64::NAN;
        let mut cumulative = 0.0;
        let mut ema_initialized = false;

        for i in self.roll_period..n {
            let period_ret = period_returns[i];
            if period_ret.is_nan() {
                continue;
            }

            // Annualize the return
            let annualized = period_ret * (self.trading_days / self.roll_period as f64);

            // Update EMA
            if !ema_initialized {
                ema = annualized;
                ema_initialized = true;
            } else {
                ema = (annualized - ema) * ema_multiplier + ema;
            }

            // Update cumulative
            cumulative += period_ret;

            result[i] = Some(RollYieldOutput {
                annualized_yield: annualized,
                period_return: period_ret,
                smoothed: ema,
                cumulative,
            });
        }

        result
    }

    /// Get the annualized yield series.
    pub fn yield_series(&self, close: &[f64]) -> Vec<f64> {
        self.calculate(close)
            .iter()
            .map(|o| o.map(|v| v.annualized_yield).unwrap_or(f64::NAN))
            .collect()
    }

    /// Get the smoothed yield series.
    pub fn smoothed_series(&self, close: &[f64]) -> Vec<f64> {
        self.calculate(close)
            .iter()
            .map(|o| o.map(|v| v.smoothed).unwrap_or(f64::NAN))
            .collect()
    }
}

impl Default for RollYield {
    fn default() -> Self {
        Self::default_params().unwrap()
    }
}

impl TechnicalIndicator for RollYield {
    fn name(&self) -> &str {
        "RollYield"
    }

    fn min_periods(&self) -> usize {
        self.roll_period + 1
    }

    fn output_features(&self) -> usize {
        2 // annualized_yield, smoothed
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
            .map(|o| o.map(|v| v.annualized_yield).unwrap_or(f64::NAN))
            .collect();

        let secondary: Vec<f64> = outputs
            .iter()
            .map(|o| o.map(|v| v.smoothed).unwrap_or(f64::NAN))
            .collect();

        Ok(IndicatorOutput::dual(primary, secondary))
    }
}

impl SignalIndicator for RollYield {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let outputs = self.calculate(&data.close);

        match outputs.last().and_then(|o| *o) {
            Some(out) => {
                if out.smoothed > 5.0 {
                    Ok(IndicatorSignal::Bullish) // Strong positive roll yield
                } else if out.smoothed < -5.0 {
                    Ok(IndicatorSignal::Bearish) // Strong negative roll yield
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
                    if out.smoothed > 5.0 {
                        IndicatorSignal::Bullish
                    } else if out.smoothed < -5.0 {
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
        (0..n).map(|i| 100.0 * (1.0 + 0.001 * i as f64).powf(i as f64 / 10.0)).collect()
    }

    fn make_downtrend_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| 100.0 * (1.0 - 0.001 * i as f64).max(0.5)).collect()
    }

    fn make_flat_data(n: usize) -> Vec<f64> {
        vec![100.0; n]
    }

    #[test]
    fn test_new_valid_params() {
        let ry = RollYield::new(20, 10);
        assert!(ry.is_ok());
    }

    #[test]
    fn test_new_invalid_roll_period() {
        let ry = RollYield::new(1, 10);
        assert!(ry.is_err());
    }

    #[test]
    fn test_new_invalid_smoothing_period() {
        let ry = RollYield::new(20, 0);
        assert!(ry.is_err());
    }

    #[test]
    fn test_uptrend_positive_yield() {
        let data = make_uptrend_data(50);
        let ry = RollYield::new(10, 5).unwrap();
        let outputs = ry.calculate(&data);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Uptrend should have positive roll yield
        let positive_count = valid_outputs
            .iter()
            .filter(|o| o.annualized_yield > 0.0)
            .count();
        assert!(positive_count > valid_outputs.len() / 2);
    }

    #[test]
    fn test_downtrend_negative_yield() {
        let data = make_downtrend_data(50);
        let ry = RollYield::new(10, 5).unwrap();
        let outputs = ry.calculate(&data);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Downtrend should have negative roll yield
        let negative_count = valid_outputs
            .iter()
            .filter(|o| o.annualized_yield < 0.0)
            .count();
        assert!(negative_count > valid_outputs.len() / 2);
    }

    #[test]
    fn test_flat_near_zero_yield() {
        let data = make_flat_data(50);
        let ry = RollYield::new(10, 5).unwrap();
        let outputs = ry.calculate(&data);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();

        // Flat data should have near-zero yield
        for out in valid_outputs {
            assert!(out.annualized_yield.abs() < 0.001);
        }
    }

    #[test]
    fn test_cumulative_increasing() {
        let data = make_uptrend_data(50);
        let ry = RollYield::new(10, 5).unwrap();
        let outputs = ry.calculate(&data);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();

        // Cumulative should generally increase in uptrend
        if valid_outputs.len() >= 2 {
            let last = valid_outputs.last().unwrap();
            let first = valid_outputs.first().unwrap();
            assert!(last.cumulative > first.cumulative);
        }
    }

    #[test]
    fn test_technical_indicator_impl() {
        let data = make_uptrend_data(50);
        let ry = RollYield::default();
        let ohlcv = OHLCVSeries::from_close(data);

        let result = ry.compute(&ohlcv);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_signal_indicator_impl() {
        let data = make_uptrend_data(50);
        let ry = RollYield::default();
        let ohlcv = OHLCVSeries::from_close(data);

        let signal = ry.signal(&ohlcv);
        assert!(signal.is_ok());
    }

    #[test]
    fn test_insufficient_data() {
        let data = vec![100.0; 10];
        let ry = RollYield::new(20, 10).unwrap();
        let ohlcv = OHLCVSeries::from_close(data);

        let result = ry.compute(&ohlcv);
        assert!(result.is_err());
    }

    #[test]
    fn test_yield_series() {
        let data = make_uptrend_data(50);
        let ry = RollYield::new(10, 5).unwrap();
        let series = ry.yield_series(&data);

        assert_eq!(series.len(), 50);
        // First values should be NaN
        assert!(series[0].is_nan());
    }

    #[test]
    fn test_smoothed_series() {
        let data = make_uptrend_data(50);
        let ry = RollYield::new(10, 5).unwrap();
        let series = ry.smoothed_series(&data);

        assert_eq!(series.len(), 50);
        // First values should be NaN
        assert!(series[0].is_nan());
    }

    #[test]
    fn test_with_trading_days() {
        let ry = RollYield::new(20, 10).unwrap().with_trading_days(260.0);
        assert!((ry.trading_days - 260.0).abs() < 0.001);
    }

    #[test]
    fn test_default_impl() {
        let ry = RollYield::default();
        assert_eq!(ry.roll_period, 20);
        assert_eq!(ry.smoothing_period, 10);
    }
}
