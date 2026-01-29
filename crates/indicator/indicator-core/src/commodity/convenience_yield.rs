//! Convenience Yield Indicator (IND-415)
//!
//! Measures the spot premium in commodity markets. Convenience yield represents
//! the benefit from holding the physical commodity rather than a futures contract,
//! including the ability to meet unexpected demand.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Output for ConvenienceYield indicator.
#[derive(Debug, Clone, Copy)]
pub struct ConvenienceYieldOutput {
    /// Estimated convenience yield (annualized percentage).
    pub yield_value: f64,
    /// Volatility component of the yield.
    pub volatility_component: f64,
    /// Trend component (backwardation indicator).
    pub trend_component: f64,
    /// Net yield after storage costs proxy.
    pub net_yield: f64,
}

/// Convenience Yield Indicator (IND-415)
///
/// Estimates the convenience yield using price behavior as a proxy for spot premium.
/// The convenience yield is the implicit return from holding physical inventory:
/// - High convenience yield indicates tight supply or high demand uncertainty
/// - Low/negative convenience yield suggests ample supply
///
/// # Algorithm
/// 1. Calculate rolling volatility as a measure of demand uncertainty
/// 2. Estimate backwardation from price trend (negative trend = backwardation)
/// 3. Combine volatility and backwardation components
/// 4. Subtract storage cost proxy for net yield
///
/// # Interpretation
/// - High convenience yield: Supply tightness, potential price support
/// - Low convenience yield: Ample supply, contango conditions likely
/// - Negative net yield: Storage costs exceed holding benefits
///
/// # Example
/// ```ignore
/// let cy = ConvenienceYield::new(20, 0.02)?;
/// let output = cy.compute(&data)?;
/// ```
#[derive(Debug, Clone)]
pub struct ConvenienceYield {
    /// Period for calculations.
    period: usize,
    /// Storage cost rate (annual, as decimal).
    storage_cost_rate: f64,
    /// Risk-free rate proxy (annual, as decimal).
    risk_free_rate: f64,
}

impl ConvenienceYield {
    /// Create a new ConvenienceYield indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for calculations
    /// * `storage_cost_rate` - Annual storage cost as decimal (e.g., 0.02 = 2%)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid.
    pub fn new(period: usize, storage_cost_rate: f64) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if storage_cost_rate < 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "storage_cost_rate".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        Ok(Self {
            period,
            storage_cost_rate,
            risk_free_rate: 0.05, // Default 5% annual rate
        })
    }

    /// Create with default parameters (20, 0.02).
    pub fn default_params() -> Result<Self> {
        Self::new(20, 0.02)
    }

    /// Set custom risk-free rate.
    pub fn with_risk_free_rate(mut self, rate: f64) -> Self {
        self.risk_free_rate = rate;
        self
    }

    /// Calculate convenience yield values.
    pub fn calculate(&self, close: &[f64]) -> Vec<Option<ConvenienceYieldOutput>> {
        let n = close.len();
        let mut result = vec![None; n];

        if n < self.period + 1 {
            return result;
        }

        for i in self.period..n {
            let start = i - self.period;

            // Calculate returns for volatility
            let mut returns = Vec::with_capacity(self.period);
            for j in (start + 1)..=i {
                if close[j - 1] > 0.0 {
                    returns.push((close[j] / close[j - 1]).ln());
                }
            }

            if returns.is_empty() {
                continue;
            }

            // Calculate volatility (annualized)
            let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean_ret).powi(2))
                .sum::<f64>() / returns.len() as f64;
            let volatility = variance.sqrt() * (252.0_f64).sqrt();

            // Calculate trend component (annualized return)
            let period_return = if close[start] > 0.0 {
                (close[i] / close[start]).ln()
            } else {
                0.0
            };
            let trend = period_return * (252.0 / self.period as f64);

            // Convenience yield approximation:
            // In backwardation (negative trend), convenience yield is high
            // Volatility adds to convenience yield (option value of inventory)
            let volatility_component = volatility * 0.5; // Half of volatility as option value
            let trend_component = -trend; // Negative trend = positive backwardation = positive convenience yield

            // Total convenience yield
            let yield_value = volatility_component + trend_component.max(0.0);

            // Net yield = convenience yield - storage costs - risk-free rate
            let net_yield = yield_value - self.storage_cost_rate - self.risk_free_rate;

            result[i] = Some(ConvenienceYieldOutput {
                yield_value: yield_value * 100.0, // Convert to percentage
                volatility_component: volatility_component * 100.0,
                trend_component: trend_component * 100.0,
                net_yield: net_yield * 100.0,
            });
        }

        result
    }

    /// Get the yield series.
    pub fn yield_series(&self, close: &[f64]) -> Vec<f64> {
        self.calculate(close)
            .iter()
            .map(|o| o.map(|v| v.yield_value).unwrap_or(f64::NAN))
            .collect()
    }

    /// Get the net yield series.
    pub fn net_yield_series(&self, close: &[f64]) -> Vec<f64> {
        self.calculate(close)
            .iter()
            .map(|o| o.map(|v| v.net_yield).unwrap_or(f64::NAN))
            .collect()
    }
}

impl Default for ConvenienceYield {
    fn default() -> Self {
        Self::default_params().unwrap()
    }
}

impl TechnicalIndicator for ConvenienceYield {
    fn name(&self) -> &str {
        "ConvenienceYield"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        2 // yield_value, net_yield
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
            .map(|o| o.map(|v| v.yield_value).unwrap_or(f64::NAN))
            .collect();

        let secondary: Vec<f64> = outputs
            .iter()
            .map(|o| o.map(|v| v.net_yield).unwrap_or(f64::NAN))
            .collect();

        Ok(IndicatorOutput::dual(primary, secondary))
    }
}

impl SignalIndicator for ConvenienceYield {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let outputs = self.calculate(&data.close);

        match outputs.last().and_then(|o| *o) {
            Some(out) => {
                if out.net_yield > 5.0 {
                    Ok(IndicatorSignal::Bullish) // High convenience yield = supply tightness
                } else if out.net_yield < -5.0 {
                    Ok(IndicatorSignal::Bearish) // Negative net yield = oversupply
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
                    if out.net_yield > 5.0 {
                        IndicatorSignal::Bullish
                    } else if out.net_yield < -5.0 {
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

    fn make_backwardation_data(n: usize) -> Vec<f64> {
        // Declining prices = backwardation scenario
        (0..n).map(|i| 100.0 * (0.999_f64).powi(i as i32)).collect()
    }

    fn make_contango_data(n: usize) -> Vec<f64> {
        // Rising prices = contango scenario
        (0..n).map(|i| 100.0 * (1.001_f64).powi(i as i32)).collect()
    }

    fn make_volatile_data(n: usize) -> Vec<f64> {
        // High volatility data
        (0..n)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 10.0)
            .collect()
    }

    #[test]
    fn test_new_valid_params() {
        let cy = ConvenienceYield::new(20, 0.02);
        assert!(cy.is_ok());
    }

    #[test]
    fn test_new_invalid_period() {
        let cy = ConvenienceYield::new(2, 0.02);
        assert!(cy.is_err());
    }

    #[test]
    fn test_new_invalid_storage_cost() {
        let cy = ConvenienceYield::new(20, -0.02);
        assert!(cy.is_err());
    }

    #[test]
    fn test_backwardation_high_yield() {
        let data = make_backwardation_data(50);
        let cy = ConvenienceYield::new(10, 0.01).unwrap();
        let outputs = cy.calculate(&data);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Backwardation should have positive trend component
        let positive_trend = valid_outputs
            .iter()
            .filter(|o| o.trend_component > 0.0)
            .count();
        assert!(positive_trend > valid_outputs.len() / 2);
    }

    #[test]
    fn test_contango_low_yield() {
        let data = make_contango_data(50);
        let cy = ConvenienceYield::new(10, 0.01).unwrap();
        let outputs = cy.calculate(&data);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Contango should have negative trend component
        let negative_trend = valid_outputs
            .iter()
            .filter(|o| o.trend_component < 0.0)
            .count();
        assert!(negative_trend > valid_outputs.len() / 2);
    }

    #[test]
    fn test_volatile_data_volatility_component() {
        let data = make_volatile_data(50);
        let cy = ConvenienceYield::new(10, 0.01).unwrap();
        let outputs = cy.calculate(&data);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Should have positive volatility component
        for out in valid_outputs {
            assert!(out.volatility_component >= 0.0);
        }
    }

    #[test]
    fn test_net_yield_calculation() {
        let data = make_backwardation_data(50);
        let cy = ConvenienceYield::new(10, 0.02)
            .unwrap()
            .with_risk_free_rate(0.05);
        let outputs = cy.calculate(&data);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();

        // Net yield should be yield minus costs
        for out in valid_outputs {
            let expected_net = out.yield_value - 2.0 - 5.0; // storage 2% + risk-free 5%
            assert!((out.net_yield - expected_net).abs() < 0.01);
        }
    }

    #[test]
    fn test_technical_indicator_impl() {
        let data = make_backwardation_data(50);
        let cy = ConvenienceYield::default();
        let ohlcv = OHLCVSeries::from_close(data);

        let result = cy.compute(&ohlcv);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_signal_indicator_impl() {
        let data = make_backwardation_data(50);
        let cy = ConvenienceYield::default();
        let ohlcv = OHLCVSeries::from_close(data);

        let signal = cy.signal(&ohlcv);
        assert!(signal.is_ok());
    }

    #[test]
    fn test_insufficient_data() {
        let data = vec![100.0; 10];
        let cy = ConvenienceYield::new(20, 0.02).unwrap();
        let ohlcv = OHLCVSeries::from_close(data);

        let result = cy.compute(&ohlcv);
        assert!(result.is_err());
    }

    #[test]
    fn test_yield_series() {
        let data = make_backwardation_data(50);
        let cy = ConvenienceYield::new(10, 0.02).unwrap();
        let series = cy.yield_series(&data);

        assert_eq!(series.len(), 50);
        assert!(series[0].is_nan());
    }

    #[test]
    fn test_net_yield_series() {
        let data = make_backwardation_data(50);
        let cy = ConvenienceYield::new(10, 0.02).unwrap();
        let series = cy.net_yield_series(&data);

        assert_eq!(series.len(), 50);
        assert!(series[0].is_nan());
    }

    #[test]
    fn test_with_risk_free_rate() {
        let cy = ConvenienceYield::new(20, 0.02).unwrap().with_risk_free_rate(0.03);
        assert!((cy.risk_free_rate - 0.03).abs() < 0.001);
    }

    #[test]
    fn test_default_impl() {
        let cy = ConvenienceYield::default();
        assert_eq!(cy.period, 20);
        assert!((cy.storage_cost_rate - 0.02).abs() < 0.001);
    }
}
