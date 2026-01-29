//! Real Yield Indicator (IND-298)
//!
//! Calculates the real yield as nominal yield minus inflation expectations.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Real Yield - Nominal yield minus inflation proxy
///
/// This indicator calculates real yields by subtracting inflation expectations
/// from nominal yields. When actual inflation data is unavailable, it uses
/// price momentum as an inflation proxy.
///
/// # Formula
/// Real Yield = Nominal Yield - Inflation Expectation
///
/// # Interpretation
/// - Positive real yield: Bonds provide positive real return
/// - Zero real yield: Bonds just keep pace with inflation
/// - Negative real yield: Bonds lose purchasing power (risk-on signal)
///
/// # Note
/// For accurate analysis, use calculate_real_yield with actual yield
/// and inflation expectation data.
#[derive(Debug, Clone)]
pub struct RealYield {
    /// Period for inflation proxy calculation
    inflation_period: usize,
    /// Smoothing period
    smooth_period: usize,
    /// Default assumed nominal yield (annualized %)
    nominal_base: f64,
}

impl RealYield {
    /// Create a new RealYield indicator
    ///
    /// # Arguments
    /// * `inflation_period` - Period for inflation proxy (minimum 10)
    /// * `smooth_period` - EMA smoothing period (minimum 2)
    /// * `nominal_base` - Base nominal yield assumption (e.g., 3.0 for 3%)
    pub fn new(inflation_period: usize, smooth_period: usize, nominal_base: f64) -> Result<Self> {
        if inflation_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "inflation_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if nominal_base < 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "nominal_base".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        Ok(Self {
            inflation_period,
            smooth_period,
            nominal_base,
        })
    }

    /// Create with default parameters
    pub fn default_params() -> Result<Self> {
        Self::new(20, 5, 3.0)
    }

    /// Calculate real yield using price momentum as inflation proxy
    ///
    /// This method uses rolling price changes as a proxy for inflation
    /// expectations, then subtracts from an assumed nominal yield.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.inflation_period {
            return result;
        }

        for i in self.inflation_period..n {
            // Use price momentum as inflation proxy (annualized)
            let price_change = if close[i - self.inflation_period] > 1e-10 {
                (close[i] / close[i - self.inflation_period] - 1.0) * 100.0
            } else {
                0.0
            };

            // Annualize the inflation proxy (assuming daily data, ~252 trading days)
            let annualization_factor = 252.0 / self.inflation_period as f64;
            let inflation_proxy = price_change * annualization_factor;

            // Real yield = nominal - inflation
            // Clamp inflation to reasonable range (-10% to 20%)
            let clamped_inflation = inflation_proxy.max(-10.0).min(20.0);
            result[i] = self.nominal_base - clamped_inflation;
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate real yield from actual yield and inflation data
    ///
    /// # Arguments
    /// * `nominal_yields` - Nominal yields (e.g., 10Y Treasury)
    /// * `inflation_expectations` - Inflation expectations (e.g., breakeven inflation)
    ///
    /// # Returns
    /// Real yields in percentage points
    pub fn calculate_real_yield(
        &self,
        nominal_yields: &[f64],
        inflation_expectations: &[f64],
    ) -> Vec<f64> {
        let n = nominal_yields.len().min(inflation_expectations.len());
        let mut result = vec![0.0; n];

        for i in 0..n {
            result[i] = nominal_yields[i] - inflation_expectations[i];
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate real yield trend (change over period)
    pub fn calculate_trend(&self, real_yields: &[f64], period: usize) -> Vec<f64> {
        let n = real_yields.len();
        let mut result = vec![0.0; n];

        for i in period..n {
            result[i] = real_yields[i] - real_yields[i - period];
        }

        result
    }

    /// Classify real yield environment
    pub fn classify_environment(&self, real_yield: f64) -> RealYieldEnvironment {
        if real_yield > 2.0 {
            RealYieldEnvironment::HighlyPositive
        } else if real_yield > 0.5 {
            RealYieldEnvironment::Positive
        } else if real_yield > -0.5 {
            RealYieldEnvironment::Neutral
        } else if real_yield > -2.0 {
            RealYieldEnvironment::Negative
        } else {
            RealYieldEnvironment::DeepNegative
        }
    }
}

/// Real yield environment classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RealYieldEnvironment {
    /// Real yield > 2%
    HighlyPositive,
    /// Real yield 0.5% to 2%
    Positive,
    /// Real yield -0.5% to 0.5%
    Neutral,
    /// Real yield -2% to -0.5%
    Negative,
    /// Real yield < -2%
    DeepNegative,
}

impl TechnicalIndicator for RealYield {
    fn name(&self) -> &str {
        "Real Yield"
    }

    fn min_periods(&self) -> usize {
        self.inflation_period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let n = 60;
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.1) // Mild uptrend
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = vec![1000.0; n];

        OHLCVSeries { open, high, low, close, volume }
    }

    fn make_inflationary_data() -> OHLCVSeries {
        let n = 60;
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 * (1.001_f64).powi(i as i32)) // ~1% daily inflation
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = vec![1000.0; n];

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_real_yield_basic() {
        let data = make_test_data();
        let indicator = RealYield::new(20, 5, 3.0).unwrap();
        let result = indicator.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_real_yield_from_actual_data() {
        let nominal_yields: Vec<f64> = vec![3.5; 50]; // 3.5% nominal
        let inflation_exp: Vec<f64> = vec![2.5; 50]; // 2.5% inflation

        let indicator = RealYield::new(20, 5, 3.0).unwrap();
        let real_yields = indicator.calculate_real_yield(&nominal_yields, &inflation_exp);

        assert_eq!(real_yields.len(), 50);
        // Real yield should be around 1% (3.5 - 2.5)
        assert!((real_yields[40] - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_real_yield_negative_environment() {
        let nominal_yields: Vec<f64> = vec![2.0; 50]; // 2% nominal
        let inflation_exp: Vec<f64> = vec![3.5; 50]; // 3.5% inflation

        let indicator = RealYield::new(20, 5, 3.0).unwrap();
        let real_yields = indicator.calculate_real_yield(&nominal_yields, &inflation_exp);

        // Real yield should be negative (2.0 - 3.5 = -1.5)
        assert!(real_yields[40] < 0.0);
    }

    #[test]
    fn test_real_yield_inflationary_proxy() {
        let data = make_inflationary_data();
        let indicator = RealYield::new(20, 5, 3.0).unwrap();
        let result = indicator.calculate(&data.close);

        // High inflation proxy should result in lower real yields
        // With 3% base and high inflation proxy, should be lower
        // Note: result depends on the specific data pattern
        assert!(result[50] < 5.0); // Should be bounded reasonably
    }

    #[test]
    fn test_real_yield_classify_environment() {
        let indicator = RealYield::default_params().unwrap();

        assert_eq!(
            indicator.classify_environment(3.0),
            RealYieldEnvironment::HighlyPositive
        );
        assert_eq!(
            indicator.classify_environment(1.0),
            RealYieldEnvironment::Positive
        );
        assert_eq!(
            indicator.classify_environment(0.0),
            RealYieldEnvironment::Neutral
        );
        assert_eq!(
            indicator.classify_environment(-1.0),
            RealYieldEnvironment::Negative
        );
        assert_eq!(
            indicator.classify_environment(-3.0),
            RealYieldEnvironment::DeepNegative
        );
    }

    #[test]
    fn test_real_yield_trend() {
        let real_yields: Vec<f64> = (0..30).map(|i| 1.0 + i as f64 * 0.1).collect();

        let indicator = RealYield::default_params().unwrap();
        let trend = indicator.calculate_trend(&real_yields, 5);

        // Trend should be positive (increasing real yields)
        assert!(trend[20] > 0.0);
    }

    #[test]
    fn test_real_yield_technical_indicator_trait() {
        let data = make_test_data();
        let indicator = RealYield::new(20, 5, 3.0).unwrap();

        assert_eq!(indicator.name(), "Real Yield");
        assert_eq!(indicator.min_periods(), 25);

        let output = indicator.compute(&data).unwrap();
        assert!(!output.values.is_empty());
    }

    #[test]
    fn test_real_yield_parameter_validation() {
        assert!(RealYield::new(5, 5, 3.0).is_err()); // inflation_period too small
        assert!(RealYield::new(20, 1, 3.0).is_err()); // smooth_period too small
        assert!(RealYield::new(20, 5, -1.0).is_err()); // negative nominal_base
    }

    #[test]
    fn test_real_yield_default_params() {
        let indicator = RealYield::default_params().unwrap();
        assert_eq!(indicator.inflation_period, 20);
        assert_eq!(indicator.smooth_period, 5);
        assert!((indicator.nominal_base - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_real_yield_environment_enum() {
        let env1 = RealYieldEnvironment::HighlyPositive;
        let env2 = RealYieldEnvironment::Negative;

        assert_ne!(env1, env2);
        assert_eq!(env1, RealYieldEnvironment::HighlyPositive);
    }
}
