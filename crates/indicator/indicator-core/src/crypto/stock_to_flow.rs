//! Stock-to-Flow Model (S2F) - IND-275
//!
//! A scarcity-based valuation model for Bitcoin and other finite-supply assets.
//!
//! Stock = Total existing supply
//! Flow = New supply per period (e.g., annual production/mining)
//!
//! S2F Ratio = Stock / Flow (higher = more scarce)
//!
//! Interpretation:
//! - Higher S2F = Greater scarcity = Potential price appreciation
//! - Bitcoin's S2F doubles approximately every 4 years (halvings)
//! - Gold S2F ~62, Silver S2F ~22, Bitcoin post-2024 ~120+

use indicator_spi::IndicatorSignal;

/// Stock-to-Flow output.
#[derive(Debug, Clone)]
pub struct StockToFlowOutput {
    /// Raw S2F ratio values.
    pub s2f_ratio: Vec<f64>,
    /// Modeled price based on S2F (log-linear model).
    pub model_price: Vec<f64>,
    /// Deviation from model price (percentage).
    pub deviation: Vec<f64>,
}

/// Stock-to-Flow signal interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StockToFlowSignal {
    /// Price significantly below model - undervalued.
    SignificantlyUndervalued,
    /// Price below model - undervalued.
    Undervalued,
    /// Price near model - fair value.
    FairValue,
    /// Price above model - overvalued.
    Overvalued,
    /// Price significantly above model - extremely overvalued.
    SignificantlyOvervalued,
}

/// Stock-to-Flow Model (S2F) - IND-275
///
/// A scarcity model comparing existing supply (stock) to new supply (flow).
///
/// # Formula
/// ```text
/// S2F = Stock / Flow
/// Model Price = exp(a + b * ln(S2F))  // Log-linear relationship
/// Deviation = (Actual Price - Model Price) / Model Price * 100
/// ```
///
/// # Example
/// ```
/// use indicator_core::crypto::StockToFlow;
///
/// let s2f = StockToFlow::new(14.6, 3.3); // Bitcoin model parameters
/// let stock = vec![19_000_000.0; 10]; // ~19M BTC
/// let flow = vec![328_500.0; 10]; // ~328.5K new BTC per year post-halving
/// let prices = vec![50000.0; 10];
/// let output = s2f.calculate(&stock, &flow, &prices);
/// ```
#[derive(Debug, Clone)]
pub struct StockToFlow {
    /// Coefficient 'a' in model: ln(price) = a + b * ln(S2F).
    model_intercept: f64,
    /// Coefficient 'b' in model (slope).
    model_slope: f64,
    /// Undervalued threshold (negative deviation %).
    undervalued_threshold: f64,
    /// Overvalued threshold (positive deviation %).
    overvalued_threshold: f64,
}

impl StockToFlow {
    /// Create a new Stock-to-Flow indicator with model parameters.
    ///
    /// Default Bitcoin model: intercept ~14.6, slope ~3.3 (PlanB's original model).
    pub fn new(model_intercept: f64, model_slope: f64) -> Self {
        Self {
            model_intercept,
            model_slope,
            undervalued_threshold: -30.0,
            overvalued_threshold: 50.0,
        }
    }

    /// Create with custom deviation thresholds.
    pub fn with_thresholds(
        model_intercept: f64,
        model_slope: f64,
        undervalued_threshold: f64,
        overvalued_threshold: f64,
    ) -> Self {
        Self {
            model_intercept,
            model_slope,
            undervalued_threshold,
            overvalued_threshold,
        }
    }

    /// Calculate S2F metrics from stock, flow, and price series.
    pub fn calculate(&self, stock: &[f64], flow: &[f64], prices: &[f64]) -> StockToFlowOutput {
        let n = stock.len().min(flow.len()).min(prices.len());

        if n == 0 {
            return StockToFlowOutput {
                s2f_ratio: vec![],
                model_price: vec![],
                deviation: vec![],
            };
        }

        let mut s2f_ratio = vec![f64::NAN; n];
        let mut model_price = vec![f64::NAN; n];
        let mut deviation = vec![f64::NAN; n];

        for i in 0..n {
            if flow[i] > 0.0 && stock[i] > 0.0 {
                // Calculate S2F ratio
                let s2f = stock[i] / flow[i];
                s2f_ratio[i] = s2f;

                // Calculate model price using log-linear relationship
                // ln(price) = a + b * ln(S2F)
                // price = exp(a + b * ln(S2F))
                if s2f > 0.0 {
                    let ln_model = self.model_intercept + self.model_slope * s2f.ln();
                    let mp = ln_model.exp();
                    model_price[i] = mp;

                    // Calculate deviation from model
                    if mp > 0.0 && prices[i] > 0.0 {
                        deviation[i] = (prices[i] - mp) / mp * 100.0;
                    }
                }
            }
        }

        StockToFlowOutput {
            s2f_ratio,
            model_price,
            deviation,
        }
    }

    /// Calculate from S2F ratio directly (when stock/flow not available).
    pub fn calculate_from_ratio(&self, s2f_ratio: &[f64], prices: &[f64]) -> StockToFlowOutput {
        let n = s2f_ratio.len().min(prices.len());

        if n == 0 {
            return StockToFlowOutput {
                s2f_ratio: vec![],
                model_price: vec![],
                deviation: vec![],
            };
        }

        let mut model_price = vec![f64::NAN; n];
        let mut deviation = vec![f64::NAN; n];

        for i in 0..n {
            let s2f = s2f_ratio[i];
            if s2f > 0.0 && !s2f.is_nan() {
                let ln_model = self.model_intercept + self.model_slope * s2f.ln();
                let mp = ln_model.exp();
                model_price[i] = mp;

                if mp > 0.0 && prices[i] > 0.0 {
                    deviation[i] = (prices[i] - mp) / mp * 100.0;
                }
            }
        }

        StockToFlowOutput {
            s2f_ratio: s2f_ratio.to_vec(),
            model_price,
            deviation,
        }
    }

    /// Get signal interpretation for a deviation value.
    pub fn interpret(&self, deviation: f64) -> StockToFlowSignal {
        if deviation.is_nan() {
            StockToFlowSignal::FairValue
        } else if deviation < self.undervalued_threshold * 1.5 {
            StockToFlowSignal::SignificantlyUndervalued
        } else if deviation < self.undervalued_threshold {
            StockToFlowSignal::Undervalued
        } else if deviation > self.overvalued_threshold * 1.5 {
            StockToFlowSignal::SignificantlyOvervalued
        } else if deviation > self.overvalued_threshold {
            StockToFlowSignal::Overvalued
        } else {
            StockToFlowSignal::FairValue
        }
    }

    /// Convert S2F signal to trading signal.
    pub fn to_indicator_signal(&self, signal: StockToFlowSignal) -> IndicatorSignal {
        match signal {
            StockToFlowSignal::SignificantlyUndervalued => IndicatorSignal::Bullish,
            StockToFlowSignal::Undervalued => IndicatorSignal::Bullish,
            StockToFlowSignal::FairValue => IndicatorSignal::Neutral,
            StockToFlowSignal::Overvalued => IndicatorSignal::Bearish,
            StockToFlowSignal::SignificantlyOvervalued => IndicatorSignal::Bearish,
        }
    }

    /// Calculate years to next halving impact on model.
    pub fn years_until_halving(current_s2f: f64, target_s2f: f64) -> f64 {
        if target_s2f <= current_s2f {
            return 0.0;
        }
        // Assuming S2F roughly doubles every 4 years (halving cycle)
        let ratio = target_s2f / current_s2f;
        ratio.log2() * 4.0
    }
}

impl Default for StockToFlow {
    fn default() -> Self {
        // PlanB's original Bitcoin S2F model parameters
        Self::new(14.6, 3.3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s2f_basic() {
        let s2f = StockToFlow::default();
        // Bitcoin-like data
        let stock = vec![19_000_000.0; 10]; // ~19M coins
        let flow = vec![328_500.0; 10]; // ~328.5K new per year
        let prices = vec![50_000.0; 10];

        let output = s2f.calculate(&stock, &flow, &prices);

        assert_eq!(output.s2f_ratio.len(), 10);
        // S2F should be ~57.8 (19M / 328.5K)
        assert!((output.s2f_ratio[0] - 57.8).abs() < 1.0);
    }

    #[test]
    fn test_s2f_model_price() {
        let s2f = StockToFlow::default();
        let s2f_ratios = vec![56.0, 58.0, 60.0, 100.0, 120.0];
        let prices = vec![50000.0; 5];

        let output = s2f.calculate_from_ratio(&s2f_ratios, &prices);

        // Model price should increase with S2F
        assert!(output.model_price[4] > output.model_price[0]);
    }

    #[test]
    fn test_s2f_interpretation() {
        let s2f = StockToFlow::default();

        assert_eq!(s2f.interpret(-50.0), StockToFlowSignal::SignificantlyUndervalued);
        assert_eq!(s2f.interpret(-35.0), StockToFlowSignal::Undervalued);
        assert_eq!(s2f.interpret(0.0), StockToFlowSignal::FairValue);
        assert_eq!(s2f.interpret(60.0), StockToFlowSignal::Overvalued);
        assert_eq!(s2f.interpret(100.0), StockToFlowSignal::SignificantlyOvervalued);
    }

    #[test]
    fn test_s2f_zero_flow() {
        let s2f = StockToFlow::default();
        let stock = vec![19_000_000.0, 19_000_000.0];
        let flow = vec![0.0, 328_500.0];
        let prices = vec![50_000.0; 2];

        let output = s2f.calculate(&stock, &flow, &prices);

        assert!(output.s2f_ratio[0].is_nan());
        assert!(!output.s2f_ratio[1].is_nan());
    }

    #[test]
    fn test_s2f_empty_input() {
        let s2f = StockToFlow::default();
        let output = s2f.calculate(&[], &[], &[]);

        assert!(output.s2f_ratio.is_empty());
        assert!(output.model_price.is_empty());
        assert!(output.deviation.is_empty());
    }

    #[test]
    fn test_s2f_signal_conversion() {
        let s2f = StockToFlow::default();

        assert_eq!(
            s2f.to_indicator_signal(StockToFlowSignal::SignificantlyUndervalued),
            IndicatorSignal::Bullish
        );
        assert_eq!(
            s2f.to_indicator_signal(StockToFlowSignal::SignificantlyOvervalued),
            IndicatorSignal::Bearish
        );
        assert_eq!(
            s2f.to_indicator_signal(StockToFlowSignal::FairValue),
            IndicatorSignal::Neutral
        );
    }

    #[test]
    fn test_years_until_halving() {
        // S2F doubles in ~4 years
        let years = StockToFlow::years_until_halving(56.0, 112.0);
        assert!((years - 4.0).abs() < 0.1);

        // S2F quadruples in ~8 years
        let years = StockToFlow::years_until_halving(56.0, 224.0);
        assert!((years - 8.0).abs() < 0.1);
    }
}
