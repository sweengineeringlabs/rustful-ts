//! SOPR (Spent Output Profit Ratio) - IND-087
//!
//! Measures the profit/loss of spent outputs on the blockchain.
//!
//! Formula: SOPR = Spent Value / Realized Value of spent outputs
//!
//! Interpretation:
//! - SOPR > 1: Spent outputs are in profit (profit taking)
//! - SOPR < 1: Spent outputs are at loss (capitulation)
//! - SOPR = 1: Break-even point (key support/resistance)

use indicator_spi::IndicatorSignal;

/// SOPR output.
#[derive(Debug, Clone)]
pub struct SOPROutput {
    /// Raw SOPR values.
    pub sopr: Vec<f64>,
    /// Smoothed SOPR (SMA).
    pub sopr_smoothed: Vec<f64>,
    /// Adjusted SOPR (excluding outputs < 1 hour old).
    pub sopr_adjusted: Vec<f64>,
}

/// SOPR signal interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SOPRSignal {
    /// Heavy profit taking (SOPR > 1.05).
    HeavyProfitTaking,
    /// Moderate profit taking (1.0 < SOPR <= 1.05).
    ProfitTaking,
    /// Break-even zone (SOPR ≈ 1.0).
    BreakEven,
    /// Moderate loss realization (0.95 <= SOPR < 1.0).
    LossRealization,
    /// Heavy capitulation (SOPR < 0.95).
    Capitulation,
}

/// SOPR (Spent Output Profit Ratio) - IND-087
///
/// An on-chain metric that measures whether coins being spent are in profit or loss.
///
/// # Formula
/// ```text
/// SOPR = Σ(spent_value) / Σ(realized_value_at_creation)
/// ```
///
/// # Example
/// ```
/// use indicator_core::crypto::SOPR;
///
/// let sopr = SOPR::new(7);
/// let spent_values = vec![100.0, 105.0, 98.0];
/// let realized_values = vec![100.0, 100.0, 100.0];
/// let output = sopr.calculate(&spent_values, &realized_values);
/// ```
#[derive(Debug, Clone)]
pub struct SOPR {
    /// Smoothing period for SMA.
    smoothing_period: usize,
}

impl SOPR {
    /// Create a new SOPR indicator.
    pub fn new(smoothing_period: usize) -> Self {
        Self { smoothing_period }
    }

    /// Calculate SOPR from spent value and realized value series.
    ///
    /// # Arguments
    /// * `spent_values` - Value of outputs when spent (current price)
    /// * `realized_values` - Value of outputs when created (cost basis)
    /// * `adjusted_spent` - Optional: spent values excluding short-term (<1h) outputs
    /// * `adjusted_realized` - Optional: realized values for adjusted calculation
    pub fn calculate(&self, spent_values: &[f64], realized_values: &[f64]) -> SOPROutput {
        self.calculate_with_adjusted(spent_values, realized_values, None, None)
    }

    /// Calculate SOPR with optional adjusted values.
    pub fn calculate_with_adjusted(
        &self,
        spent_values: &[f64],
        realized_values: &[f64],
        adjusted_spent: Option<&[f64]>,
        adjusted_realized: Option<&[f64]>,
    ) -> SOPROutput {
        let n = spent_values.len().min(realized_values.len());

        if n == 0 {
            return SOPROutput {
                sopr: vec![],
                sopr_smoothed: vec![],
                sopr_adjusted: vec![],
            };
        }

        // Calculate raw SOPR
        let sopr: Vec<f64> = (0..n)
            .map(|i| {
                if realized_values[i] > 0.0 {
                    spent_values[i] / realized_values[i]
                } else {
                    f64::NAN
                }
            })
            .collect();

        // Calculate smoothed SOPR
        let sopr_smoothed = self.calculate_sma(&sopr);

        // Calculate adjusted SOPR if provided
        let sopr_adjusted = match (adjusted_spent, adjusted_realized) {
            (Some(adj_s), Some(adj_r)) => {
                let adj_n = adj_s.len().min(adj_r.len()).min(n);
                (0..n)
                    .map(|i| {
                        if i < adj_n && adj_r[i] > 0.0 {
                            adj_s[i] / adj_r[i]
                        } else {
                            f64::NAN
                        }
                    })
                    .collect()
            }
            _ => sopr.clone(),
        };

        SOPROutput {
            sopr,
            sopr_smoothed,
            sopr_adjusted,
        }
    }

    /// Calculate SMA.
    fn calculate_sma(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < self.smoothing_period {
            return result;
        }

        for i in (self.smoothing_period - 1)..n {
            let start = i + 1 - self.smoothing_period;
            let valid: Vec<f64> = data[start..=i]
                .iter()
                .filter(|x| !x.is_nan())
                .copied()
                .collect();

            if !valid.is_empty() {
                result[i] = valid.iter().sum::<f64>() / valid.len() as f64;
            }
        }

        result
    }

    /// Get signal interpretation for a single SOPR value.
    pub fn interpret(&self, sopr_value: f64) -> SOPRSignal {
        if sopr_value.is_nan() {
            SOPRSignal::BreakEven
        } else if sopr_value > 1.05 {
            SOPRSignal::HeavyProfitTaking
        } else if sopr_value > 1.0 {
            SOPRSignal::ProfitTaking
        } else if sopr_value >= 0.995 {
            SOPRSignal::BreakEven
        } else if sopr_value >= 0.95 {
            SOPRSignal::LossRealization
        } else {
            SOPRSignal::Capitulation
        }
    }

    /// Convert SOPR signal to trading signal.
    ///
    /// In a bull market: SOPR < 1 can be a buy opportunity (capitulation)
    /// In a bear market: SOPR > 1 can indicate local tops
    pub fn to_indicator_signal(&self, sopr_signal: SOPRSignal) -> IndicatorSignal {
        match sopr_signal {
            SOPRSignal::Capitulation => IndicatorSignal::Bullish,
            SOPRSignal::LossRealization => IndicatorSignal::Bullish,
            SOPRSignal::BreakEven => IndicatorSignal::Neutral,
            SOPRSignal::ProfitTaking => IndicatorSignal::Neutral,
            SOPRSignal::HeavyProfitTaking => IndicatorSignal::Bearish,
        }
    }

    /// Detect SOPR crossing the 1.0 level.
    pub fn detect_breakeven_cross(&self, output: &SOPROutput) -> Vec<i32> {
        let n = output.sopr_smoothed.len();
        let mut crosses = vec![0; n];

        for i in 1..n {
            let prev = output.sopr_smoothed[i - 1];
            let curr = output.sopr_smoothed[i];

            if prev.is_nan() || curr.is_nan() {
                continue;
            }

            // Cross above 1.0 = bullish (holders moving to profit)
            if prev < 1.0 && curr >= 1.0 {
                crosses[i] = 1;
            }
            // Cross below 1.0 = bearish (holders moving to loss)
            else if prev >= 1.0 && curr < 1.0 {
                crosses[i] = -1;
            }
        }

        crosses
    }
}

impl Default for SOPR {
    fn default() -> Self {
        Self::new(7) // 7-day smoothing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sopr_basic() {
        let sopr = SOPR::new(3);
        let spent = vec![100.0, 105.0, 98.0, 110.0, 95.0];
        let realized = vec![100.0, 100.0, 100.0, 100.0, 100.0];

        let output = sopr.calculate(&spent, &realized);

        assert_eq!(output.sopr.len(), 5);
        assert!((output.sopr[0] - 1.0).abs() < 1e-10);
        assert!((output.sopr[1] - 1.05).abs() < 1e-10);
        assert!((output.sopr[2] - 0.98).abs() < 1e-10);
    }

    #[test]
    fn test_sopr_smoothing() {
        let sopr = SOPR::new(3);
        let spent = vec![100.0, 105.0, 98.0, 110.0, 95.0];
        let realized = vec![100.0, 100.0, 100.0, 100.0, 100.0];

        let output = sopr.calculate(&spent, &realized);

        // Smoothed should be average of last 3
        // At index 2: (1.0 + 1.05 + 0.98) / 3 = 1.01
        assert!(!output.sopr_smoothed[2].is_nan());
        let expected = (1.0 + 1.05 + 0.98) / 3.0;
        assert!((output.sopr_smoothed[2] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_sopr_interpretation() {
        let sopr = SOPR::default();

        assert_eq!(sopr.interpret(1.10), SOPRSignal::HeavyProfitTaking);
        assert_eq!(sopr.interpret(1.02), SOPRSignal::ProfitTaking);
        assert_eq!(sopr.interpret(1.0), SOPRSignal::BreakEven);
        assert_eq!(sopr.interpret(0.97), SOPRSignal::LossRealization);
        assert_eq!(sopr.interpret(0.90), SOPRSignal::Capitulation);
    }

    #[test]
    fn test_sopr_breakeven_cross() {
        let sopr = SOPR::new(1); // No smoothing for clear test
        let spent = vec![95.0, 98.0, 102.0, 105.0, 98.0, 95.0];
        let realized = vec![100.0; 6];

        let output = sopr.calculate(&spent, &realized);
        let crosses = sopr.detect_breakeven_cross(&output);

        // Cross above 1.0 at index 2 (0.98 -> 1.02)
        assert_eq!(crosses[2], 1);
        // Cross below 1.0 at index 4 (1.05 -> 0.98)
        assert_eq!(crosses[4], -1);
    }

    #[test]
    fn test_sopr_zero_realized() {
        let sopr = SOPR::default();
        let spent = vec![100.0, 100.0, 100.0];
        let realized = vec![100.0, 0.0, 100.0];

        let output = sopr.calculate(&spent, &realized);

        assert!(!output.sopr[0].is_nan());
        assert!(output.sopr[1].is_nan());
        assert!(!output.sopr[2].is_nan());
    }

    #[test]
    fn test_sopr_empty_input() {
        let sopr = SOPR::default();
        let output = sopr.calculate(&[], &[]);

        assert!(output.sopr.is_empty());
    }

    #[test]
    fn test_sopr_signal_conversion() {
        let sopr = SOPR::default();

        assert_eq!(
            sopr.to_indicator_signal(SOPRSignal::Capitulation),
            IndicatorSignal::Bullish
        );
        assert_eq!(
            sopr.to_indicator_signal(SOPRSignal::HeavyProfitTaking),
            IndicatorSignal::Bearish
        );
    }
}
