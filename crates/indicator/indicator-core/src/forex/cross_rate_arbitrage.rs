//! Cross Rate Arbitrage Indicator (IND-313)
//!
//! Triangular arbitrage opportunity detection for forex markets.
//! Identifies deviations from no-arbitrage conditions in currency triangles.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Cross Rate Arbitrage - Triangular arbitrage detector (IND-313)
///
/// This indicator detects triangular arbitrage opportunities by
/// measuring the deviation from no-arbitrage conditions.
///
/// For three currencies A, B, C:
/// The arbitrage-free condition is: (A/B) * (B/C) = (A/C)
///
/// # Interpretation
/// - Values near 0 indicate no arbitrage opportunity
/// - Positive/negative values indicate potential arbitrage (mispricing)
/// - Values are expressed in basis points
///
/// # Example
/// ```ignore
/// use indicator_core::forex::CrossRateArbitrage;
///
/// let arb = CrossRateArbitrage::new(5, 0.5).unwrap();
/// // Using proxy calculation from a single pair
/// let opportunities = arb.calculate(&close);
/// ```
#[derive(Debug, Clone)]
pub struct CrossRateArbitrage {
    /// Smoothing period for the calculation
    period: usize,
    /// Transaction cost threshold in basis points
    cost_threshold: f64,
}

impl CrossRateArbitrage {
    /// Create a new CrossRateArbitrage indicator.
    ///
    /// # Arguments
    /// * `period` - Smoothing period (minimum 2)
    /// * `cost_threshold` - Transaction cost in basis points (minimum 0)
    pub fn new(period: usize, cost_threshold: f64) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if cost_threshold < 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "cost_threshold".to_string(),
                reason: "must be non-negative".to_string(),
            });
        }
        Ok(Self { period, cost_threshold })
    }

    /// Calculate triangular arbitrage proxy from single pair.
    ///
    /// This uses the synthetic cross-rate deviation approach,
    /// measuring the deviation between the direct rate and
    /// its moving average as a proxy for arbitrage opportunity.
    ///
    /// # Arguments
    /// * `close` - Closing prices of the currency pair
    ///
    /// # Returns
    /// Vector of arbitrage opportunity values in basis points
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.period {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);

            // Calculate moving average as "fair value"
            let ma = close[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;

            // Deviation from fair value in basis points
            if ma > 0.0 {
                let deviation_bps = (close[i] - ma) / ma * 10000.0;

                // Apply cost threshold - only report if deviation exceeds costs
                if deviation_bps.abs() > self.cost_threshold {
                    result[i] = deviation_bps;
                }
            }
        }

        result
    }

    /// Calculate arbitrage from three rates (proper triangular arbitrage).
    ///
    /// For currencies A, B, C with rates:
    /// - rate_ab: A/B rate
    /// - rate_bc: B/C rate
    /// - rate_ac: A/C rate
    ///
    /// The arbitrage signal is: (rate_ab * rate_bc) / rate_ac - 1
    ///
    /// # Arguments
    /// * `rate_ab` - A/B exchange rate series
    /// * `rate_bc` - B/C exchange rate series
    /// * `rate_ac` - A/C exchange rate series
    ///
    /// # Returns
    /// Vector of arbitrage opportunities in basis points
    pub fn calculate_triangular(
        &self,
        rate_ab: &[f64],
        rate_bc: &[f64],
        rate_ac: &[f64],
    ) -> Vec<f64> {
        let n = rate_ab.len().min(rate_bc.len()).min(rate_ac.len());
        if n < self.period {
            return vec![0.0; n];
        }

        let mut raw_arb = vec![0.0; n];

        // Calculate raw arbitrage signal
        for i in 0..n {
            if rate_ac[i] > 0.0 {
                // Synthetic rate vs actual rate
                let synthetic = rate_ab[i] * rate_bc[i];
                let deviation = (synthetic / rate_ac[i]) - 1.0;
                raw_arb[i] = deviation * 10000.0; // Convert to basis points
            }
        }

        // Apply smoothing
        let mut result = vec![0.0; n];
        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let avg = raw_arb[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;

            // Apply cost threshold
            if avg.abs() > self.cost_threshold {
                result[i] = avg;
            }
        }

        result
    }

    /// Calculate with extended output including direction and magnitude.
    pub fn calculate_extended(&self, close: &[f64]) -> CrossRateArbitrageOutput {
        let n = close.len();
        if n < self.period {
            return CrossRateArbitrageOutput {
                arbitrage_signal: vec![0.0; n],
                signal_strength: vec![0.0; n],
                direction: vec![ArbitrageDirection::None; n],
                is_profitable: vec![false; n],
            };
        }

        let mut signal = vec![0.0; n];
        let mut strength = vec![0.0; n];
        let mut direction = vec![ArbitrageDirection::None; n];
        let mut profitable = vec![false; n];

        // Calculate rolling standard deviation for normalization
        let mut rolling_std = vec![0.0; n];
        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let returns: Vec<f64> = ((start + 1)..=i)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();

            if !returns.is_empty() {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance = returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                rolling_std[i] = variance.sqrt() * 10000.0; // In basis points
            }
        }

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let ma = close[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;

            if ma > 0.0 {
                let deviation_bps = (close[i] - ma) / ma * 10000.0;

                // Normalize strength using rolling volatility
                let vol = rolling_std[i].max(1.0);
                strength[i] = (deviation_bps.abs() / vol).min(5.0);

                if deviation_bps.abs() > self.cost_threshold {
                    signal[i] = deviation_bps;
                    profitable[i] = true;

                    direction[i] = if deviation_bps > 0.0 {
                        ArbitrageDirection::LongSynthetic
                    } else {
                        ArbitrageDirection::ShortSynthetic
                    };
                }
            }
        }

        CrossRateArbitrageOutput {
            arbitrage_signal: signal,
            signal_strength: strength,
            direction,
            is_profitable: profitable,
        }
    }
}

/// Direction of the arbitrage trade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArbitrageDirection {
    /// No arbitrage opportunity
    None,
    /// Long the synthetic, short the direct
    LongSynthetic,
    /// Short the synthetic, long the direct
    ShortSynthetic,
}

/// Extended output for CrossRateArbitrage indicator.
#[derive(Debug, Clone)]
pub struct CrossRateArbitrageOutput {
    /// Arbitrage signal in basis points
    pub arbitrage_signal: Vec<f64>,
    /// Signal strength (volatility-normalized)
    pub signal_strength: Vec<f64>,
    /// Trade direction
    pub direction: Vec<ArbitrageDirection>,
    /// Whether signal exceeds cost threshold
    pub is_profitable: Vec<bool>,
}

impl TechnicalIndicator for CrossRateArbitrage {
    fn name(&self) -> &str {
        "Cross Rate Arbitrage"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        vec![
            1.1000, 1.1010, 1.1005, 1.1015, 1.1020, 1.1008, 1.1025, 1.1030, 1.1015, 1.1040,
            1.1035, 1.1050, 1.1045, 1.1060, 1.1055, 1.1070, 1.1065, 1.1080, 1.1075, 1.1090,
            1.1085, 1.1100, 1.1095, 1.1110, 1.1105, 1.1120, 1.1115, 1.1130, 1.1125, 1.1140,
        ];
    }

    #[test]
    fn test_cross_rate_arbitrage_new() {
        assert!(CrossRateArbitrage::new(5, 0.5).is_ok());
        assert!(CrossRateArbitrage::new(1, 0.5).is_err()); // period too small
        assert!(CrossRateArbitrage::new(5, -0.5).is_err()); // negative cost
    }

    #[test]
    fn test_cross_rate_arbitrage_calculate() {
        let close = make_test_data();
        let arb = CrossRateArbitrage::new(5, 0.5).unwrap();
        let result = arb.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Results should be reasonably bounded
        for &val in &result[5..] {
            assert!(val.abs() < 1000.0); // Less than 10%
        }
    }

    #[test]
    fn test_cross_rate_arbitrage_triangular() {
        // Create synthetic rates where A/B * B/C should equal A/C
        let rate_ab: Vec<f64> = (0..30).map(|i| 1.10 + (i as f64 * 0.001)).collect();
        let rate_bc: Vec<f64> = (0..30).map(|i| 0.85 + (i as f64 * 0.0005)).collect();
        // Introduce small arbitrage
        let rate_ac: Vec<f64> = rate_ab.iter()
            .zip(rate_bc.iter())
            .map(|(ab, bc)| ab * bc * 1.0001) // Slight deviation
            .collect();

        let arb = CrossRateArbitrage::new(5, 0.5).unwrap();
        let result = arb.calculate_triangular(&rate_ab, &rate_bc, &rate_ac);

        assert_eq!(result.len(), 30);
        // Should detect the small deviation
        assert!(result[20].abs() < 10.0); // Should be around 1 bp
    }

    #[test]
    fn test_cross_rate_arbitrage_extended() {
        let close = make_test_data();
        let arb = CrossRateArbitrage::new(5, 0.5).unwrap();
        let output = arb.calculate_extended(&close);

        assert_eq!(output.arbitrage_signal.len(), close.len());
        assert_eq!(output.signal_strength.len(), close.len());
        assert_eq!(output.direction.len(), close.len());
        assert_eq!(output.is_profitable.len(), close.len());

        // Strength should be non-negative
        for &s in &output.signal_strength {
            assert!(s >= 0.0);
        }
    }

    #[test]
    fn test_cross_rate_arbitrage_cost_threshold() {
        let close = make_test_data();

        let arb_low = CrossRateArbitrage::new(5, 0.1).unwrap();
        let arb_high = CrossRateArbitrage::new(5, 50.0).unwrap();

        let result_low = arb_low.calculate(&close);
        let result_high = arb_high.calculate(&close);

        // Higher threshold should filter more signals
        let non_zero_low = result_low.iter().filter(|&&x| x != 0.0).count();
        let non_zero_high = result_high.iter().filter(|&&x| x != 0.0).count();

        assert!(non_zero_low >= non_zero_high);
    }

    #[test]
    fn test_cross_rate_arbitrage_technical_indicator() {
        let close = make_test_data();
        let arb = CrossRateArbitrage::new(5, 0.5).unwrap();

        assert_eq!(arb.name(), "Cross Rate Arbitrage");
        assert_eq!(arb.min_periods(), 5);

        let data = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x * 1.01).collect(),
            low: close.iter().map(|x| x * 0.99).collect(),
            close: close.clone(),
            volume: vec![1000.0; close.len()],
        };

        let output = arb.compute(&data).unwrap();
        assert!(output.primary.len() == close.len());
    }
}
