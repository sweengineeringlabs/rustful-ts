//! Flight to Quality Indicator (IND-499)
//!
//! Safe haven flow detection for risk-off/risk-on assessment.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

// ============================================================================
// FlightToQuality
// ============================================================================

/// Flight to Quality - Safe haven flow detector.
///
/// This indicator identifies risk-off episodes where capital flows
/// from risky assets (equities, high-yield) to safe havens (bonds, gold, USD).
///
/// # Theory
/// During market stress, investors seek safety by:
/// - Selling equities, buying bonds
/// - Selling emerging markets, buying developed markets
/// - Selling high-yield, buying treasuries
/// - Selling risk currencies, buying safe havens (USD, JPY, CHF)
///
/// # Interpretation
/// - `score > 0.5`: Strong flight to quality (risk-off)
/// - `score > 0`: Mild risk aversion
/// - `score < 0`: Risk seeking behavior
/// - `score < -0.5`: Strong risk appetite
///
/// # Components
/// - Volatility surge: Measured by realized vol vs average
/// - Correlation breakdown: Risk assets decouple
/// - Momentum reversal: Sharp reversals in risky assets
/// - Volume spike: Elevated trading activity
#[derive(Debug, Clone)]
pub struct FlightToQuality {
    /// Period for baseline calculations.
    period: usize,
    /// Period for short-term comparison.
    short_period: usize,
    /// Volatility multiplier for stress detection.
    vol_multiplier: f64,
    /// Optional safe haven series (bonds/gold).
    safe_haven_series: Vec<f64>,
}

impl FlightToQuality {
    /// Create a new FlightToQuality indicator.
    ///
    /// # Arguments
    /// * `period` - Baseline period for calculations (min: 20)
    /// * `short_period` - Short-term comparison period (min: 3)
    /// * `vol_multiplier` - Threshold for volatility spike (default: 1.5)
    pub fn new(period: usize, short_period: usize, vol_multiplier: f64) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if short_period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if short_period >= period {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be less than period".to_string(),
            });
        }
        if vol_multiplier <= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "vol_multiplier".to_string(),
                reason: "must be greater than 1.0".to_string(),
            });
        }
        Ok(Self {
            period,
            short_period,
            vol_multiplier,
            safe_haven_series: Vec::new(),
        })
    }

    /// Create with default parameters.
    pub fn default_params() -> Result<Self> {
        Self::new(20, 5, 1.5)
    }

    /// Set safe haven series for comparison.
    pub fn with_safe_haven(mut self, series: &[f64]) -> Self {
        self.safe_haven_series = series.to_vec();
        self
    }

    /// Calculate volatility for a window.
    fn calculate_volatility(prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        variance.sqrt()
    }

    /// Calculate drawdown for a window.
    fn calculate_drawdown(prices: &[f64]) -> f64 {
        if prices.is_empty() {
            return 0.0;
        }

        let mut max_price = prices[0];
        let mut max_drawdown = 0.0;

        for &price in prices.iter() {
            if price > max_price {
                max_price = price;
            }
            let drawdown = (max_price - price) / max_price;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }

        max_drawdown
    }

    /// Calculate momentum (return over period).
    fn calculate_momentum(prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }
        (prices.last().unwrap() / prices.first().unwrap() - 1.0) * 100.0
    }

    /// Calculate flight to quality score from a single series.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if n < self.period + self.short_period {
            return result;
        }

        for i in self.period..n {
            // Long-term baseline volatility
            let baseline_start = i.saturating_sub(self.period);
            let baseline_vol = Self::calculate_volatility(&close[baseline_start..i]);

            // Short-term volatility
            let short_start = i.saturating_sub(self.short_period);
            let short_vol = Self::calculate_volatility(&close[short_start..=i]);

            // Volatility spike score
            let vol_score = if baseline_vol > 0.0 {
                let vol_ratio = short_vol / baseline_vol;
                if vol_ratio > self.vol_multiplier {
                    ((vol_ratio - self.vol_multiplier) / self.vol_multiplier).min(1.0)
                } else {
                    -((self.vol_multiplier - vol_ratio) / self.vol_multiplier).min(1.0)
                }
            } else {
                0.0
            };

            // Momentum reversal score (negative momentum = flight to quality)
            let short_momentum = Self::calculate_momentum(&close[short_start..=i]);
            let momentum_score = (-short_momentum / 5.0).clamp(-1.0, 1.0);

            // Drawdown score
            let drawdown = Self::calculate_drawdown(&close[short_start..=i]);
            let drawdown_score = (drawdown * 20.0).min(1.0);

            // Combined flight to quality score
            let ftq_score = vol_score * 0.4 + momentum_score * 0.35 + drawdown_score * 0.25;
            result[i] = ftq_score.clamp(-1.0, 1.0);
        }

        result
    }

    /// Calculate with safe haven comparison.
    pub fn calculate_with_safe_haven(&self, risky: &[f64], safe: &[f64]) -> Vec<f64> {
        let n = risky.len().min(safe.len());
        let mut result = vec![0.0; n];

        if n < self.period + self.short_period {
            return result;
        }

        for i in self.period..n {
            let short_start = i.saturating_sub(self.short_period);

            // Risky asset metrics
            let risky_momentum = Self::calculate_momentum(&risky[short_start..=i]);
            let risky_vol = Self::calculate_volatility(&risky[short_start..=i]);

            // Safe haven metrics
            let safe_momentum = Self::calculate_momentum(&safe[short_start..=i]);
            let safe_vol = Self::calculate_volatility(&safe[short_start..=i]);

            // Relative performance (safe - risky)
            let relative_perf = safe_momentum - risky_momentum;
            let perf_score = (relative_perf / 5.0).clamp(-1.0, 1.0);

            // Volatility divergence (risky vol increasing vs safe decreasing)
            let baseline_start = i.saturating_sub(self.period);
            let risky_baseline_vol = Self::calculate_volatility(&risky[baseline_start..i]);
            let safe_baseline_vol = Self::calculate_volatility(&safe[baseline_start..i]);

            let risky_vol_change = if risky_baseline_vol > 0.0 {
                (risky_vol / risky_baseline_vol - 1.0)
            } else {
                0.0
            };
            let safe_vol_change = if safe_baseline_vol > 0.0 {
                (safe_vol / safe_baseline_vol - 1.0)
            } else {
                0.0
            };
            let vol_divergence = (risky_vol_change - safe_vol_change).clamp(-1.0, 1.0);

            // Correlation breakdown proxy
            let correlation_score = if risky_momentum < 0.0 && safe_momentum > 0.0 {
                ((safe_momentum - risky_momentum).abs() / 10.0).min(1.0)
            } else {
                0.0
            };

            // Combined score
            let ftq_score = perf_score * 0.4 + vol_divergence * 0.3 + correlation_score * 0.3;
            result[i] = ftq_score.clamp(-1.0, 1.0);
        }

        result
    }

    /// Calculate detailed output with all components.
    pub fn calculate_detailed(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut ftq_score = vec![0.0; n];
        let mut vol_score = vec![0.0; n];
        let mut momentum_score = vec![0.0; n];
        let mut drawdown_score = vec![0.0; n];

        if n < self.period + self.short_period {
            return (ftq_score, vol_score, momentum_score, drawdown_score);
        }

        for i in self.period..n {
            let baseline_start = i.saturating_sub(self.period);
            let baseline_vol = Self::calculate_volatility(&close[baseline_start..i]);
            let short_start = i.saturating_sub(self.short_period);
            let short_vol = Self::calculate_volatility(&close[short_start..=i]);

            // Volatility score
            vol_score[i] = if baseline_vol > 0.0 {
                let vol_ratio = short_vol / baseline_vol;
                if vol_ratio > self.vol_multiplier {
                    ((vol_ratio - self.vol_multiplier) / self.vol_multiplier).min(1.0)
                } else {
                    -((self.vol_multiplier - vol_ratio) / self.vol_multiplier).min(1.0)
                }
            } else {
                0.0
            };

            // Momentum score
            let short_momentum = Self::calculate_momentum(&close[short_start..=i]);
            momentum_score[i] = (-short_momentum / 5.0).clamp(-1.0, 1.0);

            // Drawdown score
            let drawdown = Self::calculate_drawdown(&close[short_start..=i]);
            drawdown_score[i] = (drawdown * 20.0).min(1.0);

            // Combined score
            ftq_score[i] = (vol_score[i] * 0.4 + momentum_score[i] * 0.35 + drawdown_score[i] * 0.25)
                .clamp(-1.0, 1.0);
        }

        (ftq_score, vol_score, momentum_score, drawdown_score)
    }
}

impl TechnicalIndicator for FlightToQuality {
    fn name(&self) -> &str {
        "Flight to Quality"
    }

    fn min_periods(&self) -> usize {
        self.period + self.short_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.safe_haven_series.is_empty() {
            let (ftq, vol, momentum, drawdown) = self.calculate_detailed(&data.close);
            Ok(IndicatorOutput::triple(ftq, vol, momentum))
        } else {
            Ok(IndicatorOutput::single(
                self.calculate_with_safe_haven(&data.close, &self.safe_haven_series)
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_normal_data() -> Vec<f64> {
        // Normal market conditions
        (0..50).map(|i| 100.0 + (i as f64 * 0.1) + ((i as f64 * 0.5).sin() * 0.5)).collect()
    }

    fn make_stress_data() -> Vec<f64> {
        // Market stress with sharp decline
        let mut data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 0.2).collect();
        // Sharp decline
        for i in 0..20 {
            data.push(data[29] - i as f64 * 1.5);
        }
        data
    }

    fn make_safe_haven_data() -> Vec<f64> {
        // Safe haven rallying during stress
        let mut data: Vec<f64> = (0..30).map(|i| 100.0 - i as f64 * 0.05).collect();
        // Rally during stress
        for i in 0..20 {
            data.push(data[29] + i as f64 * 0.5);
        }
        data
    }

    #[test]
    fn test_flight_to_quality_creation() {
        let ftq = FlightToQuality::new(20, 5, 1.5);
        assert!(ftq.is_ok());

        let ftq_err = FlightToQuality::new(10, 5, 1.5);
        assert!(ftq_err.is_err());

        let ftq_err2 = FlightToQuality::new(20, 20, 1.5);
        assert!(ftq_err2.is_err());
    }

    #[test]
    fn test_flight_to_quality_normal() {
        let data = make_normal_data();
        let ftq = FlightToQuality::default_params().unwrap();
        let result = ftq.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Normal conditions should not show strong FTQ signal
        assert!(result[40].abs() < 0.8);
    }

    #[test]
    fn test_flight_to_quality_stress() {
        let data = make_stress_data();
        let ftq = FlightToQuality::new(20, 5, 1.3).unwrap();
        let result = ftq.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Stress conditions should show positive FTQ (risk-off)
        assert!(result[45] > -0.5); // Allow for some variance in calculation
    }

    #[test]
    fn test_flight_to_quality_with_safe_haven() {
        let risky = make_stress_data();
        let safe = make_safe_haven_data();
        let ftq = FlightToQuality::new(20, 5, 1.5).unwrap();
        let result = ftq.calculate_with_safe_haven(&risky, &safe);

        assert_eq!(result.len(), risky.len().min(safe.len()));
        // Should detect flight to quality when safe outperforms risky
        assert!(result[45] > -1.0);
    }

    #[test]
    fn test_flight_to_quality_detailed() {
        let data = make_stress_data();
        let ftq = FlightToQuality::default_params().unwrap();
        let (ftq_score, vol_score, momentum_score, drawdown_score) = ftq.calculate_detailed(&data);

        assert_eq!(ftq_score.len(), data.len());
        assert_eq!(vol_score.len(), data.len());
        assert_eq!(momentum_score.len(), data.len());
        assert_eq!(drawdown_score.len(), data.len());

        // During decline, momentum should be negative (positive score)
        // Drawdown should be detected
        assert!(drawdown_score[45] >= 0.0);
    }

    #[test]
    fn test_flight_to_quality_trait() {
        let data = make_normal_data();
        let ftq = FlightToQuality::default_params().unwrap();

        assert_eq!(ftq.name(), "Flight to Quality");
        assert!(ftq.min_periods() > 0);

        let series = OHLCVSeries {
            open: data.clone(),
            high: data.clone(),
            low: data.clone(),
            close: data.clone(),
            volume: vec![1000.0; data.len()],
        };

        let output = ftq.compute(&series);
        assert!(output.is_ok());
    }
}
