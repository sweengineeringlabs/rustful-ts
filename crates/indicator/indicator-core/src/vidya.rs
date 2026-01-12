//! Variable Index Dynamic Average (VIDYA) implementation.
//!
//! An adaptive moving average using Chande Momentum Oscillator for volatility.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::VIDYAConfig;

/// Variable Index Dynamic Average (VIDYA).
///
/// VIDYA uses the Chande Momentum Oscillator (CMO) to dynamically adjust
/// the smoothing constant. When the market shows strong momentum, VIDYA
/// becomes more responsive. When momentum is weak, it applies more smoothing.
///
/// Formula: VIDYA(i) = alpha * |CMO| * price(i) + (1 - alpha * |CMO|) * VIDYA(i-1)
#[derive(Debug, Clone)]
pub struct VIDYA {
    period: usize,
    cmo_period: usize,
}

impl VIDYA {
    pub fn new(period: usize, cmo_period: usize) -> Self {
        Self { period, cmo_period }
    }

    pub fn from_config(config: VIDYAConfig) -> Self {
        Self {
            period: config.period,
            cmo_period: config.cmo_period,
        }
    }

    /// Calculate VIDYA values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let min_required = self.period.max(self.cmo_period);
        if data.len() < min_required || self.period == 0 || self.cmo_period == 0 {
            return vec![f64::NAN; data.len()];
        }

        let alpha = 2.0 / (self.period as f64 + 1.0);

        // Calculate CMO values
        let cmo_values = self.calculate_cmo(data);

        let mut result = vec![f64::NAN; min_required - 1];

        // Initialize with first valid price
        result.push(data[min_required - 1]);
        let mut vidya = data[min_required - 1];

        for i in min_required..data.len() {
            let cmo_abs = cmo_values[i].abs();
            let sc = alpha * cmo_abs;
            vidya = sc * data[i] + (1.0 - sc) * vidya;
            result.push(vidya);
        }

        result
    }

    /// Calculate Chande Momentum Oscillator values.
    fn calculate_cmo(&self, data: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; data.len()];

        if data.len() < self.cmo_period + 1 {
            return result;
        }

        // Calculate price changes
        let mut gains = vec![0.0; data.len()];
        let mut losses = vec![0.0; data.len()];

        for i in 1..data.len() {
            let change = data[i] - data[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Calculate CMO using rolling sums
        for i in self.cmo_period..data.len() {
            let sum_gains: f64 = gains[(i + 1 - self.cmo_period)..=i].iter().sum();
            let sum_losses: f64 = losses[(i + 1 - self.cmo_period)..=i].iter().sum();

            if sum_gains + sum_losses != 0.0 {
                result[i] = (sum_gains - sum_losses) / (sum_gains + sum_losses);
            }
        }

        result
    }
}

impl Default for VIDYA {
    fn default() -> Self {
        Self::from_config(VIDYAConfig::default())
    }
}

impl TechnicalIndicator for VIDYA {
    fn name(&self) -> &str {
        "VIDYA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.period.max(self.cmo_period);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period.max(self.cmo_period)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vidya() {
        let vidya = VIDYA::new(10, 9);
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let result = vidya.calculate(&data);

        // First 9 values should be NaN
        for i in 0..9 {
            assert!(result[i].is_nan());
        }
        // Subsequent values should be valid
        for i in 9..30 {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_vidya_trending() {
        let vidya = VIDYA::new(10, 9);
        // Strong uptrend - CMO should be high
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 2.0).collect();
        let result = vidya.calculate(&data);

        // In strong trend, VIDYA should follow price relatively closely
        let last_price = data[29];
        let last_vidya = result[29];
        // Should be within reasonable range of price
        assert!(last_vidya > 100.0);
        assert!(last_vidya < last_price);
    }

    #[test]
    fn test_vidya_insufficient_data() {
        let vidya = VIDYA::new(10, 9);
        let data = vec![1.0, 2.0, 3.0];
        let result = vidya.calculate(&data);

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_vidya_default() {
        let vidya = VIDYA::default();
        assert_eq!(vidya.period, 14);
        assert_eq!(vidya.cmo_period, 9);
    }

    #[test]
    fn test_vidya_technical_indicator_trait() {
        let vidya = VIDYA::new(10, 9);
        assert_eq!(vidya.name(), "VIDYA");
        assert_eq!(vidya.min_periods(), 10);
    }

    #[test]
    fn test_cmo_calculation() {
        let vidya = VIDYA::new(10, 5);
        // Constant gains
        let data = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        let cmo = vidya.calculate_cmo(&data);

        // CMO should be 1.0 (all gains, no losses)
        assert!((cmo[5] - 1.0).abs() < 1e-10);
    }
}
