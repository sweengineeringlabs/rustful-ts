//! VolumeDeltaPercentage (IND-219) - Delta as percentage of volume
//!
//! Normalizes delta by expressing it as a percentage of total volume,
//! making it easier to compare across different volume levels.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator, SignalIndicator, IndicatorSignal,
};

/// Volume Delta Percentage Output
#[derive(Debug, Clone)]
pub struct VolumeDeltaPercentageOutput {
    /// Delta percentage (-100 to 100)
    pub delta_percentage: Vec<f64>,
    /// Smoothed delta percentage (EMA)
    pub smoothed: Vec<f64>,
    /// Histogram (delta percentage - smoothed)
    pub histogram: Vec<f64>,
}

/// Volume Delta Percentage Configuration
#[derive(Debug, Clone)]
pub struct VolumeDeltaPercentageConfig {
    /// EMA smoothing period
    pub smoothing_period: usize,
    /// Overbought threshold
    pub overbought: f64,
    /// Oversold threshold
    pub oversold: f64,
}

impl Default for VolumeDeltaPercentageConfig {
    fn default() -> Self {
        Self {
            smoothing_period: 14,
            overbought: 40.0,
            oversold: -40.0,
        }
    }
}

/// VolumeDeltaPercentage (IND-219)
///
/// Calculates delta as a percentage of total volume, providing a
/// normalized view of buying/selling pressure.
///
/// Formula:
/// - Delta% = (Buy Volume - Sell Volume) / Total Volume * 100
/// - Range: -100% (all selling) to +100% (all buying)
///
/// Interpretation:
/// - Values > 0: Net buying pressure
/// - Values < 0: Net selling pressure
/// - Extreme values (>40 or <-40): Potential exhaustion
/// - Histogram crosses signal line: Momentum shifts
#[derive(Debug, Clone)]
pub struct VolumeDeltaPercentage {
    config: VolumeDeltaPercentageConfig,
}

impl VolumeDeltaPercentage {
    pub fn new(config: VolumeDeltaPercentageConfig) -> Result<Self> {
        if config.smoothing_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if config.overbought <= config.oversold {
            return Err(IndicatorError::InvalidParameter {
                name: "overbought".to_string(),
                reason: "must be greater than oversold".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self {
            config: VolumeDeltaPercentageConfig::default(),
        }
    }

    /// Calculate volume delta percentage with full output
    pub fn calculate_full(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> VolumeDeltaPercentageOutput {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut delta_percentage = vec![0.0; n];
        let mut smoothed = vec![0.0; n];
        let mut histogram = vec![0.0; n];

        if n == 0 {
            return VolumeDeltaPercentageOutput {
                delta_percentage,
                smoothed,
                histogram,
            };
        }

        // Calculate delta percentage for each bar
        for i in 0..n {
            let range = high[i] - low[i];
            if range > 0.0 && volume[i] > 0.0 {
                // Position: 0 = low, 1 = high
                let position = (close[i] - low[i]) / range;
                // Delta percentage: -100 to +100
                delta_percentage[i] = (2.0 * position - 1.0) * 100.0;
            }
        }

        // Calculate EMA smoothing
        let alpha = 2.0 / (self.config.smoothing_period as f64 + 1.0);
        smoothed[0] = delta_percentage[0];
        for i in 1..n {
            smoothed[i] = alpha * delta_percentage[i] + (1.0 - alpha) * smoothed[i - 1];
        }

        // Calculate histogram
        for i in 0..n {
            histogram[i] = delta_percentage[i] - smoothed[i];
        }

        VolumeDeltaPercentageOutput {
            delta_percentage,
            smoothed,
            histogram,
        }
    }

    /// Calculate delta percentage values only
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        self.calculate_full(high, low, close, volume).delta_percentage
    }

    /// Check if current value is in overbought zone
    pub fn is_overbought(&self, value: f64) -> bool {
        value >= self.config.overbought
    }

    /// Check if current value is in oversold zone
    pub fn is_oversold(&self, value: f64) -> bool {
        value <= self.config.oversold
    }
}

impl TechnicalIndicator for VolumeDeltaPercentage {
    fn name(&self) -> &str {
        "Volume Delta Percentage"
    }

    fn min_periods(&self) -> usize {
        self.config.smoothing_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.smoothing_period {
            return Err(IndicatorError::InsufficientData {
                required: self.config.smoothing_period,
                got: data.close.len(),
            });
        }

        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            output.delta_percentage,
            output.smoothed,
            output.histogram,
        ))
    }
}

impl SignalIndicator for VolumeDeltaPercentage {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);

        if let (Some(&last_dp), Some(&last_smooth)) = (
            output.delta_percentage.last(),
            output.smoothed.last(),
        ) {
            // Generate signal based on delta percentage and histogram
            if last_dp > self.config.overbought {
                return Ok(IndicatorSignal::Bullish);
            } else if last_dp < self.config.oversold {
                return Ok(IndicatorSignal::Bearish);
            } else if last_dp > last_smooth && last_dp > 0.0 {
                return Ok(IndicatorSignal::Bullish);
            } else if last_dp < last_smooth && last_dp < 0.0 {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);

        Ok(output
            .delta_percentage
            .iter()
            .zip(output.smoothed.iter())
            .map(|(&dp, &smooth)| {
                if dp > self.config.overbought {
                    IndicatorSignal::Bullish
                } else if dp < self.config.oversold {
                    IndicatorSignal::Bearish
                } else if dp > smooth && dp > 0.0 {
                    IndicatorSignal::Bullish
                } else if dp < smooth && dp < 0.0 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect())
    }
}

impl Default for VolumeDeltaPercentage {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // Uptrending data with closes near highs
        let high = vec![105.0, 107.0, 109.0, 111.0, 113.0, 115.0, 117.0, 119.0, 121.0, 123.0,
                       125.0, 127.0, 129.0, 131.0, 133.0, 135.0, 137.0, 139.0, 141.0, 143.0];
        let low = vec![100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0,
                      120.0, 122.0, 124.0, 126.0, 128.0, 130.0, 132.0, 134.0, 136.0, 138.0];
        let close = vec![104.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0, 120.0, 122.0,
                        124.0, 126.0, 128.0, 130.0, 132.0, 134.0, 136.0, 138.0, 140.0, 142.0];
        let volume = vec![1000.0; 20];
        (high, low, close, volume)
    }

    #[test]
    fn test_volume_delta_percentage_basic() {
        let vdp = VolumeDeltaPercentage::default_config();
        let (high, low, close, volume) = make_test_data();

        let result = vdp.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 20);

        // Closes are at 80% of range, so delta% should be 60%
        // (2 * 0.8 - 1) * 100 = 60
        for &dp in &result {
            assert!((dp - 60.0).abs() < 1.0, "Expected ~60%, got {}", dp);
        }
    }

    #[test]
    fn test_volume_delta_percentage_full_output() {
        let vdp = VolumeDeltaPercentage::default_config();
        let (high, low, close, volume) = make_test_data();

        let output = vdp.calculate_full(&high, &low, &close, &volume);

        assert_eq!(output.delta_percentage.len(), 20);
        assert_eq!(output.smoothed.len(), 20);
        assert_eq!(output.histogram.len(), 20);

        // Smoothed should converge to delta_percentage when constant
        assert!((output.smoothed[19] - output.delta_percentage[19]).abs() < 5.0);
    }

    #[test]
    fn test_volume_delta_percentage_range() {
        let vdp = VolumeDeltaPercentage::default_config();

        // Close at high (100% buying)
        let high = vec![110.0];
        let low = vec![100.0];
        let close = vec![110.0];
        let volume = vec![1000.0];

        let result = vdp.calculate(&high, &low, &close, &volume);
        assert!((result[0] - 100.0).abs() < 1e-10, "Close at high should be 100%");

        // Close at low (100% selling)
        let close = vec![100.0];
        let result = vdp.calculate(&high, &low, &close, &volume);
        assert!((result[0] - (-100.0)).abs() < 1e-10, "Close at low should be -100%");

        // Close at middle (neutral)
        let close = vec![105.0];
        let result = vdp.calculate(&high, &low, &close, &volume);
        assert!(result[0].abs() < 1e-10, "Close at middle should be 0%");
    }

    #[test]
    fn test_volume_delta_percentage_overbought_oversold() {
        let config = VolumeDeltaPercentageConfig {
            smoothing_period: 14,
            overbought: 50.0,
            oversold: -50.0,
        };
        let vdp = VolumeDeltaPercentage::new(config).unwrap();

        assert!(vdp.is_overbought(60.0));
        assert!(!vdp.is_overbought(40.0));
        assert!(vdp.is_oversold(-60.0));
        assert!(!vdp.is_oversold(-40.0));
    }

    #[test]
    fn test_volume_delta_percentage_invalid_config() {
        let config = VolumeDeltaPercentageConfig {
            smoothing_period: 0,
            overbought: 40.0,
            oversold: -40.0,
        };
        assert!(VolumeDeltaPercentage::new(config).is_err());

        let config = VolumeDeltaPercentageConfig {
            smoothing_period: 14,
            overbought: -50.0, // Less than oversold
            oversold: -40.0,
        };
        assert!(VolumeDeltaPercentage::new(config).is_err());
    }

    #[test]
    fn test_volume_delta_percentage_signal() {
        let vdp = VolumeDeltaPercentage::default_config();
        let (high, low, close, volume) = make_test_data();

        let data = OHLCVSeries {
            open: vec![100.0; 20],
            high,
            low,
            close,
            volume,
        };

        let signal = vdp.signal(&data).unwrap();
        // With 60% delta percentage (above overbought), should be bullish
        assert!(matches!(signal, IndicatorSignal::Bullish));
    }

    #[test]
    fn test_volume_delta_percentage_zero_range() {
        let vdp = VolumeDeltaPercentage::default_config();
        let high = vec![100.0; 10];
        let low = vec![100.0; 10];
        let close = vec![100.0; 10];
        let volume = vec![1000.0; 10];

        let result = vdp.calculate(&high, &low, &close, &volume);

        // Zero range should produce zero delta percentage
        for &dp in &result {
            assert_eq!(dp, 0.0);
        }
    }
}
