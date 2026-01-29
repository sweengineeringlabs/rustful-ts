//! CumulativeDelta (IND-217) - Running delta total
//!
//! Tracks the cumulative sum of delta (buy - sell volume) over time,
//! similar to On-Balance Volume but using delta instead of directional volume.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator, SignalIndicator, IndicatorSignal,
};

/// Cumulative Delta Output
#[derive(Debug, Clone)]
pub struct CumulativeDeltaOutput {
    /// Cumulative delta values
    pub cumulative_delta: Vec<f64>,
    /// Rate of change of cumulative delta
    pub delta_roc: Vec<f64>,
    /// Smoothed cumulative delta (EMA)
    pub smoothed: Vec<f64>,
}

/// Cumulative Delta Configuration
#[derive(Debug, Clone)]
pub struct CumulativeDeltaConfig {
    /// Period for rate of change calculation
    pub roc_period: usize,
    /// Period for EMA smoothing
    pub smoothing_period: usize,
    /// Whether to reset cumulative on session (not implemented in basic version)
    pub reset_on_session: bool,
}

impl Default for CumulativeDeltaConfig {
    fn default() -> Self {
        Self {
            roc_period: 10,
            smoothing_period: 14,
            reset_on_session: false,
        }
    }
}

/// CumulativeDelta (IND-217)
///
/// Calculates the running cumulative total of delta (buy volume - sell volume).
/// This indicator helps identify the overall buying/selling pressure over time.
///
/// Features:
/// - Cumulative delta tracking
/// - Rate of change for momentum analysis
/// - Smoothed version for trend identification
///
/// Interpretation:
/// - Rising cumulative delta = sustained buying pressure
/// - Falling cumulative delta = sustained selling pressure
/// - Divergence with price = potential reversal signal
#[derive(Debug, Clone)]
pub struct CumulativeDelta {
    config: CumulativeDeltaConfig,
}

impl CumulativeDelta {
    pub fn new(config: CumulativeDeltaConfig) -> Result<Self> {
        if config.roc_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "roc_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if config.smoothing_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self {
            config: CumulativeDeltaConfig::default(),
        }
    }

    /// Calculate cumulative delta with full output
    pub fn calculate_full(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> CumulativeDeltaOutput {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut cumulative_delta = vec![0.0; n];
        let mut delta_roc = vec![0.0; n];
        let mut smoothed = vec![0.0; n];

        if n == 0 {
            return CumulativeDeltaOutput {
                cumulative_delta,
                delta_roc,
                smoothed,
            };
        }

        // Calculate cumulative delta
        let mut cumsum = 0.0;
        for i in 0..n {
            let range = high[i] - low[i];
            if range > 0.0 {
                let position = (close[i] - low[i]) / range;
                let delta = volume[i] * (2.0 * position - 1.0);
                cumsum += delta;
            }
            cumulative_delta[i] = cumsum;
        }

        // Calculate rate of change
        for i in self.config.roc_period..n {
            let prev = cumulative_delta[i - self.config.roc_period];
            if prev.abs() > 1e-10 {
                delta_roc[i] = (cumulative_delta[i] - prev) / prev.abs() * 100.0;
            } else {
                delta_roc[i] = cumulative_delta[i].signum() * 100.0;
            }
        }

        // Calculate EMA smoothing
        let alpha = 2.0 / (self.config.smoothing_period as f64 + 1.0);
        smoothed[0] = cumulative_delta[0];
        for i in 1..n {
            smoothed[i] = alpha * cumulative_delta[i] + (1.0 - alpha) * smoothed[i - 1];
        }

        CumulativeDeltaOutput {
            cumulative_delta,
            delta_roc,
            smoothed,
        }
    }

    /// Calculate cumulative delta values only
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        self.calculate_full(high, low, close, volume).cumulative_delta
    }
}

impl TechnicalIndicator for CumulativeDelta {
    fn name(&self) -> &str {
        "Cumulative Delta"
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            output.cumulative_delta,
            output.delta_roc,
            output.smoothed,
        ))
    }
}

impl SignalIndicator for CumulativeDelta {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);

        if output.delta_roc.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        if let Some(&last_roc) = output.delta_roc.last() {
            if last_roc > 10.0 {
                return Ok(IndicatorSignal::Bullish);
            } else if last_roc < -10.0 {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);

        Ok(output
            .delta_roc
            .iter()
            .map(|&roc| {
                if roc > 10.0 {
                    IndicatorSignal::Bullish
                } else if roc < -10.0 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect())
    }
}

impl Default for CumulativeDelta {
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
        let volume = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0,
                         2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0, 2600.0, 2700.0, 2800.0, 2900.0];
        (high, low, close, volume)
    }

    #[test]
    fn test_cumulative_delta_basic() {
        let cd = CumulativeDelta::default_config();
        let (high, low, close, volume) = make_test_data();

        let result = cd.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 20);

        // Cumulative delta should be increasing in uptrend with closes near highs
        assert!(result[10] > result[5], "Cumulative delta should increase");
        assert!(result[15] > result[10], "Cumulative delta should continue increasing");
    }

    #[test]
    fn test_cumulative_delta_full_output() {
        let cd = CumulativeDelta::default_config();
        let (high, low, close, volume) = make_test_data();

        let output = cd.calculate_full(&high, &low, &close, &volume);

        assert_eq!(output.cumulative_delta.len(), 20);
        assert_eq!(output.delta_roc.len(), 20);
        assert_eq!(output.smoothed.len(), 20);

        // Smoothed should track cumulative delta
        // Last smoothed should be positive in uptrend
        assert!(output.smoothed[19] > 0.0);
    }

    #[test]
    fn test_cumulative_delta_downtrend() {
        let cd = CumulativeDelta::default_config();
        // Downtrending data with closes near lows
        let high = vec![105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0];
        let low = vec![100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0];
        let close = vec![101.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0]; // Near lows
        let volume = vec![1000.0; 10];

        let result = cd.calculate(&high, &low, &close, &volume);

        // Cumulative delta should be decreasing in downtrend
        assert!(result[9] < result[5], "Cumulative delta should decrease in downtrend");
        assert!(result[9] < 0.0, "Final cumulative delta should be negative");
    }

    #[test]
    fn test_cumulative_delta_config() {
        let config = CumulativeDeltaConfig {
            roc_period: 5,
            smoothing_period: 10,
            reset_on_session: false,
        };
        let cd = CumulativeDelta::new(config).unwrap();
        let (high, low, close, volume) = make_test_data();

        let output = cd.calculate_full(&high, &low, &close, &volume);
        assert_eq!(output.cumulative_delta.len(), 20);
    }

    #[test]
    fn test_cumulative_delta_invalid_config() {
        let config = CumulativeDeltaConfig {
            roc_period: 0,
            smoothing_period: 14,
            reset_on_session: false,
        };
        assert!(CumulativeDelta::new(config).is_err());
    }

    #[test]
    fn test_cumulative_delta_signal() {
        let cd = CumulativeDelta::default_config();
        let (high, low, close, volume) = make_test_data();

        let data = OHLCVSeries {
            open: vec![100.0; 20],
            high,
            low,
            close,
            volume,
        };

        let signal = cd.signal(&data).unwrap();
        // In strong uptrend, should be bullish
        assert!(matches!(signal, IndicatorSignal::Bullish | IndicatorSignal::Neutral));
    }
}
