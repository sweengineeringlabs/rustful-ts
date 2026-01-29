//! COT Forex Indicator (IND-311)
//!
//! Net positioning in forex markets based on Commitment of Traders report proxy.
//! Uses volume and price momentum as a proxy for institutional positioning.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// COT Forex - Net positioning proxy for forex markets (IND-311)
///
/// This indicator approximates the Commitment of Traders positioning
/// using price momentum and volume as proxies for institutional flows.
///
/// # Interpretation
/// - Positive values indicate net long positioning
/// - Negative values indicate net short positioning
/// - Extreme readings may signal crowded trades
///
/// # Example
/// ```ignore
/// use indicator_core::forex::COTForex;
///
/// let cot = COTForex::new(14, 5).unwrap();
/// let positioning = cot.calculate(&close, &volume);
/// ```
#[derive(Debug, Clone)]
pub struct COTForex {
    /// Lookback period for positioning calculation
    period: usize,
    /// Smoothing period for the result
    smooth_period: usize,
}

impl COTForex {
    /// Create a new COTForex indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for positioning (minimum 5)
    /// * `smooth_period` - Smoothing period for result (minimum 2)
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, smooth_period })
    }

    /// Calculate COT forex net positioning proxy.
    ///
    /// # Arguments
    /// * `close` - Closing prices
    /// * `volume` - Trading volumes
    ///
    /// # Returns
    /// Vector of net positioning values (normalized -100 to 100)
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        if n < self.period + self.smooth_period {
            return vec![0.0; n];
        }

        let mut raw_positioning = vec![0.0; n];

        // Calculate raw positioning using volume-weighted momentum
        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            let mut long_pressure = 0.0;
            let mut short_pressure = 0.0;
            let avg_volume = volume[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;

            for j in (start + 1)..=i {
                let price_change = close[j] - close[j - 1];
                let relative_volume = if avg_volume > 0.0 {
                    volume[j] / avg_volume
                } else {
                    1.0
                };

                if price_change > 0.0 {
                    long_pressure += price_change.abs() * relative_volume;
                } else {
                    short_pressure += price_change.abs() * relative_volume;
                }
            }

            // Net positioning as ratio
            let total = long_pressure + short_pressure;
            if total > 0.0 {
                raw_positioning[i] = (long_pressure - short_pressure) / total * 100.0;
            }
        }

        // Apply EMA smoothing
        let mut result = vec![0.0; n];
        let alpha = 2.0 / (self.smooth_period + 1) as f64;

        for i in self.period..n {
            if i == self.period {
                result[i] = raw_positioning[i];
            } else {
                result[i] = alpha * raw_positioning[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }

    /// Calculate with extended output including long and short components.
    pub fn calculate_extended(&self, close: &[f64], volume: &[f64]) -> COTForexOutput {
        let n = close.len().min(volume.len());
        if n < self.period + self.smooth_period {
            return COTForexOutput {
                net_positioning: vec![0.0; n],
                long_positioning: vec![0.0; n],
                short_positioning: vec![0.0; n],
                positioning_change: vec![0.0; n],
            };
        }

        let mut net = vec![0.0; n];
        let mut long_pos = vec![0.0; n];
        let mut short_pos = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            let mut long_pressure = 0.0;
            let mut short_pressure = 0.0;
            let avg_volume = volume[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;

            for j in (start + 1)..=i {
                let price_change = close[j] - close[j - 1];
                let relative_volume = if avg_volume > 0.0 {
                    volume[j] / avg_volume
                } else {
                    1.0
                };

                if price_change > 0.0 {
                    long_pressure += price_change.abs() * relative_volume;
                } else {
                    short_pressure += price_change.abs() * relative_volume;
                }
            }

            let total = long_pressure + short_pressure;
            if total > 0.0 {
                long_pos[i] = long_pressure / total * 100.0;
                short_pos[i] = short_pressure / total * 100.0;
                net[i] = long_pos[i] - short_pos[i];
            }
        }

        // Smooth the results
        let alpha = 2.0 / (self.smooth_period + 1) as f64;
        for i in (self.period + 1)..n {
            net[i] = alpha * net[i] + (1.0 - alpha) * net[i - 1];
            long_pos[i] = alpha * long_pos[i] + (1.0 - alpha) * long_pos[i - 1];
            short_pos[i] = alpha * short_pos[i] + (1.0 - alpha) * short_pos[i - 1];
        }

        // Calculate positioning change
        let mut change = vec![0.0; n];
        for i in 1..n {
            change[i] = net[i] - net[i - 1];
        }

        COTForexOutput {
            net_positioning: net,
            long_positioning: long_pos,
            short_positioning: short_pos,
            positioning_change: change,
        }
    }
}

/// Extended output for COTForex indicator.
#[derive(Debug, Clone)]
pub struct COTForexOutput {
    /// Net positioning (-100 to 100)
    pub net_positioning: Vec<f64>,
    /// Long positioning percentage (0 to 100)
    pub long_positioning: Vec<f64>,
    /// Short positioning percentage (0 to 100)
    pub short_positioning: Vec<f64>,
    /// Change in net positioning
    pub positioning_change: Vec<f64>,
}

impl TechnicalIndicator for COTForex {
    fn name(&self) -> &str {
        "COT Forex"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        let close = vec![
            1.1000, 1.1020, 1.0980, 1.1050, 1.1030, 1.1080, 1.1060, 1.1100, 1.1080, 1.1120,
            1.1100, 1.1150, 1.1130, 1.1180, 1.1160, 1.1200, 1.1180, 1.1220, 1.1200, 1.1250,
            1.1230, 1.1280, 1.1260, 1.1300, 1.1280, 1.1330, 1.1310, 1.1350, 1.1330, 1.1380,
        ];
        let volume = vec![
            1000.0, 1200.0, 1100.0, 1300.0, 1500.0, 1400.0, 1600.0, 1800.0, 1700.0, 1900.0,
            2000.0, 1800.0, 2100.0, 2200.0, 2000.0, 2300.0, 2100.0, 2400.0, 2200.0, 2500.0,
            2300.0, 2600.0, 2400.0, 2700.0, 2500.0, 2800.0, 2600.0, 2900.0, 2700.0, 3000.0,
        ];
        (close, volume)
    }

    #[test]
    fn test_cot_forex_new() {
        assert!(COTForex::new(14, 5).is_ok());
        assert!(COTForex::new(4, 5).is_err()); // period too small
        assert!(COTForex::new(14, 1).is_err()); // smooth_period too small
    }

    #[test]
    fn test_cot_forex_calculate() {
        let (close, volume) = make_test_data();
        let cot = COTForex::new(10, 3).unwrap();
        let result = cot.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Uptrend should show net long positioning
        assert!(result[25] > 0.0);
        // Values should be bounded
        assert!(result[25] >= -100.0 && result[25] <= 100.0);
    }

    #[test]
    fn test_cot_forex_extended() {
        let (close, volume) = make_test_data();
        let cot = COTForex::new(10, 3).unwrap();
        let output = cot.calculate_extended(&close, &volume);

        assert_eq!(output.net_positioning.len(), close.len());
        assert_eq!(output.long_positioning.len(), close.len());
        assert_eq!(output.short_positioning.len(), close.len());
        assert_eq!(output.positioning_change.len(), close.len());

        // Net should equal long - short
        let idx = 25;
        let diff = (output.long_positioning[idx] - output.short_positioning[idx] - output.net_positioning[idx]).abs();
        assert!(diff < 1.0); // Allow for smoothing differences
    }

    #[test]
    fn test_cot_forex_technical_indicator() {
        let (close, volume) = make_test_data();
        let cot = COTForex::new(10, 3).unwrap();

        assert_eq!(cot.name(), "COT Forex");
        assert_eq!(cot.min_periods(), 13);

        let data = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x * 1.01).collect(),
            low: close.iter().map(|x| x * 0.99).collect(),
            close: close.clone(),
            volume,
        };

        let output = cot.compute(&data).unwrap();
        assert!(output.primary.len() == close.len());
    }
}
