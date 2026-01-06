//! Donchian Channels implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::DonchianConfig;

/// Donchian Channels.
///
/// Price channel based on highest high and lowest low over a period.
/// - Upper Band: Highest high over period
/// - Lower Band: Lowest low over period
/// - Middle Band: (Upper + Lower) / 2
#[derive(Debug, Clone)]
pub struct DonchianChannels {
    period: usize,
}

impl DonchianChannels {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn from_config(config: DonchianConfig) -> Self {
        Self { period: config.period }
    }

    /// Calculate Donchian Channels (upper, middle, lower).
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len();
        if n < self.period || self.period == 0 {
            return (
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
            );
        }

        let mut upper = vec![f64::NAN; self.period - 1];
        let mut middle = vec![f64::NAN; self.period - 1];
        let mut lower = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let highest = high[start..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lowest = low[start..=i].iter().cloned().fold(f64::INFINITY, f64::min);

            upper.push(highest);
            lower.push(lowest);
            middle.push((highest + lowest) / 2.0);
        }

        (upper, middle, lower)
    }
}

impl Default for DonchianChannels {
    fn default() -> Self {
        Self::from_config(DonchianConfig::default())
    }
}

impl TechnicalIndicator for DonchianChannels {
    fn name(&self) -> &str {
        "DonchianChannels"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.high.len(),
            });
        }

        let (upper, middle, lower) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_donchian_channels() {
        let dc = DonchianChannels::new(5);
        let high = vec![105.0, 106.0, 107.0, 106.0, 108.0, 109.0, 108.0];
        let low = vec![95.0, 96.0, 97.0, 96.0, 98.0, 99.0, 98.0];

        let (upper, middle, lower) = dc.calculate(&high, &low);

        // Check first valid values
        assert!((upper[4] - 108.0).abs() < 1e-10); // Highest high in first 5 bars
        assert!((lower[4] - 95.0).abs() < 1e-10); // Lowest low in first 5 bars
        assert!((middle[4] - 101.5).abs() < 1e-10); // (108 + 95) / 2
    }
}
