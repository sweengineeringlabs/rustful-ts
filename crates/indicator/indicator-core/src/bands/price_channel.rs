//! Price Channel implementation.
//!
//! Simple price channel based on highest high and lowest low with an offset.

use crate::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Price Channel.
///
/// A simple price channel indicator consisting of:
/// - Upper Channel: Highest high over period
/// - Lower Channel: Lowest low over period
/// - Middle Channel: (Upper + Lower) / 2
///
/// Similar to Donchian Channels but often uses different periods
/// for upper and lower channels, and may include an offset.
#[derive(Debug, Clone)]
pub struct PriceChannel {
    /// Period for the upper channel (highest high).
    upper_period: usize,
    /// Period for the lower channel (lowest low).
    lower_period: usize,
    /// Offset for lagging the channel (typically 0 or 1).
    offset: usize,
}

impl PriceChannel {
    /// Create a new Price Channel indicator.
    ///
    /// # Arguments
    /// * `upper_period` - Period for highest high calculation
    /// * `lower_period` - Period for lowest low calculation
    /// * `offset` - Number of bars to offset (lag) the channel
    pub fn new(upper_period: usize, lower_period: usize, offset: usize) -> Self {
        Self {
            upper_period,
            lower_period,
            offset,
        }
    }

    /// Create with symmetric periods (same for upper and lower).
    pub fn symmetric(period: usize) -> Self {
        Self::new(period, period, 0)
    }

    /// Create with symmetric periods and offset.
    pub fn symmetric_with_offset(period: usize, offset: usize) -> Self {
        Self::new(period, period, offset)
    }

    /// Calculate Price Channel (upper, middle, lower).
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len();
        let min_period = self.upper_period.max(self.lower_period);

        if n < min_period || min_period == 0 {
            return (
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
            );
        }

        let mut upper = vec![f64::NAN; n];
        let mut lower = vec![f64::NAN; n];
        let mut middle = vec![f64::NAN; n];

        // Calculate raw channels
        let mut raw_upper = vec![f64::NAN; n];
        let mut raw_lower = vec![f64::NAN; n];

        // Calculate upper channel (highest high)
        for i in (self.upper_period - 1)..n {
            let start = i + 1 - self.upper_period;
            raw_upper[i] = high[start..=i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
        }

        // Calculate lower channel (lowest low)
        for i in (self.lower_period - 1)..n {
            let start = i + 1 - self.lower_period;
            raw_lower[i] = low[start..=i]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
        }

        // Apply offset
        if self.offset == 0 {
            for i in 0..n {
                upper[i] = raw_upper[i];
                lower[i] = raw_lower[i];
                if !upper[i].is_nan() && !lower[i].is_nan() {
                    middle[i] = (upper[i] + lower[i]) / 2.0;
                }
            }
        } else {
            for i in self.offset..n {
                upper[i] = raw_upper[i - self.offset];
                lower[i] = raw_lower[i - self.offset];
                if !upper[i].is_nan() && !lower[i].is_nan() {
                    middle[i] = (upper[i] + lower[i]) / 2.0;
                }
            }
        }

        (upper, middle, lower)
    }

    /// Calculate channel width.
    pub fn channel_width(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let (upper, _, lower) = self.calculate(high, low);
        upper
            .iter()
            .zip(lower.iter())
            .map(|(&u, &l)| {
                if u.is_nan() || l.is_nan() {
                    f64::NAN
                } else {
                    u - l
                }
            })
            .collect()
    }
}

impl Default for PriceChannel {
    fn default() -> Self {
        Self::symmetric(20)
    }
}

impl TechnicalIndicator for PriceChannel {
    fn name(&self) -> &str {
        "PriceChannel"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.upper_period.max(self.lower_period);
        if data.high.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.high.len(),
            });
        }

        let (upper, middle, lower) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }

    fn min_periods(&self) -> usize {
        self.upper_period.max(self.lower_period)
    }

    fn output_features(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_channel() {
        let pc = PriceChannel::symmetric(5);
        let high = vec![105.0, 106.0, 107.0, 106.0, 108.0, 109.0, 108.0];
        let low = vec![95.0, 96.0, 97.0, 96.0, 98.0, 99.0, 98.0];

        let (upper, middle, lower) = pc.calculate(&high, &low);

        // Check first valid values (at index 4)
        assert!((upper[4] - 108.0).abs() < 1e-10); // Highest high in first 5 bars
        assert!((lower[4] - 95.0).abs() < 1e-10); // Lowest low in first 5 bars
        assert!((middle[4] - 101.5).abs() < 1e-10); // (108 + 95) / 2
    }

    #[test]
    fn test_price_channel_with_offset() {
        let pc = PriceChannel::symmetric_with_offset(5, 1);
        let high = vec![105.0, 106.0, 107.0, 106.0, 108.0, 109.0, 108.0];
        let low = vec![95.0, 96.0, 97.0, 96.0, 98.0, 99.0, 98.0];

        let (upper, _, lower) = pc.calculate(&high, &low);

        // With offset 1, values at index 5 should be from index 4
        assert!((upper[5] - 108.0).abs() < 1e-10);
        assert!((lower[5] - 95.0).abs() < 1e-10);
    }

    #[test]
    fn test_asymmetric_channel() {
        let pc = PriceChannel::new(10, 5, 0);
        assert_eq!(pc.upper_period, 10);
        assert_eq!(pc.lower_period, 5);
        assert_eq!(pc.min_periods(), 10);
    }

    #[test]
    fn test_channel_width() {
        let pc = PriceChannel::symmetric(5);
        let high = vec![105.0, 106.0, 107.0, 106.0, 108.0, 109.0, 108.0];
        let low = vec![95.0, 96.0, 97.0, 96.0, 98.0, 99.0, 98.0];

        let width = pc.channel_width(&high, &low);
        assert!((width[4] - 13.0).abs() < 1e-10); // 108 - 95
    }
}
