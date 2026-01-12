//! Vertical Horizontal Filter (VHF).

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Vertical Horizontal Filter (VHF) - IND-187
///
/// Determines if market is trending or ranging.
/// VHF = (Highest High - Lowest Low) / Sum of absolute changes
#[derive(Debug, Clone)]
pub struct VerticalHorizontalFilter {
    period: usize,
}

impl VerticalHorizontalFilter {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period];

        for i in self.period..n {
            let window = &data[(i - self.period)..=i];

            // Numerator: highest - lowest in period
            let highest = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let numerator = highest - lowest;

            // Denominator: sum of absolute price changes
            let mut denominator = 0.0;
            for j in 1..=self.period {
                let idx = i - self.period + j;
                denominator += (data[idx] - data[idx - 1]).abs();
            }

            if denominator != 0.0 {
                result.push(numerator / denominator);
            } else {
                result.push(f64::NAN);
            }
        }

        result
    }
}

impl Default for VerticalHorizontalFilter {
    fn default() -> Self {
        Self::new(28)
    }
}

impl TechnicalIndicator for VerticalHorizontalFilter {
    fn name(&self) -> &str {
        "VHF"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }
}

impl SignalIndicator for VerticalHorizontalFilter {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // High VHF = trending, Low VHF = ranging
        // Standard thresholds: > 0.40 trending, < 0.25 ranging
        if last > 0.40 {
            Ok(IndicatorSignal::Bullish) // Trend-following signals work
        } else if last < 0.25 {
            Ok(IndicatorSignal::Bearish) // Mean-reversion signals work
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let signals = values.iter().map(|&val| {
            if val.is_nan() {
                IndicatorSignal::Neutral
            } else if val > 0.40 {
                IndicatorSignal::Bullish
            } else if val < 0.25 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect();
        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vhf_trending() {
        let vhf = VerticalHorizontalFilter::new(14);
        // Strong trend: VHF should be high
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64 * 2.0).collect();
        let result = vhf.calculate(&data);

        let last = result.last().unwrap();
        assert!(!last.is_nan());
        // In a strong trend, VHF should be relatively high
    }

    #[test]
    fn test_vhf_ranging() {
        let vhf = VerticalHorizontalFilter::new(14);
        // Ranging market: VHF should be low
        let data: Vec<f64> = (0..30).map(|i| 100.0 + ((i as f64) % 2.0) - 0.5).collect();
        let result = vhf.calculate(&data);

        let last = result.last().unwrap();
        assert!(!last.is_nan());
    }
}
