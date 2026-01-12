//! Net Volume implementation.

use crate::{TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal, IndicatorError, Result, OHLCVSeries};

/// Net Volume.
///
/// Simple volume indicator that assigns volume as positive or negative
/// based on price movement.
///
/// - If Close > Open: Net Volume = +Volume (buying pressure)
/// - If Close < Open: Net Volume = -Volume (selling pressure)
/// - If Close = Open: Net Volume = 0
///
/// Can be viewed as cumulative or per-bar.
#[derive(Debug, Clone)]
pub struct NetVolume {
    cumulative: bool,
}

impl NetVolume {
    pub fn new() -> Self {
        Self { cumulative: false }
    }

    pub fn cumulative() -> Self {
        Self { cumulative: true }
    }

    /// Calculate Net Volume values.
    pub fn calculate(&self, open: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();

        if n == 0 {
            return vec![];
        }

        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let net = if close[i] > open[i] {
                volume[i]
            } else if close[i] < open[i] {
                -volume[i]
            } else {
                0.0
            };

            if self.cumulative && i > 0 {
                result.push(result[i - 1] + net);
            } else {
                result.push(net);
            }
        }

        result
    }
}

impl Default for NetVolume {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for NetVolume {
    fn name(&self) -> &str {
        "Net Volume"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let values = self.calculate(&data.open, &data.close, &data.volume);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for NetVolume {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.open, &data.close, &data.volume);

        if let Some(&last) = values.last() {
            if last > 0.0 {
                return Ok(IndicatorSignal::Bullish);
            } else if last < 0.0 {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.open, &data.close, &data.volume);
        Ok(values
            .iter()
            .map(|&v| {
                if v > 0.0 {
                    IndicatorSignal::Bullish
                } else if v < 0.0 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_net_volume_per_bar() {
        let nv = NetVolume::new();
        let open = vec![100.0, 101.0, 102.0];
        let close = vec![102.0, 100.0, 102.0]; // Up, Down, Flat
        let volume = vec![1000.0, 1500.0, 1200.0];

        let result = nv.calculate(&open, &close, &volume);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 1000.0).abs() < 1e-10); // Bullish
        assert!((result[1] - (-1500.0)).abs() < 1e-10); // Bearish
        assert!((result[2] - 0.0).abs() < 1e-10); // Neutral
    }

    #[test]
    fn test_net_volume_cumulative() {
        let nv = NetVolume::cumulative();
        let open = vec![100.0, 101.0, 100.0];
        let close = vec![102.0, 103.0, 98.0]; // Up, Up, Down
        let volume = vec![1000.0, 1500.0, 2000.0];

        let result = nv.calculate(&open, &close, &volume);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 1000.0).abs() < 1e-10);
        assert!((result[1] - 2500.0).abs() < 1e-10); // 1000 + 1500
        assert!((result[2] - 500.0).abs() < 1e-10); // 2500 - 2000
    }
}
