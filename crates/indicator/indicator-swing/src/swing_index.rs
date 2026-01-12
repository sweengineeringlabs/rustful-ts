//! Wilder's Swing Index implementation.
//!
//! The Swing Index attempts to isolate price swings within the context of a single bar.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Wilder's Swing Index.
///
/// Developed by J. Welles Wilder, the Swing Index attempts to isolate the
/// "real" price movement by comparing the relationships between the current
/// open/high/low/close and the previous bar's values.
///
/// The Swing Index ranges from -100 to +100, with positive values indicating
/// bullish pressure and negative values indicating bearish pressure.
///
/// Formula:
/// SI = 50 * (Cy - C + 0.5*(Cy - Oy) + 0.25*(C - O)) / R * K / T
///
/// Where:
/// - C = Current Close
/// - O = Current Open
/// - H = Current High
/// - L = Current Low
/// - Cy = Previous Close
/// - Oy = Previous Open
/// - Hy = Previous High
/// - Ly = Previous Low
/// - K = max(|H - Cy|, |L - Cy|)
/// - T = Limit move value (typically the ATR or fixed value)
/// - R = Determined by the largest of several true range components
#[derive(Debug, Clone)]
pub struct SwingIndex {
    /// Limit move value for normalization.
    limit_move: f64,
}

impl SwingIndex {
    /// Create a new Swing Index indicator.
    ///
    /// # Arguments
    /// * `limit_move` - Maximum expected price move (e.g., daily limit or ATR)
    pub fn new(limit_move: f64) -> Self {
        Self { limit_move }
    }

    /// Create with default limit move of 1.0 (percentage-based).
    pub fn default_limit() -> Self {
        Self { limit_move: 1.0 }
    }

    /// Calculate Swing Index values.
    pub fn calculate(
        &self,
        open: &[f64],
        high: &[f64],
        low: &[f64],
        close: &[f64],
    ) -> Vec<f64> {
        let n = close.len();
        if n < 2 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN];

        for i in 1..n {
            let c = close[i];
            let o = open[i];
            let h = high[i];
            let l = low[i];
            let cy = close[i - 1];
            let oy = open[i - 1];
            let _hy = high[i - 1];
            let _ly = low[i - 1];

            // Calculate K (larger of High-PrevClose or Low-PrevClose)
            let k = (h - cy).abs().max((l - cy).abs());

            // Calculate R (true range component)
            let hc = (h - cy).abs();
            let lc = (l - cy).abs();
            let hl = h - l;
            let ch = (cy - oy).abs();

            let r = if hc >= lc && hc >= hl {
                hc - 0.5 * lc + 0.25 * ch
            } else if lc >= hc && lc >= hl {
                lc - 0.5 * hc + 0.25 * ch
            } else {
                hl + 0.25 * ch
            };

            if r == 0.0 || self.limit_move == 0.0 {
                result.push(0.0);
                continue;
            }

            // Calculate Swing Index
            let numerator = (cy - c) + 0.5 * (cy - oy) + 0.25 * (c - o);
            let si = 50.0 * numerator * (k / self.limit_move) / r;

            // Clamp to -100 to +100 range
            let clamped = si.clamp(-100.0, 100.0);
            result.push(clamped);
        }

        result
    }
}

impl TechnicalIndicator for SwingIndex {
    fn name(&self) -> &str {
        "SwingIndex"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 2 {
            return Err(IndicatorError::InsufficientData {
                required: 2,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.open, &data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swing_index_basic() {
        let si = SwingIndex::new(3.0);

        let open = vec![100.0, 101.0, 102.0, 101.5, 103.0];
        let high = vec![102.0, 103.0, 104.0, 103.5, 105.0];
        let low = vec![99.0, 100.0, 101.0, 100.5, 102.0];
        let close = vec![101.0, 102.0, 103.0, 102.5, 104.0];

        let result = si.calculate(&open, &high, &low, &close);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());

        // Values should be in valid range
        for i in 1..5 {
            assert!(result[i] >= -100.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_swing_index_default_limit() {
        let si = SwingIndex::default_limit();

        let open = vec![100.0, 100.5, 101.0];
        let high = vec![101.0, 101.5, 102.0];
        let low = vec![99.5, 100.0, 100.5];
        let close = vec![100.5, 101.0, 101.5];

        let result = si.calculate(&open, &high, &low, &close);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_swing_index_technical_indicator() {
        let si = SwingIndex::new(2.0);

        let mut data = OHLCVSeries::new();
        for i in 0..10 {
            data.open.push(100.0 + i as f64);
            data.high.push(102.0 + i as f64);
            data.low.push(98.0 + i as f64);
            data.close.push(101.0 + i as f64);
            data.volume.push(1000.0);
        }

        let output = si.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 10);
    }
}
