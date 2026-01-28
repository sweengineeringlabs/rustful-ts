//! Price Pattern Indicators
//!
//! Pattern detection for common price formations.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Double Top Pattern Detection
#[derive(Debug, Clone)]
pub struct DoubleTop {
    lookback: usize,
    tolerance: f64,
}

impl DoubleTop {
    pub fn new(lookback: usize, tolerance: f64) -> Result<Self> {
        if lookback < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if tolerance <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "tolerance".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { lookback, tolerance })
    }

    /// Returns 1 for pattern detected, 0 otherwise
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);
            let window_high = &high[start..=i];
            let window_low = &low[start..=i];

            // Find two local highs
            let max_idx1 = window_high.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let max_val1 = window_high[max_idx1];

            // Find second high (excluding first peak area)
            let mut max_idx2 = 0;
            let mut max_val2 = f64::NEG_INFINITY;
            for (j, &val) in window_high.iter().enumerate() {
                if (j as i32 - max_idx1 as i32).abs() > 3 && val > max_val2 {
                    max_val2 = val;
                    max_idx2 = j;
                }
            }

            // Check if double top pattern
            if max_val2 != f64::NEG_INFINITY {
                let diff = (max_val1 - max_val2).abs() / max_val1;
                if diff < self.tolerance / 100.0 {
                    // Find the valley between peaks
                    let valley_start = max_idx1.min(max_idx2);
                    let valley_end = max_idx1.max(max_idx2);
                    if valley_end > valley_start + 1 {
                        let valley_low = window_low[valley_start..valley_end].iter()
                            .fold(f64::INFINITY, |a, &b| a.min(b));

                        // Confirm breakdown
                        if close[i] < valley_low {
                            result[i] = 1.0;
                        }
                    }
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for DoubleTop {
    fn name(&self) -> &str {
        "Double Top"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Double Bottom Pattern Detection
#[derive(Debug, Clone)]
pub struct DoubleBottom {
    lookback: usize,
    tolerance: f64,
}

impl DoubleBottom {
    pub fn new(lookback: usize, tolerance: f64) -> Result<Self> {
        if lookback < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if tolerance <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "tolerance".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { lookback, tolerance })
    }

    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);
            let window_high = &high[start..=i];
            let window_low = &low[start..=i];

            // Find two local lows
            let min_idx1 = window_low.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let min_val1 = window_low[min_idx1];

            // Find second low
            let mut min_idx2 = 0;
            let mut min_val2 = f64::INFINITY;
            for (j, &val) in window_low.iter().enumerate() {
                if (j as i32 - min_idx1 as i32).abs() > 3 && val < min_val2 {
                    min_val2 = val;
                    min_idx2 = j;
                }
            }

            // Check if double bottom pattern
            if min_val2 != f64::INFINITY {
                let diff = (min_val1 - min_val2).abs() / min_val1;
                if diff < self.tolerance / 100.0 {
                    let peak_start = min_idx1.min(min_idx2);
                    let peak_end = min_idx1.max(min_idx2);
                    if peak_end > peak_start + 1 {
                        let peak_high = window_high[peak_start..peak_end].iter()
                            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                        if close[i] > peak_high {
                            result[i] = 1.0;
                        }
                    }
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for DoubleBottom {
    fn name(&self) -> &str {
        "Double Bottom"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Head and Shoulders Detection (simplified)
#[derive(Debug, Clone)]
pub struct HeadShoulders {
    lookback: usize,
}

impl HeadShoulders {
    pub fn new(lookback: usize) -> Result<Self> {
        if lookback < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self { lookback })
    }

    pub fn calculate(&self, high: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);
            let window = &high[start..=i];

            // Find 3 peaks
            let mut peaks: Vec<(usize, f64)> = Vec::new();
            for j in 2..(window.len() - 2) {
                if window[j] > window[j - 1] && window[j] > window[j - 2]
                    && window[j] > window[j + 1] && window[j] > window[j + 2]
                {
                    peaks.push((j, window[j]));
                }
            }

            if peaks.len() >= 3 {
                // Check for H&S pattern in last 3 peaks
                let last_peaks = &peaks[peaks.len() - 3..];
                let left_shoulder = last_peaks[0].1;
                let head = last_peaks[1].1;
                let right_shoulder = last_peaks[2].1;

                // Head should be highest, shoulders roughly equal
                if head > left_shoulder && head > right_shoulder {
                    let shoulder_diff = (left_shoulder - right_shoulder).abs() / left_shoulder;
                    if shoulder_diff < 0.05 {
                        result[i] = 1.0;
                    }
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for HeadShoulders {
    fn name(&self) -> &str {
        "Head and Shoulders"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.close)))
    }
}

/// Triangle Pattern Detection
#[derive(Debug, Clone)]
pub struct Triangle {
    lookback: usize,
}

impl Triangle {
    pub fn new(lookback: usize) -> Result<Self> {
        if lookback < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { lookback })
    }

    /// Returns: 1 = ascending, -1 = descending, 0.5 = symmetrical, 0 = none
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            // Calculate trend of highs and lows
            let high_slope = linear_slope(&high[start..=i]);
            let low_slope = linear_slope(&low[start..=i]);

            // Determine triangle type
            if low_slope > 0.001 && high_slope.abs() < 0.001 {
                result[i] = 1.0; // Ascending
            } else if high_slope < -0.001 && low_slope.abs() < 0.001 {
                result[i] = -1.0; // Descending
            } else if high_slope < -0.001 && low_slope > 0.001 {
                result[i] = 0.5; // Symmetrical
            }
        }
        result
    }
}

fn linear_slope(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let sum_x: f64 = (0..data.len()).map(|x| x as f64).sum();
    let sum_y: f64 = data.iter().sum();
    let sum_xy: f64 = data.iter().enumerate().map(|(x, &y)| x as f64 * y).sum();
    let sum_xx: f64 = (0..data.len()).map(|x| (x as f64).powi(2)).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return 0.0;
    }

    (n * sum_xy - sum_x * sum_y) / denom
}

impl TechnicalIndicator for Triangle {
    fn name(&self) -> &str {
        "Triangle"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low)))
    }
}

/// Channel Detection
#[derive(Debug, Clone)]
pub struct Channel {
    lookback: usize,
}

impl Channel {
    pub fn new(lookback: usize) -> Result<Self> {
        if lookback < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { lookback })
    }

    /// Returns: 1 = upward channel, -1 = downward, 0.5 = horizontal, 0 = none
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            let high_slope = linear_slope(&high[start..=i]);
            let low_slope = linear_slope(&low[start..=i]);

            // Check if slopes are similar (parallel channel)
            let slope_diff = (high_slope - low_slope).abs();
            let avg_slope = (high_slope + low_slope) / 2.0;

            if slope_diff < 0.001 {
                if avg_slope > 0.001 {
                    result[i] = 1.0; // Upward channel
                } else if avg_slope < -0.001 {
                    result[i] = -1.0; // Downward channel
                } else {
                    result[i] = 0.5; // Horizontal channel
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for Channel {
    fn name(&self) -> &str {
        "Channel"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low)))
    }
}

/// Wedge Pattern Detection
#[derive(Debug, Clone)]
pub struct Wedge {
    lookback: usize,
}

impl Wedge {
    pub fn new(lookback: usize) -> Result<Self> {
        if lookback < 15 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 15".to_string(),
            });
        }
        Ok(Self { lookback })
    }

    /// Returns: 1 = rising wedge, -1 = falling wedge, 0 = none
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len().min(low.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            let high_slope = linear_slope(&high[start..=i]);
            let low_slope = linear_slope(&low[start..=i]);

            // Wedge: both slopes same direction, converging
            if high_slope > 0.001 && low_slope > 0.001 && low_slope > high_slope {
                result[i] = 1.0; // Rising wedge (bearish)
            } else if high_slope < -0.001 && low_slope < -0.001 && high_slope < low_slope {
                result[i] = -1.0; // Falling wedge (bullish)
            }
        }
        result
    }
}

impl TechnicalIndicator for Wedge {
    fn name(&self) -> &str {
        "Wedge"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 109.5, 108.0, 109.0, 110.0,
                       109.5, 108.0, 107.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0,
                       113.0, 114.0, 113.5, 112.0, 111.0, 110.0, 111.0, 112.0, 113.0, 114.0];
        let low = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.5, 103.0, 104.0, 105.0,
                      104.5, 103.0, 102.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0,
                      108.0, 109.0, 108.5, 107.0, 106.0, 105.0, 106.0, 107.0, 108.0, 109.0];
        let close = vec![102.0, 104.0, 105.0, 106.0, 107.0, 108.0, 107.0, 105.0, 106.0, 108.0,
                        107.0, 105.0, 104.0, 103.0, 104.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                        111.0, 112.0, 111.0, 109.0, 108.0, 107.0, 108.0, 110.0, 111.0, 112.0];
        (high, low, close)
    }

    #[test]
    fn test_double_top() {
        let (high, low, close) = make_test_data();
        let dt = DoubleTop::new(15, 3.0).unwrap();
        let result = dt.calculate(&high, &low, &close);
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_double_bottom() {
        let (high, low, close) = make_test_data();
        let db = DoubleBottom::new(15, 3.0).unwrap();
        let result = db.calculate(&high, &low, &close);
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_head_shoulders() {
        let (high, _, close) = make_test_data();
        let hs = HeadShoulders::new(20).unwrap();
        let result = hs.calculate(&high, &close);
        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_triangle() {
        let (high, low, _) = make_test_data();
        let tri = Triangle::new(15).unwrap();
        let result = tri.calculate(&high, &low);
        assert_eq!(result.len(), high.len());
    }

    #[test]
    fn test_channel() {
        let (high, low, _) = make_test_data();
        let ch = Channel::new(15).unwrap();
        let result = ch.calculate(&high, &low);
        assert_eq!(result.len(), high.len());
    }

    #[test]
    fn test_wedge() {
        let (high, low, _) = make_test_data();
        let wedge = Wedge::new(15).unwrap();
        let result = wedge.calculate(&high, &low);
        assert_eq!(result.len(), high.len());
    }
}
