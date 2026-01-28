//! Extended Support and Resistance Indicators
//!
//! Additional support/resistance detection and analysis indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Dynamic Support/Resistance - Adaptive levels based on recent price action
#[derive(Debug, Clone)]
pub struct DynamicSupportResistance {
    lookback: usize,
    threshold: f64,
}

impl DynamicSupportResistance {
    pub fn new(lookback: usize, threshold: f64) -> Result<Self> {
        if lookback < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { lookback, threshold })
    }

    /// Calculate dynamic support (lower band) and resistance (upper band)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut support = vec![0.0; n];
        let mut resistance = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            // Find recent significant highs and lows
            let mut max_high = high[start];
            let mut min_low = low[start];

            for j in start..=i {
                if high[j] > max_high {
                    max_high = high[j];
                }
                if low[j] < min_low {
                    min_low = low[j];
                }
            }

            // Dynamic levels with threshold adjustment
            let range = max_high - min_low;
            resistance[i] = max_high - range * self.threshold / 100.0;
            support[i] = min_low + range * self.threshold / 100.0;
        }

        (support, resistance)
    }
}

impl TechnicalIndicator for DynamicSupportResistance {
    fn name(&self) -> &str {
        "Dynamic Support/Resistance"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (support, resistance) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(support, resistance))
    }
}

/// Price Clusters - Identifies price clustering/congestion zones
#[derive(Debug, Clone)]
pub struct PriceClusters {
    lookback: usize,
    num_bins: usize,
}

impl PriceClusters {
    pub fn new(lookback: usize, num_bins: usize) -> Result<Self> {
        if lookback < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if num_bins < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_bins".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { lookback, num_bins })
    }

    /// Calculate price cluster strength (higher = more price congestion at current level)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);
            let slice = &close[start..=i];

            let min_price = slice.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_price = slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = max_price - min_price;

            if range < 1e-10 {
                result[i] = 100.0; // Maximum clustering if no range
                continue;
            }

            // Create histogram bins
            let bin_size = range / self.num_bins as f64;
            let current_bin = ((close[i] - min_price) / bin_size).floor() as usize;
            let current_bin = current_bin.min(self.num_bins - 1);

            // Count prices in current bin
            let mut count = 0;
            for &price in slice {
                let bin = ((price - min_price) / bin_size).floor() as usize;
                let bin = bin.min(self.num_bins - 1);
                if bin == current_bin {
                    count += 1;
                }
            }

            // Normalize to percentage
            result[i] = (count as f64 / slice.len() as f64) * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for PriceClusters {
    fn name(&self) -> &str {
        "Price Clusters"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Volume Support/Resistance - S/R levels based on volume distribution
#[derive(Debug, Clone)]
pub struct VolumeSupportResistance {
    lookback: usize,
    num_levels: usize,
}

impl VolumeSupportResistance {
    pub fn new(lookback: usize, num_levels: usize) -> Result<Self> {
        if lookback < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if num_levels < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_levels".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { lookback, num_levels })
    }

    /// Calculate volume-weighted price level (POC approximation)
    pub fn calculate(&self, high: &[f64], low: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut poc = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            // Find price range
            let mut min_price = low[start];
            let mut max_price = high[start];
            for j in start..=i {
                if low[j] < min_price {
                    min_price = low[j];
                }
                if high[j] > max_price {
                    max_price = high[j];
                }
            }

            let range = max_price - min_price;
            if range < 1e-10 {
                poc[i] = (high[i] + low[i]) / 2.0;
                continue;
            }

            // Create volume profile bins
            let bin_size = range / self.num_levels as f64;
            let mut volume_at_price = vec![0.0; self.num_levels];

            for j in start..=i {
                let mid_price = (high[j] + low[j]) / 2.0;
                let bin = ((mid_price - min_price) / bin_size).floor() as usize;
                let bin = bin.min(self.num_levels - 1);
                volume_at_price[bin] += volume[j];
            }

            // Find POC (bin with max volume)
            let max_vol_bin = volume_at_price
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(self.num_levels / 2);

            poc[i] = min_price + (max_vol_bin as f64 + 0.5) * bin_size;
        }

        poc
    }
}

impl TechnicalIndicator for VolumeSupportResistance {
    fn name(&self) -> &str {
        "Volume Support/Resistance"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.volume)))
    }
}

/// Swing Level Detector - S/R from swing highs/lows
#[derive(Debug, Clone)]
pub struct SwingLevelDetector {
    swing_strength: usize,
}

impl SwingLevelDetector {
    pub fn new(swing_strength: usize) -> Result<Self> {
        if swing_strength < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "swing_strength".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { swing_strength })
    }

    /// Calculate swing support and resistance levels
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        let mut support = vec![0.0; n];
        let mut resistance = vec![0.0; n];

        let mut last_swing_high = f64::NAN;
        let mut last_swing_low = f64::NAN;

        let min_period = 2 * self.swing_strength + 1;

        for i in min_period..n {
            // Check for swing high
            let pivot_idx = i - self.swing_strength;
            let mut is_swing_high = true;
            let mut is_swing_low = true;

            for j in (pivot_idx.saturating_sub(self.swing_strength))..pivot_idx {
                if high[j] >= high[pivot_idx] {
                    is_swing_high = false;
                }
                if low[j] <= low[pivot_idx] {
                    is_swing_low = false;
                }
            }

            for j in (pivot_idx + 1)..=i {
                if high[j] >= high[pivot_idx] {
                    is_swing_high = false;
                }
                if low[j] <= low[pivot_idx] {
                    is_swing_low = false;
                }
            }

            if is_swing_high {
                last_swing_high = high[pivot_idx];
            }
            if is_swing_low {
                last_swing_low = low[pivot_idx];
            }

            resistance[i] = last_swing_high;
            support[i] = last_swing_low;
        }

        (support, resistance)
    }
}

impl TechnicalIndicator for SwingLevelDetector {
    fn name(&self) -> &str {
        "Swing Level Detector"
    }

    fn min_periods(&self) -> usize {
        2 * self.swing_strength + 2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (support, resistance) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::dual(support, resistance))
    }
}

/// Trendline Break Detector - Detects breaks of dynamic support/resistance
#[derive(Debug, Clone)]
pub struct TrendlineBreak {
    lookback: usize,
    break_threshold: f64,
}

impl TrendlineBreak {
    pub fn new(lookback: usize, break_threshold: f64) -> Result<Self> {
        if lookback < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if break_threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "break_threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { lookback, break_threshold })
    }

    /// Calculate trendline break signal: positive = break above, negative = break below
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            // Simple linear regression on highs for resistance trendline
            let mut sum_x = 0.0;
            let mut sum_y_high = 0.0;
            let mut sum_y_low = 0.0;
            let mut sum_xy_high = 0.0;
            let mut sum_xy_low = 0.0;
            let mut sum_x2 = 0.0;
            let count = (i - start) as f64;

            for (idx, j) in (start..i).enumerate() {
                let x = idx as f64;
                sum_x += x;
                sum_y_high += high[j];
                sum_y_low += low[j];
                sum_xy_high += x * high[j];
                sum_xy_low += x * low[j];
                sum_x2 += x * x;
            }

            let denom = count * sum_x2 - sum_x * sum_x;
            if denom.abs() < 1e-10 {
                continue;
            }

            // Resistance trendline slope and intercept
            let slope_r = (count * sum_xy_high - sum_x * sum_y_high) / denom;
            let intercept_r = (sum_y_high - slope_r * sum_x) / count;
            let trendline_resistance = intercept_r + slope_r * count;

            // Support trendline slope and intercept
            let slope_s = (count * sum_xy_low - sum_x * sum_y_low) / denom;
            let intercept_s = (sum_y_low - slope_s * sum_x) / count;
            let trendline_support = intercept_s + slope_s * count;

            // Check for breaks
            let break_amount_above = (close[i] - trendline_resistance) / trendline_resistance * 100.0;
            let break_amount_below = (trendline_support - close[i]) / trendline_support * 100.0;

            if break_amount_above > self.break_threshold {
                result[i] = break_amount_above;
            } else if break_amount_below > self.break_threshold {
                result[i] = -break_amount_below;
            }
        }

        result
    }
}

impl TechnicalIndicator for TrendlineBreak {
    fn name(&self) -> &str {
        "Trendline Break"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

/// Psychological Levels - Round number support/resistance proximity
#[derive(Debug, Clone)]
pub struct PsychologicalLevels {
    round_size: f64,
}

impl PsychologicalLevels {
    pub fn new(round_size: f64) -> Result<Self> {
        if round_size <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "round_size".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self { round_size })
    }

    /// Calculate proximity to nearest psychological level (0-100, 100 = at level)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in 0..n {
            let nearest_level = (close[i] / self.round_size).round() * self.round_size;
            let distance = (close[i] - nearest_level).abs();
            let max_distance = self.round_size / 2.0;

            // Proximity: 100 = at level, 0 = furthest from level
            result[i] = (1.0 - distance / max_distance) * 100.0;
        }

        result
    }

    /// Get nearest support and resistance psychological levels
    pub fn calculate_levels(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut support = vec![0.0; n];
        let mut resistance = vec![0.0; n];

        for i in 0..n {
            let lower = (close[i] / self.round_size).floor() * self.round_size;
            let upper = lower + self.round_size;

            support[i] = lower;
            resistance[i] = upper;
        }

        (support, resistance)
    }
}

impl TechnicalIndicator for PsychologicalLevels {
    fn name(&self) -> &str {
        "Psychological Levels"
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.3).sin() * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let volume: Vec<f64> = (0..30).map(|i| 1000.0 + (i as f64 * 0.5).sin() * 500.0).collect();
        (high, low, close, volume)
    }

    #[test]
    fn test_dynamic_support_resistance() {
        let (high, low, close, _) = make_test_data();
        let dsr = DynamicSupportResistance::new(10, 5.0).unwrap();
        let (support, resistance) = dsr.calculate(&high, &low, &close);

        assert_eq!(support.len(), close.len());
        assert_eq!(resistance.len(), close.len());

        // Resistance should be above support
        for i in 15..close.len() {
            assert!(resistance[i] >= support[i]);
        }
    }

    #[test]
    fn test_price_clusters() {
        let (_, _, close, _) = make_test_data();
        let pc = PriceClusters::new(15, 5).unwrap();
        let result = pc.calculate(&close);

        assert_eq!(result.len(), close.len());

        // Values should be percentages
        for i in 20..close.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_volume_support_resistance() {
        let (high, low, _, volume) = make_test_data();
        let vsr = VolumeSupportResistance::new(15, 10).unwrap();
        let poc = vsr.calculate(&high, &low, &volume);

        assert_eq!(poc.len(), high.len());

        // POC should be within price range
        for i in 20..poc.len() {
            assert!(poc[i] >= low[i] - 10.0 && poc[i] <= high[i] + 10.0);
        }
    }

    #[test]
    fn test_swing_level_detector() {
        let (high, low, _, _) = make_test_data();
        let sld = SwingLevelDetector::new(3).unwrap();
        let (support, resistance) = sld.calculate(&high, &low);

        assert_eq!(support.len(), high.len());
        assert_eq!(resistance.len(), high.len());
    }

    #[test]
    fn test_trendline_break() {
        let (high, low, close, _) = make_test_data();
        let tb = TrendlineBreak::new(10, 0.5).unwrap();
        let result = tb.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_psychological_levels() {
        let close = vec![98.5, 99.0, 100.0, 100.5, 101.0, 99.5, 100.0];
        let pl = PsychologicalLevels::new(10.0).unwrap();
        let result = pl.calculate(&close);

        assert_eq!(result.len(), close.len());

        // At round number (100.0), proximity should be 100
        assert!((result[2] - 100.0).abs() < 1e-10);

        // Further from round number should have lower proximity
        assert!(result[0] < result[2]);
    }

    #[test]
    fn test_psychological_levels_support_resistance() {
        let close = vec![105.0];
        let pl = PsychologicalLevels::new(10.0).unwrap();
        let (support, resistance) = pl.calculate_levels(&close);

        assert_eq!(support[0], 100.0);
        assert_eq!(resistance[0], 110.0);
    }
}
