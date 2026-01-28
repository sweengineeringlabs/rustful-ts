//! Extended Pattern Indicators
//!
//! Additional pattern recognition indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Gap Analysis - Detect and classify gaps
#[derive(Debug, Clone)]
pub struct GapAnalysis {
    min_gap_percent: f64,
}

impl GapAnalysis {
    pub fn new(min_gap_percent: f64) -> Result<Self> {
        if min_gap_percent <= 0.0 || min_gap_percent > 10.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_gap_percent".to_string(),
                reason: "must be between 0 and 10".to_string(),
            });
        }
        Ok(Self { min_gap_percent })
    }

    /// Calculate gap signals (1 = gap up, -1 = gap down, 0 = no gap)
    /// Also returns gap size as percentage
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut signal = vec![0.0; n];
        let mut gap_size = vec![0.0; n];

        for i in 1..n {
            // Gap up: low[i] > high[i-1]
            if low[i] > high[i - 1] {
                let gap_pct = (low[i] - high[i - 1]) / close[i - 1] * 100.0;
                if gap_pct >= self.min_gap_percent {
                    signal[i] = 1.0;
                    gap_size[i] = gap_pct;
                }
            }
            // Gap down: high[i] < low[i-1]
            else if high[i] < low[i - 1] {
                let gap_pct = (low[i - 1] - high[i]) / close[i - 1] * 100.0;
                if gap_pct >= self.min_gap_percent {
                    signal[i] = -1.0;
                    gap_size[i] = -gap_pct;
                }
            }
        }
        (signal, gap_size)
    }
}

impl TechnicalIndicator for GapAnalysis {
    fn name(&self) -> &str {
        "Gap Analysis"
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (signal, gap_size) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(signal, gap_size))
    }
}

/// Inside Bar - Inside bar pattern detection
#[derive(Debug, Clone)]
pub struct InsideBar;

impl InsideBar {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    /// Detect inside bars (1 = inside bar, 0 = not)
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut result = vec![0.0; n];

        for i in 1..n {
            // Inside bar: current range is within previous range
            if high[i] <= high[i - 1] && low[i] >= low[i - 1] {
                result[i] = 1.0;
            }
        }
        result
    }
}

impl Default for InsideBar {
    fn default() -> Self {
        Self
    }
}

impl TechnicalIndicator for InsideBar {
    fn name(&self) -> &str {
        "Inside Bar"
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low)))
    }
}

/// Outside Bar - Outside bar pattern detection
#[derive(Debug, Clone)]
pub struct OutsideBar;

impl OutsideBar {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    /// Detect outside bars (1 = bullish outside, -1 = bearish outside, 0 = not)
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut result = vec![0.0; n];

        for i in 1..n {
            // Outside bar: current range engulfs previous range
            if high[i] > high[i - 1] && low[i] < low[i - 1] {
                // Determine direction by close relative to open
                if close[i] > open[i] {
                    result[i] = 1.0; // Bullish outside bar
                } else if close[i] < open[i] {
                    result[i] = -1.0; // Bearish outside bar
                }
            }
        }
        result
    }
}

impl Default for OutsideBar {
    fn default() -> Self {
        Self
    }
}

impl TechnicalIndicator for OutsideBar {
    fn name(&self) -> &str {
        "Outside Bar"
    }

    fn min_periods(&self) -> usize {
        2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close)))
    }
}

/// Narrow Range - NR4/NR7 pattern detection
#[derive(Debug, Clone)]
pub struct NarrowRange {
    lookback: usize,
}

impl NarrowRange {
    pub fn new(lookback: usize) -> Result<Self> {
        if lookback < 4 || lookback > 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be between 4 and 20".to_string(),
            });
        }
        Ok(Self { lookback })
    }

    /// Detect narrow range bars (1 = narrowest in lookback, 0 = not)
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let current_range = high[i] - low[i];

            // Check if current range is smallest in lookback period
            let start = i.saturating_sub(self.lookback - 1);
            let is_narrowest = (start..i)
                .all(|j| (high[j] - low[j]) >= current_range);

            if is_narrowest {
                result[i] = 1.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for NarrowRange {
    fn name(&self) -> &str {
        "Narrow Range"
    }

    fn min_periods(&self) -> usize {
        self.lookback
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low)))
    }
}

/// Wide Range Bar - Wide range bar detection
#[derive(Debug, Clone)]
pub struct WideRangeBar {
    lookback: usize,
    multiplier: f64,
}

impl WideRangeBar {
    pub fn new(lookback: usize, multiplier: f64) -> Result<Self> {
        if lookback < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if multiplier <= 1.0 || multiplier > 5.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "multiplier".to_string(),
                reason: "must be between 1 and 5".to_string(),
            });
        }
        Ok(Self { lookback, multiplier })
    }

    /// Detect wide range bars (1 = bullish WRB, -1 = bearish WRB, 0 = not)
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let current_range = high[i] - low[i];

            // Calculate average range
            let avg_range: f64 = (i - self.lookback..i)
                .map(|j| high[j] - low[j])
                .sum::<f64>() / self.lookback as f64;

            // Wide range bar if current > multiplier * average
            if current_range > avg_range * self.multiplier {
                if close[i] > open[i] {
                    result[i] = 1.0; // Bullish WRB
                } else if close[i] < open[i] {
                    result[i] = -1.0; // Bearish WRB
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for WideRangeBar {
    fn name(&self) -> &str {
        "Wide Range Bar"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close)))
    }
}

/// Trend Bar - Classify bars as trend or non-trend
#[derive(Debug, Clone)]
pub struct TrendBar {
    body_ratio_threshold: f64,
}

impl TrendBar {
    pub fn new(body_ratio_threshold: f64) -> Result<Self> {
        if body_ratio_threshold <= 0.0 || body_ratio_threshold > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "body_ratio_threshold".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { body_ratio_threshold })
    }

    /// Classify trend bars (1 = bullish trend, -1 = bearish trend, 0 = non-trend)
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut result = vec![0.0; n];

        for i in 0..n {
            let range = high[i] - low[i];
            if range > 1e-10 {
                let body = (close[i] - open[i]).abs();
                let body_ratio = body / range;

                // Trend bar if body is significant portion of range
                if body_ratio >= self.body_ratio_threshold {
                    if close[i] > open[i] {
                        result[i] = 1.0; // Bullish trend bar
                    } else {
                        result[i] = -1.0; // Bearish trend bar
                    }
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for TrendBar {
    fn name(&self) -> &str {
        "Trend Bar"
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.open, &data.high, &data.low, &data.close)))
    }
}

/// Consolidation Pattern - Detect consolidation periods
#[derive(Debug, Clone)]
pub struct ConsolidationPattern {
    period: usize,
    threshold: f64,
}

impl ConsolidationPattern {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if threshold <= 0.0 || threshold > 0.2 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be between 0 and 0.2".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Detect consolidation (1 = consolidating, 0 = not)
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i - self.period;

            // Find range over period
            let period_high = high[start..=i].iter().cloned().fold(f64::MIN, f64::max);
            let period_low = low[start..=i].iter().cloned().fold(f64::MAX, f64::min);
            let range = period_high - period_low;

            // Calculate range as percentage of price
            let avg_price = close[start..=i].iter().sum::<f64>() / (self.period + 1) as f64;
            let range_pct = range / avg_price;

            // Consolidation if range is below threshold
            if range_pct < self.threshold {
                result[i] = 1.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for ConsolidationPattern {
    fn name(&self) -> &str {
        "Consolidation Pattern"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let open = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5,
                       107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0,
                       114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0];
        let high = vec![102.0, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0, 108.0, 107.5,
                       109.0, 110.0, 109.5, 111.0, 112.0, 111.5, 113.0, 114.0, 113.5, 115.0,
                       116.0, 115.5, 117.0, 118.0, 117.5, 119.0, 120.0, 119.5, 121.0, 122.0];
        let low = vec![98.0, 99.0, 100.0, 99.5, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5,
                      105.0, 106.0, 105.5, 107.0, 108.0, 107.5, 109.0, 110.0, 109.5, 111.0,
                      112.0, 111.5, 113.0, 114.0, 113.5, 115.0, 116.0, 115.5, 117.0, 118.0];
        let close = vec![101.0, 102.0, 103.0, 102.5, 104.0, 105.0, 104.5, 106.0, 107.0, 106.5,
                        108.0, 109.0, 108.5, 110.0, 111.0, 110.5, 112.0, 113.0, 112.5, 114.0,
                        115.0, 114.5, 116.0, 117.0, 116.5, 118.0, 119.0, 118.5, 120.0, 121.0];
        (open, high, low, close)
    }

    #[test]
    fn test_gap_analysis() {
        let (_, high, low, close) = make_test_data();
        let ga = GapAnalysis::new(0.5).unwrap();
        let (signal, gap_size) = ga.calculate(&high, &low, &close);

        assert_eq!(signal.len(), close.len());
        assert_eq!(gap_size.len(), close.len());
    }

    #[test]
    fn test_inside_bar() {
        let (_, high, low, _) = make_test_data();
        let ib = InsideBar::new().unwrap();
        let result = ib.calculate(&high, &low);

        assert_eq!(result.len(), high.len());
    }

    #[test]
    fn test_outside_bar() {
        let (open, high, low, close) = make_test_data();
        let ob = OutsideBar::new().unwrap();
        let result = ob.calculate(&open, &high, &low, &close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_narrow_range() {
        let (_, high, low, _) = make_test_data();
        let nr = NarrowRange::new(7).unwrap();
        let result = nr.calculate(&high, &low);

        assert_eq!(result.len(), high.len());
    }

    #[test]
    fn test_wide_range_bar() {
        let (open, high, low, close) = make_test_data();
        let wrb = WideRangeBar::new(10, 1.5).unwrap();
        let result = wrb.calculate(&open, &high, &low, &close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_trend_bar() {
        let (open, high, low, close) = make_test_data();
        let tb = TrendBar::new(0.6).unwrap();
        let result = tb.calculate(&open, &high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Should detect some trend bars in trending data
    }

    #[test]
    fn test_consolidation_pattern() {
        let (_, high, low, close) = make_test_data();
        let cp = ConsolidationPattern::new(10, 0.1).unwrap();
        let result = cp.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
    }
}
