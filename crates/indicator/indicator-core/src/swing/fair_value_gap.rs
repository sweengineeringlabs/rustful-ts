//! Fair Value Gap (FVG) / Imbalance detection implementation.
//!
//! Identifies price imbalances where there are gaps between candle wicks.

use crate::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};
use serde::{Deserialize, Serialize};

/// Fair Value Gap type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FVGType {
    /// Bullish FVG (gap up - demand imbalance)
    Bullish,
    /// Bearish FVG (gap down - supply imbalance)
    Bearish,
}

/// Represents an identified Fair Value Gap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FVGZone {
    /// Type of FVG
    pub fvg_type: FVGType,
    /// Index where the FVG was formed (middle candle)
    pub index: usize,
    /// Upper boundary of the gap
    pub upper: f64,
    /// Lower boundary of the gap
    pub lower: f64,
    /// Whether the gap has been filled
    pub filled: bool,
    /// Percentage of gap that has been filled (0.0 to 1.0)
    pub fill_percentage: f64,
}

impl FVGZone {
    /// Calculate the size of the gap.
    pub fn size(&self) -> f64 {
        self.upper - self.lower
    }

    /// Check if price is within the gap zone.
    pub fn contains(&self, price: f64) -> bool {
        price >= self.lower && price <= self.upper
    }

    /// Update fill status based on price action.
    pub fn update_fill(&mut self, high: f64, low: f64) {
        match self.fvg_type {
            FVGType::Bullish => {
                // Bullish FVG filled when price returns down into it
                if low <= self.upper {
                    let penetration = (self.upper - low.max(self.lower)) / self.size();
                    self.fill_percentage = penetration.clamp(0.0, 1.0);
                    if low <= self.lower {
                        self.filled = true;
                        self.fill_percentage = 1.0;
                    }
                }
            }
            FVGType::Bearish => {
                // Bearish FVG filled when price returns up into it
                if high >= self.lower {
                    let penetration = (high.min(self.upper) - self.lower) / self.size();
                    self.fill_percentage = penetration.clamp(0.0, 1.0);
                    if high >= self.upper {
                        self.filled = true;
                        self.fill_percentage = 1.0;
                    }
                }
            }
        }
    }
}

/// Fair Value Gap (FVG) indicator.
///
/// Identifies imbalances in price where there is a gap between the wicks
/// of adjacent candles. These gaps often act as support/resistance and
/// price tends to return to fill them.
///
/// Bullish FVG: Current candle's low > Previous candle's high (gap up)
/// Bearish FVG: Current candle's high < Previous candle's low (gap down)
///
/// More specifically, an FVG forms when:
/// - Bullish: Candle 1's high < Candle 3's low (gap in candle 2's range)
/// - Bearish: Candle 1's low > Candle 3's high (gap in candle 2's range)
///
/// Output:
/// - Primary: FVG signal (1 = bullish, -1 = bearish, 0 = none)
/// - Secondary: Upper boundary of FVG
/// - Tertiary: Lower boundary of FVG
#[derive(Debug, Clone)]
pub struct FairValueGap {
    /// Minimum gap size as percentage of price.
    min_gap_percent: f64,
    /// Whether to track gap fill status.
    track_fills: bool,
}

impl FairValueGap {
    /// Create a new Fair Value Gap indicator.
    ///
    /// # Arguments
    /// * `min_gap_percent` - Minimum gap size as percentage (e.g., 0.1 = 0.1%)
    pub fn new(min_gap_percent: f64) -> Self {
        Self {
            min_gap_percent,
            track_fills: true,
        }
    }

    /// Create with default minimum gap of 0.05%.
    pub fn default_gap() -> Self {
        Self::new(0.05)
    }

    /// Disable fill tracking.
    pub fn without_fill_tracking(mut self) -> Self {
        self.track_fills = false;
        self
    }

    /// Calculate FVG values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = high.len();
        let mut fvg_signal = vec![0.0; n];
        let mut fvg_upper = vec![f64::NAN; n];
        let mut fvg_lower = vec![f64::NAN; n];

        if n < 3 {
            return (fvg_signal, fvg_upper, fvg_lower);
        }

        for i in 2..n {
            let candle1_high = high[i - 2];
            let candle1_low = low[i - 2];
            let candle3_high = high[i];
            let candle3_low = low[i];

            let mid_price = (high[i - 1] + low[i - 1]) / 2.0;
            let min_gap = mid_price * (self.min_gap_percent / 100.0);

            // Bullish FVG: Gap between candle 1's high and candle 3's low
            if candle3_low > candle1_high {
                let gap_size = candle3_low - candle1_high;
                if gap_size >= min_gap {
                    fvg_signal[i] = 1.0;
                    fvg_upper[i] = candle3_low;
                    fvg_lower[i] = candle1_high;
                }
            }

            // Bearish FVG: Gap between candle 1's low and candle 3's high
            if candle3_high < candle1_low {
                let gap_size = candle1_low - candle3_high;
                if gap_size >= min_gap {
                    fvg_signal[i] = -1.0;
                    fvg_upper[i] = candle1_low;
                    fvg_lower[i] = candle3_high;
                }
            }
        }

        (fvg_signal, fvg_upper, fvg_lower)
    }

    /// Detect FVGs and return structured data.
    pub fn detect_gaps(&self, high: &[f64], low: &[f64]) -> Vec<FVGZone> {
        let n = high.len();
        let mut gaps = Vec::new();

        if n < 3 {
            return gaps;
        }

        for i in 2..n {
            let candle1_high = high[i - 2];
            let candle1_low = low[i - 2];
            let candle3_high = high[i];
            let candle3_low = low[i];

            let mid_price = (high[i - 1] + low[i - 1]) / 2.0;
            let min_gap = mid_price * (self.min_gap_percent / 100.0);

            // Bullish FVG
            if candle3_low > candle1_high {
                let gap_size = candle3_low - candle1_high;
                if gap_size >= min_gap {
                    let mut zone = FVGZone {
                        fvg_type: FVGType::Bullish,
                        index: i,
                        upper: candle3_low,
                        lower: candle1_high,
                        filled: false,
                        fill_percentage: 0.0,
                    };

                    // Track fills if enabled
                    if self.track_fills {
                        for j in (i + 1)..n {
                            zone.update_fill(high[j], low[j]);
                            if zone.filled {
                                break;
                            }
                        }
                    }

                    gaps.push(zone);
                }
            }

            // Bearish FVG
            if candle3_high < candle1_low {
                let gap_size = candle1_low - candle3_high;
                if gap_size >= min_gap {
                    let mut zone = FVGZone {
                        fvg_type: FVGType::Bearish,
                        index: i,
                        upper: candle1_low,
                        lower: candle3_high,
                        filled: false,
                        fill_percentage: 0.0,
                    };

                    if self.track_fills {
                        for j in (i + 1)..n {
                            zone.update_fill(high[j], low[j]);
                            if zone.filled {
                                break;
                            }
                        }
                    }

                    gaps.push(zone);
                }
            }
        }

        gaps
    }

    /// Get only unfilled gaps.
    pub fn unfilled_gaps(&self, high: &[f64], low: &[f64]) -> Vec<FVGZone> {
        self.detect_gaps(high, low)
            .into_iter()
            .filter(|g| !g.filled)
            .collect()
    }
}

impl TechnicalIndicator for FairValueGap {
    fn name(&self) -> &str {
        "FairValueGap"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < 3 {
            return Err(IndicatorError::InsufficientData {
                required: 3,
                got: data.high.len(),
            });
        }

        let (signal, upper, lower) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::triple(signal, upper, lower))
    }

    fn min_periods(&self) -> usize {
        3
    }

    fn output_features(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fvg_bullish() {
        let fvg = FairValueGap::new(0.01);

        // Bullish FVG: gap between candle 1 high and candle 3 low
        let high = vec![100.0, 102.0, 105.0, 107.0, 108.0];
        let low = vec![98.0, 100.0, 101.0, 105.0, 106.0];

        let (signal, upper, lower) = fvg.calculate(&high, &low);

        assert_eq!(signal.len(), 5);
        // Check for bullish FVG where candle 3's low > candle 1's high
    }

    #[test]
    fn test_fvg_bearish() {
        let fvg = FairValueGap::new(0.01);

        // Bearish FVG: gap between candle 1 low and candle 3 high
        let high = vec![107.0, 105.0, 102.0, 100.0, 99.0];
        let low = vec![105.0, 103.0, 100.0, 98.0, 97.0];

        let (signal, upper, lower) = fvg.calculate(&high, &low);

        assert_eq!(signal.len(), 5);
    }

    #[test]
    fn test_fvg_zone_size() {
        let zone = FVGZone {
            fvg_type: FVGType::Bullish,
            index: 5,
            upper: 105.0,
            lower: 100.0,
            filled: false,
            fill_percentage: 0.0,
        };

        assert_eq!(zone.size(), 5.0);
        assert!(zone.contains(102.0));
        assert!(!zone.contains(99.0));
    }

    #[test]
    fn test_fvg_zone_fill_bullish() {
        let mut zone = FVGZone {
            fvg_type: FVGType::Bullish,
            index: 5,
            upper: 105.0,
            lower: 100.0,
            filled: false,
            fill_percentage: 0.0,
        };

        // Price doesn't reach zone
        zone.update_fill(110.0, 107.0);
        assert!(!zone.filled);
        assert_eq!(zone.fill_percentage, 0.0);

        // Price partially fills zone
        zone.update_fill(106.0, 103.0);
        assert!(!zone.filled);
        assert!(zone.fill_percentage > 0.0);

        // Reset and fully fill
        zone.filled = false;
        zone.fill_percentage = 0.0;
        zone.update_fill(101.0, 98.0);
        assert!(zone.filled);
        assert_eq!(zone.fill_percentage, 1.0);
    }

    #[test]
    fn test_fvg_zone_fill_bearish() {
        let mut zone = FVGZone {
            fvg_type: FVGType::Bearish,
            index: 5,
            upper: 105.0,
            lower: 100.0,
            filled: false,
            fill_percentage: 0.0,
        };

        // Price doesn't reach zone
        zone.update_fill(98.0, 95.0);
        assert!(!zone.filled);

        // Price fully fills zone
        zone.update_fill(107.0, 104.0);
        assert!(zone.filled);
        assert_eq!(zone.fill_percentage, 1.0);
    }

    #[test]
    fn test_fvg_detect_gaps() {
        let fvg = FairValueGap::new(0.01);

        // Create clear gaps
        let high = vec![100.0, 102.0, 110.0, 112.0, 114.0];
        let low = vec![98.0, 100.0, 103.0, 110.0, 112.0];

        let gaps = fvg.detect_gaps(&high, &low);

        for gap in &gaps {
            assert!(gap.upper >= gap.lower);
        }
    }

    #[test]
    fn test_fvg_technical_indicator() {
        let fvg = FairValueGap::new(0.1);

        let mut data = OHLCVSeries::new();
        for i in 0..10 {
            data.open.push(100.0 + i as f64 * 2.0);
            data.high.push(102.0 + i as f64 * 2.0);
            data.low.push(98.0 + i as f64 * 2.0);
            data.close.push(101.0 + i as f64 * 2.0);
            data.volume.push(1000.0);
        }

        let output = fvg.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 10);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }
}
