//! Market Structure indicator implementation.
//!
//! Identifies higher highs (HH), lower lows (LL), higher lows (HL), and lower highs (LH).

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Market structure point type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StructurePoint {
    /// Higher High - bullish continuation
    HigherHigh,
    /// Higher Low - bullish continuation
    HigherLow,
    /// Lower High - bearish continuation
    LowerHigh,
    /// Lower Low - bearish continuation
    LowerLow,
    /// No significant structure point
    None,
}

impl StructurePoint {
    /// Convert to numeric value for output.
    /// HH = 2, HL = 1, LH = -1, LL = -2, None = 0
    pub fn to_numeric(&self) -> f64 {
        match self {
            StructurePoint::HigherHigh => 2.0,
            StructurePoint::HigherLow => 1.0,
            StructurePoint::LowerHigh => -1.0,
            StructurePoint::LowerLow => -2.0,
            StructurePoint::None => 0.0,
        }
    }
}

/// Market Structure trend state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketTrend {
    /// Bullish trend (HH + HL)
    Bullish,
    /// Bearish trend (LH + LL)
    Bearish,
    /// Ranging/Consolidation
    Ranging,
}

/// Market Structure indicator.
///
/// Analyzes price action to identify higher highs (HH), higher lows (HL),
/// lower highs (LH), and lower lows (LL). These are fundamental building
/// blocks of price action analysis.
///
/// Bullish structure: Series of HH and HL
/// Bearish structure: Series of LH and LL
///
/// Output:
/// - Primary: Structure point type (HH=2, HL=1, LH=-1, LL=-2, None=0)
/// - Secondary: Market trend (1=Bullish, -1=Bearish, 0=Ranging)
#[derive(Debug, Clone)]
pub struct MarketStructure {
    /// Lookback period for swing detection.
    lookback: usize,
}

impl MarketStructure {
    /// Create a new Market Structure indicator.
    ///
    /// # Arguments
    /// * `lookback` - Number of bars to look back for swing detection
    pub fn new(lookback: usize) -> Self {
        Self {
            lookback: lookback.max(2),
        }
    }

    /// Create with default lookback of 5.
    pub fn default_lookback() -> Self {
        Self::new(5)
    }

    /// Detect swing highs and lows.
    fn detect_swing_points(
        &self,
        high: &[f64],
        low: &[f64],
    ) -> (Vec<Option<f64>>, Vec<Option<f64>>) {
        let n = high.len();
        let mut swing_highs = vec![None; n];
        let mut swing_lows = vec![None; n];

        if n < 2 * self.lookback + 1 {
            return (swing_highs, swing_lows);
        }

        for i in self.lookback..(n - self.lookback) {
            // Check for swing high
            let is_swing_high = (0..self.lookback)
                .all(|j| high[i] >= high[i - j - 1] && high[i] >= high[i + j + 1]);

            if is_swing_high {
                swing_highs[i] = Some(high[i]);
            }

            // Check for swing low
            let is_swing_low = (0..self.lookback)
                .all(|j| low[i] <= low[i - j - 1] && low[i] <= low[i + j + 1]);

            if is_swing_low {
                swing_lows[i] = Some(low[i]);
            }
        }

        (swing_highs, swing_lows)
    }

    /// Calculate market structure.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        let mut structure_points = vec![0.0; n];
        let mut trend = vec![0.0; n];

        let (swing_highs, swing_lows) = self.detect_swing_points(high, low);

        // Track previous swing high and low
        let mut prev_swing_high: Option<f64> = None;
        let mut prev_swing_low: Option<f64> = None;
        let mut last_hh = false;
        let mut last_hl = false;
        let mut last_lh = false;
        let mut last_ll = false;

        for i in 0..n {
            // Check for structure point at swing high
            if let Some(sh) = swing_highs[i] {
                if let Some(prev_sh) = prev_swing_high {
                    if sh > prev_sh {
                        structure_points[i] = StructurePoint::HigherHigh.to_numeric();
                        last_hh = true;
                        last_lh = false;
                    } else if sh < prev_sh {
                        structure_points[i] = StructurePoint::LowerHigh.to_numeric();
                        last_lh = true;
                        last_hh = false;
                    }
                }
                prev_swing_high = Some(sh);
            }

            // Check for structure point at swing low
            if let Some(sl) = swing_lows[i] {
                if let Some(prev_sl) = prev_swing_low {
                    if sl > prev_sl {
                        // Only mark HL if we haven't marked something else this bar
                        if structure_points[i] == 0.0 {
                            structure_points[i] = StructurePoint::HigherLow.to_numeric();
                        }
                        last_hl = true;
                        last_ll = false;
                    } else if sl < prev_sl {
                        if structure_points[i] == 0.0 {
                            structure_points[i] = StructurePoint::LowerLow.to_numeric();
                        }
                        last_ll = true;
                        last_hl = false;
                    }
                }
                prev_swing_low = Some(sl);
            }

            // Determine trend based on recent structure
            if last_hh && last_hl {
                trend[i] = 1.0; // Bullish
            } else if last_lh && last_ll {
                trend[i] = -1.0; // Bearish
            } else if i > 0 {
                trend[i] = trend[i - 1]; // Continue previous trend
            }
        }

        (structure_points, trend)
    }

    /// Get the current market trend.
    pub fn get_trend(&self, high: &[f64], low: &[f64]) -> MarketTrend {
        let (_, trend) = self.calculate(high, low);
        let last = trend.last().copied().unwrap_or(0.0);

        if last > 0.0 {
            MarketTrend::Bullish
        } else if last < 0.0 {
            MarketTrend::Bearish
        } else {
            MarketTrend::Ranging
        }
    }
}

impl TechnicalIndicator for MarketStructure {
    fn name(&self) -> &str {
        "MarketStructure"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < 2 * self.lookback + 1 {
            return Err(IndicatorError::InsufficientData {
                required: 2 * self.lookback + 1,
                got: data.high.len(),
            });
        }

        let (structure_points, trend) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::dual(structure_points, trend))
    }

    fn min_periods(&self) -> usize {
        2 * self.lookback + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for MarketStructure {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (_, trend) = self.calculate(&data.high, &data.low);
        let last = trend.last().copied().unwrap_or(0.0);

        if last > 0.0 {
            Ok(IndicatorSignal::Bullish)
        } else if last < 0.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (_, trend) = self.calculate(&data.high, &data.low);

        let signals = trend
            .iter()
            .map(|&t| {
                if t > 0.0 {
                    IndicatorSignal::Bullish
                } else if t < 0.0 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_structure_bullish() {
        let ms = MarketStructure::new(2);

        // Clear bullish structure: ascending highs and lows
        let high = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0];
        let low = vec![98.0, 99.0, 100.0, 99.5, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0];

        let (structure, trend) = ms.calculate(&high, &low);

        assert_eq!(structure.len(), 11);
        assert_eq!(trend.len(), 11);
    }

    #[test]
    fn test_market_structure_bearish() {
        let ms = MarketStructure::new(2);

        // Clear bearish structure: descending highs and lows
        let high = vec![107.0, 106.0, 105.0, 105.5, 104.0, 103.0, 103.5, 102.0, 101.0, 101.5, 100.0];
        let low = vec![105.0, 104.0, 103.0, 103.5, 102.0, 101.0, 101.5, 100.0, 99.0, 99.5, 98.0];

        let (structure, trend) = ms.calculate(&high, &low);

        assert_eq!(structure.len(), 11);
        assert_eq!(trend.len(), 11);
    }

    #[test]
    fn test_market_structure_technical_indicator() {
        let ms = MarketStructure::new(3);

        let mut data = OHLCVSeries::new();
        for i in 0..20 {
            let base = 100.0 + (i as f64 * 0.3).sin() * 5.0;
            data.open.push(base);
            data.high.push(base + 2.0);
            data.low.push(base - 2.0);
            data.close.push(base + 1.0);
            data.volume.push(1000.0);
        }

        let output = ms.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 20);
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_structure_point_numeric() {
        assert_eq!(StructurePoint::HigherHigh.to_numeric(), 2.0);
        assert_eq!(StructurePoint::HigherLow.to_numeric(), 1.0);
        assert_eq!(StructurePoint::LowerHigh.to_numeric(), -1.0);
        assert_eq!(StructurePoint::LowerLow.to_numeric(), -2.0);
        assert_eq!(StructurePoint::None.to_numeric(), 0.0);
    }
}
