//! DeltaDivergence (IND-218) - Price vs delta divergence detection
//!
//! Detects divergences between price action and cumulative delta,
//! which can signal potential trend reversals or exhaustion.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator, SignalIndicator, IndicatorSignal,
};

/// Type of divergence detected
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeltaDivergenceType {
    /// No divergence
    None,
    /// Bullish divergence: price making lower lows, delta making higher lows
    BullishRegular,
    /// Bearish divergence: price making higher highs, delta making lower highs
    BearishRegular,
    /// Hidden bullish: price making higher lows, delta making lower lows
    BullishHidden,
    /// Hidden bearish: price making lower highs, delta making higher highs
    BearishHidden,
}

/// Delta Divergence Output
#[derive(Debug, Clone)]
pub struct DeltaDivergenceOutput {
    /// Divergence type at each bar
    pub divergence_type: Vec<DeltaDivergenceType>,
    /// Divergence strength (-100 to 100)
    pub divergence_strength: Vec<f64>,
    /// Cumulative delta for reference
    pub cumulative_delta: Vec<f64>,
    /// Signal values: 1 = bullish divergence, -1 = bearish, 0 = none
    pub signal: Vec<f64>,
}

/// Delta Divergence Configuration
#[derive(Debug, Clone)]
pub struct DeltaDivergenceConfig {
    /// Lookback period for finding pivots
    pub pivot_lookback: usize,
    /// Minimum bars between pivots
    pub min_pivot_distance: usize,
    /// Threshold for significant divergence
    pub divergence_threshold: f64,
    /// Whether to detect hidden divergences
    pub detect_hidden: bool,
}

impl Default for DeltaDivergenceConfig {
    fn default() -> Self {
        Self {
            pivot_lookback: 5,
            min_pivot_distance: 3,
            divergence_threshold: 5.0,
            detect_hidden: true,
        }
    }
}

/// DeltaDivergence (IND-218)
///
/// Detects divergences between price and cumulative delta.
///
/// Regular Divergences (trend reversal signals):
/// - Bullish: Price makes lower low, delta makes higher low
/// - Bearish: Price makes higher high, delta makes lower high
///
/// Hidden Divergences (trend continuation signals):
/// - Bullish Hidden: Price makes higher low, delta makes lower low
/// - Bearish Hidden: Price makes lower high, delta makes higher high
///
/// A divergence suggests that the underlying buying/selling pressure
/// is not supporting the current price movement.
#[derive(Debug, Clone)]
pub struct DeltaDivergence {
    config: DeltaDivergenceConfig,
}

impl DeltaDivergence {
    pub fn new(config: DeltaDivergenceConfig) -> Result<Self> {
        if config.pivot_lookback < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "pivot_lookback".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if config.min_pivot_distance < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "min_pivot_distance".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self {
            config: DeltaDivergenceConfig::default(),
        }
    }

    /// Find local highs/lows for divergence detection
    fn find_pivots(&self, data: &[f64]) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
        let n = data.len();
        let lookback = self.config.pivot_lookback;

        // Track most recent high/low pivot at each index
        let mut highs = vec![None; n];
        let mut lows = vec![None; n];

        let mut last_high: Option<usize> = None;
        let mut last_low: Option<usize> = None;

        for i in lookback..(n.saturating_sub(lookback)) {
            let is_high = (0..lookback).all(|j| data[i] >= data[i - j - 1])
                && (0..lookback).all(|j| data[i] >= data[i + j + 1]);
            let is_low = (0..lookback).all(|j| data[i] <= data[i - j - 1])
                && (0..lookback).all(|j| data[i] <= data[i + j + 1]);

            if is_high {
                last_high = Some(i);
            }
            if is_low {
                last_low = Some(i);
            }

            highs[i] = last_high;
            lows[i] = last_low;
        }

        // Fill forward
        for i in 1..n {
            if highs[i].is_none() && highs[i - 1].is_some() {
                highs[i] = highs[i - 1];
            }
            if lows[i].is_none() && lows[i - 1].is_some() {
                lows[i] = lows[i - 1];
            }
        }

        (highs, lows)
    }

    /// Calculate cumulative delta
    fn calculate_cumulative_delta(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut cumulative = vec![0.0; n];
        let mut cumsum = 0.0;

        for i in 0..n {
            let range = high[i] - low[i];
            if range > 0.0 {
                let position = (close[i] - low[i]) / range;
                let delta = volume[i] * (2.0 * position - 1.0);
                cumsum += delta;
            }
            cumulative[i] = cumsum;
        }

        cumulative
    }

    /// Calculate divergence with full output
    pub fn calculate_full(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> DeltaDivergenceOutput {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut divergence_type = vec![DeltaDivergenceType::None; n];
        let mut divergence_strength = vec![0.0; n];
        let mut signal = vec![0.0; n];

        let cumulative_delta = self.calculate_cumulative_delta(high, low, close, volume);

        if n < self.config.pivot_lookback * 2 + self.config.min_pivot_distance {
            return DeltaDivergenceOutput {
                divergence_type,
                divergence_strength,
                cumulative_delta,
                signal,
            };
        }

        // Find pivots in price and delta
        let (price_highs, price_lows) = self.find_pivots(close);
        let (delta_highs, delta_lows) = self.find_pivots(&cumulative_delta);

        // Detect divergences
        for i in (self.config.pivot_lookback * 2 + self.config.min_pivot_distance)..n {
            // Check for bullish regular divergence (lower price lows, higher delta lows)
            if let (Some(curr_price_low), Some(curr_delta_low)) = (price_lows[i], delta_lows[i]) {
                // Look for previous low
                let search_start = i.saturating_sub(20).max(self.config.pivot_lookback);
                for j in search_start..curr_price_low.saturating_sub(self.config.min_pivot_distance) {
                    if let (Some(prev_price_low), Some(prev_delta_low)) = (price_lows[j], delta_lows[j]) {
                        if prev_price_low != curr_price_low && prev_delta_low != curr_delta_low {
                            let price_lower = close[curr_price_low] < close[prev_price_low];
                            let delta_higher = cumulative_delta[curr_delta_low] > cumulative_delta[prev_delta_low];

                            if price_lower && delta_higher {
                                divergence_type[i] = DeltaDivergenceType::BullishRegular;
                                let price_change = (close[prev_price_low] - close[curr_price_low]) / close[prev_price_low] * 100.0;
                                let delta_change = cumulative_delta[curr_delta_low] - cumulative_delta[prev_delta_low];
                                divergence_strength[i] = price_change.abs() + delta_change.abs().min(50.0);
                                signal[i] = 1.0;
                                break;
                            }

                            // Hidden bullish: higher price low, lower delta low
                            if self.config.detect_hidden && !price_lower && !delta_higher {
                                divergence_type[i] = DeltaDivergenceType::BullishHidden;
                                divergence_strength[i] = 25.0;
                                signal[i] = 0.5;
                                break;
                            }
                        }
                    }
                }
            }

            // Check for bearish regular divergence (higher price highs, lower delta highs)
            if divergence_type[i] == DeltaDivergenceType::None {
                if let (Some(curr_price_high), Some(curr_delta_high)) = (price_highs[i], delta_highs[i]) {
                    let search_start = i.saturating_sub(20).max(self.config.pivot_lookback);
                    for j in search_start..curr_price_high.saturating_sub(self.config.min_pivot_distance) {
                        if let (Some(prev_price_high), Some(prev_delta_high)) = (price_highs[j], delta_highs[j]) {
                            if prev_price_high != curr_price_high && prev_delta_high != curr_delta_high {
                                let price_higher = close[curr_price_high] > close[prev_price_high];
                                let delta_lower = cumulative_delta[curr_delta_high] < cumulative_delta[prev_delta_high];

                                if price_higher && delta_lower {
                                    divergence_type[i] = DeltaDivergenceType::BearishRegular;
                                    let price_change = (close[curr_price_high] - close[prev_price_high]) / close[prev_price_high] * 100.0;
                                    let delta_change = cumulative_delta[prev_delta_high] - cumulative_delta[curr_delta_high];
                                    divergence_strength[i] = price_change.abs() + delta_change.abs().min(50.0);
                                    signal[i] = -1.0;
                                    break;
                                }

                                // Hidden bearish: lower price high, higher delta high
                                if self.config.detect_hidden && !price_higher && !delta_lower {
                                    divergence_type[i] = DeltaDivergenceType::BearishHidden;
                                    divergence_strength[i] = 25.0;
                                    signal[i] = -0.5;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        DeltaDivergenceOutput {
            divergence_type,
            divergence_strength,
            cumulative_delta,
            signal,
        }
    }

    /// Calculate divergence signal only
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        self.calculate_full(high, low, close, volume).signal
    }
}

impl TechnicalIndicator for DeltaDivergence {
    fn name(&self) -> &str {
        "Delta Divergence"
    }

    fn min_periods(&self) -> usize {
        self.config.pivot_lookback * 2 + self.config.min_pivot_distance + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min = self.min_periods();
        if data.close.len() < min {
            return Err(IndicatorError::InsufficientData {
                required: min,
                got: data.close.len(),
            });
        }

        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            output.signal,
            output.divergence_strength,
            output.cumulative_delta,
        ))
    }
}

impl SignalIndicator for DeltaDivergence {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);

        if let Some(&last) = output.signal.last() {
            if last > 0.0 {
                return Ok(IndicatorSignal::Bullish);
            } else if last < 0.0 {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let output = self.calculate_full(&data.high, &data.low, &data.close, &data.volume);

        Ok(output
            .signal
            .iter()
            .map(|&s| {
                if s > 0.0 {
                    IndicatorSignal::Bullish
                } else if s < 0.0 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect())
    }
}

impl Default for DeltaDivergence {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_divergence_basic() {
        let dd = DeltaDivergence::default_config();

        // Create data with potential bullish divergence
        // Price makes lower low, but delta (closes near highs) should make higher low
        let high = vec![110.0, 109.0, 108.0, 107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0,
                       100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0,
                       96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0];
        let low = vec![105.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0,
                      95.0, 96.0, 97.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0,
                      91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0];
        let close = vec![109.0, 108.0, 107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0,
                        99.0, 100.0, 101.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0,
                        95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0];
        let volume = vec![1000.0; 30];

        let result = dd.calculate(&high, &low, &close, &volume);
        assert_eq!(result.len(), 30);
    }

    #[test]
    fn test_delta_divergence_full_output() {
        let dd = DeltaDivergence::default_config();
        let high = vec![105.0; 30];
        let low = vec![100.0; 30];
        let close = vec![103.0; 30];
        let volume = vec![1000.0; 30];

        let output = dd.calculate_full(&high, &low, &close, &volume);

        assert_eq!(output.divergence_type.len(), 30);
        assert_eq!(output.divergence_strength.len(), 30);
        assert_eq!(output.cumulative_delta.len(), 30);
        assert_eq!(output.signal.len(), 30);
    }

    #[test]
    fn test_delta_divergence_config() {
        let config = DeltaDivergenceConfig {
            pivot_lookback: 3,
            min_pivot_distance: 2,
            divergence_threshold: 10.0,
            detect_hidden: false,
        };
        let dd = DeltaDivergence::new(config).unwrap();

        assert_eq!(dd.min_periods(), 3 * 2 + 2 + 1);
    }

    #[test]
    fn test_delta_divergence_invalid_config() {
        let config = DeltaDivergenceConfig {
            pivot_lookback: 1,
            min_pivot_distance: 1,
            divergence_threshold: 5.0,
            detect_hidden: true,
        };
        assert!(DeltaDivergence::new(config).is_err());
    }

    #[test]
    fn test_divergence_type_enum() {
        let dt = DeltaDivergenceType::BullishRegular;
        assert_eq!(dt, DeltaDivergenceType::BullishRegular);

        let dt2 = DeltaDivergenceType::None;
        assert_ne!(dt, dt2);
    }

    #[test]
    fn test_delta_divergence_insufficient_data() {
        let dd = DeltaDivergence::default_config();
        let high = vec![105.0; 5];
        let low = vec![100.0; 5];
        let close = vec![103.0; 5];
        let volume = vec![1000.0; 5];

        let data = OHLCVSeries {
            open: vec![101.0; 5],
            high,
            low,
            close,
            volume,
        };

        let result = dd.compute(&data);
        assert!(result.is_err());
    }
}
