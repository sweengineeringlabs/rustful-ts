//! Fibonacci Channels - Parallel channel levels based on Fibonacci ratios.
//!
//! IND-390: Fibonacci Channels create parallel trend channels where the
//! width between channels is determined by Fibonacci ratios.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};

/// Fibonacci channel level data.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FibChannelLevels {
    /// Base channel line (0%)
    pub base: f64,
    /// 23.6% channel extension
    pub level_236: f64,
    /// 38.2% channel extension
    pub level_382: f64,
    /// 50% channel extension
    pub level_500: f64,
    /// 61.8% channel extension
    pub level_618: f64,
    /// 100% channel extension
    pub level_1000: f64,
    /// 161.8% channel extension
    pub level_1618: f64,
}

/// Fibonacci Channels Indicator
///
/// Creates parallel trend channels extending from a base trend line at
/// Fibonacci-ratio distances. The base channel is drawn between two swing
/// points, and parallel channels are projected at Fibonacci levels.
///
/// # Interpretation
/// - Channel lines act as potential support/resistance
/// - Price often reverses at channel boundaries
/// - Breaking through a channel suggests move to next level
/// - 161.8% extension is often a final target
///
/// # Usage
/// - Identify the primary trend direction
/// - Use channels to set profit targets
/// - Look for reversal signals at channel boundaries
#[derive(Debug, Clone)]
pub struct FibonacciChannels {
    /// Lookback period to find anchor points
    lookback: usize,
    /// Swing detection strength
    swing_strength: usize,
    /// Whether to use high/low for channels or close
    use_extremes: bool,
}

impl FibonacciChannels {
    /// Create a new Fibonacci Channels indicator.
    ///
    /// # Arguments
    /// * `lookback` - Period to find swing points (minimum 10)
    /// * `swing_strength` - Bars required to confirm swing (minimum 2)
    /// * `use_extremes` - Use high/low (true) or close (false)
    pub fn new(lookback: usize, swing_strength: usize, use_extremes: bool) -> Result<Self> {
        if lookback < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if swing_strength < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "swing_strength".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self {
            lookback,
            swing_strength,
            use_extremes,
        })
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self {
            lookback: 20,
            swing_strength: 3,
            use_extremes: true,
        }
    }

    /// Find the trend line slope and intercept using linear regression.
    fn fit_trend_line(&self, prices: &[f64], start: usize, end: usize) -> (f64, f64) {
        let n = (end - start) as f64;
        if n < 2.0 {
            return (0.0, prices.get(start).copied().unwrap_or(0.0));
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (idx, i) in (start..end).enumerate() {
            let x = idx as f64;
            let y = prices[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let denom = n * sum_x2 - sum_x * sum_x;
        if denom.abs() < 1e-10 {
            return (0.0, sum_y / n);
        }

        let slope = (n * sum_xy - sum_x * sum_y) / denom;
        let intercept = (sum_y - slope * sum_x) / n;

        (slope, intercept)
    }

    /// Calculate the channel width (perpendicular distance).
    fn channel_width(&self, high: &[f64], low: &[f64], slope: f64, intercept: f64, start: usize, end: usize) -> f64 {
        let mut max_dist_above = 0.0f64;
        let mut max_dist_below = 0.0f64;

        for (idx, i) in (start..end).enumerate() {
            let x = idx as f64;
            let trend_value = slope * x + intercept;

            let dist_above = high[i] - trend_value;
            let dist_below = trend_value - low[i];

            max_dist_above = max_dist_above.max(dist_above);
            max_dist_below = max_dist_below.max(dist_below);
        }

        max_dist_above.max(max_dist_below)
    }

    /// Calculate Fibonacci channel values.
    ///
    /// Returns base line and channel extensions at various Fibonacci levels.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<Vec<f64>> {
        let n = close.len();
        let mut base = vec![f64::NAN; n];
        let mut level_236 = vec![f64::NAN; n];
        let mut level_382 = vec![f64::NAN; n];
        let mut level_500 = vec![f64::NAN; n];
        let mut level_618 = vec![f64::NAN; n];
        let mut level_1000 = vec![f64::NAN; n];
        let mut level_1618 = vec![f64::NAN; n];

        if n < self.lookback {
            return vec![base, level_236, level_382, level_500, level_618, level_1000, level_1618];
        }

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            // Fit trend line to closes
            let (slope, intercept) = self.fit_trend_line(close, start, i);

            // Calculate channel width
            let width = if self.use_extremes {
                self.channel_width(high, low, slope, intercept, start, i)
            } else {
                // Use close deviation
                let mut max_dev = 0.0f64;
                for (idx, j) in (start..i).enumerate() {
                    let x = idx as f64;
                    let trend_value = slope * x + intercept;
                    let dev = (close[j] - trend_value).abs();
                    max_dev = max_dev.max(dev);
                }
                max_dev
            };

            // Calculate current bar's position
            let x = (i - start) as f64;
            let base_value = slope * x + intercept;

            base[i] = base_value;
            level_236[i] = base_value + width * 0.236;
            level_382[i] = base_value + width * 0.382;
            level_500[i] = base_value + width * 0.500;
            level_618[i] = base_value + width * 0.618;
            level_1000[i] = base_value + width * 1.000;
            level_1618[i] = base_value + width * 1.618;
        }

        vec![base, level_236, level_382, level_500, level_618, level_1000, level_1618]
    }

    /// Get channel levels at a specific bar.
    pub fn get_levels(&self, high: &[f64], low: &[f64], close: &[f64], bar_index: usize) -> Option<FibChannelLevels> {
        let channels = self.calculate(high, low, close);

        if bar_index >= close.len() || channels[0][bar_index].is_nan() {
            return None;
        }

        Some(FibChannelLevels {
            base: channels[0][bar_index],
            level_236: channels[1][bar_index],
            level_382: channels[2][bar_index],
            level_500: channels[3][bar_index],
            level_618: channels[4][bar_index],
            level_1000: channels[5][bar_index],
            level_1618: channels[6][bar_index],
        })
    }

    /// Determine which channel zone the price is currently in.
    pub fn get_channel_zone(&self, price: f64, levels: &FibChannelLevels) -> ChannelZone {
        if price < levels.base {
            ChannelZone::BelowBase
        } else if price < levels.level_236 {
            ChannelZone::Zone0To236
        } else if price < levels.level_382 {
            ChannelZone::Zone236To382
        } else if price < levels.level_500 {
            ChannelZone::Zone382To500
        } else if price < levels.level_618 {
            ChannelZone::Zone500To618
        } else if price < levels.level_1000 {
            ChannelZone::Zone618To1000
        } else if price < levels.level_1618 {
            ChannelZone::Zone1000To1618
        } else {
            ChannelZone::Above1618
        }
    }
}

/// Channel zone classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelZone {
    /// Price below base channel
    BelowBase,
    /// Price between base and 23.6%
    Zone0To236,
    /// Price between 23.6% and 38.2%
    Zone236To382,
    /// Price between 38.2% and 50%
    Zone382To500,
    /// Price between 50% and 61.8%
    Zone500To618,
    /// Price between 61.8% and 100%
    Zone618To1000,
    /// Price between 100% and 161.8%
    Zone1000To1618,
    /// Price above 161.8%
    Above1618,
}

impl Default for FibonacciChannels {
    fn default() -> Self {
        Self::default_params()
    }
}

impl TechnicalIndicator for FibonacciChannels {
    fn name(&self) -> &str {
        "Fibonacci Channels"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let channels = self.calculate(&data.high, &data.low, &data.close);
        // Use base, 50%, and 61.8% channel levels as the three outputs
        Ok(IndicatorOutput::triple(channels[0].clone(), channels[3].clone(), channels[4].clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create trending data
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();

        for i in 0..50 {
            let base = 100.0 + (i as f64) * 0.5;
            let noise = (i as f64 * 0.5).sin() * 2.0;
            close.push(base + noise);
            high.push(base + noise + 2.0);
            low.push(base + noise - 2.0);
        }

        (high, low, close)
    }

    #[test]
    fn test_fib_channels_creation() {
        let channels = FibonacciChannels::new(20, 3, true);
        assert!(channels.is_ok());

        let channels = FibonacciChannels::new(5, 3, true);
        assert!(channels.is_err());

        let channels = FibonacciChannels::new(20, 1, true);
        assert!(channels.is_err());
    }

    #[test]
    fn test_fib_channels_calculation() {
        let (high, low, close) = make_test_data();
        let channels = FibonacciChannels::new(15, 2, true).unwrap();
        let result = channels.calculate(&high, &low, &close);

        assert_eq!(result.len(), 7); // 7 channel levels
        assert_eq!(result[0].len(), close.len());

        // Check that valid values exist after lookback
        let valid_count = result[0].iter().filter(|v| !v.is_nan()).count();
        assert!(valid_count > 0);
    }

    #[test]
    fn test_fib_channels_ordering() {
        let (high, low, close) = make_test_data();
        let channels = FibonacciChannels::new(15, 2, true).unwrap();
        let result = channels.calculate(&high, &low, &close);

        // Channel levels should be in ascending order
        for i in 20..close.len() {
            if !result[0][i].is_nan() {
                assert!(result[1][i] >= result[0][i]); // 23.6% >= base
                assert!(result[2][i] >= result[1][i]); // 38.2% >= 23.6%
                assert!(result[3][i] >= result[2][i]); // 50% >= 38.2%
                assert!(result[4][i] >= result[3][i]); // 61.8% >= 50%
                assert!(result[5][i] >= result[4][i]); // 100% >= 61.8%
                assert!(result[6][i] >= result[5][i]); // 161.8% >= 100%
            }
        }
    }

    #[test]
    fn test_fib_channels_get_levels() {
        let (high, low, close) = make_test_data();
        let channels = FibonacciChannels::default_params();

        let levels = channels.get_levels(&high, &low, &close, 30);
        assert!(levels.is_some());

        let lvl = levels.unwrap();
        assert!(lvl.level_236 > lvl.base);
        assert!(lvl.level_382 > lvl.level_236);
        assert!(lvl.level_500 > lvl.level_382);
        assert!(lvl.level_618 > lvl.level_500);
        assert!(lvl.level_1000 > lvl.level_618);
        assert!(lvl.level_1618 > lvl.level_1000);
    }

    #[test]
    fn test_fib_channels_zone() {
        let channels = FibonacciChannels::default_params();
        let levels = FibChannelLevels {
            base: 100.0,
            level_236: 102.36,
            level_382: 103.82,
            level_500: 105.0,
            level_618: 106.18,
            level_1000: 110.0,
            level_1618: 116.18,
        };

        assert_eq!(channels.get_channel_zone(99.0, &levels), ChannelZone::BelowBase);
        assert_eq!(channels.get_channel_zone(101.0, &levels), ChannelZone::Zone0To236);
        assert_eq!(channels.get_channel_zone(103.0, &levels), ChannelZone::Zone236To382);
        assert_eq!(channels.get_channel_zone(104.0, &levels), ChannelZone::Zone382To500);
        assert_eq!(channels.get_channel_zone(105.5, &levels), ChannelZone::Zone500To618);
        assert_eq!(channels.get_channel_zone(108.0, &levels), ChannelZone::Zone618To1000);
        assert_eq!(channels.get_channel_zone(112.0, &levels), ChannelZone::Zone1000To1618);
        assert_eq!(channels.get_channel_zone(120.0, &levels), ChannelZone::Above1618);
    }

    #[test]
    fn test_fib_channels_technical_indicator() {
        let channels = FibonacciChannels::default_params();
        assert_eq!(channels.name(), "Fibonacci Channels");
        assert_eq!(channels.min_periods(), 21);
    }

    #[test]
    fn test_fib_channels_compute() {
        let (high, low, close) = make_test_data();
        let volume = vec![1000.0; close.len()];

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let channels = FibonacciChannels::default_params();
        let result = channels.compute(&data);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.values.len(), 7);
    }

    #[test]
    fn test_fib_channels_use_extremes() {
        let (high, low, close) = make_test_data();

        let channels_extremes = FibonacciChannels::new(15, 2, true).unwrap();
        let channels_close = FibonacciChannels::new(15, 2, false).unwrap();

        let result_extremes = channels_extremes.calculate(&high, &low, &close);
        let result_close = channels_close.calculate(&high, &low, &close);

        // Both should produce valid results
        let valid_extremes = result_extremes[0].iter().filter(|v| !v.is_nan()).count();
        let valid_close = result_close[0].iter().filter(|v| !v.is_nan()).count();

        assert!(valid_extremes > 0);
        assert!(valid_close > 0);
    }
}
