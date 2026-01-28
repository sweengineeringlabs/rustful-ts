//! Volume Profile implementation.
//!
//! Shows volume traded at each price level as a histogram.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::VolumeProfileConfig;

/// Volume Profile output containing POC and Value Area levels.
#[derive(Debug, Clone)]
pub struct VolumeProfileOutput {
    /// Point of Control - price level with highest volume.
    pub poc: f64,
    /// Value Area High - upper bound of 70% volume area.
    pub vah: f64,
    /// Value Area Low - lower bound of 70% volume area.
    pub val: f64,
    /// Volume at each price bin (indexed from lowest to highest price).
    pub profile: Vec<f64>,
    /// Price levels corresponding to each bin (bin center prices).
    pub price_levels: Vec<f64>,
}

/// Volume Profile indicator.
///
/// Creates a histogram of volume traded at each price level.
/// Key outputs:
/// - POC (Point of Control): Price level with highest volume
/// - VAH (Value Area High): Upper bound containing 70% of volume
/// - VAL (Value Area Low): Lower bound containing 70% of volume
#[derive(Debug, Clone)]
pub struct VolumeProfile {
    /// Price bin size (auto-calculated if None).
    tick_size: Option<f64>,
    /// Number of bins if tick_size is None.
    num_bins: usize,
    /// Value area percentage (default: 0.70).
    value_area_pct: f64,
    /// Profile period in bars (0 = entire series).
    period: usize,
    /// Use close only vs full OHLC range.
    close_only: bool,
}

impl VolumeProfile {
    pub fn new(num_bins: usize) -> Self {
        Self {
            tick_size: None,
            num_bins,
            value_area_pct: 0.70,
            period: 0,
            close_only: false,
        }
    }

    pub fn with_tick_size(tick_size: f64) -> Self {
        Self {
            tick_size: Some(tick_size),
            num_bins: 50,
            value_area_pct: 0.70,
            period: 0,
            close_only: false,
        }
    }

    pub fn from_config(config: VolumeProfileConfig) -> Self {
        Self {
            tick_size: config.tick_size,
            num_bins: config.num_bins,
            value_area_pct: config.value_area_pct,
            period: config.period,
            close_only: config.close_only,
        }
    }

    /// Calculate volume profile from OHLCV data.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Option<VolumeProfileOutput> {
        let n = close.len();
        if n == 0 {
            return None;
        }

        // Determine range to analyze
        let start_idx = if self.period > 0 && self.period < n {
            n - self.period
        } else {
            0
        };

        let high_slice = &high[start_idx..];
        let low_slice = &low[start_idx..];
        let close_slice = &close[start_idx..];
        let volume_slice = &volume[start_idx..];

        // Find price range
        let (session_high, session_low) = if self.close_only {
            let max = close_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = close_slice.iter().cloned().fold(f64::INFINITY, f64::min);
            (max, min)
        } else {
            let max = high_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min = low_slice.iter().cloned().fold(f64::INFINITY, f64::min);
            (max, min)
        };

        if session_high <= session_low || !session_high.is_finite() || !session_low.is_finite() {
            return None;
        }

        let range = session_high - session_low;

        // Determine bin size
        let (bin_size, num_bins) = if let Some(tick) = self.tick_size {
            let bins = ((range / tick).ceil() as usize).max(1);
            (tick, bins)
        } else {
            let bins = self.num_bins.max(1);
            (range / bins as f64, bins)
        };

        // Create price bins and accumulate volume
        let mut profile = vec![0.0; num_bins];
        let mut price_levels = Vec::with_capacity(num_bins);

        for i in 0..num_bins {
            // Bin center price
            price_levels.push(session_low + (i as f64 + 0.5) * bin_size);
        }

        // Distribute volume across touched price levels
        for i in 0..volume_slice.len() {
            let bar_volume = volume_slice[i];
            if bar_volume <= 0.0 {
                continue;
            }

            let (bar_high, bar_low) = if self.close_only {
                (close_slice[i], close_slice[i])
            } else {
                (high_slice[i], low_slice[i])
            };

            // Find bins touched by this bar
            let low_bin = ((bar_low - session_low) / bin_size).floor() as i64;
            let high_bin = ((bar_high - session_low) / bin_size).floor() as i64;

            let low_bin = (low_bin.max(0) as usize).min(num_bins - 1);
            let high_bin = (high_bin.max(0) as usize).min(num_bins - 1);

            // Distribute volume equally across touched bins
            let bins_touched = (high_bin - low_bin + 1) as f64;
            let volume_per_bin = bar_volume / bins_touched;

            for bin in low_bin..=high_bin {
                profile[bin] += volume_per_bin;
            }
        }

        // Find POC (max volume bin)
        let (poc_bin, _max_vol) = profile
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        let poc = price_levels[poc_bin];

        // Calculate Value Area by expanding from POC
        let total_volume: f64 = profile.iter().sum();
        let target_volume = total_volume * self.value_area_pct;

        let mut va_volume = profile[poc_bin];
        let mut va_low_bin = poc_bin;
        let mut va_high_bin = poc_bin;

        // Expand Value Area from POC until we reach target percentage
        while va_volume < target_volume {
            // Check which direction to expand
            let can_go_lower = va_low_bin > 0;
            let can_go_higher = va_high_bin < num_bins - 1;

            if !can_go_lower && !can_go_higher {
                break;
            }

            let lower_vol = if can_go_lower { profile[va_low_bin - 1] } else { 0.0 };
            let higher_vol = if can_go_higher { profile[va_high_bin + 1] } else { 0.0 };

            // Expand in direction with more volume
            if lower_vol >= higher_vol && can_go_lower {
                va_low_bin -= 1;
                va_volume += profile[va_low_bin];
            } else if can_go_higher {
                va_high_bin += 1;
                va_volume += profile[va_high_bin];
            } else if can_go_lower {
                va_low_bin -= 1;
                va_volume += profile[va_low_bin];
            } else {
                break;
            }
        }

        let val = session_low + va_low_bin as f64 * bin_size;
        let vah = session_low + (va_high_bin + 1) as f64 * bin_size;

        Some(VolumeProfileOutput {
            poc,
            vah,
            val,
            profile,
            price_levels,
        })
    }

    /// Calculate POC series over rolling windows.
    pub fn calculate_poc_series(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n == 0 {
            return result;
        }

        let period = if self.period > 0 { self.period } else { n };

        for i in (period - 1)..n {
            let start = i + 1 - period;
            if let Some(output) = self.calculate(
                &high[start..=i],
                &low[start..=i],
                &close[start..=i],
                &volume[start..=i],
            ) {
                result[i] = output.poc;
            }
        }

        result
    }
}

impl Default for VolumeProfile {
    fn default() -> Self {
        Self::from_config(VolumeProfileConfig::default())
    }
}

impl TechnicalIndicator for VolumeProfile {
    fn name(&self) -> &str {
        "VolumeProfile"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let poc_series = self.calculate_poc_series(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::single(poc_series))
    }

    fn min_periods(&self) -> usize {
        if self.period > 0 { self.period } else { 1 }
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_profile_basic() {
        let vp = VolumeProfile::new(10);

        // Create simple test data with clear volume distribution
        let high = vec![105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let close = vec![102.0, 103.0, 104.0, 105.0, 106.0];
        let volume = vec![1000.0, 2000.0, 5000.0, 2000.0, 1000.0];

        let output = vp.calculate(&high, &low, &close, &volume);
        assert!(output.is_some());

        let result = output.unwrap();
        // POC should be around middle prices where volume is highest
        assert!(result.poc >= 100.0 && result.poc <= 109.0);
        // VAH should be >= VAL
        assert!(result.vah >= result.val);
        // POC should be within Value Area
        assert!(result.poc >= result.val && result.poc <= result.vah);
    }

    #[test]
    fn test_volume_profile_with_tick_size() {
        let vp = VolumeProfile::with_tick_size(1.0);

        let high = vec![105.0, 106.0, 107.0];
        let low = vec![100.0, 101.0, 102.0];
        let close = vec![102.0, 103.0, 104.0];
        let volume = vec![1000.0, 2000.0, 1000.0];

        let output = vp.calculate(&high, &low, &close, &volume);
        assert!(output.is_some());
    }

    #[test]
    fn test_volume_profile_empty() {
        let vp = VolumeProfile::new(10);
        let output = vp.calculate(&[], &[], &[], &[]);
        assert!(output.is_none());
    }

    #[test]
    fn test_volume_profile_poc_series() {
        let mut config = VolumeProfileConfig::default();
        config.period = 3;
        let vp = VolumeProfile::from_config(config);

        let high = vec![105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let close = vec![102.0, 103.0, 104.0, 105.0, 106.0];
        let volume = vec![1000.0, 2000.0, 5000.0, 2000.0, 1000.0];

        let result = vp.calculate_poc_series(&high, &low, &close, &volume);

        assert_eq!(result.len(), 5);
        // First two should be NaN (period = 3)
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Rest should have values
        assert!(!result[2].is_nan());
        assert!(!result[3].is_nan());
        assert!(!result[4].is_nan());
    }
}
