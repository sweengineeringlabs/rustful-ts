//! Market Profile (TPO) implementation.
//!
//! Time Price Opportunity based analysis showing time spent at each price level.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::MarketProfileConfig;

/// Market Profile output containing TPO-based levels.
#[derive(Debug, Clone)]
pub struct MarketProfileOutput {
    /// Point of Control - price level with most TPOs.
    pub poc: f64,
    /// Value Area High - upper bound of 70% TPO area.
    pub vah: f64,
    /// Value Area Low - lower bound of 70% TPO area.
    pub val: f64,
    /// Initial Balance High (first N TPO periods).
    pub ib_high: f64,
    /// Initial Balance Low (first N TPO periods).
    pub ib_low: f64,
    /// Initial Balance Range width.
    pub ib_range: f64,
    /// TPO count at each price bin.
    pub tpo_counts: Vec<usize>,
    /// Price levels corresponding to each bin.
    pub price_levels: Vec<f64>,
}

/// Market Profile indicator.
///
/// Creates a TPO (Time Price Opportunity) profile showing time spent at each price level.
/// Key outputs:
/// - POC (Point of Control): Price level with most TPOs
/// - VAH/VAL: Value Area boundaries (70% of TPOs)
/// - IB High/Low: Initial Balance range (typically first hour)
#[derive(Debug, Clone)]
pub struct MarketProfile {
    /// Price bin size (auto-calculated if None).
    tick_size: Option<f64>,
    /// Number of bins if tick_size is None.
    num_bins: usize,
    /// Value area percentage (default: 0.70).
    value_area_pct: f64,
    /// Bars per TPO period.
    tpo_period: usize,
    /// Initial Balance TPO periods.
    ib_periods: usize,
    /// Session length in bars (0 = entire series).
    session_bars: usize,
}

impl MarketProfile {
    pub fn new(num_bins: usize) -> Self {
        Self {
            tick_size: None,
            num_bins,
            value_area_pct: 0.70,
            tpo_period: 1,
            ib_periods: 2,
            session_bars: 0,
        }
    }

    pub fn with_tick_size(tick_size: f64) -> Self {
        Self {
            tick_size: Some(tick_size),
            num_bins: 50,
            value_area_pct: 0.70,
            tpo_period: 1,
            ib_periods: 2,
            session_bars: 0,
        }
    }

    pub fn from_config(config: MarketProfileConfig) -> Self {
        Self {
            tick_size: config.tick_size,
            num_bins: config.num_bins,
            value_area_pct: config.value_area_pct,
            tpo_period: config.tpo_period.max(1),
            ib_periods: config.ib_periods,
            session_bars: config.session_bars,
        }
    }

    /// Calculate market profile from OHLCV data.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        _close: &[f64],
        _volume: &[f64],
    ) -> Option<MarketProfileOutput> {
        let n = high.len();
        if n == 0 {
            return None;
        }

        // Determine range to analyze
        let start_idx = if self.session_bars > 0 && self.session_bars < n {
            n - self.session_bars
        } else {
            0
        };

        let high_slice = &high[start_idx..];
        let low_slice = &low[start_idx..];
        let slice_len = high_slice.len();

        // Find price range
        let session_high = high_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let session_low = low_slice.iter().cloned().fold(f64::INFINITY, f64::min);

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

        // Create price bins
        let mut tpo_counts = vec![0usize; num_bins];
        let mut price_levels = Vec::with_capacity(num_bins);

        for i in 0..num_bins {
            price_levels.push(session_low + (i as f64 + 0.5) * bin_size);
        }

        // Calculate Initial Balance from first N TPO periods
        let ib_bars = (self.ib_periods * self.tpo_period).min(slice_len);
        let ib_high = if ib_bars > 0 {
            high_slice[..ib_bars].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        } else {
            session_high
        };
        let ib_low = if ib_bars > 0 {
            low_slice[..ib_bars].iter().cloned().fold(f64::INFINITY, f64::min)
        } else {
            session_low
        };
        let ib_range = ib_high - ib_low;

        // Group bars into TPO periods and mark touched price levels
        let num_tpo_periods = (slice_len + self.tpo_period - 1) / self.tpo_period;

        for tpo_idx in 0..num_tpo_periods {
            let period_start = tpo_idx * self.tpo_period;
            let period_end = ((tpo_idx + 1) * self.tpo_period).min(slice_len);

            // Find high/low for this TPO period
            let period_high = high_slice[period_start..period_end]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let period_low = low_slice[period_start..period_end]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);

            // Mark all bins touched by this TPO period
            let low_bin = ((period_low - session_low) / bin_size).floor() as i64;
            let high_bin = ((period_high - session_low) / bin_size).floor() as i64;

            let low_bin = (low_bin.max(0) as usize).min(num_bins - 1);
            let high_bin = (high_bin.max(0) as usize).min(num_bins - 1);

            for bin in low_bin..=high_bin {
                tpo_counts[bin] += 1;
            }
        }

        // Find POC (max TPO count bin)
        let (poc_bin, _max_tpo) = tpo_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .unwrap_or((0, &0));

        let poc = price_levels[poc_bin];

        // Calculate Value Area by expanding from POC
        let total_tpos: usize = tpo_counts.iter().sum();
        let target_tpos = (total_tpos as f64 * self.value_area_pct) as usize;

        let mut va_tpos = tpo_counts[poc_bin];
        let mut va_low_bin = poc_bin;
        let mut va_high_bin = poc_bin;

        // Expand Value Area from POC until we reach target percentage
        while va_tpos < target_tpos {
            let can_go_lower = va_low_bin > 0;
            let can_go_higher = va_high_bin < num_bins - 1;

            if !can_go_lower && !can_go_higher {
                break;
            }

            let lower_tpo = if can_go_lower { tpo_counts[va_low_bin - 1] } else { 0 };
            let higher_tpo = if can_go_higher { tpo_counts[va_high_bin + 1] } else { 0 };

            // Expand in direction with more TPOs
            if lower_tpo >= higher_tpo && can_go_lower {
                va_low_bin -= 1;
                va_tpos += tpo_counts[va_low_bin];
            } else if can_go_higher {
                va_high_bin += 1;
                va_tpos += tpo_counts[va_high_bin];
            } else if can_go_lower {
                va_low_bin -= 1;
                va_tpos += tpo_counts[va_low_bin];
            } else {
                break;
            }
        }

        let val = session_low + va_low_bin as f64 * bin_size;
        let vah = session_low + (va_high_bin + 1) as f64 * bin_size;

        Some(MarketProfileOutput {
            poc,
            vah,
            val,
            ib_high,
            ib_low,
            ib_range,
            tpo_counts,
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

        let period = if self.session_bars > 0 { self.session_bars } else { n };

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

impl Default for MarketProfile {
    fn default() -> Self {
        Self::from_config(MarketProfileConfig::default())
    }
}

impl TechnicalIndicator for MarketProfile {
    fn name(&self) -> &str {
        "MarketProfile"
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
        if self.session_bars > 0 { self.session_bars } else { 1 }
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_profile_basic() {
        let mp = MarketProfile::new(10);

        // Create test data
        let high = vec![105.0, 106.0, 107.0, 108.0, 109.0, 108.0, 107.0, 106.0, 105.0, 104.0];
        let low = vec![100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0];
        let close = vec![102.0, 103.0, 104.0, 105.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0];
        let volume = vec![1000.0; 10];

        let output = mp.calculate(&high, &low, &close, &volume);
        assert!(output.is_some());

        let result = output.unwrap();
        // POC should be within price range
        assert!(result.poc >= 99.0 && result.poc <= 109.0);
        // VAH should be >= VAL
        assert!(result.vah >= result.val);
        // POC should be within Value Area
        assert!(result.poc >= result.val && result.poc <= result.vah);
        // IB should be calculated
        assert!(result.ib_high >= result.ib_low);
        assert!(result.ib_range >= 0.0);
    }

    #[test]
    fn test_market_profile_with_tick_size() {
        let mp = MarketProfile::with_tick_size(1.0);

        let high = vec![105.0, 106.0, 107.0];
        let low = vec![100.0, 101.0, 102.0];
        let close = vec![102.0, 103.0, 104.0];
        let volume = vec![1000.0, 2000.0, 1000.0];

        let output = mp.calculate(&high, &low, &close, &volume);
        assert!(output.is_some());
    }

    #[test]
    fn test_market_profile_empty() {
        let mp = MarketProfile::new(10);
        let output = mp.calculate(&[], &[], &[], &[]);
        assert!(output.is_none());
    }

    #[test]
    fn test_market_profile_initial_balance() {
        let mut config = MarketProfileConfig::default();
        config.ib_periods = 3;
        config.tpo_period = 2;
        let mp = MarketProfile::from_config(config);

        let high = vec![105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0];
        let low = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let close = vec![102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0];
        let volume = vec![1000.0; 8];

        let output = mp.calculate(&high, &low, &close, &volume);
        assert!(output.is_some());

        let result = output.unwrap();
        // IB is calculated from first 6 bars (3 periods * 2 bars each)
        // First 6 highs: 105, 106, 107, 108, 109, 110 -> max = 110
        // First 6 lows: 100, 101, 102, 103, 104, 105 -> min = 100
        assert_eq!(result.ib_high, 110.0);
        assert_eq!(result.ib_low, 100.0);
        assert_eq!(result.ib_range, 10.0);
    }

    #[test]
    fn test_market_profile_tpo_periods() {
        let mut config = MarketProfileConfig::default();
        config.tpo_period = 2;
        config.num_bins = 5;
        let mp = MarketProfile::from_config(config);

        // 4 bars = 2 TPO periods
        let high = vec![105.0, 105.0, 105.0, 105.0];
        let low = vec![100.0, 100.0, 100.0, 100.0];
        let close = vec![102.0, 103.0, 103.0, 102.0];
        let volume = vec![1000.0; 4];

        let output = mp.calculate(&high, &low, &close, &volume);
        assert!(output.is_some());

        let result = output.unwrap();
        // With 2 TPO periods touching all bins, all bins should have count 2
        assert!(result.tpo_counts.iter().all(|&c| c == 2));
    }

    #[test]
    fn test_market_profile_poc_series() {
        let mut config = MarketProfileConfig::default();
        config.session_bars = 3;
        let mp = MarketProfile::from_config(config);

        let high = vec![105.0, 106.0, 107.0, 108.0, 109.0];
        let low = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let close = vec![102.0, 103.0, 104.0, 105.0, 106.0];
        let volume = vec![1000.0; 5];

        let result = mp.calculate_poc_series(&high, &low, &close, &volume);

        assert_eq!(result.len(), 5);
        // First two should be NaN (session_bars = 3)
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // Rest should have values
        assert!(!result[2].is_nan());
        assert!(!result[3].is_nan());
        assert!(!result[4].is_nan());
    }
}
