//! Bullish Percent Index indicator.
//!
//! The Bullish Percent Index (BPI) measures the percentage of stocks in an index
//! that are on Point & Figure buy signals. Since P&F data is not typically available,
//! this implementation uses a proxy based on percentage of stocks above their
//! moving average (commonly 50-day MA).

use super::{BreadthIndicator, BreadthSeries};
use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

/// Bullish Percent Index (BPI)
///
/// Measures market breadth by tracking the percentage of stocks on Point & Figure
/// buy signals. Since P&F data is often unavailable, this implementation uses
/// a proxy: percentage of stocks above their N-day moving average.
///
/// # Formula
/// BPI = (Number of stocks on P&F buy signals / Total stocks) * 100
///
/// Proxy: BPI = (Number of stocks above N-day MA / Total stocks) * 100
///
/// # Interpretation
/// - Above 70%: Bull alert (overbought, potential reversal down)
/// - Above 50%: Bull confirmed (bullish)
/// - Below 50%: Bear confirmed (bearish)
/// - Below 30%: Bear alert (oversold, potential reversal up)
///
/// # Signals
/// - BPI crossing above 70 from below: Market getting overheated
/// - BPI crossing below 30 from above: Market getting oversold (potential buy)
/// - BPI crossing above 50 from below: Bull market confirmation
/// - BPI crossing below 50 from above: Bear market confirmation
///
/// # Note
/// This indicator is commonly used with NYSE, S&P 500, or sector indices
/// to gauge overall market sentiment and participation.
#[derive(Debug, Clone)]
pub struct BullishPercent {
    /// MA period for proxy calculation (default: 50)
    ma_period: usize,
    /// Optional smoothing period (0 = no smoothing)
    smoothing_period: usize,
    /// Bull alert threshold (default: 70%)
    bull_alert_threshold: f64,
    /// Bull confirmed threshold (default: 50%)
    bull_confirmed_threshold: f64,
    /// Bear alert threshold (default: 30%)
    bear_alert_threshold: f64,
}

impl Default for BullishPercent {
    fn default() -> Self {
        Self::new()
    }
}

impl BullishPercent {
    pub fn new() -> Self {
        Self {
            ma_period: 50,
            smoothing_period: 0,
            bull_alert_threshold: 70.0,
            bull_confirmed_threshold: 50.0,
            bear_alert_threshold: 30.0,
        }
    }

    /// Create with custom MA period
    pub fn with_ma_period(ma_period: usize) -> Self {
        Self {
            ma_period,
            ..Self::new()
        }
    }

    /// Create for 200-day MA (long-term)
    pub fn ma_200() -> Self {
        Self::with_ma_period(200)
    }

    /// Create for 50-day MA (intermediate, default)
    pub fn ma_50() -> Self {
        Self::with_ma_period(50)
    }

    /// Create for 20-day MA (short-term)
    pub fn ma_20() -> Self {
        Self::with_ma_period(20)
    }

    /// Add smoothing (e.g., 10-day SMA of BPI)
    pub fn with_smoothing(mut self, period: usize) -> Self {
        self.smoothing_period = period;
        self
    }

    /// Set custom thresholds
    pub fn with_thresholds(
        mut self,
        bull_alert: f64,
        bull_confirmed: f64,
        bear_alert: f64,
    ) -> Self {
        self.bull_alert_threshold = bull_alert;
        self.bull_confirmed_threshold = bull_confirmed;
        self.bear_alert_threshold = bear_alert;
        self
    }

    /// Get the MA period used for proxy calculation
    pub fn ma_period(&self) -> usize {
        self.ma_period
    }

    /// Calculate SMA for a data series
    fn calculate_sma(data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return data.to_vec();
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().sum();
        result.push(sum / period as f64);

        for i in period..data.len() {
            sum = sum - data[i - period] + data[i];
            result.push(sum / period as f64);
        }

        result
    }

    /// Calculate BPI from counts of stocks above MA and total stocks
    ///
    /// # Arguments
    /// * `above_count` - Number of stocks above their MA each period
    /// * `total_count` - Total number of stocks each period
    pub fn calculate(&self, above_count: &[f64], total_count: &[f64]) -> Vec<f64> {
        let raw: Vec<f64> = above_count
            .iter()
            .zip(total_count.iter())
            .map(|(above, total)| {
                if *total == 0.0 {
                    0.0
                } else {
                    (above / total) * 100.0
                }
            })
            .collect();

        if self.smoothing_period > 0 {
            Self::calculate_sma(&raw, self.smoothing_period)
        } else {
            raw
        }
    }

    /// Calculate from pre-computed BPI percentage values (0-100)
    pub fn calculate_from_percent(&self, percent_values: &[f64]) -> Vec<f64> {
        if self.smoothing_period > 0 {
            Self::calculate_sma(percent_values, self.smoothing_period)
        } else {
            percent_values.to_vec()
        }
    }

    /// Calculate BPI from BreadthSeries using advances as proxy for "above MA"
    ///
    /// This approximation uses advances/(advances+declines) as a proxy for
    /// the percentage of stocks above their moving average.
    pub fn calculate_from_breadth(&self, data: &BreadthSeries) -> Vec<f64> {
        let raw: Vec<f64> = data
            .advances
            .iter()
            .zip(data.declines.iter())
            .map(|(a, d)| {
                let total = a + d;
                if total == 0.0 {
                    50.0 // Neutral when no data
                } else {
                    (a / total) * 100.0
                }
            })
            .collect();

        if self.smoothing_period > 0 {
            Self::calculate_sma(&raw, self.smoothing_period)
        } else {
            raw
        }
    }

    /// Calculate BPI from OHLCV data by computing how many closes are above MA
    ///
    /// This is a single-stock proxy - in practice, BPI is computed across
    /// many stocks. This method can be used when you have component data
    /// for multiple stocks merged into a single series.
    pub fn calculate_from_ohlcv(&self, data: &OHLCVSeries) -> Vec<f64> {
        let n = data.len();
        if n < self.ma_period {
            return vec![f64::NAN; n];
        }

        // Calculate SMA of close prices
        let ma = Self::calculate_sma(&data.close, self.ma_period);

        // Calculate percentage above MA (for single stock this is 0 or 100)
        // This is typically used as a building block for multi-stock BPI
        let raw: Vec<f64> = data
            .close
            .iter()
            .zip(ma.iter())
            .map(|(close, ma_val)| {
                if ma_val.is_nan() {
                    f64::NAN
                } else if *close >= *ma_val {
                    100.0
                } else {
                    0.0
                }
            })
            .collect();

        if self.smoothing_period > 0 {
            Self::calculate_sma(&raw, self.smoothing_period)
        } else {
            raw
        }
    }

    /// Check if current value indicates bull alert (overbought)
    pub fn is_bull_alert(&self, value: f64) -> bool {
        !value.is_nan() && value >= self.bull_alert_threshold
    }

    /// Check if current value indicates bull confirmed
    pub fn is_bull_confirmed(&self, value: f64) -> bool {
        !value.is_nan() && value >= self.bull_confirmed_threshold
    }

    /// Check if current value indicates bear alert (oversold)
    pub fn is_bear_alert(&self, value: f64) -> bool {
        !value.is_nan() && value <= self.bear_alert_threshold
    }

    /// Check if current value indicates bear confirmed
    pub fn is_bear_confirmed(&self, value: f64) -> bool {
        !value.is_nan() && value < self.bull_confirmed_threshold
    }

    /// Get market status based on current BPI value
    pub fn market_status(&self, value: f64) -> BPIStatus {
        if value.is_nan() {
            BPIStatus::Unknown
        } else if value >= self.bull_alert_threshold {
            BPIStatus::BullAlert
        } else if value >= self.bull_confirmed_threshold {
            BPIStatus::BullConfirmed
        } else if value <= self.bear_alert_threshold {
            BPIStatus::BearAlert
        } else {
            BPIStatus::BearConfirmed
        }
    }

    /// Detect BPI signals (crossovers of key levels)
    ///
    /// Returns a vector of BPISignal for each bar
    pub fn detect_signals(&self, values: &[f64]) -> Vec<BPISignal> {
        let mut signals = vec![BPISignal::None; values.len()];

        for i in 1..values.len() {
            if values[i].is_nan() || values[i - 1].is_nan() {
                continue;
            }

            let curr = values[i];
            let prev = values[i - 1];

            // Bull alert crossover (crossing above 70)
            if prev < self.bull_alert_threshold && curr >= self.bull_alert_threshold {
                signals[i] = BPISignal::BullAlertCrossUp;
            }
            // Bear alert crossover (crossing below 30)
            else if prev > self.bear_alert_threshold && curr <= self.bear_alert_threshold {
                signals[i] = BPISignal::BearAlertCrossDown;
            }
            // Bull confirmed crossover (crossing above 50)
            else if prev < self.bull_confirmed_threshold && curr >= self.bull_confirmed_threshold
            {
                signals[i] = BPISignal::BullConfirmedCrossUp;
            }
            // Bear confirmed crossover (crossing below 50)
            else if prev >= self.bull_confirmed_threshold && curr < self.bull_confirmed_threshold
            {
                signals[i] = BPISignal::BearConfirmedCrossDown;
            }
        }

        signals
    }
}

/// BPI market status categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BPIStatus {
    /// Above 70% - market is overbought, potential reversal down
    BullAlert,
    /// 50-70% - market is bullish
    BullConfirmed,
    /// 30-50% - market is bearish
    BearConfirmed,
    /// Below 30% - market is oversold, potential reversal up
    BearAlert,
    /// Insufficient data
    Unknown,
}

/// BPI trading signals based on level crossovers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BPISignal {
    /// No signal
    None,
    /// BPI crossed above 70 (bull alert - caution)
    BullAlertCrossUp,
    /// BPI crossed below 30 (bear alert - potential buy)
    BearAlertCrossDown,
    /// BPI crossed above 50 (bull confirmed - bullish)
    BullConfirmedCrossUp,
    /// BPI crossed below 50 (bear confirmed - bearish)
    BearConfirmedCrossDown,
}

impl BreadthIndicator for BullishPercent {
    fn name(&self) -> &str {
        "Bullish Percent Index"
    }

    fn compute_breadth(&self, data: &BreadthSeries) -> Result<IndicatorOutput> {
        let min_required = if self.smoothing_period > 0 {
            self.smoothing_period
        } else {
            1
        };

        if data.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.len(),
            });
        }

        let values = self.calculate_from_breadth(data);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        if self.smoothing_period > 0 {
            self.smoothing_period
        } else {
            1
        }
    }
}

impl TechnicalIndicator for BullishPercent {
    fn name(&self) -> &str {
        "Bullish Percent Index"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.len() < self.ma_period {
            return Err(IndicatorError::InsufficientData {
                required: self.ma_period,
                got: data.len(),
            });
        }

        let values = self.calculate_from_ohlcv(data);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.ma_period
    }
}

impl SignalIndicator for BullishPercent {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate_from_ohlcv(data);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Oversold (bear alert) is bullish signal (buy opportunity)
        // Overbought (bull alert) is bearish signal (sell opportunity)
        if last <= self.bear_alert_threshold {
            Ok(IndicatorSignal::Bullish)
        } else if last >= self.bull_alert_threshold {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate_from_ohlcv(data);
        let signals = values
            .iter()
            .map(|&bpi| {
                if bpi.is_nan() {
                    IndicatorSignal::Neutral
                } else if bpi <= self.bear_alert_threshold {
                    IndicatorSignal::Bullish
                } else if bpi >= self.bull_alert_threshold {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();
        Ok(signals)
    }
}

/// Data structure for pre-calculated BPI values
#[derive(Debug, Clone, Default)]
pub struct BPISeries {
    /// BPI percentage values (0-100)
    pub values: Vec<f64>,
}

impl BPISeries {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, percent: f64) {
        self.values.push(percent);
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::breadth::BreadthData;
    use crate::OHLCV;

    #[test]
    fn test_bullish_percent_basic() {
        let bpi = BullishPercent::new();

        let above = vec![1500.0, 1600.0, 1400.0, 1800.0, 1200.0];
        let total = vec![2000.0, 2000.0, 2000.0, 2000.0, 2000.0];

        let result = bpi.calculate(&above, &total);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 75.0).abs() < 1e-10);
        assert!((result[1] - 80.0).abs() < 1e-10);
        assert!((result[2] - 70.0).abs() < 1e-10);
        assert!((result[3] - 90.0).abs() < 1e-10);
        assert!((result[4] - 60.0).abs() < 1e-10);
    }

    #[test]
    fn test_bullish_percent_with_smoothing() {
        let bpi = BullishPercent::new().with_smoothing(3);

        let above = vec![1500.0, 1600.0, 1400.0, 1800.0, 1200.0];
        let total = vec![2000.0, 2000.0, 2000.0, 2000.0, 2000.0];

        let result = bpi.calculate(&above, &total);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // SMA of 75, 80, 70 = 75
        assert!((result[2] - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_bullish_percent_market_status() {
        let bpi = BullishPercent::new();

        assert_eq!(bpi.market_status(75.0), BPIStatus::BullAlert);
        assert_eq!(bpi.market_status(60.0), BPIStatus::BullConfirmed);
        assert_eq!(bpi.market_status(40.0), BPIStatus::BearConfirmed);
        assert_eq!(bpi.market_status(25.0), BPIStatus::BearAlert);
        assert_eq!(bpi.market_status(f64::NAN), BPIStatus::Unknown);
    }

    #[test]
    fn test_bullish_percent_thresholds() {
        let bpi = BullishPercent::new().with_thresholds(80.0, 50.0, 20.0);

        assert!(bpi.is_bull_alert(85.0));
        assert!(!bpi.is_bull_alert(75.0));
        assert!(bpi.is_bear_alert(15.0));
        assert!(!bpi.is_bear_alert(25.0));
        assert!(bpi.is_bull_confirmed(55.0));
        assert!(bpi.is_bear_confirmed(45.0));
    }

    #[test]
    fn test_from_breadth_series() {
        let bpi = BullishPercent::new();

        let mut series = BreadthSeries::new();
        series.push(BreadthData::from_ad(1500.0, 500.0)); // 1500/2000 = 75%
        series.push(BreadthData::from_ad(1600.0, 400.0)); // 1600/2000 = 80%

        let result = bpi.calculate_from_breadth(&series);

        assert_eq!(result.len(), 2);
        assert!((result[0] - 75.0).abs() < 1e-10);
        assert!((result[1] - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_percent_values() {
        let bpi = BullishPercent::new();
        let percent = vec![65.0, 70.0, 72.0, 68.0, 75.0];

        let result = bpi.calculate_from_percent(&percent);

        assert_eq!(result, percent);
    }

    #[test]
    fn test_preset_ma_periods() {
        let ma200 = BullishPercent::ma_200();
        let ma50 = BullishPercent::ma_50();
        let ma20 = BullishPercent::ma_20();

        assert_eq!(ma200.ma_period(), 200);
        assert_eq!(ma50.ma_period(), 50);
        assert_eq!(ma20.ma_period(), 20);
    }

    #[test]
    fn test_detect_signals() {
        let bpi = BullishPercent::new();
        let values = vec![45.0, 55.0, 65.0, 75.0, 65.0, 55.0, 45.0, 35.0, 25.0];

        let signals = bpi.detect_signals(&values);

        assert_eq!(signals.len(), 9);
        assert_eq!(signals[0], BPISignal::None);
        assert_eq!(signals[1], BPISignal::BullConfirmedCrossUp); // 45 -> 55
        assert_eq!(signals[3], BPISignal::BullAlertCrossUp); // 65 -> 75
        assert_eq!(signals[6], BPISignal::BearConfirmedCrossDown); // 55 -> 45
        assert_eq!(signals[8], BPISignal::BearAlertCrossDown); // 35 -> 25
    }

    #[test]
    fn test_zero_total() {
        let bpi = BullishPercent::new();

        let above = vec![100.0, 0.0];
        let total = vec![0.0, 0.0];

        let result = bpi.calculate(&above, &total);

        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_from_ohlcv() {
        let bpi = BullishPercent::with_ma_period(5);

        // Create uptrending OHLCV data
        let mut series = OHLCVSeries::new();
        for i in 0..20 {
            let price = 100.0 + i as f64;
            series.push(OHLCV::new(price, price + 1.0, price - 0.5, price, 1000.0));
        }

        let result = bpi.calculate_from_ohlcv(&series);

        assert_eq!(result.len(), 20);
        // First 4 should be NaN (need 5 periods for MA)
        for i in 0..4 {
            assert!(result[i].is_nan());
        }
        // After MA is calculated, in uptrend close should be above MA
        // So result should be 100
        assert!(!result[5].is_nan());
    }

    #[test]
    fn test_technical_indicator_trait() {
        let bpi = BullishPercent::with_ma_period(5);

        let mut series = OHLCVSeries::new();
        for i in 0..20 {
            let price = 100.0 + i as f64;
            series.push(OHLCV::new(price, price + 1.0, price - 0.5, price, 1000.0));
        }

        let output = bpi.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 20);
    }

    #[test]
    fn test_signal_indicator_trait() {
        let bpi = BullishPercent::with_ma_period(5);

        // Create OHLCV with price consistently above MA (bullish - but actually overbought)
        let mut series = OHLCVSeries::new();
        for i in 0..20 {
            let price = 100.0 + i as f64;
            series.push(OHLCV::new(price, price + 1.0, price - 0.5, price, 1000.0));
        }

        // For single stock, BPI is either 0 or 100
        // In uptrend, close > MA so BPI = 100, which is bull alert (overbought)
        // Signal should be Bearish (sell opportunity)
        let signal = bpi.signal(&series).unwrap();
        assert_eq!(signal, IndicatorSignal::Bearish);

        let signals = bpi.signals(&series).unwrap();
        assert_eq!(signals.len(), 20);
    }

    #[test]
    fn test_breadth_indicator_trait() {
        let bpi = BullishPercent::new();

        let mut series = BreadthSeries::new();
        for _ in 0..10 {
            series.push(BreadthData::from_ad(1500.0, 500.0));
        }

        let output = bpi.compute_breadth(&series).unwrap();
        assert_eq!(output.primary.len(), 10);
    }

    #[test]
    fn test_insufficient_data_ohlcv() {
        let bpi = BullishPercent::with_ma_period(20);

        let mut series = OHLCVSeries::new();
        for i in 0..10 {
            series.push(OHLCV::new(100.0 + i as f64, 101.0, 99.0, 100.0, 1000.0));
        }

        let result = bpi.compute(&series);
        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_data_breadth() {
        let bpi = BullishPercent::new().with_smoothing(10);

        let mut series = BreadthSeries::new();
        for _ in 0..5 {
            series.push(BreadthData::from_ad(1500.0, 500.0));
        }

        let result = bpi.compute_breadth(&series);
        assert!(result.is_err());
    }

    #[test]
    fn test_bpi_series() {
        let mut series = BPISeries::new();
        assert!(series.is_empty());

        series.push(65.0);
        series.push(70.0);

        assert_eq!(series.len(), 2);
        assert!(!series.is_empty());
        assert_eq!(series.values, vec![65.0, 70.0]);
    }
}
