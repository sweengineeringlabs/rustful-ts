//! Kase Dev Stops (IND-182)
//!
//! Deviation-based trailing stops using True Range and standard deviation,
//! developed by Cynthia Kase.
//!
//! The indicator calculates stop levels at multiple standard deviation distances
//! from the average true range, providing adaptive trailing stop placement based
//! on current market volatility.
//!
//! Algorithm:
//! 1. Calculate True Range for each bar
//! 2. Calculate average True Range over period (ATR)
//! 3. Calculate standard deviation of True Range over period
//! 4. Dev Stop = ATR + (num_devs × StdDev of TR)
//! 5. Long stop = price - (dev_stop × multiplier)
//! 6. Short stop = price + (dev_stop × multiplier)
//!
//! Outputs three stop levels at 1, 2, and 3 standard deviations.

use crate::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Kase Dev Stops Output containing stop levels at multiple deviations.
#[derive(Debug, Clone)]
pub struct KaseDevStopsOutput {
    /// Dev Stop 1: ATR + (1 × StdDev of TR)
    pub dev_stop_1: Vec<f64>,
    /// Dev Stop 2: ATR + (2 × StdDev of TR)
    pub dev_stop_2: Vec<f64>,
    /// Dev Stop 3: ATR + (3 × StdDev of TR)
    pub dev_stop_3: Vec<f64>,
    /// Long stop level using num_devs
    pub long_stop: Vec<f64>,
    /// Short stop level using num_devs
    pub short_stop: Vec<f64>,
}

/// Kase Dev Stops (IND-182)
///
/// A deviation-based trailing stop indicator using True Range and standard deviation.
/// Provides adaptive stop levels that adjust to market volatility.
///
/// # Parameters
/// - `period`: Lookback period for ATR and StdDev calculations (default: 30)
/// - `num_devs`: Number of standard deviations for stop calculation (default: 2.0)
///
/// # Example
/// ```ignore
/// use indicator_core::volatility::KaseDevStops;
///
/// let kds = KaseDevStops::new(30, 2.0);
/// let output = kds.calculate(&high, &low, &close);
/// // Use output.long_stop for long position trailing stops
/// // Use output.short_stop for short position trailing stops
/// ```
#[derive(Debug, Clone)]
pub struct KaseDevStops {
    /// Lookback period for ATR and standard deviation.
    period: usize,
    /// Number of standard deviations for the primary stop calculation.
    num_devs: f64,
}

impl KaseDevStops {
    /// Create a new Kase Dev Stops indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for calculations (typical: 30)
    /// * `num_devs` - Standard deviation multiplier (typical: 2.0)
    pub fn new(period: usize, num_devs: f64) -> Self {
        Self { period, num_devs }
    }

    /// Create from configuration.
    pub fn from_config(config: indicator_api::KaseDevStopsConfig) -> Self {
        Self {
            period: config.period,
            num_devs: config.num_devs,
        }
    }

    /// Calculate Kase Dev Stops.
    ///
    /// Returns stop levels at 1, 2, and 3 standard deviations plus
    /// long/short stop levels using the configured num_devs.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> KaseDevStopsOutput {
        let n = high.len();

        if n < self.period {
            return KaseDevStopsOutput {
                dev_stop_1: vec![f64::NAN; n],
                dev_stop_2: vec![f64::NAN; n],
                dev_stop_3: vec![f64::NAN; n],
                long_stop: vec![f64::NAN; n],
                short_stop: vec![f64::NAN; n],
            };
        }

        // Step 1: Calculate True Range
        let tr = self.true_range(high, low, close);

        // Initialize output vectors
        let mut dev_stop_1 = vec![f64::NAN; n];
        let mut dev_stop_2 = vec![f64::NAN; n];
        let mut dev_stop_3 = vec![f64::NAN; n];
        let mut long_stop = vec![f64::NAN; n];
        let mut short_stop = vec![f64::NAN; n];

        // Calculate rolling ATR and StdDev of TR
        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let tr_window = &tr[start..=i];

            // Step 2: Average True Range (SMA of TR)
            let atr = tr_window.iter().sum::<f64>() / self.period as f64;

            // Step 3: Standard deviation of True Range
            let mean = atr;
            let variance: f64 = tr_window
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>()
                / self.period as f64;
            let std_dev = variance.sqrt();

            // Step 4: Calculate Dev Stop levels
            // Dev Stop = ATR + (num_devs × StdDev)
            dev_stop_1[i] = atr + std_dev;
            dev_stop_2[i] = atr + 2.0 * std_dev;
            dev_stop_3[i] = atr + 3.0 * std_dev;

            let dev_stop = atr + self.num_devs * std_dev;

            // Step 5 & 6: Calculate long and short stop levels
            // Long stop: below price
            // Short stop: above price
            long_stop[i] = close[i] - dev_stop;
            short_stop[i] = close[i] + dev_stop;
        }

        KaseDevStopsOutput {
            dev_stop_1,
            dev_stop_2,
            dev_stop_3,
            long_stop,
            short_stop,
        }
    }

    /// Calculate True Range for each bar.
    fn true_range(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n == 0 {
            return vec![];
        }

        let mut tr = Vec::with_capacity(n);
        tr.push(high[0] - low[0]); // First TR is just high-low

        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr.push(hl.max(hc).max(lc));
        }

        tr
    }
}

impl Default for KaseDevStops {
    fn default() -> Self {
        Self::new(30, 2.0)
    }
}

impl TechnicalIndicator for KaseDevStops {
    fn name(&self) -> &str {
        "KaseDevStops"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.high.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);

        // Return long_stop and short_stop as the primary outputs
        Ok(IndicatorOutput::dual(result.long_stop, result.short_stop))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_ohlc_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        (high, low, close)
    }

    #[test]
    fn test_kase_dev_stops_basic() {
        let kds = KaseDevStops::new(20, 2.0);
        let (high, low, close) = generate_ohlc_data(50);

        let result = kds.calculate(&high, &low, &close);

        assert_eq!(result.dev_stop_1.len(), 50);
        assert_eq!(result.dev_stop_2.len(), 50);
        assert_eq!(result.dev_stop_3.len(), 50);
        assert_eq!(result.long_stop.len(), 50);
        assert_eq!(result.short_stop.len(), 50);
    }

    #[test]
    fn test_kase_dev_stops_nan_prefix() {
        let kds = KaseDevStops::new(20, 2.0);
        let (high, low, close) = generate_ohlc_data(50);

        let result = kds.calculate(&high, &low, &close);

        // First period-1 values should be NaN
        for i in 0..19 {
            assert!(result.dev_stop_1[i].is_nan());
            assert!(result.long_stop[i].is_nan());
        }

        // Values at period-1 and beyond should be valid
        assert!(!result.dev_stop_1[19].is_nan());
        assert!(!result.long_stop[19].is_nan());
    }

    #[test]
    fn test_kase_dev_stops_order() {
        let kds = KaseDevStops::new(20, 2.0);

        // Generate data with varying volatility for non-zero std dev
        let high: Vec<f64> = (0..50)
            .map(|i| 105.0 + (i as f64 * 0.3).sin() * 5.0 + (i as f64 * 0.1))
            .collect();
        let low: Vec<f64> = (0..50)
            .map(|i| 95.0 + (i as f64 * 0.3).sin() * 5.0 - (i as f64 * 0.05))
            .collect();
        let close: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();

        let result = kds.calculate(&high, &low, &close);

        // Dev stops should be non-decreasing: dev_stop_1 <= dev_stop_2 <= dev_stop_3
        // With varying volatility, the std dev should be positive
        for i in 19..50 {
            assert!(
                result.dev_stop_1[i] <= result.dev_stop_2[i] + 1e-10,
                "dev_stop_1 should be <= dev_stop_2 at index {}",
                i
            );
            assert!(
                result.dev_stop_2[i] <= result.dev_stop_3[i] + 1e-10,
                "dev_stop_2 should be <= dev_stop_3 at index {}",
                i
            );
        }
    }

    #[test]
    fn test_kase_dev_stops_long_short_relationship() {
        let kds = KaseDevStops::new(20, 2.0);
        let (high, low, close) = generate_ohlc_data(50);

        let result = kds.calculate(&high, &low, &close);

        // Long stop should be below close, short stop should be above close
        for i in 19..50 {
            assert!(
                result.long_stop[i] < close[i],
                "Long stop should be below close at index {}",
                i
            );
            assert!(
                result.short_stop[i] > close[i],
                "Short stop should be above close at index {}",
                i
            );
            // Long stop + Short stop should equal 2 × close
            let sum = result.long_stop[i] + result.short_stop[i];
            let expected = 2.0 * close[i];
            assert!(
                (sum - expected).abs() < 1e-10,
                "Long + Short should equal 2×close at index {}",
                i
            );
        }
    }

    #[test]
    fn test_kase_dev_stops_default() {
        let kds = KaseDevStops::default();
        assert_eq!(kds.period, 30);
        assert!((kds.num_devs - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_kase_dev_stops_technical_indicator() {
        let kds = KaseDevStops::new(20, 2.0);
        assert_eq!(kds.name(), "KaseDevStops");
        assert_eq!(kds.min_periods(), 20);
        assert_eq!(kds.output_features(), 2);
    }

    #[test]
    fn test_kase_dev_stops_compute() {
        let kds = KaseDevStops::new(20, 2.0);
        let (high, low, close) = generate_ohlc_data(50);

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 50],
        };

        let result = kds.compute(&data).unwrap();

        // Should return 2 features (long_stop and short_stop)
        assert_eq!(result.primary.len(), 50);
        assert!(result.secondary.is_some());
        assert_eq!(result.secondary.as_ref().unwrap().len(), 50);
    }

    #[test]
    fn test_kase_dev_stops_insufficient_data() {
        let kds = KaseDevStops::new(20, 2.0);
        let (high, low, close) = generate_ohlc_data(10);

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 10],
        };

        let result = kds.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_true_range_calculation() {
        let kds = KaseDevStops::new(3, 2.0);

        // Manual TR calculation test
        let high = vec![102.0, 105.0, 104.0];
        let low = vec![98.0, 99.0, 100.0];
        let close = vec![100.0, 103.0, 101.0];

        let tr = kds.true_range(&high, &low, &close);

        // TR[0] = high[0] - low[0] = 102 - 98 = 4
        assert!((tr[0] - 4.0).abs() < 1e-10);

        // TR[1] = max(105-99, |105-100|, |99-100|) = max(6, 5, 1) = 6
        assert!((tr[1] - 6.0).abs() < 1e-10);

        // TR[2] = max(104-100, |104-103|, |100-103|) = max(4, 1, 3) = 4
        assert!((tr[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_kase_dev_stops_volatility_sensitivity() {
        // Low volatility data
        let high_low: Vec<f64> = (0..50).map(|_| 101.0).collect();
        let low_low: Vec<f64> = (0..50).map(|_| 99.0).collect();
        let close_low: Vec<f64> = (0..50).map(|_| 100.0).collect();

        // High volatility data
        let high_high: Vec<f64> = (0..50).map(|_| 110.0).collect();
        let low_high: Vec<f64> = (0..50).map(|_| 90.0).collect();
        let close_high: Vec<f64> = (0..50).map(|_| 100.0).collect();

        let kds = KaseDevStops::new(20, 2.0);

        let result_low = kds.calculate(&high_low, &low_low, &close_low);
        let result_high = kds.calculate(&high_high, &low_high, &close_high);

        // High volatility should have wider stops
        let last = 49;
        let low_vol_spread = result_high.short_stop[last] - result_high.long_stop[last];
        let high_vol_spread = result_low.short_stop[last] - result_low.long_stop[last];

        assert!(
            low_vol_spread > high_vol_spread,
            "High volatility data should produce wider stop spread"
        );
    }
}
