//! Point of Control (POC) Indicator (IND-222)
//!
//! The Point of Control represents the price level with the highest traded volume
//! within a given period. It is a key concept in Market Profile and Volume Profile
//! analysis, indicating the "fair value" price where most trading activity occurred.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator,
};

/// Configuration for Point of Control indicator.
#[derive(Debug, Clone)]
pub struct PointOfControlConfig {
    /// Lookback period for volume profile calculation.
    pub period: usize,
    /// Number of price bins for volume distribution.
    pub bins: usize,
    /// Whether to use typical price (true) or close price (false).
    pub use_typical_price: bool,
}

impl Default for PointOfControlConfig {
    fn default() -> Self {
        Self {
            period: 20,
            bins: 24,
            use_typical_price: true,
        }
    }
}

/// Output structure for Point of Control indicator.
#[derive(Debug, Clone)]
pub struct PointOfControlOutput {
    /// POC price level for each bar.
    pub poc: Vec<f64>,
    /// Volume at POC level.
    pub poc_volume: Vec<f64>,
    /// POC as percentage of price range.
    pub poc_position: Vec<f64>,
}

/// Point of Control (POC) - Highest volume price level.
///
/// The POC identifies the price level where the most volume was traded
/// during the lookback period. This level often acts as a magnet for price
/// and represents the "value area" center.
///
/// # Interpretation
/// - Price above POC suggests bullish sentiment
/// - Price below POC suggests bearish sentiment
/// - POC migration indicates trend direction
/// - Multiple POCs can indicate consolidation zones
#[derive(Debug, Clone)]
pub struct PointOfControl {
    config: PointOfControlConfig,
}

impl PointOfControl {
    /// Create a new Point of Control indicator with default settings.
    pub fn new() -> Self {
        Self {
            config: PointOfControlConfig::default(),
        }
    }

    /// Create from configuration.
    pub fn from_config(config: PointOfControlConfig) -> Result<Self> {
        if config.period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if config.bins < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "bins".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with custom period and bins.
    pub fn with_params(period: usize, bins: usize) -> Result<Self> {
        Self::from_config(PointOfControlConfig {
            period,
            bins,
            ..Default::default()
        })
    }

    /// Calculate POC values.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> PointOfControlOutput {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut poc = vec![0.0; n];
        let mut poc_volume = vec![0.0; n];
        let mut poc_position = vec![0.0; n];

        for i in (self.config.period - 1)..n {
            let start = i + 1 - self.config.period;

            // Find price range for the period
            let min_price = low[start..=i]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let max_price = high[start..=i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let range = max_price - min_price;

            if range <= 0.0 {
                poc[i] = close[i];
                continue;
            }

            let bin_size = range / self.config.bins as f64;
            let mut bin_volume = vec![0.0; self.config.bins];

            // Distribute volume to bins
            for j in start..=i {
                let price = if self.config.use_typical_price {
                    (high[j] + low[j] + close[j]) / 3.0
                } else {
                    close[j]
                };

                let bin_idx = ((price - min_price) / bin_size).floor() as usize;
                let bin_idx = bin_idx.min(self.config.bins - 1);
                bin_volume[bin_idx] += volume[j];
            }

            // Find POC (highest volume bin)
            let (poc_bin, max_vol) = bin_volume
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, &v)| (i, v))
                .unwrap_or((0, 0.0));

            // Convert bin to price level (center of bin)
            let poc_price = min_price + (poc_bin as f64 + 0.5) * bin_size;

            poc[i] = poc_price;
            poc_volume[i] = max_vol;
            poc_position[i] = (poc_price - min_price) / range;
        }

        // Fill initial values
        for i in 0..(self.config.period - 1).min(n) {
            poc[i] = close[i];
            poc_position[i] = 0.5;
        }

        PointOfControlOutput {
            poc,
            poc_volume,
            poc_position,
        }
    }

    /// Calculate POC distance from current price (normalized).
    pub fn poc_distance(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        let output = self.calculate(high, low, close, volume);
        let n = close.len();
        let mut distance = vec![0.0; n];

        for i in 0..n {
            if output.poc[i] > 0.0 {
                distance[i] = (close[i] - output.poc[i]) / output.poc[i] * 100.0;
            }
        }

        distance
    }
}

impl Default for PointOfControl {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for PointOfControl {
    fn name(&self) -> &str {
        "Point of Control"
    }

    fn min_periods(&self) -> usize {
        self.config.period
    }

    fn output_features(&self) -> usize {
        3
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.period {
            return Err(IndicatorError::InsufficientData {
                required: self.config.period,
                got: data.close.len(),
            });
        }

        let output = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::triple(
            output.poc,
            output.poc_volume,
            output.poc_position,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![
            105.0, 106.0, 107.0, 106.5, 108.0, 109.0, 108.5, 110.0, 109.5, 111.0,
            110.5, 112.0, 111.5, 113.0, 112.5, 114.0, 113.5, 115.0, 114.5, 116.0,
            115.5, 117.0, 116.5, 118.0, 117.5,
        ];
        let low = vec![
            100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 104.5, 106.0,
            105.5, 107.0, 106.5, 108.0, 107.5, 109.0, 108.5, 110.0, 109.5, 111.0,
            110.5, 112.0, 111.5, 113.0, 112.5,
        ];
        let close = vec![
            104.0, 105.0, 106.0, 105.0, 107.0, 108.0, 107.0, 109.0, 108.0, 110.0,
            109.0, 111.0, 110.0, 112.0, 111.0, 113.0, 112.0, 114.0, 113.0, 115.0,
            114.0, 116.0, 115.0, 117.0, 116.0,
        ];
        let volume = vec![
            1000.0, 1200.0, 1100.0, 1300.0, 1500.0, 1400.0, 1600.0, 1800.0, 1700.0, 1900.0,
            2000.0, 1800.0, 2100.0, 2200.0, 2000.0, 2300.0, 2100.0, 2400.0, 2200.0, 2500.0,
            2400.0, 2600.0, 2500.0, 2700.0, 2600.0,
        ];
        (high, low, close, volume)
    }

    #[test]
    fn test_poc_basic() {
        let poc = PointOfControl::with_params(10, 10).unwrap();
        let (high, low, close, volume) = make_test_data();

        let output = poc.calculate(&high, &low, &close, &volume);

        assert_eq!(output.poc.len(), close.len());
        assert_eq!(output.poc_volume.len(), close.len());
        assert_eq!(output.poc_position.len(), close.len());

        // POC should be within price range
        for i in 9..close.len() {
            assert!(output.poc[i] >= low[i - 9..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b)));
            assert!(output.poc[i] <= high[i - 9..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        }
    }

    #[test]
    fn test_poc_position() {
        let poc = PointOfControl::with_params(10, 10).unwrap();
        let (high, low, close, volume) = make_test_data();

        let output = poc.calculate(&high, &low, &close, &volume);

        // POC position should be between 0 and 1
        for i in 9..close.len() {
            assert!(output.poc_position[i] >= 0.0);
            assert!(output.poc_position[i] <= 1.0);
        }
    }

    #[test]
    fn test_poc_distance() {
        let poc = PointOfControl::with_params(10, 10).unwrap();
        let (high, low, close, volume) = make_test_data();

        let distance = poc.poc_distance(&high, &low, &close, &volume);

        assert_eq!(distance.len(), close.len());
        // Distance values should be reasonable percentages
        for d in &distance[9..] {
            assert!(d.abs() < 50.0);
        }
    }

    #[test]
    fn test_poc_compute_trait() {
        let poc = PointOfControl::with_params(10, 10).unwrap();
        let (high, low, close, volume) = make_test_data();

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let result = poc.compute(&data).unwrap();
        assert!(!result.primary.is_empty());
        assert!(result.secondary.is_some());
        assert!(result.tertiary.is_some());
    }

    #[test]
    fn test_poc_invalid_params() {
        assert!(PointOfControl::with_params(1, 10).is_err());
        assert!(PointOfControl::with_params(10, 3).is_err());
    }

    #[test]
    fn test_poc_insufficient_data() {
        let poc = PointOfControl::with_params(10, 10).unwrap();

        let data = OHLCVSeries {
            open: vec![100.0; 5],
            high: vec![105.0; 5],
            low: vec![95.0; 5],
            close: vec![100.0; 5],
            volume: vec![1000.0; 5],
        };

        let result = poc.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_poc_default() {
        let poc = PointOfControl::default();
        assert_eq!(poc.config.period, 20);
        assert_eq!(poc.config.bins, 24);
    }
}
