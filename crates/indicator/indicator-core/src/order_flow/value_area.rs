//! Value Area Indicator (IND-223)
//!
//! The Value Area represents the price range where a specified percentage
//! (typically 70%) of the total volume was traded. It consists of the Value
//! Area High (VAH) and Value Area Low (VAL), which define the upper and
//! lower boundaries of this high-volume zone.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator,
};

/// Configuration for Value Area indicator.
#[derive(Debug, Clone)]
pub struct ValueAreaConfig {
    /// Lookback period for volume profile calculation.
    pub period: usize,
    /// Number of price bins for volume distribution.
    pub bins: usize,
    /// Percentage of volume to include in value area (0.0 to 1.0).
    pub value_percentage: f64,
}

impl Default for ValueAreaConfig {
    fn default() -> Self {
        Self {
            period: 20,
            bins: 24,
            value_percentage: 0.70,
        }
    }
}

/// Output structure for Value Area indicator.
#[derive(Debug, Clone)]
pub struct ValueAreaOutput {
    /// Point of Control (highest volume price).
    pub poc: Vec<f64>,
    /// Value Area High.
    pub vah: Vec<f64>,
    /// Value Area Low.
    pub val: Vec<f64>,
    /// Value Area width as percentage of price.
    pub va_width: Vec<f64>,
}

/// Value Area - 70% (configurable) volume range.
///
/// The Value Area identifies the price range containing a specified percentage
/// of the total traded volume, centered around the Point of Control.
///
/// # Components
/// - POC: Price level with highest volume
/// - VAH: Value Area High - upper boundary
/// - VAL: Value Area Low - lower boundary
///
/// # Interpretation
/// - Price above VAH suggests potential overbought
/// - Price below VAL suggests potential oversold
/// - Price within VA is considered "fair value"
/// - VA expansion indicates increased volatility/acceptance
/// - VA contraction indicates consolidation
#[derive(Debug, Clone)]
pub struct ValueArea {
    config: ValueAreaConfig,
}

impl ValueArea {
    /// Create a new Value Area indicator with default settings.
    pub fn new() -> Self {
        Self {
            config: ValueAreaConfig::default(),
        }
    }

    /// Create from configuration.
    pub fn from_config(config: ValueAreaConfig) -> Result<Self> {
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
        if config.value_percentage <= 0.0 || config.value_percentage > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "value_percentage".to_string(),
                reason: "must be between 0 and 1".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with custom parameters.
    pub fn with_params(period: usize, bins: usize, value_percentage: f64) -> Result<Self> {
        Self::from_config(ValueAreaConfig {
            period,
            bins,
            value_percentage,
        })
    }

    /// Calculate Value Area.
    pub fn calculate(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> ValueAreaOutput {
        let n = close.len().min(volume.len()).min(high.len()).min(low.len());
        let mut poc = vec![0.0; n];
        let mut vah = vec![0.0; n];
        let mut val = vec![0.0; n];
        let mut va_width = vec![0.0; n];

        for i in (self.config.period - 1)..n {
            let start = i + 1 - self.config.period;

            // Find price range
            let min_price = low[start..=i]
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let max_price = high[start..=i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let range = max_price - min_price;

            if range <= 0.0 {
                poc[i] = close[i];
                vah[i] = close[i];
                val[i] = close[i];
                continue;
            }

            let bin_size = range / self.config.bins as f64;
            let mut bin_volume = vec![0.0; self.config.bins];
            let mut total_volume = 0.0;

            // Distribute volume to bins
            for j in start..=i {
                let typical = (high[j] + low[j] + close[j]) / 3.0;
                let bin_idx = ((typical - min_price) / bin_size).floor() as usize;
                let bin_idx = bin_idx.min(self.config.bins - 1);
                bin_volume[bin_idx] += volume[j];
                total_volume += volume[j];
            }

            // Find POC bin
            let poc_bin = bin_volume
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // Calculate value area by expanding from POC
            let target_volume = total_volume * self.config.value_percentage;
            let mut va_volume = bin_volume[poc_bin];
            let mut va_low_bin = poc_bin;
            let mut va_high_bin = poc_bin;

            // Expand value area alternating up and down
            while va_volume < target_volume {
                let can_expand_up = va_high_bin < self.config.bins - 1;
                let can_expand_down = va_low_bin > 0;

                if !can_expand_up && !can_expand_down {
                    break;
                }

                let vol_up = if can_expand_up {
                    bin_volume[va_high_bin + 1]
                } else {
                    0.0
                };

                let vol_down = if can_expand_down {
                    bin_volume[va_low_bin - 1]
                } else {
                    0.0
                };

                // Expand in direction of higher volume
                if vol_up >= vol_down && can_expand_up {
                    va_high_bin += 1;
                    va_volume += bin_volume[va_high_bin];
                } else if can_expand_down {
                    va_low_bin -= 1;
                    va_volume += bin_volume[va_low_bin];
                } else if can_expand_up {
                    va_high_bin += 1;
                    va_volume += bin_volume[va_high_bin];
                }
            }

            // Convert bins to prices
            poc[i] = min_price + (poc_bin as f64 + 0.5) * bin_size;
            val[i] = min_price + va_low_bin as f64 * bin_size;
            vah[i] = min_price + (va_high_bin as f64 + 1.0) * bin_size;
            va_width[i] = (vah[i] - val[i]) / poc[i] * 100.0;
        }

        // Fill initial values
        for i in 0..(self.config.period - 1).min(n) {
            poc[i] = close[i];
            vah[i] = high[i];
            val[i] = low[i];
            if close[i] > 0.0 {
                va_width[i] = (high[i] - low[i]) / close[i] * 100.0;
            }
        }

        ValueAreaOutput {
            poc,
            vah,
            val,
            va_width,
        }
    }

    /// Calculate position within value area.
    /// Returns: -1 below VAL, 0-1 within VA, >1 above VAH
    pub fn position_in_va(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        let output = self.calculate(high, low, close, volume);
        let n = close.len();
        let mut position = vec![0.5; n];

        for i in 0..n {
            let va_range = output.vah[i] - output.val[i];
            if va_range > 0.0 {
                position[i] = (close[i] - output.val[i]) / va_range;
            }
        }

        position
    }
}

impl Default for ValueArea {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for ValueArea {
    fn name(&self) -> &str {
        "Value Area"
    }

    fn min_periods(&self) -> usize {
        self.config.period
    }

    fn output_features(&self) -> usize {
        4
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.period {
            return Err(IndicatorError::InsufficientData {
                required: self.config.period,
                got: data.close.len(),
            });
        }

        let output = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        // Return POC as primary, VAH as secondary, VAL as tertiary
        Ok(IndicatorOutput::triple(output.poc, output.vah, output.val))
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
    fn test_value_area_basic() {
        let va = ValueArea::with_params(10, 10, 0.70).unwrap();
        let (high, low, close, volume) = make_test_data();

        let output = va.calculate(&high, &low, &close, &volume);

        assert_eq!(output.poc.len(), close.len());
        assert_eq!(output.vah.len(), close.len());
        assert_eq!(output.val.len(), close.len());

        // VAH >= POC >= VAL
        for i in 9..close.len() {
            assert!(
                output.vah[i] >= output.poc[i],
                "VAH {} should be >= POC {} at index {}",
                output.vah[i],
                output.poc[i],
                i
            );
            assert!(
                output.poc[i] >= output.val[i],
                "POC {} should be >= VAL {} at index {}",
                output.poc[i],
                output.val[i],
                i
            );
        }
    }

    #[test]
    fn test_value_area_boundaries() {
        let va = ValueArea::with_params(10, 10, 0.70).unwrap();
        let (high, low, close, volume) = make_test_data();

        let output = va.calculate(&high, &low, &close, &volume);

        // VA should be within price range
        for i in 9..close.len() {
            let min_low = low[i - 9..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_high = high[i - 9..=i]
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            assert!(output.val[i] >= min_low - 0.01);
            assert!(output.vah[i] <= max_high + 0.01);
        }
    }

    #[test]
    fn test_value_area_position() {
        let va = ValueArea::with_params(10, 10, 0.70).unwrap();
        let (high, low, close, volume) = make_test_data();

        let position = va.position_in_va(&high, &low, &close, &volume);

        assert_eq!(position.len(), close.len());
        // Positions should be finite
        for p in &position {
            assert!(p.is_finite());
        }
    }

    #[test]
    fn test_value_area_width() {
        let va = ValueArea::with_params(10, 10, 0.70).unwrap();
        let (high, low, close, volume) = make_test_data();

        let output = va.calculate(&high, &low, &close, &volume);

        // Width should be positive
        for i in 9..close.len() {
            assert!(output.va_width[i] >= 0.0);
        }
    }

    #[test]
    fn test_value_area_compute_trait() {
        let va = ValueArea::with_params(10, 10, 0.70).unwrap();
        let (high, low, close, volume) = make_test_data();

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let result = va.compute(&data).unwrap();
        assert!(!result.primary.is_empty());
        assert!(result.secondary.is_some());
        assert!(result.tertiary.is_some());
    }

    #[test]
    fn test_value_area_invalid_params() {
        assert!(ValueArea::with_params(1, 10, 0.70).is_err());
        assert!(ValueArea::with_params(10, 3, 0.70).is_err());
        assert!(ValueArea::with_params(10, 10, 0.0).is_err());
        assert!(ValueArea::with_params(10, 10, 1.5).is_err());
    }

    #[test]
    fn test_value_area_default() {
        let va = ValueArea::default();
        assert_eq!(va.config.period, 20);
        assert_eq!(va.config.bins, 24);
        assert!((va.config.value_percentage - 0.70).abs() < 0.01);
    }

    #[test]
    fn test_value_area_different_percentages() {
        let (high, low, close, volume) = make_test_data();

        let va_50 = ValueArea::with_params(10, 10, 0.50).unwrap();
        let va_70 = ValueArea::with_params(10, 10, 0.70).unwrap();
        let va_90 = ValueArea::with_params(10, 10, 0.90).unwrap();

        let out_50 = va_50.calculate(&high, &low, &close, &volume);
        let out_70 = va_70.calculate(&high, &low, &close, &volume);
        let out_90 = va_90.calculate(&high, &low, &close, &volume);

        // Higher percentage should result in wider value area
        for i in 9..close.len() {
            let width_50 = out_50.vah[i] - out_50.val[i];
            let width_70 = out_70.vah[i] - out_70.val[i];
            let width_90 = out_90.vah[i] - out_90.val[i];

            assert!(width_70 >= width_50 - 0.01);
            assert!(width_90 >= width_70 - 0.01);
        }
    }
}
