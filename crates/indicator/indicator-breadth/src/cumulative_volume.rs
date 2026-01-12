//! Cumulative Volume Index (CVI) indicator.

use crate::{BreadthIndicator, BreadthSeries};
use indicator_spi::{IndicatorError, IndicatorOutput, Result};

/// Cumulative Volume Index (CVI)
///
/// A breadth indicator that tracks the cumulative sum of net advancing
/// volume (advancing volume minus declining volume). It measures the
/// flow of volume into advancing versus declining stocks.
///
/// # Formula
/// CVI = Previous CVI + (Advancing Volume - Declining Volume)
///
/// Or: CVI = Cumulative Sum of Net Advancing Volume
///
/// # Variants
/// - Raw CVI: Cumulative net advancing volume
/// - Normalized CVI: Net advancing volume / total volume
///
/// # Interpretation
/// - Rising CVI: Volume flowing into advancing stocks
/// - Falling CVI: Volume flowing into declining stocks
/// - Divergence from price: Volume not confirming price trend
/// - Zero line crossovers: Shift in volume sentiment
#[derive(Debug, Clone)]
pub struct CumulativeVolumeIndex {
    /// Starting value for cumulative index (default: 0)
    start_value: f64,
    /// Whether to normalize by total volume
    normalized: bool,
    /// Multiplier for normalized values (default: 1000)
    normalization_multiplier: f64,
}

impl Default for CumulativeVolumeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl CumulativeVolumeIndex {
    pub fn new() -> Self {
        Self {
            start_value: 0.0,
            normalized: false,
            normalization_multiplier: 1000.0,
        }
    }

    pub fn with_start_value(mut self, start_value: f64) -> Self {
        self.start_value = start_value;
        self
    }

    pub fn normalized(mut self) -> Self {
        self.normalized = true;
        self
    }

    pub fn with_multiplier(mut self, multiplier: f64) -> Self {
        self.normalization_multiplier = multiplier;
        self
    }

    /// Calculate CVI from advance/decline volume arrays
    pub fn calculate(&self, advance_volume: &[f64], decline_volume: &[f64]) -> Vec<f64> {
        if advance_volume.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(advance_volume.len());
        let mut cumulative = self.start_value;

        if self.normalized {
            for (av, dv) in advance_volume.iter().zip(decline_volume.iter()) {
                let total = av + dv;
                let net = if total == 0.0 {
                    0.0
                } else {
                    ((av - dv) / total) * self.normalization_multiplier
                };
                cumulative += net;
                result.push(cumulative);
            }
        } else {
            for (av, dv) in advance_volume.iter().zip(decline_volume.iter()) {
                cumulative += av - dv;
                result.push(cumulative);
            }
        }

        result
    }

    /// Calculate from BreadthSeries
    pub fn calculate_series(&self, data: &BreadthSeries) -> Vec<f64> {
        self.calculate(&data.advance_volume, &data.decline_volume)
    }

    /// Calculate net advancing volume (non-cumulative)
    pub fn net_advancing_volume(&self, data: &BreadthSeries) -> Vec<f64> {
        data.net_advance_volume()
    }

    /// Calculate volume ratio: advancing volume / total volume
    pub fn volume_ratio(&self, data: &BreadthSeries) -> Vec<f64> {
        data.advance_volume
            .iter()
            .zip(data.decline_volume.iter())
            .zip(data.unchanged_volume.iter())
            .map(|((av, dv), uv)| {
                let total = av + dv + uv;
                if total == 0.0 {
                    0.5 // Neutral when no volume
                } else {
                    av / total
                }
            })
            .collect()
    }
}

impl BreadthIndicator for CumulativeVolumeIndex {
    fn name(&self) -> &str {
        "Cumulative Volume Index"
    }

    fn compute_breadth(&self, data: &BreadthSeries) -> Result<IndicatorOutput> {
        if data.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let values = self.calculate_series(data);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        1
    }
}

/// Up/Down Volume indicator
///
/// A simpler volume breadth measure that tracks the ratio or difference
/// between advancing and declining volume without cumulation.
#[derive(Debug, Clone)]
pub struct UpDownVolume {
    /// Whether to use ratio (true) or difference (false)
    use_ratio: bool,
    /// Smoothing period (0 = no smoothing)
    smoothing_period: usize,
}

impl Default for UpDownVolume {
    fn default() -> Self {
        Self::new()
    }
}

impl UpDownVolume {
    pub fn new() -> Self {
        Self {
            use_ratio: false,
            smoothing_period: 0,
        }
    }

    pub fn ratio() -> Self {
        Self {
            use_ratio: true,
            smoothing_period: 0,
        }
    }

    pub fn with_smoothing(mut self, period: usize) -> Self {
        self.smoothing_period = period;
        self
    }

    /// Calculate SMA for smoothing
    fn calculate_sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return data.to_vec();
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().filter(|v| !v.is_nan()).sum();
        result.push(sum / period as f64);

        for i in period..data.len() {
            if !data[i - period].is_nan() {
                sum -= data[i - period];
            }
            if !data[i].is_nan() {
                sum += data[i];
            }
            result.push(sum / period as f64);
        }

        result
    }

    /// Calculate up/down volume from BreadthSeries
    pub fn calculate(&self, data: &BreadthSeries) -> Vec<f64> {
        let raw: Vec<f64> = if self.use_ratio {
            data.advance_volume
                .iter()
                .zip(data.decline_volume.iter())
                .map(|(av, dv)| {
                    if *dv == 0.0 {
                        if *av > 0.0 {
                            f64::INFINITY
                        } else {
                            1.0
                        }
                    } else {
                        av / dv
                    }
                })
                .collect()
        } else {
            data.net_advance_volume()
        };

        if self.smoothing_period > 0 {
            self.calculate_sma(&raw, self.smoothing_period)
        } else {
            raw
        }
    }
}

impl BreadthIndicator for UpDownVolume {
    fn name(&self) -> &str {
        "Up/Down Volume"
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

        let values = self.calculate(data);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BreadthData;

    fn create_test_series() -> BreadthSeries {
        let mut series = BreadthSeries::new();
        // Day 1: More advancing volume
        series.push(BreadthData::from_ad_volume(
            1500.0, 1300.0, 1_000_000.0, 800_000.0,
        ));
        // Day 2: More declining volume
        series.push(BreadthData::from_ad_volume(
            1400.0, 1500.0, 700_000.0, 900_000.0,
        ));
        // Day 3: Strong advancing volume
        series.push(BreadthData::from_ad_volume(
            1800.0, 1200.0, 1_200_000.0, 600_000.0,
        ));
        // Day 4: Equal
        series.push(BreadthData::from_ad_volume(
            1500.0, 1500.0, 800_000.0, 800_000.0,
        ));
        // Day 5: Declining
        series.push(BreadthData::from_ad_volume(
            1300.0, 1700.0, 500_000.0, 1_000_000.0,
        ));
        series
    }

    #[test]
    fn test_cvi_basic() {
        let cvi = CumulativeVolumeIndex::new();
        let series = create_test_series();
        let result = cvi.calculate_series(&series);

        assert_eq!(result.len(), 5);
        // Day 1: 1,000,000 - 800,000 = 200,000
        assert!((result[0] - 200_000.0).abs() < 1e-10);
        // Day 2: 200,000 + (700,000 - 900,000) = 0
        assert!((result[1] - 0.0).abs() < 1e-10);
        // Day 3: 0 + (1,200,000 - 600,000) = 600,000
        assert!((result[2] - 600_000.0).abs() < 1e-10);
        // Day 4: 600,000 + 0 = 600,000
        assert!((result[3] - 600_000.0).abs() < 1e-10);
        // Day 5: 600,000 + (500,000 - 1,000,000) = 100,000
        assert!((result[4] - 100_000.0).abs() < 1e-10);
    }

    #[test]
    fn test_cvi_with_start_value() {
        let cvi = CumulativeVolumeIndex::new().with_start_value(1_000_000.0);
        let series = create_test_series();
        let result = cvi.calculate_series(&series);

        // Should be offset by start value
        assert!((result[0] - 1_200_000.0).abs() < 1e-10);
    }

    #[test]
    fn test_cvi_normalized() {
        let cvi = CumulativeVolumeIndex::new().normalized().with_multiplier(1000.0);
        let series = create_test_series();
        let result = cvi.calculate_series(&series);

        // Day 1: (1,000,000 - 800,000) / 1,800,000 * 1000 = 111.11...
        let expected_day1 = (200_000.0 / 1_800_000.0) * 1000.0;
        assert!((result[0] - expected_day1).abs() < 0.01);
    }

    #[test]
    fn test_cvi_empty() {
        let cvi = CumulativeVolumeIndex::new();
        let series = BreadthSeries::new();
        let result = cvi.compute_breadth(&series);

        assert!(result.is_err());
    }

    #[test]
    fn test_net_advancing_volume() {
        let cvi = CumulativeVolumeIndex::new();
        let series = create_test_series();
        let result = cvi.net_advancing_volume(&series);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 200_000.0).abs() < 1e-10);
        assert!((result[1] - (-200_000.0)).abs() < 1e-10);
    }

    #[test]
    fn test_volume_ratio() {
        let cvi = CumulativeVolumeIndex::new();
        let series = create_test_series();
        let result = cvi.volume_ratio(&series);

        // Day 1: 1,000,000 / 1,800,000 = 0.555...
        assert!((result[0] - 0.5555555555555556).abs() < 1e-10);
        // Day 4: 800,000 / 1,600,000 = 0.5
        assert!((result[3] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_up_down_volume_difference() {
        let udv = UpDownVolume::new();
        let series = create_test_series();
        let result = udv.calculate(&series);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 200_000.0).abs() < 1e-10);
        assert!((result[1] - (-200_000.0)).abs() < 1e-10);
    }

    #[test]
    fn test_up_down_volume_ratio() {
        let udv = UpDownVolume::ratio();
        let series = create_test_series();
        let result = udv.calculate(&series);

        assert_eq!(result.len(), 5);
        // Day 1: 1,000,000 / 800,000 = 1.25
        assert!((result[0] - 1.25).abs() < 1e-10);
        // Day 4: 800,000 / 800,000 = 1.0
        assert!((result[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_up_down_volume_with_smoothing() {
        let udv = UpDownVolume::new().with_smoothing(3);
        let series = create_test_series();
        let result = udv.calculate(&series);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // SMA of 200000, -200000, 600000 = 200000
        assert!((result[2] - 200_000.0).abs() < 1e-10);
    }
}
