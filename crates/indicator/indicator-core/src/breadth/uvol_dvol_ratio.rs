//! UVOL/DVOL Ratio - Up/Down Volume Ratio (IND-400)

use super::{BreadthIndicator, BreadthSeries};
use crate::{IndicatorError, IndicatorOutput, Result};

/// UVOL/DVOL Ratio Configuration
#[derive(Debug, Clone)]
pub struct UVOLDVOLRatioConfig {
    /// Smoothing period (0 = no smoothing)
    pub smoothing_period: usize,
    /// Use EMA instead of SMA for smoothing
    pub use_ema: bool,
    /// Strong bullish threshold
    pub strong_bullish_threshold: f64,
    /// Bullish threshold
    pub bullish_threshold: f64,
    /// Bearish threshold (below 1.0)
    pub bearish_threshold: f64,
    /// Strong bearish threshold
    pub strong_bearish_threshold: f64,
}

impl Default for UVOLDVOLRatioConfig {
    fn default() -> Self {
        Self {
            smoothing_period: 0,
            use_ema: false,
            strong_bullish_threshold: 3.0,
            bullish_threshold: 1.5,
            bearish_threshold: 0.67,
            strong_bearish_threshold: 0.33,
        }
    }
}

/// UVOL/DVOL Ratio Indicator
///
/// Measures the ratio of up volume (volume in advancing stocks) to down volume
/// (volume in declining stocks). This ratio shows where the money is flowing
/// in the market.
///
/// # Formula
/// UVOL/DVOL = Advancing Volume / Declining Volume
///
/// # Interpretation
/// - Ratio > 3.0: Strong bullish, heavy accumulation
/// - Ratio > 1.5: Bullish, buying pressure
/// - Ratio = 1.0: Neutral, balanced
/// - Ratio < 0.67: Bearish, selling pressure
/// - Ratio < 0.33: Strong bearish, heavy distribution
///
/// # Use Cases
/// - Identifying institutional accumulation/distribution
/// - Confirming price breakouts with volume
/// - Measuring conviction behind market moves
/// - Divergence analysis with price
#[derive(Debug, Clone)]
pub struct UVOLDVOLRatio {
    config: UVOLDVOLRatioConfig,
}

impl Default for UVOLDVOLRatio {
    fn default() -> Self {
        Self::new()
    }
}

impl UVOLDVOLRatio {
    pub fn new() -> Self {
        Self {
            config: UVOLDVOLRatioConfig::default(),
        }
    }

    pub fn with_config(config: UVOLDVOLRatioConfig) -> Self {
        Self { config }
    }

    pub fn with_smoothing(mut self, period: usize) -> Self {
        self.config.smoothing_period = period;
        self
    }

    pub fn with_ema(mut self) -> Self {
        self.config.use_ema = true;
        self
    }

    /// Calculate SMA
    fn calculate_sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return data.to_vec();
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().filter(|v| !v.is_nan()).sum();
        let count = data[..period].iter().filter(|v| !v.is_nan()).count();
        if count > 0 {
            result.push(sum / count as f64);
        } else {
            result.push(f64::NAN);
        }

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

    /// Calculate EMA
    fn calculate_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return data.to_vec();
        }

        let mut result = vec![f64::NAN; period - 1];
        let multiplier = 2.0 / (period as f64 + 1.0);

        let valid: Vec<f64> = data[..period]
            .iter()
            .filter(|v| !v.is_nan())
            .copied()
            .collect();
        let mut ema = if valid.is_empty() {
            f64::NAN
        } else {
            valid.iter().sum::<f64>() / valid.len() as f64
        };
        result.push(ema);

        for i in period..data.len() {
            if !data[i].is_nan() && !ema.is_nan() {
                ema = (data[i] - ema) * multiplier + ema;
            }
            result.push(ema);
        }

        result
    }

    /// Calculate raw ratio values
    pub fn calculate_raw(&self, up_volume: &[f64], down_volume: &[f64]) -> Vec<f64> {
        up_volume
            .iter()
            .zip(down_volume.iter())
            .map(|(uv, dv)| {
                if *dv == 0.0 {
                    if *uv > 0.0 {
                        f64::INFINITY
                    } else {
                        f64::NAN
                    }
                } else {
                    uv / dv
                }
            })
            .collect()
    }

    /// Calculate UVOL/DVOL ratio from arrays
    pub fn calculate(&self, up_volume: &[f64], down_volume: &[f64]) -> Vec<f64> {
        let raw = self.calculate_raw(up_volume, down_volume);

        if self.config.smoothing_period > 0 {
            if self.config.use_ema {
                self.calculate_ema(&raw, self.config.smoothing_period)
            } else {
                self.calculate_sma(&raw, self.config.smoothing_period)
            }
        } else {
            raw
        }
    }

    /// Calculate from BreadthSeries
    pub fn calculate_series(&self, data: &BreadthSeries) -> Vec<f64> {
        self.calculate(&data.advance_volume, &data.decline_volume)
    }

    /// Interpret ratio value
    pub fn interpret(&self, value: f64) -> UVOLDVOLSignal {
        if value.is_nan() {
            UVOLDVOLSignal::Unknown
        } else if value >= self.config.strong_bullish_threshold {
            UVOLDVOLSignal::StrongBullish
        } else if value >= self.config.bullish_threshold {
            UVOLDVOLSignal::Bullish
        } else if value <= self.config.strong_bearish_threshold {
            UVOLDVOLSignal::StrongBearish
        } else if value <= self.config.bearish_threshold {
            UVOLDVOLSignal::Bearish
        } else {
            UVOLDVOLSignal::Neutral
        }
    }

    /// Calculate net volume (up - down)
    pub fn calculate_net(&self, up_volume: &[f64], down_volume: &[f64]) -> Vec<f64> {
        up_volume
            .iter()
            .zip(down_volume.iter())
            .map(|(uv, dv)| uv - dv)
            .collect()
    }

    /// Calculate cumulative net volume
    pub fn calculate_cumulative_net(&self, up_volume: &[f64], down_volume: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(up_volume.len());
        let mut cumulative = 0.0;

        for (uv, dv) in up_volume.iter().zip(down_volume.iter()) {
            cumulative += uv - dv;
            result.push(cumulative);
        }

        result
    }

    /// Calculate volume percentage (up volume / total volume)
    pub fn calculate_percentage(&self, up_volume: &[f64], down_volume: &[f64]) -> Vec<f64> {
        up_volume
            .iter()
            .zip(down_volume.iter())
            .map(|(uv, dv)| {
                let total = uv + dv;
                if total == 0.0 {
                    0.5
                } else {
                    uv / total
                }
            })
            .collect()
    }

    /// Analyze session volume breadth
    pub fn session_analysis(
        &self,
        up_volume: &[f64],
        down_volume: &[f64],
    ) -> UVOLDVOLSessionAnalysis {
        if up_volume.is_empty() {
            return UVOLDVOLSessionAnalysis::default();
        }

        let ratios = self.calculate_raw(up_volume, down_volume);
        let valid: Vec<f64> = ratios
            .iter()
            .filter(|v| !v.is_nan() && !v.is_infinite())
            .copied()
            .collect();

        if valid.is_empty() {
            return UVOLDVOLSessionAnalysis::default();
        }

        let high = valid.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let low = valid.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let close = *valid.last().unwrap();
        let sum: f64 = valid.iter().sum();
        let average = sum / valid.len() as f64;

        // Total volumes
        let total_up: f64 = up_volume.iter().sum();
        let total_down: f64 = down_volume.iter().sum();
        let session_ratio = if total_down == 0.0 {
            f64::INFINITY
        } else {
            total_up / total_down
        };

        let bias = if session_ratio >= self.config.strong_bullish_threshold {
            VolumeBias::StronglyBullish
        } else if session_ratio >= self.config.bullish_threshold {
            VolumeBias::Bullish
        } else if session_ratio <= self.config.strong_bearish_threshold {
            VolumeBias::StronglyBearish
        } else if session_ratio <= self.config.bearish_threshold {
            VolumeBias::Bearish
        } else {
            VolumeBias::Neutral
        };

        UVOLDVOLSessionAnalysis {
            high,
            low,
            close,
            average,
            total_up_volume: total_up,
            total_down_volume: total_down,
            session_ratio,
            bias,
        }
    }
}

/// UVOL/DVOL signal interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UVOLDVOLSignal {
    /// Very high ratio: Heavy accumulation
    StrongBullish,
    /// High ratio: Buying pressure
    Bullish,
    /// Near 1.0: Balanced
    Neutral,
    /// Low ratio: Selling pressure
    Bearish,
    /// Very low ratio: Heavy distribution
    StrongBearish,
    /// Invalid data
    Unknown,
}

/// Session analysis results
#[derive(Debug, Clone, Default)]
pub struct UVOLDVOLSessionAnalysis {
    /// Highest ratio reading
    pub high: f64,
    /// Lowest ratio reading
    pub low: f64,
    /// Closing ratio reading
    pub close: f64,
    /// Average ratio reading
    pub average: f64,
    /// Total up volume
    pub total_up_volume: f64,
    /// Total down volume
    pub total_down_volume: f64,
    /// Session-wide ratio (total up / total down)
    pub session_ratio: f64,
    /// Overall bias
    pub bias: VolumeBias,
}

/// Session volume bias
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VolumeBias {
    StronglyBullish,
    Bullish,
    #[default]
    Neutral,
    Bearish,
    StronglyBearish,
}

impl BreadthIndicator for UVOLDVOLRatio {
    fn name(&self) -> &str {
        "UVOL/DVOL Ratio"
    }

    fn compute_breadth(&self, data: &BreadthSeries) -> Result<IndicatorOutput> {
        let min_required = if self.config.smoothing_period > 0 {
            self.config.smoothing_period
        } else {
            1
        };

        if data.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.len(),
            });
        }

        let values = self.calculate_series(data);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        if self.config.smoothing_period > 0 {
            self.config.smoothing_period
        } else {
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::breadth::BreadthData;

    fn create_test_series() -> BreadthSeries {
        let mut series = BreadthSeries::new();
        series.push(BreadthData::from_ad_volume(
            2000.0, 1000.0, 3_000_000.0, 1_000_000.0,
        )); // 3.0
        series.push(BreadthData::from_ad_volume(
            1500.0, 1500.0, 2_000_000.0, 2_000_000.0,
        )); // 1.0
        series.push(BreadthData::from_ad_volume(
            1000.0, 2000.0, 1_000_000.0, 3_000_000.0,
        )); // 0.333
        series.push(BreadthData::from_ad_volume(
            1800.0, 1200.0, 2_500_000.0, 1_500_000.0,
        )); // 1.667
        series.push(BreadthData::from_ad_volume(
            1200.0, 1800.0, 800_000.0, 2_400_000.0,
        )); // 0.333
        series
    }

    #[test]
    fn test_uvol_dvol_basic() {
        let ratio = UVOLDVOLRatio::new();
        let series = create_test_series();
        let result = ratio.calculate_series(&series);

        assert_eq!(result.len(), 5);
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 0.3333333333333333).abs() < 1e-10);
    }

    #[test]
    fn test_uvol_dvol_smoothed() {
        let ratio = UVOLDVOLRatio::new().with_smoothing(3);
        let series = create_test_series();
        let result = ratio.calculate_series(&series);

        assert_eq!(result.len(), 5);
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        // SMA(3) of 3.0, 1.0, 0.333 = 1.444...
        assert!((result[2] - 1.4444444444444444).abs() < 1e-10);
    }

    #[test]
    fn test_uvol_dvol_ema() {
        let ratio = UVOLDVOLRatio::new().with_smoothing(3).with_ema();
        let series = create_test_series();
        let result = ratio.calculate_series(&series);

        assert_eq!(result.len(), 5);
        assert!(!result[2].is_nan());
    }

    #[test]
    fn test_interpretation() {
        let ratio = UVOLDVOLRatio::new();

        assert_eq!(ratio.interpret(4.0), UVOLDVOLSignal::StrongBullish);
        assert_eq!(ratio.interpret(2.0), UVOLDVOLSignal::Bullish);
        assert_eq!(ratio.interpret(1.0), UVOLDVOLSignal::Neutral);
        assert_eq!(ratio.interpret(0.5), UVOLDVOLSignal::Bearish);
        assert_eq!(ratio.interpret(0.2), UVOLDVOLSignal::StrongBearish);
        assert_eq!(ratio.interpret(f64::NAN), UVOLDVOLSignal::Unknown);
    }

    #[test]
    fn test_net_volume() {
        let ratio = UVOLDVOLRatio::new();
        let up = vec![3_000_000.0, 2_000_000.0, 1_000_000.0];
        let down = vec![1_000_000.0, 2_000_000.0, 3_000_000.0];

        let result = ratio.calculate_net(&up, &down);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 2_000_000.0).abs() < 1e-10);
        assert!((result[1] - 0.0).abs() < 1e-10);
        assert!((result[2] - (-2_000_000.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cumulative_net() {
        let ratio = UVOLDVOLRatio::new();
        let up = vec![3_000_000.0, 2_000_000.0, 1_000_000.0];
        let down = vec![1_000_000.0, 2_000_000.0, 3_000_000.0];

        let result = ratio.calculate_cumulative_net(&up, &down);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 2_000_000.0).abs() < 1e-10);
        assert!((result[1] - 2_000_000.0).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentage() {
        let ratio = UVOLDVOLRatio::new();
        let up = vec![3_000_000.0, 2_000_000.0, 1_000_000.0];
        let down = vec![1_000_000.0, 2_000_000.0, 3_000_000.0];

        let result = ratio.calculate_percentage(&up, &down);

        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.75).abs() < 1e-10);
        assert!((result[1] - 0.5).abs() < 1e-10);
        assert!((result[2] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_session_analysis() {
        let ratio = UVOLDVOLRatio::new();
        let up = vec![3_000_000.0, 2_000_000.0, 2_500_000.0];
        let down = vec![1_000_000.0, 2_000_000.0, 1_500_000.0];

        let analysis = ratio.session_analysis(&up, &down);

        assert!((analysis.high - 3.0).abs() < 1e-10);
        assert!((analysis.low - 1.0).abs() < 1e-10);
        assert!((analysis.total_up_volume - 7_500_000.0).abs() < 1e-10);
        assert!((analysis.total_down_volume - 4_500_000.0).abs() < 1e-10);
        assert!((analysis.session_ratio - 1.6666666666666667).abs() < 1e-10);
        assert_eq!(analysis.bias, VolumeBias::Bullish);
    }

    #[test]
    fn test_zero_down_volume() {
        let ratio = UVOLDVOLRatio::new();
        let up = vec![1_000_000.0];
        let down = vec![0.0];

        let result = ratio.calculate_raw(&up, &down);

        assert_eq!(result.len(), 1);
        assert!(result[0].is_infinite());
    }

    #[test]
    fn test_breadth_indicator_trait() {
        let ratio = UVOLDVOLRatio::new();
        let series = create_test_series();

        let result = ratio.compute_breadth(&series);
        assert!(result.is_ok());

        assert_eq!(ratio.min_periods(), 1);
        assert_eq!(ratio.name(), "UVOL/DVOL Ratio");
    }

    #[test]
    fn test_empty_series() {
        let ratio = UVOLDVOLRatio::new();
        let series = BreadthSeries::new();

        let result = ratio.compute_breadth(&series);
        assert!(result.is_err());
    }
}
