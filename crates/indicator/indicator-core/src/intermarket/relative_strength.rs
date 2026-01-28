//! Relative Strength (Comparative) Indicator
//!
//! Compares asset performance vs benchmark by calculating the ratio of their
//! prices over time. Useful for identifying outperformance/underperformance
//! and momentum analysis.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

use super::DualSeries;

/// Relative Strength (Comparative) output values.
#[derive(Debug, Clone, Copy)]
pub struct RelativeStrengthOutput {
    /// Raw ratio: asset_price / benchmark_price.
    pub ratio: f64,
    /// Normalized ratio (if enabled, starts at 100).
    pub normalized: f64,
    /// Rate of change of the ratio (momentum).
    pub roc: f64,
}

/// Relative Strength (Comparative) signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelativeStrengthSignal {
    /// Asset is outperforming benchmark (positive momentum).
    Outperforming,
    /// Asset is underperforming benchmark (negative momentum).
    Underperforming,
    /// Neutral - no clear relative performance signal.
    Neutral,
}

/// Relative Strength (Comparative) Indicator.
///
/// Compares asset performance versus a benchmark by calculating the ratio
/// of their prices over time. This is different from RSI (Relative Strength Index)
/// which measures price momentum of a single asset.
///
/// # Algorithm
/// 1. Calculate ratio: asset_price / benchmark_price
/// 2. Optionally normalize to starting value of 100
/// 3. Calculate rate of change (ROC) of ratio for momentum
///
/// # Interpretation
/// - Rising ratio = asset outperforming benchmark
/// - Falling ratio = asset underperforming benchmark
/// - ROC > 0 = accelerating outperformance
/// - ROC < 0 = accelerating underperformance
///
/// # Example Usage
/// ```ignore
/// let rs = RelativeStrength::new()
///     .with_normalize(true)
///     .with_roc_period(Some(14))
///     .with_benchmark(&sp500_prices);
/// let outputs = rs.calculate(&asset_prices);
/// ```
#[derive(Debug, Clone)]
pub struct RelativeStrength {
    /// Whether to normalize the ratio to start at 100.
    normalize: bool,
    /// Period for rate of change calculation (None = no ROC).
    roc_period: Option<usize>,
    /// Benchmark price series.
    benchmark_series: Vec<f64>,
    /// Threshold for ROC to signal outperformance.
    roc_threshold: f64,
}

impl RelativeStrength {
    /// Create a new Relative Strength indicator with default parameters.
    pub fn new() -> Self {
        Self {
            normalize: false,
            roc_period: None,
            benchmark_series: Vec::new(),
            roc_threshold: 0.0,
        }
    }

    /// Set whether to normalize the ratio to start at 100.
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set the ROC period for momentum calculation.
    pub fn with_roc_period(mut self, period: Option<usize>) -> Self {
        self.roc_period = period;
        self
    }

    /// Set the benchmark price series.
    pub fn with_benchmark(mut self, series: &[f64]) -> Self {
        self.benchmark_series = series.to_vec();
        self
    }

    /// Set the ROC threshold for signaling outperformance.
    pub fn with_roc_threshold(mut self, threshold: f64) -> Self {
        self.roc_threshold = threshold;
        self
    }

    /// Calculate the raw ratio between asset and benchmark.
    fn calculate_ratio(asset: &[f64], benchmark: &[f64]) -> Vec<f64> {
        let n = asset.len().min(benchmark.len());
        asset[..n]
            .iter()
            .zip(benchmark[..n].iter())
            .map(|(a, b)| {
                if b.abs() < 1e-10 {
                    f64::NAN
                } else {
                    a / b
                }
            })
            .collect()
    }

    /// Normalize ratio series to start at 100.
    fn normalize_ratio(ratio: &[f64]) -> Vec<f64> {
        // Find first valid ratio value
        let first_valid = ratio.iter().find(|r| !r.is_nan()).copied();

        match first_valid {
            Some(base) if base.abs() > 1e-10 => {
                ratio
                    .iter()
                    .map(|r| {
                        if r.is_nan() {
                            f64::NAN
                        } else {
                            (r / base) * 100.0
                        }
                    })
                    .collect()
            }
            _ => vec![f64::NAN; ratio.len()],
        }
    }

    /// Calculate rate of change.
    fn calculate_roc(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period + 1 || period == 0 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period];
        for i in period..n {
            let prev = data[i - period];
            if prev.abs() < 1e-10 || prev.is_nan() || data[i].is_nan() {
                result.push(f64::NAN);
            } else {
                let roc_val = ((data[i] - prev) / prev) * 100.0;
                result.push(roc_val);
            }
        }
        result
    }

    /// Calculate relative strength for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<Option<RelativeStrengthOutput>> {
        let ratio = Self::calculate_ratio(&dual.series1, &dual.series2);
        let normalized = if self.normalize {
            Self::normalize_ratio(&ratio)
        } else {
            ratio.clone()
        };

        let roc = match self.roc_period {
            Some(period) if period > 0 => Self::calculate_roc(&ratio, period),
            _ => vec![f64::NAN; ratio.len()],
        };

        ratio
            .iter()
            .zip(normalized.iter())
            .zip(roc.iter())
            .map(|((&r, &n), &rc)| {
                if r.is_nan() {
                    None
                } else {
                    Some(RelativeStrengthOutput {
                        ratio: r,
                        normalized: n,
                        roc: rc,
                    })
                }
            })
            .collect()
    }

    /// Calculate relative strength using two series directly.
    pub fn calculate_between(
        &self,
        asset: &[f64],
        benchmark: &[f64],
    ) -> Vec<Option<RelativeStrengthOutput>> {
        let dual = DualSeries::from_slices(asset, benchmark);
        self.calculate(&dual)
    }

    /// Get trading signals based on ROC of relative strength.
    pub fn signals(&self, dual: &DualSeries) -> Vec<RelativeStrengthSignal> {
        let outputs = self.calculate(dual);

        outputs
            .iter()
            .map(|opt| match opt {
                None => RelativeStrengthSignal::Neutral,
                Some(out) => {
                    if out.roc.is_nan() {
                        RelativeStrengthSignal::Neutral
                    } else if out.roc > self.roc_threshold {
                        RelativeStrengthSignal::Outperforming
                    } else if out.roc < -self.roc_threshold {
                        RelativeStrengthSignal::Underperforming
                    } else {
                        RelativeStrengthSignal::Neutral
                    }
                }
            })
            .collect()
    }

    /// Extract ratio series from dual series.
    pub fn ratio_series(&self, dual: &DualSeries) -> Vec<f64> {
        Self::calculate_ratio(&dual.series1, &dual.series2)
    }

    /// Extract normalized ratio series from dual series.
    pub fn normalized_series(&self, dual: &DualSeries) -> Vec<f64> {
        let ratio = Self::calculate_ratio(&dual.series1, &dual.series2);
        Self::normalize_ratio(&ratio)
    }

    /// Extract ROC series from dual series.
    pub fn roc_series(&self, dual: &DualSeries) -> Vec<f64> {
        let ratio = Self::calculate_ratio(&dual.series1, &dual.series2);
        match self.roc_period {
            Some(period) if period > 0 => Self::calculate_roc(&ratio, period),
            _ => vec![f64::NAN; ratio.len()],
        }
    }
}

impl Default for RelativeStrength {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for RelativeStrength {
    fn name(&self) -> &str {
        "RelativeStrength"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.benchmark_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "benchmark_series".to_string(),
                reason: "Benchmark series must be set before computing Relative Strength"
                    .to_string(),
            });
        }

        if self.benchmark_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "benchmark_series".to_string(),
                reason: format!(
                    "Benchmark series length ({}) must match asset series length ({})",
                    self.benchmark_series.len(),
                    data.close.len()
                ),
            });
        }

        let min_required = self.roc_period.unwrap_or(1);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.benchmark_series);
        let outputs = self.calculate(&dual);

        // Primary: ratio (or normalized if enabled)
        // Secondary: ROC (if enabled)
        let primary: Vec<f64> = outputs
            .iter()
            .map(|opt| {
                opt.map(|o| if self.normalize { o.normalized } else { o.ratio })
                    .unwrap_or(f64::NAN)
            })
            .collect();

        let secondary: Vec<f64> = outputs
            .iter()
            .map(|opt| opt.map(|o| o.roc).unwrap_or(f64::NAN))
            .collect();

        Ok(IndicatorOutput::dual(primary, secondary))
    }

    fn min_periods(&self) -> usize {
        self.roc_period.unwrap_or(1)
    }

    fn output_features(&self) -> usize {
        2 // ratio/normalized, ROC
    }
}

impl SignalIndicator for RelativeStrength {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if self.benchmark_series.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let dual = DualSeries::from_slices(&data.close, &self.benchmark_series);
        let signals = self.signals(&dual);

        match signals.last() {
            Some(RelativeStrengthSignal::Outperforming) => Ok(IndicatorSignal::Bullish),
            Some(RelativeStrengthSignal::Underperforming) => Ok(IndicatorSignal::Bearish),
            _ => Ok(IndicatorSignal::Neutral),
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        if self.benchmark_series.is_empty() {
            return Ok(vec![IndicatorSignal::Neutral; data.close.len()]);
        }

        let dual = DualSeries::from_slices(&data.close, &self.benchmark_series);
        let rs_signals = self.signals(&dual);

        let signals = rs_signals
            .iter()
            .map(|s| match s {
                RelativeStrengthSignal::Outperforming => IndicatorSignal::Bullish,
                RelativeStrengthSignal::Underperforming => IndicatorSignal::Bearish,
                RelativeStrengthSignal::Neutral => IndicatorSignal::Neutral,
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_outperforming_pair(n: usize) -> DualSeries {
        // Asset rises faster than benchmark
        let asset: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 1.5).collect();
        let benchmark: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.5).collect();
        DualSeries::new(asset, benchmark)
    }

    fn create_underperforming_pair(n: usize) -> DualSeries {
        // Asset rises slower than benchmark
        let asset: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 0.3).collect();
        let benchmark: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 1.0).collect();
        DualSeries::new(asset, benchmark)
    }

    fn create_stable_pair(n: usize) -> DualSeries {
        // Asset and benchmark rise at same rate
        let asset: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64) * 1.0).collect();
        let benchmark: Vec<f64> = (0..n).map(|i| 50.0 + (i as f64) * 0.5).collect();
        DualSeries::new(asset, benchmark)
    }

    #[test]
    fn test_basic_ratio() {
        let dual = create_stable_pair(10);
        let rs = RelativeStrength::new();
        let outputs = rs.calculate(&dual);

        // All should be valid
        assert!(outputs.iter().all(|o| o.is_some()));

        // Ratio should be constant at 2.0 (100/50, 101/50.5, etc.)
        for out in outputs.iter().flatten() {
            assert!(
                (out.ratio - 2.0).abs() < 0.01,
                "Ratio should be close to 2.0"
            );
        }
    }

    #[test]
    fn test_normalized_ratio() {
        let dual = create_outperforming_pair(50);
        let rs = RelativeStrength::new().with_normalize(true);
        let outputs = rs.calculate(&dual);

        // First valid value should be 100
        let first = outputs.iter().find(|o| o.is_some()).unwrap().unwrap();
        assert!(
            (first.normalized - 100.0).abs() < 0.01,
            "First normalized value should be 100"
        );

        // Later values should be > 100 (outperforming)
        let last = outputs.last().unwrap().unwrap();
        assert!(
            last.normalized > 100.0,
            "Normalized should be > 100 for outperforming asset"
        );
    }

    #[test]
    fn test_roc_calculation() {
        let dual = create_outperforming_pair(30);
        let rs = RelativeStrength::new().with_roc_period(Some(5));
        let outputs = rs.calculate(&dual);

        // First 5 values should have NaN ROC
        for out in outputs[..5].iter().flatten() {
            assert!(out.roc.is_nan(), "ROC should be NaN during warmup");
        }

        // After warmup, ROC should be positive (ratio is increasing)
        for out in outputs[10..].iter().flatten() {
            assert!(
                !out.roc.is_nan(),
                "ROC should be valid after warmup"
            );
            assert!(
                out.roc > 0.0,
                "ROC should be positive for outperforming asset"
            );
        }
    }

    #[test]
    fn test_underperforming_signals() {
        let dual = create_underperforming_pair(30);
        let rs = RelativeStrength::new()
            .with_roc_period(Some(5))
            .with_roc_threshold(0.0);
        let signals = rs.signals(&dual);

        // After warmup, should have underperforming signals
        let underperforming_count = signals
            .iter()
            .skip(10)
            .filter(|s| **s == RelativeStrengthSignal::Underperforming)
            .count();

        assert!(
            underperforming_count > 0,
            "Should have underperforming signals"
        );
    }

    #[test]
    fn test_outperforming_signals() {
        let dual = create_outperforming_pair(30);
        let rs = RelativeStrength::new()
            .with_roc_period(Some(5))
            .with_roc_threshold(0.0);
        let signals = rs.signals(&dual);

        // After warmup, should have outperforming signals
        let outperforming_count = signals
            .iter()
            .skip(10)
            .filter(|s| **s == RelativeStrengthSignal::Outperforming)
            .count();

        assert!(
            outperforming_count > 0,
            "Should have outperforming signals"
        );
    }

    #[test]
    fn test_ratio_series() {
        let dual = create_stable_pair(20);
        let rs = RelativeStrength::new();
        let ratios = rs.ratio_series(&dual);

        assert_eq!(ratios.len(), 20);
        // All ratios should be ~2.0
        for r in &ratios {
            assert!((r - 2.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_normalized_series() {
        let dual = create_outperforming_pair(20);
        let rs = RelativeStrength::new();
        let normalized = rs.normalized_series(&dual);

        assert_eq!(normalized.len(), 20);
        // First value should be 100
        assert!((normalized[0] - 100.0).abs() < 0.01);
        // Last value should be > 100
        assert!(normalized.last().unwrap() > &100.0);
    }

    #[test]
    fn test_zero_benchmark_handling() {
        // Test with zero values in benchmark
        let asset = vec![100.0, 110.0, 120.0];
        let benchmark = vec![50.0, 0.0, 60.0];
        let dual = DualSeries::new(asset, benchmark);

        let rs = RelativeStrength::new();
        let outputs = rs.calculate(&dual);

        assert!(outputs[0].is_some()); // Valid
        assert!(outputs[1].is_none()); // Zero benchmark -> None
        assert!(outputs[2].is_some()); // Valid
    }

    #[test]
    fn test_technical_indicator_impl() {
        let asset: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let benchmark: Vec<f64> = (0..50).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let rs = RelativeStrength::new()
            .with_normalize(true)
            .with_roc_period(Some(10))
            .with_benchmark(&benchmark);

        let data = OHLCVSeries::from_close(asset);
        let result = rs.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_missing_benchmark_error() {
        let rs = RelativeStrength::new();
        let data = OHLCVSeries::from_close(vec![100.0; 50]);
        let result = rs.compute(&data);

        assert!(result.is_err());
    }

    #[test]
    fn test_length_mismatch_error() {
        let rs = RelativeStrength::new().with_benchmark(&[100.0; 30]);
        let data = OHLCVSeries::from_close(vec![100.0; 50]);
        let result = rs.compute(&data);

        assert!(result.is_err());
    }

    #[test]
    fn test_signal_indicator_impl() {
        let asset: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 1.5).collect();
        let benchmark: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5).collect();

        let rs = RelativeStrength::new()
            .with_roc_period(Some(5))
            .with_roc_threshold(0.0)
            .with_benchmark(&benchmark);

        let data = OHLCVSeries::from_close(asset);
        let signal = rs.signal(&data);

        assert!(signal.is_ok());
        // Should be bullish (outperforming)
        assert_eq!(signal.unwrap(), IndicatorSignal::Bullish);
    }

    #[test]
    fn test_default_impl() {
        let rs = RelativeStrength::default();
        assert!(!rs.normalize);
        assert!(rs.roc_period.is_none());
        assert!(rs.benchmark_series.is_empty());
    }

    #[test]
    fn test_roc_series() {
        let dual = create_outperforming_pair(20);
        let rs = RelativeStrength::new().with_roc_period(Some(5));
        let roc = rs.roc_series(&dual);

        assert_eq!(roc.len(), 20);
        // First 5 should be NaN
        for r in &roc[..5] {
            assert!(r.is_nan());
        }
        // Rest should be positive
        for r in &roc[10..] {
            assert!(*r > 0.0);
        }
    }

    #[test]
    fn test_roc_threshold() {
        let dual = create_stable_pair(30);
        let rs = RelativeStrength::new()
            .with_roc_period(Some(5))
            .with_roc_threshold(5.0); // High threshold
        let signals = rs.signals(&dual);

        // With stable ratio and high threshold, should be mostly neutral
        let neutral_count = signals
            .iter()
            .filter(|s| **s == RelativeStrengthSignal::Neutral)
            .count();

        assert!(
            neutral_count > signals.len() / 2,
            "High threshold should produce more neutral signals"
        );
    }
}
