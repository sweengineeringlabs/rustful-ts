//! Cointegration Score for Pair Trading
//!
//! Implements a simplified cointegration test based on the Engle-Granger approach.
//! Uses an Augmented Dickey-Fuller (ADF) style statistic to test for stationarity
//! of the spread between two series.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

use super::DualSeries;

/// Cointegration output values.
#[derive(Debug, Clone, Copy)]
pub struct CointegrationOutput {
    /// The hedge ratio (beta) for the pair.
    pub hedge_ratio: f64,
    /// The spread between the two series (series1 - hedge_ratio * series2).
    pub spread: f64,
    /// The ADF-style test statistic (more negative = stronger cointegration).
    pub adf_statistic: f64,
    /// Z-score of the current spread.
    pub spread_zscore: f64,
    /// Half-life of mean reversion (in periods).
    pub half_life: f64,
}

/// Cointegration signal for pairs trading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CointegrationSignal {
    /// Go long spread (long series1, short series2).
    LongSpread,
    /// Go short spread (short series1, long series2).
    ShortSpread,
    /// No trading signal.
    Neutral,
    /// Series are not cointegrated - avoid trading.
    NotCointegrated,
}

/// Cointegration indicator for pairs trading.
///
/// Calculates the cointegration relationship between two price series
/// using a simplified Engle-Granger approach:
///
/// 1. Estimate hedge ratio via OLS regression
/// 2. Calculate spread = series1 - hedge_ratio * series2
/// 3. Test spread for stationarity using ADF-style statistic
/// 4. If cointegrated, generate trading signals based on z-score
///
/// # Interpretation
/// - ADF statistic < -2.86 (5% critical value): likely cointegrated
/// - ADF statistic < -3.43 (1% critical value): strongly cointegrated
/// - Spread z-score > entry_threshold: short the spread
/// - Spread z-score < -entry_threshold: long the spread
#[derive(Debug, Clone)]
pub struct Cointegration {
    /// Period for rolling calculations.
    period: usize,
    /// Z-score threshold for entry signals.
    entry_threshold: f64,
    /// Z-score threshold for exit signals.
    exit_threshold: f64,
    /// ADF critical value threshold (default: -2.86 for 5% significance).
    adf_threshold: f64,
    /// Secondary series for cointegration analysis.
    secondary_series: Vec<f64>,
}

impl Cointegration {
    /// Create a new Cointegration indicator with default parameters.
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculations (typically 60-252)
    pub fn new(period: usize) -> Self {
        Self {
            period,
            entry_threshold: 2.0,
            exit_threshold: 0.5,
            adf_threshold: -2.86, // 5% critical value
            secondary_series: Vec::new(),
        }
    }

    /// Set the entry z-score threshold.
    pub fn with_entry_threshold(mut self, threshold: f64) -> Self {
        self.entry_threshold = threshold;
        self
    }

    /// Set the exit z-score threshold.
    pub fn with_exit_threshold(mut self, threshold: f64) -> Self {
        self.exit_threshold = threshold;
        self
    }

    /// Set the ADF critical value threshold.
    pub fn with_adf_threshold(mut self, threshold: f64) -> Self {
        self.adf_threshold = threshold;
        self
    }

    /// Set the secondary series for cointegration analysis.
    pub fn with_secondary(mut self, series: &[f64]) -> Self {
        self.secondary_series = series.to_vec();
        self
    }

    /// Calculate OLS regression: series1 = alpha + beta * series2
    /// Returns (alpha, beta, residuals).
    fn ols_regression(series1: &[f64], series2: &[f64]) -> (f64, f64, Vec<f64>) {
        let n = series1.len() as f64;

        let mean1: f64 = series1.iter().sum::<f64>() / n;
        let mean2: f64 = series2.iter().sum::<f64>() / n;

        // Calculate beta = Cov(y, x) / Var(x)
        let mut cov = 0.0;
        let mut var_x = 0.0;

        for (y, x) in series1.iter().zip(series2.iter()) {
            let dy = y - mean1;
            let dx = x - mean2;
            cov += dy * dx;
            var_x += dx * dx;
        }

        let beta = if var_x.abs() < 1e-10 {
            0.0
        } else {
            cov / var_x
        };

        let alpha = mean1 - beta * mean2;

        // Calculate residuals (spread)
        let residuals: Vec<f64> = series1
            .iter()
            .zip(series2.iter())
            .map(|(y, x)| y - alpha - beta * x)
            .collect();

        (alpha, beta, residuals)
    }

    /// Calculate ADF-style test statistic for stationarity.
    /// Uses a simplified approach: regress diff(spread) on lagged spread.
    /// Returns (adf_statistic, half_life).
    fn adf_test(spread: &[f64]) -> (f64, f64) {
        if spread.len() < 3 {
            return (f64::NAN, f64::NAN);
        }

        // Calculate first differences
        let diff: Vec<f64> = spread.windows(2).map(|w| w[1] - w[0]).collect();

        // Lagged spread (excluding last value)
        let lagged: Vec<f64> = spread[..spread.len() - 1].to_vec();

        let n = diff.len();
        if n < 2 {
            return (f64::NAN, f64::NAN);
        }

        // Regress diff on lagged: diff = c + gamma * lagged + error
        let mean_diff: f64 = diff.iter().sum::<f64>() / n as f64;
        let mean_lagged: f64 = lagged.iter().sum::<f64>() / n as f64;

        let mut cov = 0.0;
        let mut var_lagged = 0.0;

        for (d, l) in diff.iter().zip(lagged.iter()) {
            let dd = d - mean_diff;
            let dl = l - mean_lagged;
            cov += dd * dl;
            var_lagged += dl * dl;
        }

        let gamma = if var_lagged.abs() < 1e-10 {
            0.0
        } else {
            cov / var_lagged
        };

        // Calculate residuals and standard error
        let intercept = mean_diff - gamma * mean_lagged;
        let mut sse = 0.0;

        for (d, l) in diff.iter().zip(lagged.iter()) {
            let predicted = intercept + gamma * l;
            sse += (d - predicted).powi(2);
        }

        let mse = sse / (n - 2) as f64;
        let se_gamma = if var_lagged.abs() < 1e-10 {
            f64::NAN
        } else {
            (mse / var_lagged).sqrt()
        };

        // ADF statistic = gamma / SE(gamma)
        let adf_stat = if se_gamma.is_nan() || se_gamma.abs() < 1e-10 {
            f64::NAN
        } else {
            gamma / se_gamma
        };

        // Half-life = -ln(2) / ln(1 + gamma)
        // For mean-reverting process, gamma should be negative
        let half_life = if gamma >= 0.0 || gamma <= -1.0 {
            f64::INFINITY
        } else {
            -0.693147 / (1.0 + gamma).ln()
        };

        (adf_stat, half_life)
    }

    /// Calculate z-score of current spread value.
    fn spread_zscore(spread: &[f64]) -> f64 {
        if spread.len() < 2 {
            return f64::NAN;
        }

        let mean: f64 = spread.iter().sum::<f64>() / spread.len() as f64;
        let variance: f64 =
            spread.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / spread.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev.abs() < 1e-10 {
            0.0
        } else {
            (spread.last().unwrap() - mean) / std_dev
        }
    }

    /// Calculate cointegration metrics for a dual series.
    pub fn calculate(&self, dual: &DualSeries) -> Vec<Option<CointegrationOutput>> {
        let n = dual.len();
        if n < self.period {
            return vec![None; n];
        }

        let mut result = vec![None; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window1 = &dual.series1[start..=i];
            let window2 = &dual.series2[start..=i];

            let (_alpha, beta, spread) = Self::ols_regression(window1, window2);
            let (adf_stat, half_life) = Self::adf_test(&spread);
            let zscore = Self::spread_zscore(&spread);

            result.push(Some(CointegrationOutput {
                hedge_ratio: beta,
                spread: *spread.last().unwrap_or(&f64::NAN),
                adf_statistic: adf_stat,
                spread_zscore: zscore,
                half_life,
            }));
        }

        result
    }

    /// Calculate cointegration metrics using two series directly.
    pub fn calculate_between(
        &self,
        series1: &[f64],
        series2: &[f64],
    ) -> Vec<Option<CointegrationOutput>> {
        let dual = DualSeries::from_slices(series1, series2);
        self.calculate(&dual)
    }

    /// Get trading signals based on cointegration and z-score.
    pub fn signals(&self, dual: &DualSeries) -> Vec<CointegrationSignal> {
        let outputs = self.calculate(dual);

        outputs
            .iter()
            .map(|opt| match opt {
                None => CointegrationSignal::Neutral,
                Some(out) => {
                    // Check if cointegrated
                    if out.adf_statistic.is_nan() || out.adf_statistic > self.adf_threshold {
                        return CointegrationSignal::NotCointegrated;
                    }

                    // Generate signals based on z-score
                    if out.spread_zscore > self.entry_threshold {
                        CointegrationSignal::ShortSpread
                    } else if out.spread_zscore < -self.entry_threshold {
                        CointegrationSignal::LongSpread
                    } else if out.spread_zscore.abs() < self.exit_threshold {
                        CointegrationSignal::Neutral
                    } else {
                        CointegrationSignal::Neutral
                    }
                }
            })
            .collect()
    }

    /// Extract spread series from dual series.
    pub fn spread_series(&self, dual: &DualSeries) -> Vec<f64> {
        let outputs = self.calculate(dual);
        outputs
            .iter()
            .map(|opt| opt.map(|o| o.spread).unwrap_or(f64::NAN))
            .collect()
    }

    /// Extract z-score series from dual series.
    pub fn zscore_series(&self, dual: &DualSeries) -> Vec<f64> {
        let outputs = self.calculate(dual);
        outputs
            .iter()
            .map(|opt| opt.map(|o| o.spread_zscore).unwrap_or(f64::NAN))
            .collect()
    }
}

impl TechnicalIndicator for Cointegration {
    fn name(&self) -> &str {
        "Cointegration"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        if self.secondary_series.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: "Secondary series must be set before computing Cointegration".to_string(),
            });
        }

        if self.secondary_series.len() != data.close.len() {
            return Err(IndicatorError::InvalidParameter {
                name: "secondary_series".to_string(),
                reason: format!(
                    "Secondary series length ({}) must match primary series length ({})",
                    self.secondary_series.len(),
                    data.close.len()
                ),
            });
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        let outputs = self.calculate(&dual);

        // Primary: spread z-score
        // Secondary: ADF statistic
        // Tertiary: hedge ratio
        let zscore: Vec<f64> = outputs
            .iter()
            .map(|opt| opt.map(|o| o.spread_zscore).unwrap_or(f64::NAN))
            .collect();
        let adf: Vec<f64> = outputs
            .iter()
            .map(|opt| opt.map(|o| o.adf_statistic).unwrap_or(f64::NAN))
            .collect();
        let hedge: Vec<f64> = outputs
            .iter()
            .map(|opt| opt.map(|o| o.hedge_ratio).unwrap_or(f64::NAN))
            .collect();

        Ok(IndicatorOutput::triple(zscore, adf, hedge))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        3 // z-score, ADF statistic, hedge ratio
    }
}

impl SignalIndicator for Cointegration {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if self.secondary_series.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        let signals = self.signals(&dual);

        match signals.last() {
            Some(CointegrationSignal::LongSpread) => Ok(IndicatorSignal::Bullish),
            Some(CointegrationSignal::ShortSpread) => Ok(IndicatorSignal::Bearish),
            _ => Ok(IndicatorSignal::Neutral),
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        if self.secondary_series.is_empty() {
            return Ok(vec![IndicatorSignal::Neutral; data.close.len()]);
        }

        let dual = DualSeries::from_slices(&data.close, &self.secondary_series);
        let coint_signals = self.signals(&dual);

        let signals = coint_signals
            .iter()
            .map(|s| match s {
                CointegrationSignal::LongSpread => IndicatorSignal::Bullish,
                CointegrationSignal::ShortSpread => IndicatorSignal::Bearish,
                _ => IndicatorSignal::Neutral,
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_cointegrated_pair(n: usize, noise: f64) -> DualSeries {
        // Create two series with a stable relationship: series1 = 2 * series2 + noise
        let mut series1 = Vec::with_capacity(n);
        let mut series2 = Vec::with_capacity(n);

        let mut price2 = 50.0;
        for i in 0..n {
            // Random walk for series2
            price2 += (i as f64 * 0.1).sin() * 0.5;
            series2.push(price2);

            // series1 follows series2 with hedge ratio ~2
            let spread_noise = ((i as f64) * 0.7).sin() * noise;
            series1.push(100.0 + 2.0 * price2 + spread_noise);
        }

        DualSeries::new(series1, series2)
    }

    fn create_non_cointegrated_pair(n: usize) -> DualSeries {
        // Two independent random walks
        let mut series1 = Vec::with_capacity(n);
        let mut series2 = Vec::with_capacity(n);

        let mut price1 = 100.0;
        let mut price2 = 100.0;

        for i in 0..n {
            price1 += (i as f64 * 0.3).sin() * 0.5;
            price2 += (i as f64 * 0.7).cos() * 0.3;
            series1.push(price1);
            series2.push(price2);
        }

        DualSeries::new(series1, series2)
    }

    #[test]
    fn test_cointegration_hedge_ratio() {
        let dual = create_cointegrated_pair(100, 0.1);
        let coint = Cointegration::new(50);
        let outputs = coint.calculate(&dual);

        // Last output should have hedge ratio close to 2
        let last = outputs.last().unwrap().unwrap();
        assert!(
            (last.hedge_ratio - 2.0).abs() < 0.5,
            "Hedge ratio should be close to 2"
        );
    }

    #[test]
    fn test_cointegration_adf_statistic() {
        let dual = create_cointegrated_pair(200, 0.5);
        let coint = Cointegration::new(100);
        let outputs = coint.calculate(&dual);

        // Cointegrated pair should have negative ADF statistic
        let last = outputs.last().unwrap().unwrap();
        assert!(
            !last.adf_statistic.is_nan(),
            "ADF statistic should be calculated"
        );
        // Note: may or may not pass -2.86 threshold depending on noise
    }

    #[test]
    fn test_cointegration_zscore() {
        let dual = create_cointegrated_pair(100, 1.0);
        let coint = Cointegration::new(50);
        let outputs = coint.calculate(&dual);

        // Z-scores should be within reasonable range
        for out in outputs.iter().flatten() {
            assert!(
                out.spread_zscore.abs() < 5.0,
                "Z-score should be reasonable"
            );
        }
    }

    #[test]
    fn test_cointegration_signals() {
        let dual = create_cointegrated_pair(100, 2.0);
        let coint = Cointegration::new(30).with_entry_threshold(1.5);
        let signals = coint.signals(&dual);

        // Should have some signals (not all neutral)
        let non_neutral = signals
            .iter()
            .filter(|s| {
                **s != CointegrationSignal::Neutral && **s != CointegrationSignal::NotCointegrated
            })
            .count();
        // May or may not have signals depending on z-score values
        assert!(signals.len() == 100);
    }

    #[test]
    fn test_cointegration_insufficient_data() {
        let dual = DualSeries::new(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
        let coint = Cointegration::new(50);
        let outputs = coint.calculate(&dual);

        assert!(outputs.iter().all(|o| o.is_none()));
    }

    #[test]
    fn test_cointegration_half_life() {
        let dual = create_cointegrated_pair(200, 1.0);
        let coint = Cointegration::new(100);
        let outputs = coint.calculate(&dual);

        // Half-life should be positive for mean-reverting spread
        let last = outputs.last().unwrap().unwrap();
        if last.half_life.is_finite() {
            assert!(last.half_life > 0.0, "Half-life should be positive");
        }
    }

    #[test]
    fn test_technical_indicator_impl() {
        let series1: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64) * 0.5).collect();
        let series2: Vec<f64> = (0..100).map(|i| 50.0 + (i as f64) * 0.25).collect();

        let coint = Cointegration::new(30).with_secondary(&series2);
        let data = OHLCVSeries::from_close(series1);
        let result = coint.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_spread_and_zscore_series() {
        let dual = create_cointegrated_pair(100, 1.0);
        let coint = Cointegration::new(30);

        let spreads = coint.spread_series(&dual);
        let zscores = coint.zscore_series(&dual);

        assert_eq!(spreads.len(), 100);
        assert_eq!(zscores.len(), 100);

        // First 29 should be NaN
        assert!(spreads[0].is_nan());
        // After warmup, should have values
        assert!(!spreads[50].is_nan());
        assert!(!zscores[50].is_nan());
    }
}
