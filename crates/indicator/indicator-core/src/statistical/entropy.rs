//! Market Entropy implementation.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Entropy calculation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyMethod {
    /// Shannon entropy of discretized returns
    Shannon,
    /// Approximate entropy (regularity measure)
    Approximate,
    /// Sample entropy (improved ApEn)
    Sample,
}

/// Market Entropy.
///
/// Measures the randomness/predictability of price movements.
///
/// Shannon Entropy interpretation:
/// - Higher entropy: More random, less predictable
/// - Lower entropy: More orderly, potentially more predictable
///
/// Approximate/Sample Entropy:
/// - Lower values: More regular/predictable patterns
/// - Higher values: More complex/random behavior
///
/// Useful for:
/// - Detecting market regimes (orderly vs chaotic)
/// - Identifying periods of high/low predictability
/// - Risk assessment based on market randomness
#[derive(Debug, Clone)]
pub struct MarketEntropy {
    period: usize,
    method: EntropyMethod,
    /// Number of bins for Shannon entropy histogram
    num_bins: usize,
    /// Embedding dimension for ApEn/SampEn
    embedding_dim: usize,
    /// Tolerance for ApEn/SampEn (as multiple of std dev)
    tolerance_mult: f64,
}

impl MarketEntropy {
    /// Create a new Market Entropy indicator with Shannon entropy.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            method: EntropyMethod::Shannon,
            num_bins: 10,
            embedding_dim: 2,
            tolerance_mult: 0.2,
        }
    }

    /// Create with Shannon entropy method.
    pub fn shannon(period: usize, num_bins: usize) -> Self {
        Self {
            period,
            method: EntropyMethod::Shannon,
            num_bins: num_bins.max(2),
            embedding_dim: 2,
            tolerance_mult: 0.2,
        }
    }

    /// Create with Approximate entropy method.
    pub fn approximate(period: usize, embedding_dim: usize, tolerance_mult: f64) -> Self {
        Self {
            period,
            method: EntropyMethod::Approximate,
            num_bins: 10,
            embedding_dim: embedding_dim.max(1),
            tolerance_mult: tolerance_mult.max(0.01),
        }
    }

    /// Create with Sample entropy method.
    pub fn sample(period: usize, embedding_dim: usize, tolerance_mult: f64) -> Self {
        Self {
            period,
            method: EntropyMethod::Sample,
            num_bins: 10,
            embedding_dim: embedding_dim.max(1),
            tolerance_mult: tolerance_mult.max(0.01),
        }
    }

    /// Calculate Shannon entropy of returns.
    fn calculate_shannon(&self, window: &[f64]) -> f64 {
        let n = window.len();
        if n < 2 {
            return f64::NAN;
        }

        // Calculate returns
        let returns: Vec<f64> = window
            .windows(2)
            .map(|w| {
                if w[0].abs() < 1e-10 {
                    0.0
                } else {
                    (w[1] - w[0]) / w[0]
                }
            })
            .collect();

        if returns.is_empty() {
            return f64::NAN;
        }

        // Find min and max returns
        let (min_ret, max_ret) = returns.iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
                (min.min(x), max.max(x))
            });

        let range = max_ret - min_ret;
        if range.abs() < 1e-10 {
            return 0.0; // No variation = zero entropy
        }

        // Create histogram bins
        let bin_width = range / self.num_bins as f64;
        let mut bins = vec![0usize; self.num_bins];

        for &ret in &returns {
            let bin_idx = ((ret - min_ret) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(self.num_bins - 1); // Clamp to last bin
            bins[bin_idx] += 1;
        }

        // Calculate Shannon entropy: H = -sum(p * log2(p))
        let total = returns.len() as f64;
        let mut entropy = 0.0;

        for &count in &bins {
            if count > 0 {
                let p = count as f64 / total;
                entropy -= p * p.log2();
            }
        }

        // Normalize by max entropy (log2(num_bins))
        let max_entropy = (self.num_bins as f64).log2();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            entropy
        }
    }

    /// Calculate Approximate Entropy (ApEn).
    fn calculate_approximate(&self, window: &[f64]) -> f64 {
        let n = window.len();
        let m = self.embedding_dim;

        if n < m + 2 {
            return f64::NAN;
        }

        // Calculate standard deviation for tolerance
        let mean: f64 = window.iter().sum::<f64>() / n as f64;
        let std_dev: f64 = (window.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / n as f64)
            .sqrt();

        if std_dev < 1e-10 {
            return 0.0; // No variation
        }

        let tolerance = self.tolerance_mult * std_dev;

        // Count similar patterns
        let phi_m = self.count_patterns(window, m, tolerance);
        let phi_m1 = self.count_patterns(window, m + 1, tolerance);

        if phi_m.is_nan() || phi_m1.is_nan() {
            return f64::NAN;
        }

        phi_m - phi_m1
    }

    /// Calculate Sample Entropy (SampEn).
    fn calculate_sample(&self, window: &[f64]) -> f64 {
        let n = window.len();
        let m = self.embedding_dim;

        if n < m + 2 {
            return f64::NAN;
        }

        // Calculate standard deviation for tolerance
        let mean: f64 = window.iter().sum::<f64>() / n as f64;
        let std_dev: f64 = (window.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / n as f64)
            .sqrt();

        if std_dev < 1e-10 {
            return 0.0;
        }

        let tolerance = self.tolerance_mult * std_dev;

        // Count matches for m-length and (m+1)-length templates
        let (b, a) = self.count_matches_sample(window, m, tolerance);

        if b == 0 || a == 0 {
            return f64::NAN;
        }

        -((a as f64) / (b as f64)).ln()
    }

    /// Count similar patterns for ApEn calculation.
    fn count_patterns(&self, data: &[f64], m: usize, tolerance: f64) -> f64 {
        let n = data.len();
        if n < m {
            return f64::NAN;
        }

        let num_patterns = n - m + 1;
        let mut total_log = 0.0;

        for i in 0..num_patterns {
            let mut count = 0;
            for j in 0..num_patterns {
                // Check if patterns are similar
                let similar = (0..m).all(|k| {
                    (data[i + k] - data[j + k]).abs() <= tolerance
                });
                if similar {
                    count += 1;
                }
            }
            if count > 0 {
                total_log += (count as f64 / num_patterns as f64).ln();
            }
        }

        total_log / num_patterns as f64
    }

    /// Count matches for SampEn calculation.
    fn count_matches_sample(&self, data: &[f64], m: usize, tolerance: f64) -> (usize, usize) {
        let n = data.len();
        if n < m + 1 {
            return (0, 0);
        }

        let mut b_count = 0; // Matches for m-length
        let mut a_count = 0; // Matches for (m+1)-length

        let num_patterns = n - m;

        for i in 0..num_patterns {
            for j in (i + 1)..num_patterns {
                // Check m-length match
                let m_match = (0..m).all(|k| {
                    (data[i + k] - data[j + k]).abs() <= tolerance
                });

                if m_match {
                    b_count += 1;

                    // Check if (m+1)-length also matches
                    if i + m < n && j + m < n {
                        if (data[i + m] - data[j + m]).abs() <= tolerance {
                            a_count += 1;
                        }
                    }
                }
            }
        }

        (b_count, a_count)
    }

    /// Calculate entropy for a window.
    fn calculate_entropy(&self, window: &[f64]) -> f64 {
        match self.method {
            EntropyMethod::Shannon => self.calculate_shannon(window),
            EntropyMethod::Approximate => self.calculate_approximate(window),
            EntropyMethod::Sample => self.calculate_sample(window),
        }
    }

    /// Calculate entropy values.
    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.period || self.period < 4 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];
            let entropy = self.calculate_entropy(window);
            result.push(entropy);
        }

        result
    }
}

impl TechnicalIndicator for MarketEntropy {
    fn name(&self) -> &str {
        "MarketEntropy"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.period
    }
}

impl SignalIndicator for MarketEntropy {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // For Shannon entropy (normalized 0-1):
        // Low entropy (< 0.4): Market is orderly, trend-following may work
        // High entropy (> 0.7): Market is random, be cautious
        //
        // For ApEn/SampEn:
        // Low values: Regular patterns, may be predictable
        // High values: Complex/random, harder to predict
        match self.method {
            EntropyMethod::Shannon => {
                if last < 0.4 {
                    // Low entropy - orderly market, follow trend
                    let n = data.close.len();
                    if n >= 2 {
                        if data.close[n - 1] > data.close[n - 2] {
                            Ok(IndicatorSignal::Bullish)
                        } else if data.close[n - 1] < data.close[n - 2] {
                            Ok(IndicatorSignal::Bearish)
                        } else {
                            Ok(IndicatorSignal::Neutral)
                        }
                    } else {
                        Ok(IndicatorSignal::Neutral)
                    }
                } else if last > 0.7 {
                    // High entropy - random, stay neutral
                    Ok(IndicatorSignal::Neutral)
                } else {
                    Ok(IndicatorSignal::Neutral)
                }
            }
            _ => {
                // ApEn/SampEn: low values mean more regular
                if last < 0.3 {
                    // Regular patterns detected
                    let n = data.close.len();
                    if n >= 2 {
                        if data.close[n - 1] > data.close[n - 2] {
                            Ok(IndicatorSignal::Bullish)
                        } else if data.close[n - 1] < data.close[n - 2] {
                            Ok(IndicatorSignal::Bearish)
                        } else {
                            Ok(IndicatorSignal::Neutral)
                        }
                    } else {
                        Ok(IndicatorSignal::Neutral)
                    }
                } else {
                    Ok(IndicatorSignal::Neutral)
                }
            }
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);

        let signals = match self.method {
            EntropyMethod::Shannon => {
                values
                    .iter()
                    .enumerate()
                    .map(|(i, &entropy)| {
                        if entropy.is_nan() || i == 0 {
                            IndicatorSignal::Neutral
                        } else if entropy < 0.4 {
                            if data.close[i] > data.close[i - 1] {
                                IndicatorSignal::Bullish
                            } else if data.close[i] < data.close[i - 1] {
                                IndicatorSignal::Bearish
                            } else {
                                IndicatorSignal::Neutral
                            }
                        } else {
                            IndicatorSignal::Neutral
                        }
                    })
                    .collect()
            }
            _ => {
                values
                    .iter()
                    .enumerate()
                    .map(|(i, &entropy)| {
                        if entropy.is_nan() || i == 0 {
                            IndicatorSignal::Neutral
                        } else if entropy < 0.3 {
                            if data.close[i] > data.close[i - 1] {
                                IndicatorSignal::Bullish
                            } else if data.close[i] < data.close[i - 1] {
                                IndicatorSignal::Bearish
                            } else {
                                IndicatorSignal::Neutral
                            }
                        } else {
                            IndicatorSignal::Neutral
                        }
                    })
                    .collect()
            }
        };

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy_uniform() {
        let entropy = MarketEntropy::shannon(20, 10);
        // Uniform distribution should have high entropy
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i % 10) as f64).collect();
        let result = entropy.calculate(&data);

        for i in 19..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 0.0 && result[i] <= 1.0);
        }
    }

    #[test]
    fn test_shannon_entropy_constant() {
        let entropy = MarketEntropy::shannon(20, 10);
        // Constant data should have zero entropy
        let data = vec![100.0; 50];
        let result = entropy.calculate(&data);

        for i in 19..result.len() {
            assert!((result[i] - 0.0).abs() < 0.01, "Expected ~0 entropy for constant data");
        }
    }

    #[test]
    fn test_shannon_entropy_trending() {
        let entropy = MarketEntropy::shannon(20, 10);
        // Trending data should have moderate entropy
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.5).collect();
        let result = entropy.calculate(&data);

        for i in 19..result.len() {
            assert!(!result[i].is_nan());
            assert!(result[i] >= 0.0 && result[i] <= 1.0);
        }
    }

    #[test]
    fn test_approximate_entropy() {
        let entropy = MarketEntropy::approximate(30, 2, 0.2);
        let data: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 5.0)
            .collect();
        let result = entropy.calculate(&data);

        for i in 29..result.len() {
            if !result[i].is_nan() {
                assert!(result[i].is_finite());
            }
        }
    }

    #[test]
    fn test_sample_entropy() {
        let entropy = MarketEntropy::sample(30, 2, 0.2);
        let data: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 5.0)
            .collect();
        let result = entropy.calculate(&data);

        for i in 29..result.len() {
            if !result[i].is_nan() {
                assert!(result[i].is_finite());
            }
        }
    }

    #[test]
    fn test_entropy_insufficient_data() {
        let entropy = MarketEntropy::new(20);
        let data = vec![100.0; 10]; // Less than period
        let result = entropy.calculate(&data);

        for val in &result {
            assert!(val.is_nan());
        }
    }

    #[test]
    fn test_entropy_oscillating() {
        let entropy = MarketEntropy::shannon(20, 5);
        // Oscillating data - should have some structure
        let data: Vec<f64> = (0..50)
            .map(|i| 100.0 + if i % 2 == 0 { 5.0 } else { -5.0 })
            .collect();
        let result = entropy.calculate(&data);

        for i in 19..result.len() {
            assert!(!result[i].is_nan());
        }
    }

    #[test]
    fn test_entropy_regular_pattern() {
        let entropy = MarketEntropy::approximate(30, 2, 0.15);
        // Very regular pattern should have low ApEn
        let data: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i % 5) as f64)
            .collect();
        let result = entropy.calculate(&data);

        // Check we get values
        let valid_values: Vec<f64> = result.iter()
            .filter(|x| !x.is_nan())
            .copied()
            .collect();

        assert!(!valid_values.is_empty(), "Should have valid values");
    }
}
