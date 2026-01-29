//! Jump Detection (IND-408) - Price jump identifier
//!
//! Detects sudden, large price movements (jumps) in financial time series.
//! Distinguishes between continuous price evolution and discrete jumps
//! using statistical tests based on realized variance measures.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for Jump Detection indicator
#[derive(Debug, Clone)]
pub struct JumpDetectionConfig {
    /// Rolling window period for variance estimation
    pub period: usize,
    /// Significance level for jump detection (e.g., 0.99)
    pub confidence_level: f64,
    /// Multiplier for threshold (number of std devs)
    pub threshold_multiplier: f64,
}

impl Default for JumpDetectionConfig {
    fn default() -> Self {
        Self {
            period: 20,
            confidence_level: 0.99,
            threshold_multiplier: 3.0,
        }
    }
}

/// Jump Detection - Price jump identifier (IND-408)
///
/// Identifies discrete jumps in price series by comparing realized returns
/// to a threshold based on local volatility estimation. Uses bipower variation
/// to robustly estimate continuous volatility component.
///
/// # Method
/// 1. Calculate bipower variation (robust to jumps)
/// 2. Calculate realized variance (includes jumps)
/// 3. Jump = return exceeds threshold based on bipower volatility
///
/// # Formula
/// Bipower Variation: BV = (pi/2) * sum(|r_t| * |r_{t-1}|)
/// Jump Threshold: threshold = multiplier * sqrt(BV / period)
/// Jump Signal: |return| > threshold
///
/// # Output
/// - Jump intensity: 0 = no jump, positive = jump magnitude / threshold
/// - Values > 1 indicate increasingly significant jumps
///
/// # Example
/// ```ignore
/// let jd = JumpDetection::new(20, 0.99, 3.0).unwrap();
/// let jumps = jd.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct JumpDetection {
    config: JumpDetectionConfig,
}

impl JumpDetection {
    /// Create a new Jump Detection indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window for variance estimation (minimum 10)
    /// * `confidence_level` - Confidence level for detection (0.9 to 0.999)
    /// * `threshold_multiplier` - Number of standard deviations for threshold
    pub fn new(period: usize, confidence_level: f64, threshold_multiplier: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if confidence_level < 0.9 || confidence_level > 0.999 {
            return Err(IndicatorError::InvalidParameter {
                name: "confidence_level".to_string(),
                reason: "must be between 0.9 and 0.999".to_string(),
            });
        }
        if threshold_multiplier < 1.5 || threshold_multiplier > 10.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold_multiplier".to_string(),
                reason: "must be between 1.5 and 10.0".to_string(),
            });
        }
        Ok(Self {
            config: JumpDetectionConfig {
                period,
                confidence_level,
                threshold_multiplier,
            },
        })
    }

    /// Create with default configuration
    pub fn default_indicator() -> Self {
        Self {
            config: JumpDetectionConfig::default(),
        }
    }

    /// Calculate log returns
    fn calculate_log_returns(prices: &[f64]) -> Vec<f64> {
        if prices.len() < 2 {
            return Vec::new();
        }
        prices
            .windows(2)
            .map(|w| {
                if w[0] > 1e-10 {
                    (w[1] / w[0]).ln()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Calculate bipower variation (robust to jumps)
    /// BV = (pi/2) * sum(|r_t| * |r_{t-1}|) / (n-1)
    fn bipower_variation(returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let mu1 = (2.0_f64 / std::f64::consts::PI).sqrt();
        let bv: f64 = returns
            .windows(2)
            .map(|w| w[0].abs() * w[1].abs())
            .sum();

        bv / (mu1 * mu1 * (returns.len() - 1) as f64)
    }

    /// Calculate realized variance (includes jumps)
    fn realized_variance(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        returns.iter().map(|r| r * r).sum::<f64>() / returns.len() as f64
    }

    /// Calculate jump intensity for each point
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period + 2 {
            return vec![0.0; n];
        }

        let log_returns = Self::calculate_log_returns(close);
        let mut result = vec![0.0; self.config.period + 1];

        for i in self.config.period..log_returns.len() {
            let start = i - self.config.period + 1;
            let window = &log_returns[start..=i];

            // Calculate bipower variation for the window
            let bv = Self::bipower_variation(window);
            let bv_std = bv.sqrt();

            // Jump threshold based on bipower volatility
            let threshold = self.config.threshold_multiplier * bv_std;

            // Current return
            let current_return = log_returns[i].abs();

            // Jump intensity: how many thresholds exceeded
            let jump_intensity = if threshold > 1e-10 {
                (current_return / threshold).max(0.0)
            } else {
                0.0
            };

            // Only report if above threshold (intensity > 1)
            result.push(if jump_intensity > 1.0 { jump_intensity } else { 0.0 });
        }

        // Pad to match original length
        while result.len() < n {
            result.push(0.0);
        }

        result
    }

    /// Calculate jump ratio: realized variance / bipower variation
    /// Values significantly > 1 indicate jump presence
    pub fn calculate_jump_ratio(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.config.period + 2 {
            return vec![1.0; n];
        }

        let log_returns = Self::calculate_log_returns(close);
        let mut result = vec![1.0; self.config.period + 1];

        for i in self.config.period..log_returns.len() {
            let start = i - self.config.period + 1;
            let window = &log_returns[start..=i];

            let rv = Self::realized_variance(window);
            let bv = Self::bipower_variation(window);

            let ratio = if bv > 1e-10 {
                rv / bv
            } else {
                1.0
            };

            result.push(ratio);
        }

        while result.len() < n {
            result.push(1.0);
        }

        result
    }

    /// Detect binary jump signals (1 = jump detected, 0 = no jump)
    pub fn detect_jumps(&self, close: &[f64]) -> Vec<f64> {
        self.calculate(close)
            .into_iter()
            .map(|intensity| if intensity > 0.0 { 1.0 } else { 0.0 })
            .collect()
    }
}

impl TechnicalIndicator for JumpDetection {
    fn name(&self) -> &str {
        "Jump Detection"
    }

    fn min_periods(&self) -> usize {
        self.config.period + 2
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_prices_with_jump(n: usize, jump_idx: usize, jump_size: f64) -> Vec<f64> {
        let mut prices = Vec::with_capacity(n);
        let mut price = 100.0;
        for i in 0..n {
            let normal_ret = (i as f64 * 0.1).sin() * 0.01;
            if i == jump_idx {
                price *= 1.0 + jump_size;
            } else {
                price *= 1.0 + normal_ret;
            }
            prices.push(price);
        }
        prices
    }

    #[test]
    fn test_jump_detection_creation() {
        let jd = JumpDetection::new(20, 0.99, 3.0);
        assert!(jd.is_ok());

        let jd = JumpDetection::new(5, 0.99, 3.0);
        assert!(jd.is_err());

        let jd = JumpDetection::new(20, 0.5, 3.0);
        assert!(jd.is_err());

        let jd = JumpDetection::new(20, 0.99, 0.5);
        assert!(jd.is_err());
    }

    #[test]
    fn test_jump_detection_no_jump() {
        let jd = JumpDetection::new(20, 0.99, 3.0).unwrap();
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 * (1.0 + (i as f64 * 0.1).sin() * 0.01))
            .collect();

        let result = jd.calculate(&prices);

        // Most values should be 0 (no jump)
        let jump_count = result.iter().filter(|&&v| v > 0.0).count();
        assert!(jump_count < 10); // Allow some false positives
    }

    #[test]
    fn test_jump_detection_with_jump() {
        let jd = JumpDetection::new(20, 0.99, 3.0).unwrap();
        let prices = generate_prices_with_jump(100, 50, 0.10); // 10% jump

        let result = jd.calculate(&prices);

        // Should detect jump around index 50
        let max_idx = result
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Jump should be detected near index 50 (accounting for period offset)
        assert!(max_idx >= 45 && max_idx <= 55);
        assert!(result[max_idx] > 1.0);
    }

    #[test]
    fn test_jump_ratio() {
        let jd = JumpDetection::new(20, 0.99, 3.0).unwrap();
        let prices_normal: Vec<f64> = (0..100)
            .map(|i| 100.0 * (1.0 + (i as f64 * 0.1).sin() * 0.01))
            .collect();
        let prices_jump = generate_prices_with_jump(100, 50, 0.15);

        let ratio_normal = jd.calculate_jump_ratio(&prices_normal);
        let ratio_jump = jd.calculate_jump_ratio(&prices_jump);

        // Jump series should have higher ratio around jump point
        let max_normal = ratio_normal.iter().cloned().fold(0.0_f64, f64::max);
        let max_jump = ratio_jump.iter().cloned().fold(0.0_f64, f64::max);

        assert!(max_jump > max_normal);
    }

    #[test]
    fn test_detect_jumps_binary() {
        let jd = JumpDetection::new(20, 0.99, 3.0).unwrap();
        let prices = generate_prices_with_jump(100, 50, 0.10);

        let binary = jd.detect_jumps(&prices);

        // All values should be 0 or 1
        assert!(binary.iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn test_jump_detection_indicator_trait() {
        let jd = JumpDetection::new(20, 0.99, 3.0).unwrap();

        assert_eq!(jd.name(), "Jump Detection");
        assert_eq!(jd.min_periods(), 22);
        assert_eq!(jd.output_features(), 1);
    }

    #[test]
    fn test_jump_detection_default() {
        let jd = JumpDetection::default_indicator();
        assert_eq!(jd.config.period, 20);
        assert!((jd.config.confidence_level - 0.99).abs() < 1e-10);
        assert!((jd.config.threshold_multiplier - 3.0).abs() < 1e-10);
    }
}
