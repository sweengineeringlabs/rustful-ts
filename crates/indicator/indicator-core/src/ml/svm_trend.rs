//! SVM Trend Classifier (IND-287)
//!
//! A proxy implementation of Support Vector Machine classification for trend detection.
//! Uses a margin-based approach with multiple features to classify market trends.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// SVM trend classification result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SVMTrendClass {
    /// Strong uptrend detected.
    StrongUptrend,
    /// Weak uptrend detected.
    WeakUptrend,
    /// No clear trend (neutral zone).
    Neutral,
    /// Weak downtrend detected.
    WeakDowntrend,
    /// Strong downtrend detected.
    StrongDowntrend,
}

impl SVMTrendClass {
    /// Convert to numeric value: -2 to +2.
    pub fn to_numeric(&self) -> f64 {
        match self {
            SVMTrendClass::StrongUptrend => 2.0,
            SVMTrendClass::WeakUptrend => 1.0,
            SVMTrendClass::Neutral => 0.0,
            SVMTrendClass::WeakDowntrend => -1.0,
            SVMTrendClass::StrongDowntrend => -2.0,
        }
    }

    /// Create from numeric value.
    pub fn from_numeric(value: f64) -> Self {
        if value >= 1.5 {
            SVMTrendClass::StrongUptrend
        } else if value >= 0.5 {
            SVMTrendClass::WeakUptrend
        } else if value > -0.5 {
            SVMTrendClass::Neutral
        } else if value > -1.5 {
            SVMTrendClass::WeakDowntrend
        } else {
            SVMTrendClass::StrongDowntrend
        }
    }
}

/// SVM Trend Classifier output.
#[derive(Debug, Clone)]
pub struct SVMTrendOutput {
    /// Classification result for each bar.
    pub classification: Vec<SVMTrendClass>,
    /// Numeric classification values (-2 to +2).
    pub class_value: Vec<f64>,
    /// Confidence score (0-100).
    pub confidence: Vec<f64>,
    /// Distance from decision boundary (margin).
    pub margin_distance: Vec<f64>,
}

/// SVM Trend Classifier configuration.
#[derive(Debug, Clone)]
pub struct SVMTrendClassifierConfig {
    /// Lookback period for feature extraction (default: 20).
    pub lookback: usize,
    /// Short EMA period for momentum (default: 5).
    pub ema_short: usize,
    /// Long EMA period for trend (default: 20).
    pub ema_long: usize,
    /// Margin threshold for strong classification (default: 0.6).
    pub strong_margin: f64,
    /// Margin threshold for weak classification (default: 0.3).
    pub weak_margin: f64,
    /// Feature weight for momentum (default: 0.3).
    pub momentum_weight: f64,
    /// Feature weight for volatility (default: 0.2).
    pub volatility_weight: f64,
    /// Feature weight for trend direction (default: 0.5).
    pub trend_weight: f64,
}

impl Default for SVMTrendClassifierConfig {
    fn default() -> Self {
        Self {
            lookback: 20,
            ema_short: 5,
            ema_long: 20,
            strong_margin: 0.6,
            weak_margin: 0.3,
            momentum_weight: 0.3,
            volatility_weight: 0.2,
            trend_weight: 0.5,
        }
    }
}

/// SVM Trend Classifier (IND-287)
///
/// A proxy implementation of Support Vector Machine classification that uses
/// a margin-based approach to classify market trends. The "SVM-like" behavior
/// is simulated by:
///
/// 1. **Feature Extraction**: Computes momentum, volatility, and trend features
/// 2. **Hyperplane Decision**: Uses weighted feature sum as decision function
/// 3. **Margin Classification**: Distance from decision boundary determines confidence
///
/// Features used:
/// - Momentum: Short-term price rate of change
/// - Volatility: Normalized ATR as a stabilizing factor
/// - Trend Direction: EMA crossover strength
///
/// The margin distance simulates the SVM's distance from the separating hyperplane,
/// providing a confidence measure for the classification.
#[derive(Debug, Clone)]
pub struct SVMTrendClassifier {
    lookback: usize,
    ema_short: usize,
    ema_long: usize,
    strong_margin: f64,
    weak_margin: f64,
    momentum_weight: f64,
    volatility_weight: f64,
    trend_weight: f64,
}

impl SVMTrendClassifier {
    /// Create a new SVM Trend Classifier with default configuration.
    pub fn new() -> Self {
        Self::with_config(SVMTrendClassifierConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: SVMTrendClassifierConfig) -> Self {
        Self {
            lookback: config.lookback,
            ema_short: config.ema_short,
            ema_long: config.ema_long,
            strong_margin: config.strong_margin,
            weak_margin: config.weak_margin,
            momentum_weight: config.momentum_weight,
            volatility_weight: config.volatility_weight,
            trend_weight: config.trend_weight,
        }
    }

    /// Create with custom lookback period.
    pub fn with_lookback(lookback: usize) -> Result<Self> {
        if lookback < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        let mut config = SVMTrendClassifierConfig::default();
        config.lookback = lookback;
        Ok(Self::with_config(config))
    }

    /// Calculate SVM trend classification.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> SVMTrendOutput {
        let n = close.len().min(high.len()).min(low.len());
        let mut classification = Vec::with_capacity(n);
        let mut class_value = Vec::with_capacity(n);
        let mut confidence = Vec::with_capacity(n);
        let mut margin_distance = Vec::with_capacity(n);

        // Compute EMAs
        let ema_short_vals = self.compute_ema(close, self.ema_short);
        let ema_long_vals = self.compute_ema(close, self.ema_long);

        for i in 0..n {
            if i < self.lookback {
                classification.push(SVMTrendClass::Neutral);
                class_value.push(0.0);
                confidence.push(0.0);
                margin_distance.push(0.0);
                continue;
            }

            // Extract features
            let momentum = self.compute_momentum(close, i);
            let volatility = self.compute_volatility(high, low, close, i);
            let trend_strength = self.compute_trend_strength(
                ema_short_vals[i],
                ema_long_vals[i],
                close[i],
            );

            // SVM-like decision function: weighted sum of features
            // Volatility acts as a dampening factor
            let volatility_factor = 1.0 / (1.0 + volatility);
            let decision_value = self.momentum_weight * momentum * volatility_factor
                + self.trend_weight * trend_strength;

            // Margin distance (absolute distance from decision boundary)
            let margin = decision_value.abs();

            // Classify based on decision value and margin
            let (class, class_val) = if margin >= self.strong_margin {
                if decision_value > 0.0 {
                    (SVMTrendClass::StrongUptrend, 2.0)
                } else {
                    (SVMTrendClass::StrongDowntrend, -2.0)
                }
            } else if margin >= self.weak_margin {
                if decision_value > 0.0 {
                    (SVMTrendClass::WeakUptrend, 1.0)
                } else {
                    (SVMTrendClass::WeakDowntrend, -1.0)
                }
            } else {
                (SVMTrendClass::Neutral, 0.0)
            };

            // Confidence based on margin distance (normalized to 0-100)
            let conf = (margin / self.strong_margin).min(1.0) * 100.0;

            classification.push(class);
            class_value.push(class_val);
            confidence.push(conf);
            margin_distance.push(margin);
        }

        SVMTrendOutput {
            classification,
            class_value,
            confidence,
            margin_distance,
        }
    }

    /// Compute exponential moving average.
    fn compute_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut ema = vec![f64::NAN; n];

        if n == 0 || period == 0 {
            return ema;
        }

        let multiplier = 2.0 / (period as f64 + 1.0);

        // Initialize with SMA
        if n >= period {
            let sma: f64 = data[..period].iter().sum::<f64>() / period as f64;
            ema[period - 1] = sma;

            for i in period..n {
                ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1];
            }
        }

        ema
    }

    /// Compute momentum feature.
    fn compute_momentum(&self, close: &[f64], i: usize) -> f64 {
        if i < self.lookback || close[i - self.lookback] == 0.0 {
            return 0.0;
        }
        (close[i] / close[i - self.lookback] - 1.0).clamp(-1.0, 1.0)
    }

    /// Compute volatility feature (normalized ATR-like).
    fn compute_volatility(&self, high: &[f64], low: &[f64], close: &[f64], i: usize) -> f64 {
        if i < self.lookback {
            return 0.0;
        }

        let start = i - self.lookback;
        let mut sum_tr = 0.0;

        for j in (start + 1)..=i {
            let tr = (high[j] - low[j])
                .max((high[j] - close[j - 1]).abs())
                .max((low[j] - close[j - 1]).abs());
            sum_tr += tr;
        }

        let atr = sum_tr / self.lookback as f64;
        let avg_price = close[start..=i].iter().sum::<f64>() / (self.lookback + 1) as f64;

        if avg_price > 0.0 {
            atr / avg_price
        } else {
            0.0
        }
    }

    /// Compute trend strength from EMA crossover.
    fn compute_trend_strength(&self, ema_short: f64, ema_long: f64, price: f64) -> f64 {
        if ema_short.is_nan() || ema_long.is_nan() || ema_long == 0.0 {
            return 0.0;
        }

        // EMA crossover strength
        let ema_diff = (ema_short - ema_long) / ema_long;

        // Price position relative to EMAs
        let price_position = if price > 0.0 {
            ((price - ema_long) / price).clamp(-0.5, 0.5)
        } else {
            0.0
        };

        (ema_diff * 0.7 + price_position * 0.3).clamp(-1.0, 1.0)
    }
}

impl Default for SVMTrendClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for SVMTrendClassifier {
    fn name(&self) -> &str {
        "SVM Trend Classifier"
    }

    fn min_periods(&self) -> usize {
        self.lookback + self.ema_long
    }

    fn output_features(&self) -> usize {
        3
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.min_periods();
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);

        Ok(IndicatorOutput::triple(
            result.class_value,
            result.confidence,
            result.margin_distance,
        ))
    }
}

impl SignalIndicator for SVMTrendClassifier {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low, &data.close);

        if result.classification.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        match result.classification.last().unwrap() {
            SVMTrendClass::StrongUptrend | SVMTrendClass::WeakUptrend => {
                Ok(IndicatorSignal::Bullish)
            }
            SVMTrendClass::StrongDowntrend | SVMTrendClass::WeakDowntrend => {
                Ok(IndicatorSignal::Bearish)
            }
            SVMTrendClass::Neutral => Ok(IndicatorSignal::Neutral),
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low, &data.close);

        let signals: Vec<_> = result
            .classification
            .iter()
            .map(|c| match c {
                SVMTrendClass::StrongUptrend | SVMTrendClass::WeakUptrend => {
                    IndicatorSignal::Bullish
                }
                SVMTrendClass::StrongDowntrend | SVMTrendClass::WeakDowntrend => {
                    IndicatorSignal::Bearish
                }
                SVMTrendClass::Neutral => IndicatorSignal::Neutral,
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_uptrend_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.5).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        (high, low, close)
    }

    fn generate_downtrend_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n).map(|i| 150.0 - i as f64 * 0.5).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        (high, low, close)
    }

    fn generate_sideways_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.3).sin() * 2.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 0.5).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 0.5).collect();
        (high, low, close)
    }

    #[test]
    fn test_svm_trend_classifier_basic() {
        let classifier = SVMTrendClassifier::new();
        let (high, low, close) = generate_uptrend_data(50);

        let result = classifier.calculate(&high, &low, &close);

        assert_eq!(result.classification.len(), 50);
        assert_eq!(result.class_value.len(), 50);
        assert_eq!(result.confidence.len(), 50);
        assert_eq!(result.margin_distance.len(), 50);
    }

    #[test]
    fn test_svm_uptrend_detection() {
        let classifier = SVMTrendClassifier::new();
        let (high, low, close) = generate_uptrend_data(60);

        let result = classifier.calculate(&high, &low, &close);

        // Check later values are bullish
        let bullish_count = result.classification[40..]
            .iter()
            .filter(|c| matches!(c, SVMTrendClass::StrongUptrend | SVMTrendClass::WeakUptrend))
            .count();

        assert!(bullish_count > 10, "Expected uptrend detection");
    }

    #[test]
    fn test_svm_downtrend_detection() {
        let classifier = SVMTrendClassifier::new();
        let (high, low, close) = generate_downtrend_data(60);

        let result = classifier.calculate(&high, &low, &close);

        // Check later values are bearish
        let bearish_count = result.classification[40..]
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    SVMTrendClass::StrongDowntrend | SVMTrendClass::WeakDowntrend
                )
            })
            .count();

        assert!(bearish_count > 10, "Expected downtrend detection");
    }

    #[test]
    fn test_svm_sideways_detection() {
        let classifier = SVMTrendClassifier::new();
        let (high, low, close) = generate_sideways_data(60);

        let result = classifier.calculate(&high, &low, &close);

        // In sideways market, should have mostly neutral or mixed signals
        let neutral_count = result.classification[40..]
            .iter()
            .filter(|c| matches!(c, SVMTrendClass::Neutral))
            .count();

        // Should have some neutral readings in sideways market
        assert!(neutral_count > 0, "Expected some neutral readings in sideways market");
    }

    #[test]
    fn test_svm_confidence_range() {
        let classifier = SVMTrendClassifier::new();
        let (high, low, close) = generate_uptrend_data(50);

        let result = classifier.calculate(&high, &low, &close);

        for conf in &result.confidence[classifier.min_periods()..] {
            assert!(*conf >= 0.0 && *conf <= 100.0, "Confidence out of range");
        }
    }

    #[test]
    fn test_svm_class_numeric_conversion() {
        assert_eq!(SVMTrendClass::StrongUptrend.to_numeric(), 2.0);
        assert_eq!(SVMTrendClass::WeakUptrend.to_numeric(), 1.0);
        assert_eq!(SVMTrendClass::Neutral.to_numeric(), 0.0);
        assert_eq!(SVMTrendClass::WeakDowntrend.to_numeric(), -1.0);
        assert_eq!(SVMTrendClass::StrongDowntrend.to_numeric(), -2.0);

        assert_eq!(SVMTrendClass::from_numeric(2.0), SVMTrendClass::StrongUptrend);
        assert_eq!(SVMTrendClass::from_numeric(0.0), SVMTrendClass::Neutral);
        assert_eq!(SVMTrendClass::from_numeric(-2.0), SVMTrendClass::StrongDowntrend);
    }

    #[test]
    fn test_svm_compute() {
        let classifier = SVMTrendClassifier::new();
        let (high, low, close) = generate_uptrend_data(50);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 50],
        };

        let output = classifier.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_svm_signal() {
        let classifier = SVMTrendClassifier::new();
        let (high, low, close) = generate_uptrend_data(50);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 50],
        };

        let signal = classifier.signal(&series).unwrap();
        assert!(matches!(signal, IndicatorSignal::Bullish | IndicatorSignal::Neutral));
    }

    #[test]
    fn test_svm_custom_config() {
        let config = SVMTrendClassifierConfig {
            lookback: 10,
            ema_short: 3,
            ema_long: 10,
            strong_margin: 0.5,
            weak_margin: 0.2,
            momentum_weight: 0.4,
            volatility_weight: 0.1,
            trend_weight: 0.5,
        };

        let classifier = SVMTrendClassifier::with_config(config);
        assert_eq!(classifier.name(), "SVM Trend Classifier");
        assert_eq!(classifier.min_periods(), 20); // lookback + ema_long
    }

    #[test]
    fn test_svm_with_lookback() {
        let classifier = SVMTrendClassifier::with_lookback(15).unwrap();
        assert_eq!(classifier.lookback, 15);

        // Test invalid lookback
        let result = SVMTrendClassifier::with_lookback(2);
        assert!(result.is_err());
    }
}
