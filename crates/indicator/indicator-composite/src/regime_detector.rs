//! Market Regime Detection implementation.
//!
//! Identifies whether market is trending or ranging using multiple indicators.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_core::{ADX, BollingerBands, ATR, EMA};

/// Market regime types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    /// Strong upward trend.
    StrongUptrend,
    /// Weak upward trend.
    WeakUptrend,
    /// Ranging/consolidating market.
    Ranging,
    /// Weak downward trend.
    WeakDowntrend,
    /// Strong downward trend.
    StrongDowntrend,
}

impl MarketRegime {
    /// Convert to numeric value.
    pub fn to_numeric(&self) -> f64 {
        match self {
            MarketRegime::StrongUptrend => 2.0,
            MarketRegime::WeakUptrend => 1.0,
            MarketRegime::Ranging => 0.0,
            MarketRegime::WeakDowntrend => -1.0,
            MarketRegime::StrongDowntrend => -2.0,
        }
    }

    /// Create from numeric value.
    pub fn from_numeric(value: f64) -> Self {
        if value >= 1.5 {
            MarketRegime::StrongUptrend
        } else if value >= 0.5 {
            MarketRegime::WeakUptrend
        } else if value > -0.5 {
            MarketRegime::Ranging
        } else if value > -1.5 {
            MarketRegime::WeakDowntrend
        } else {
            MarketRegime::StrongDowntrend
        }
    }
}

/// Regime Detector output.
#[derive(Debug, Clone)]
pub struct RegimeDetectorOutput {
    /// Current market regime.
    pub regime: Vec<MarketRegime>,
    /// Regime numeric values (-2 to +2).
    pub regime_value: Vec<f64>,
    /// Trend strength (0-100).
    pub trend_strength: Vec<f64>,
    /// Volatility percentile (0-100).
    pub volatility_percentile: Vec<f64>,
    /// Probability of regime change.
    pub regime_change_prob: Vec<f64>,
}

/// Regime Detector configuration.
#[derive(Debug, Clone)]
pub struct RegimeDetectorConfig {
    /// ADX period for trend strength (default: 14).
    pub adx_period: usize,
    /// Bollinger Bands period for volatility (default: 20).
    pub bb_period: usize,
    /// ATR period for volatility (default: 14).
    pub atr_period: usize,
    /// EMA periods for trend direction (default: 20, 50).
    pub ema_short: usize,
    pub ema_long: usize,
    /// Lookback for volatility percentile (default: 100).
    pub volatility_lookback: usize,
    /// ADX threshold for trending (default: 25).
    pub trend_threshold: f64,
    /// ADX threshold for strong trend (default: 40).
    pub strong_trend_threshold: f64,
}

impl Default for RegimeDetectorConfig {
    fn default() -> Self {
        Self {
            adx_period: 14,
            bb_period: 20,
            atr_period: 14,
            ema_short: 20,
            ema_long: 50,
            volatility_lookback: 100,
            trend_threshold: 25.0,
            strong_trend_threshold: 40.0,
        }
    }
}

/// Market Regime Detector.
///
/// Identifies the current market regime by analyzing:
///
/// 1. **Trend Strength (ADX)**: Measures how strongly the market is trending
/// 2. **Trend Direction (EMA)**: Determines bullish or bearish bias
/// 3. **Volatility (BB Width, ATR)**: Measures market volatility
/// 4. **Regime Stability**: Estimates probability of regime change
///
/// Regimes:
/// - Strong Uptrend: ADX > 40, EMA short > EMA long
/// - Weak Uptrend: 25 < ADX < 40, EMA short > EMA long
/// - Ranging: ADX < 25
/// - Weak Downtrend: 25 < ADX < 40, EMA short < EMA long
/// - Strong Downtrend: ADX > 40, EMA short < EMA long
#[derive(Debug, Clone)]
pub struct RegimeDetector {
    adx: ADX,
    bb: BollingerBands,
    atr: ATR,
    ema_short: EMA,
    ema_long: EMA,
    volatility_lookback: usize,
    trend_threshold: f64,
    strong_trend_threshold: f64,
}

impl RegimeDetector {
    pub fn new(config: RegimeDetectorConfig) -> Self {
        Self {
            adx: ADX::new(config.adx_period),
            bb: BollingerBands::new(config.bb_period, 2.0),
            atr: ATR::new(config.atr_period),
            ema_short: EMA::new(config.ema_short),
            ema_long: EMA::new(config.ema_long),
            volatility_lookback: config.volatility_lookback,
            trend_threshold: config.trend_threshold,
            strong_trend_threshold: config.strong_trend_threshold,
        }
    }

    /// Calculate regime detection values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> RegimeDetectorOutput {
        let n = close.len();

        // Calculate ADX for trend strength
        let adx_output = self.adx.calculate(high, low, close);

        // Calculate EMAs for trend direction
        let ema_short_values = self.ema_short.calculate(close);
        let ema_long_values = self.ema_long.calculate(close);

        // Calculate ATR for volatility
        let atr_values = self.atr.calculate(high, low, close);

        // Calculate Bollinger Bandwidth for volatility
        let _bb_bandwidth = self.bb.bandwidth(close);

        // Calculate volatility percentile
        let volatility_percentile = self.calculate_volatility_percentile(&atr_values, close);

        // Determine regime
        let mut regime = Vec::with_capacity(n);
        let mut regime_value = Vec::with_capacity(n);
        let mut trend_strength = Vec::with_capacity(n);

        for i in 0..n {
            let adx = adx_output.adx[i];
            let plus_di = adx_output.plus_di[i];
            let minus_di = adx_output.minus_di[i];
            let ema_s = ema_short_values[i];
            let ema_l = ema_long_values[i];

            if adx.is_nan() || ema_s.is_nan() || ema_l.is_nan() {
                regime.push(MarketRegime::Ranging);
                regime_value.push(0.0);
                trend_strength.push(0.0);
                continue;
            }

            trend_strength.push(adx);

            // Determine direction
            let bullish = ema_s > ema_l || (!plus_di.is_nan() && !minus_di.is_nan() && plus_di > minus_di);
            let bearish = ema_s < ema_l || (!plus_di.is_nan() && !minus_di.is_nan() && minus_di > plus_di);

            // Determine regime
            let current_regime = if adx >= self.strong_trend_threshold {
                if bullish {
                    MarketRegime::StrongUptrend
                } else {
                    MarketRegime::StrongDowntrend
                }
            } else if adx >= self.trend_threshold {
                if bullish {
                    MarketRegime::WeakUptrend
                } else if bearish {
                    MarketRegime::WeakDowntrend
                } else {
                    MarketRegime::Ranging
                }
            } else {
                MarketRegime::Ranging
            };

            regime.push(current_regime);
            regime_value.push(current_regime.to_numeric());
        }

        // Calculate regime change probability
        let regime_change_prob = self.calculate_regime_change_prob(&regime_value, &trend_strength, &volatility_percentile);

        RegimeDetectorOutput {
            regime,
            regime_value,
            trend_strength,
            volatility_percentile,
            regime_change_prob,
        }
    }

    /// Calculate volatility percentile.
    fn calculate_volatility_percentile(&self, atr: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut percentile = vec![50.0; n];

        // Normalize ATR by price
        let normalized_atr: Vec<f64> = atr.iter()
            .zip(close.iter())
            .map(|(&a, &c)| {
                if a.is_nan() || c.abs() < 1e-10 {
                    f64::NAN
                } else {
                    (a / c) * 100.0
                }
            })
            .collect();

        for i in self.volatility_lookback..n {
            let start = i - self.volatility_lookback;
            let window: Vec<f64> = normalized_atr[start..i]
                .iter()
                .filter(|x| !x.is_nan())
                .cloned()
                .collect();

            if window.is_empty() || normalized_atr[i].is_nan() {
                continue;
            }

            let current = normalized_atr[i];
            let below_count = window.iter().filter(|&&x| x < current).count();
            percentile[i] = (below_count as f64 / window.len() as f64) * 100.0;
        }

        percentile
    }

    /// Calculate probability of regime change.
    fn calculate_regime_change_prob(
        &self,
        regime_value: &[f64],
        trend_strength: &[f64],
        volatility: &[f64],
    ) -> Vec<f64> {
        let n = regime_value.len();
        let mut prob = vec![0.0; n];

        for i in 5..n {
            // Factors that increase regime change probability:
            // 1. Trend strength declining
            // 2. High volatility
            // 3. Regime at extremes

            let trend_declining = if i >= 10 {
                let avg_recent = trend_strength[(i-5)..i].iter().sum::<f64>() / 5.0;
                let avg_prior = trend_strength[(i-10)..(i-5)].iter().sum::<f64>() / 5.0;
                if avg_prior > 0.0 {
                    ((avg_prior - avg_recent) / avg_prior).max(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let high_volatility = volatility[i] / 100.0; // 0-1 scale

            let extreme_regime = regime_value[i].abs() / 2.0; // 0-1 scale

            // Combine factors
            prob[i] = ((trend_declining * 0.4 + high_volatility * 0.3 + extreme_regime * 0.3) * 100.0).clamp(0.0, 100.0);
        }

        prob
    }
}

impl Default for RegimeDetector {
    fn default() -> Self {
        Self::new(RegimeDetectorConfig::default())
    }
}

impl TechnicalIndicator for RegimeDetector {
    fn name(&self) -> &str {
        "RegimeDetector"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = 50; // EMA long period
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);

        Ok(IndicatorOutput::triple(
            result.regime_value,
            result.trend_strength,
            result.volatility_percentile,
        ))
    }

    fn min_periods(&self) -> usize {
        50
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for RegimeDetector {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low, &data.close);
        let n = result.regime.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        match result.regime[n - 1] {
            MarketRegime::StrongUptrend | MarketRegime::WeakUptrend => {
                Ok(IndicatorSignal::Bullish)
            }
            MarketRegime::StrongDowntrend | MarketRegime::WeakDowntrend => {
                Ok(IndicatorSignal::Bearish)
            }
            MarketRegime::Ranging => {
                Ok(IndicatorSignal::Neutral)
            }
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low, &data.close);

        let signals: Vec<_> = result.regime.iter()
            .map(|r| match r {
                MarketRegime::StrongUptrend | MarketRegime::WeakUptrend => IndicatorSignal::Bullish,
                MarketRegime::StrongDowntrend | MarketRegime::WeakDowntrend => IndicatorSignal::Bearish,
                MarketRegime::Ranging => IndicatorSignal::Neutral,
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_trending_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64 * 0.8).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64 * 0.8).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.8).collect();
        (high, low, close)
    }

    fn generate_ranging_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.5).sin() * 2.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.5).sin() * 2.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.5).sin() * 2.0).collect();
        (high, low, close)
    }

    #[test]
    fn test_regime_detector_basic() {
        let detector = RegimeDetector::default();
        let (high, low, close) = generate_trending_data(120);

        let result = detector.calculate(&high, &low, &close);

        assert_eq!(result.regime.len(), 120);
        assert_eq!(result.regime_value.len(), 120);
        assert_eq!(result.trend_strength.len(), 120);
        assert_eq!(result.volatility_percentile.len(), 120);
        assert_eq!(result.regime_change_prob.len(), 120);
    }

    #[test]
    fn test_regime_detector_trending() {
        let detector = RegimeDetector::default();
        let (high, low, close) = generate_trending_data(120);

        let result = detector.calculate(&high, &low, &close);

        // In a trending market, we should see uptrend regimes
        let uptrend_count = result.regime.iter()
            .filter(|&&r| r == MarketRegime::StrongUptrend || r == MarketRegime::WeakUptrend)
            .count();

        assert!(uptrend_count > 0, "Expected uptrend regimes in trending data");
    }

    #[test]
    fn test_regime_detector_ranging() {
        let detector = RegimeDetector::default();
        let (high, low, close) = generate_ranging_data(120);

        let result = detector.calculate(&high, &low, &close);

        // In a ranging market, we should see ranging regimes
        let ranging_count = result.regime.iter()
            .filter(|&&r| r == MarketRegime::Ranging)
            .count();

        assert!(ranging_count > 0, "Expected ranging regimes in ranging data");
    }

    #[test]
    fn test_regime_numeric_conversion() {
        assert_eq!(MarketRegime::StrongUptrend.to_numeric(), 2.0);
        assert_eq!(MarketRegime::WeakUptrend.to_numeric(), 1.0);
        assert_eq!(MarketRegime::Ranging.to_numeric(), 0.0);
        assert_eq!(MarketRegime::WeakDowntrend.to_numeric(), -1.0);
        assert_eq!(MarketRegime::StrongDowntrend.to_numeric(), -2.0);

        assert_eq!(MarketRegime::from_numeric(2.0), MarketRegime::StrongUptrend);
        assert_eq!(MarketRegime::from_numeric(0.0), MarketRegime::Ranging);
        assert_eq!(MarketRegime::from_numeric(-2.0), MarketRegime::StrongDowntrend);
    }

    #[test]
    fn test_regime_detector_config() {
        let config = RegimeDetectorConfig {
            adx_period: 10,
            bb_period: 15,
            atr_period: 10,
            ema_short: 15,
            ema_long: 40,
            volatility_lookback: 50,
            trend_threshold: 20.0,
            strong_trend_threshold: 35.0,
        };

        let detector = RegimeDetector::new(config);
        assert_eq!(detector.name(), "RegimeDetector");
    }

    #[test]
    fn test_regime_detector_compute() {
        let detector = RegimeDetector::default();
        let (high, low, close) = generate_trending_data(120);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 120],
        };

        let output = detector.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 120);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }
}
