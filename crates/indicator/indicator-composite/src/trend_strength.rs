//! Trend Strength Index implementation.
//!
//! Composite indicator combining multiple trend measurements.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_core::{EMA, ADX, RSI, MACD};

/// Trend Strength Index output.
#[derive(Debug, Clone)]
pub struct TrendStrengthOutput {
    /// Composite trend strength (0-100 scale).
    pub strength: Vec<f64>,
    /// Trend direction: 1 = up, -1 = down, 0 = neutral.
    pub direction: Vec<i8>,
    /// Individual component scores.
    pub components: TrendComponents,
}

/// Individual trend components.
#[derive(Debug, Clone)]
pub struct TrendComponents {
    /// EMA alignment score.
    pub ema_score: Vec<f64>,
    /// ADX trend strength.
    pub adx_score: Vec<f64>,
    /// RSI momentum score.
    pub rsi_score: Vec<f64>,
    /// MACD momentum score.
    pub macd_score: Vec<f64>,
}

/// Trend Strength Index configuration.
#[derive(Debug, Clone)]
pub struct TrendStrengthConfig {
    /// Fast EMA period (default: 8).
    pub ema_fast: usize,
    /// Medium EMA period (default: 21).
    pub ema_medium: usize,
    /// Slow EMA period (default: 55).
    pub ema_slow: usize,
    /// ADX period (default: 14).
    pub adx_period: usize,
    /// RSI period (default: 14).
    pub rsi_period: usize,
    /// MACD fast period (default: 12).
    pub macd_fast: usize,
    /// MACD slow period (default: 26).
    pub macd_slow: usize,
    /// MACD signal period (default: 9).
    pub macd_signal: usize,
    /// Weight for EMA component (default: 0.25).
    pub weight_ema: f64,
    /// Weight for ADX component (default: 0.30).
    pub weight_adx: f64,
    /// Weight for RSI component (default: 0.20).
    pub weight_rsi: f64,
    /// Weight for MACD component (default: 0.25).
    pub weight_macd: f64,
}

impl Default for TrendStrengthConfig {
    fn default() -> Self {
        Self {
            ema_fast: 8,
            ema_medium: 21,
            ema_slow: 55,
            adx_period: 14,
            rsi_period: 14,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
            weight_ema: 0.25,
            weight_adx: 0.30,
            weight_rsi: 0.20,
            weight_macd: 0.25,
        }
    }
}

/// Trend Strength Index.
///
/// A composite indicator that combines multiple trend measurements:
///
/// 1. **EMA Alignment**: Measures how well EMAs are stacked (bullish/bearish order)
/// 2. **ADX Strength**: Measures trend strength regardless of direction
/// 3. **RSI Momentum**: Measures momentum with overbought/oversold zones
/// 4. **MACD Momentum**: Measures trend momentum and crossovers
///
/// Output is a weighted combination of these components on a 0-100 scale.
/// Values above 60 indicate strong trend, below 40 indicate weak trend.
#[derive(Debug, Clone)]
pub struct TrendStrengthIndex {
    ema_fast: EMA,
    ema_medium: EMA,
    ema_slow: EMA,
    adx: ADX,
    rsi: RSI,
    macd: MACD,
    weight_ema: f64,
    weight_adx: f64,
    weight_rsi: f64,
    weight_macd: f64,
}

impl TrendStrengthIndex {
    pub fn new(config: TrendStrengthConfig) -> Self {
        Self {
            ema_fast: EMA::new(config.ema_fast),
            ema_medium: EMA::new(config.ema_medium),
            ema_slow: EMA::new(config.ema_slow),
            adx: ADX::new(config.adx_period),
            rsi: RSI::new(config.rsi_period),
            macd: MACD::new(config.macd_fast, config.macd_slow, config.macd_signal),
            weight_ema: config.weight_ema,
            weight_adx: config.weight_adx,
            weight_rsi: config.weight_rsi,
            weight_macd: config.weight_macd,
        }
    }

    /// Calculate Trend Strength Index values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> TrendStrengthOutput {
        let n = close.len();

        // Calculate EMAs
        let ema_fast = self.ema_fast.calculate(close);
        let ema_medium = self.ema_medium.calculate(close);
        let ema_slow = self.ema_slow.calculate(close);

        // Calculate EMA alignment score
        let mut ema_score = vec![50.0; n];
        for i in 0..n {
            if ema_fast[i].is_nan() || ema_medium[i].is_nan() || ema_slow[i].is_nan() {
                continue;
            }

            // Bullish alignment: fast > medium > slow
            // Bearish alignment: fast < medium < slow
            let bullish_aligned = ema_fast[i] > ema_medium[i] && ema_medium[i] > ema_slow[i];
            let bearish_aligned = ema_fast[i] < ema_medium[i] && ema_medium[i] < ema_slow[i];

            if bullish_aligned {
                // Calculate spread for strength
                let spread = (ema_fast[i] - ema_slow[i]) / ema_slow[i] * 100.0;
                ema_score[i] = (50.0 + spread.min(50.0)).clamp(0.0, 100.0);
            } else if bearish_aligned {
                let spread = (ema_slow[i] - ema_fast[i]) / ema_slow[i] * 100.0;
                ema_score[i] = (50.0 - spread.min(50.0)).clamp(0.0, 100.0);
            }
        }

        // Calculate ADX score (already 0-100)
        let adx_output = self.adx.calculate(high, low, close);
        let mut adx_score = vec![0.0; n];
        for i in 0..n {
            if !adx_output.adx[i].is_nan() {
                // Transform ADX to trend strength
                // ADX > 25 = trending, ADX < 20 = ranging
                adx_score[i] = adx_output.adx[i].clamp(0.0, 100.0);
            }
        }

        // Calculate RSI score (already 0-100, but need to interpret for trend)
        let rsi_values = self.rsi.calculate(close);
        let mut rsi_score = vec![50.0; n];
        for i in 0..n {
            if !rsi_values[i].is_nan() {
                // RSI > 50 = bullish momentum, RSI < 50 = bearish momentum
                rsi_score[i] = rsi_values[i];
            }
        }

        // Calculate MACD score
        let (macd_line, signal_line, histogram) = self.macd.calculate(close);
        let mut macd_score = vec![50.0; n];
        for i in 0..n {
            if macd_line[i].is_nan() || signal_line[i].is_nan() {
                continue;
            }

            // Normalize histogram to 0-100 scale
            let hist_normalized = histogram[i] / close[i] * 1000.0; // Scale factor
            macd_score[i] = (50.0 + hist_normalized * 10.0).clamp(0.0, 100.0);
        }

        // Calculate composite strength
        let mut strength = vec![50.0; n];
        let mut direction = vec![0i8; n];

        for i in 0..n {
            let composite = self.weight_ema * ema_score[i]
                + self.weight_adx * adx_score[i]
                + self.weight_rsi * rsi_score[i]
                + self.weight_macd * macd_score[i];

            strength[i] = composite.clamp(0.0, 100.0);

            // Determine direction based on component consensus
            let bullish_votes = [
                ema_score[i] > 55.0,
                rsi_score[i] > 50.0,
                macd_score[i] > 50.0,
                adx_output.plus_di[i] > adx_output.minus_di[i],
            ].iter().filter(|&&x| x).count();

            if bullish_votes >= 3 && strength[i] > 50.0 {
                direction[i] = 1;
            } else if bullish_votes <= 1 && strength[i] < 50.0 {
                direction[i] = -1;
            }
        }

        TrendStrengthOutput {
            strength,
            direction,
            components: TrendComponents {
                ema_score,
                adx_score,
                rsi_score,
                macd_score,
            },
        }
    }
}

impl Default for TrendStrengthIndex {
    fn default() -> Self {
        Self::new(TrendStrengthConfig::default())
    }
}

impl TechnicalIndicator for TrendStrengthIndex {
    fn name(&self) -> &str {
        "TrendStrengthIndex"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = 55; // Slowest component
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);

        let direction_values: Vec<f64> = result.direction.iter()
            .map(|&d| d as f64)
            .collect();

        Ok(IndicatorOutput::dual(result.strength, direction_values))
    }

    fn min_periods(&self) -> usize {
        55
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for TrendStrengthIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low, &data.close);
        let n = result.strength.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let strength = result.strength[n - 1];
        let direction = result.direction[n - 1];

        // Strong bullish: high strength + bullish direction
        if strength > 60.0 && direction == 1 {
            return Ok(IndicatorSignal::Bullish);
        }
        // Strong bearish: high strength (inverse) + bearish direction
        else if strength < 40.0 && direction == -1 {
            return Ok(IndicatorSignal::Bearish);
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low, &data.close);

        let signals: Vec<_> = result.strength.iter()
            .zip(result.direction.iter())
            .map(|(&strength, &direction)| {
                if strength > 60.0 && direction == 1 {
                    IndicatorSignal::Bullish
                } else if strength < 40.0 && direction == -1 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_uptrend_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64 * 0.8).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64 * 0.8).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.8).collect();
        (high, low, close)
    }

    fn generate_ranging_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.3).sin() * 3.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.3).sin() * 3.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.3).sin() * 3.0).collect();
        (high, low, close)
    }

    #[test]
    fn test_trend_strength_basic() {
        let tsi = TrendStrengthIndex::default();
        let (high, low, close) = generate_uptrend_data(80);

        let result = tsi.calculate(&high, &low, &close);

        assert_eq!(result.strength.len(), 80);
        assert_eq!(result.direction.len(), 80);
    }

    #[test]
    fn test_trend_strength_range() {
        let tsi = TrendStrengthIndex::default();
        let (high, low, close) = generate_uptrend_data(80);

        let result = tsi.calculate(&high, &low, &close);

        // Strength should be 0-100
        for &s in &result.strength {
            assert!(s >= 0.0 && s <= 100.0);
        }

        // Direction should be -1, 0, or 1
        for &d in &result.direction {
            assert!(d >= -1 && d <= 1);
        }
    }

    #[test]
    fn test_trend_strength_uptrend() {
        let tsi = TrendStrengthIndex::default();
        let (high, low, close) = generate_uptrend_data(80);

        let result = tsi.calculate(&high, &low, &close);

        // In a strong uptrend, we should see bullish direction
        let bullish_count = result.direction.iter().filter(|&&d| d == 1).count();
        assert!(bullish_count > 0, "Expected bullish directions in uptrend");
    }

    #[test]
    fn test_trend_strength_ranging() {
        let tsi = TrendStrengthIndex::default();
        let (high, low, close) = generate_ranging_data(80);

        let result = tsi.calculate(&high, &low, &close);

        // In ranging market, strength should be more moderate
        let avg_strength: f64 = result.strength[60..80].iter().sum::<f64>() / 20.0;
        // Should be closer to 50 in ranging market
        assert!(avg_strength > 30.0 && avg_strength < 70.0);
    }

    #[test]
    fn test_trend_strength_config() {
        let config = TrendStrengthConfig {
            ema_fast: 5,
            ema_medium: 15,
            ema_slow: 30,
            adx_period: 10,
            rsi_period: 10,
            macd_fast: 8,
            macd_slow: 17,
            macd_signal: 6,
            weight_ema: 0.30,
            weight_adx: 0.25,
            weight_rsi: 0.25,
            weight_macd: 0.20,
        };

        let tsi = TrendStrengthIndex::new(config);
        assert_eq!(tsi.name(), "TrendStrengthIndex");
    }
}
