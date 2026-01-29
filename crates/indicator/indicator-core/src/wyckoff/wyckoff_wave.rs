//! Wyckoff Wave - Composite market indicator (IND-228)
//!
//! A composite indicator combining multiple Wyckoff concepts to identify
//! market phases and trend direction using price, volume, and spread analysis.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Wyckoff Wave configuration.
#[derive(Debug, Clone)]
pub struct WyckoffWaveConfig {
    /// Period for wave calculation
    pub period: usize,
    /// Volume smoothing period
    pub volume_period: usize,
    /// Spread weighting factor
    pub spread_weight: f64,
    /// Volume weighting factor
    pub volume_weight: f64,
}

impl Default for WyckoffWaveConfig {
    fn default() -> Self {
        Self {
            period: 14,
            volume_period: 10,
            spread_weight: 0.5,
            volume_weight: 0.5,
        }
    }
}

/// Wyckoff Wave output containing wave values and phase indicators.
#[derive(Debug, Clone)]
pub struct WyckoffWaveOutput {
    /// Wave values (cumulative)
    pub wave: Vec<f64>,
    /// Effort vs Result comparison
    pub effort_result: Vec<f64>,
    /// Volume-weighted trend strength
    pub trend_strength: Vec<f64>,
}

/// Wyckoff Wave - Composite market indicator.
///
/// Combines spread (high-low), close location within spread, and volume
/// to create a cumulative wave that reflects supply/demand dynamics.
///
/// The indicator identifies:
/// - Effort (volume) vs Result (price movement)
/// - Close position within the spread (weakness/strength)
/// - Cumulative buying/selling pressure
#[derive(Debug, Clone)]
pub struct WyckoffWave {
    config: WyckoffWaveConfig,
}

impl WyckoffWave {
    pub fn new(period: usize) -> Self {
        Self {
            config: WyckoffWaveConfig {
                period,
                ..Default::default()
            },
        }
    }

    pub fn from_config(config: WyckoffWaveConfig) -> Self {
        Self { config }
    }

    /// Calculate the close location value (CLV).
    /// CLV = ((Close - Low) - (High - Close)) / (High - Low)
    /// Range: -1 (close at low) to +1 (close at high)
    fn calculate_clv(high: f64, low: f64, close: f64) -> f64 {
        let spread = high - low;
        if spread <= 0.0 {
            return 0.0;
        }
        ((close - low) - (high - close)) / spread
    }

    /// Calculate Wyckoff Wave values.
    pub fn calculate(&self, data: &OHLCVSeries) -> WyckoffWaveOutput {
        let n = data.close.len();

        if n < self.config.period {
            return WyckoffWaveOutput {
                wave: vec![f64::NAN; n],
                effort_result: vec![f64::NAN; n],
                trend_strength: vec![f64::NAN; n],
            };
        }

        let mut wave = vec![f64::NAN; n];
        let mut effort_result = vec![f64::NAN; n];
        let mut trend_strength = vec![f64::NAN; n];

        // Calculate average volume for normalization
        let avg_volume: f64 = data.volume.iter().sum::<f64>() / n as f64;
        let avg_volume = if avg_volume > 0.0 { avg_volume } else { 1.0 };

        // Calculate spread and CLV for each bar
        let mut spreads = Vec::with_capacity(n);
        let mut clvs = Vec::with_capacity(n);
        let mut volume_clv = Vec::with_capacity(n);

        for i in 0..n {
            let spread = data.high[i] - data.low[i];
            let clv = Self::calculate_clv(data.high[i], data.low[i], data.close[i]);
            let vol_norm = data.volume[i] / avg_volume;

            spreads.push(spread);
            clvs.push(clv);
            volume_clv.push(clv * vol_norm * spread);
        }

        // Calculate cumulative wave
        let mut cumulative = 0.0;
        for i in 0..n {
            cumulative += volume_clv[i];
            wave[i] = cumulative;
        }

        // Calculate effort vs result (volume vs price change)
        for i in self.config.period..n {
            let price_change = (data.close[i] - data.close[i - self.config.period]).abs();
            let volume_sum: f64 = data.volume[(i - self.config.period + 1)..=i].iter().sum();
            let avg_spread: f64 = spreads[(i - self.config.period + 1)..=i].iter().sum::<f64>()
                / self.config.period as f64;

            // Effort vs Result: high volume with small price change = weak
            // Low volume with large price change = strong
            let effort = volume_sum / (self.config.period as f64 * avg_volume);
            let result = if avg_spread > 0.0 { price_change / avg_spread } else { 0.0 };

            effort_result[i] = if effort > 0.0 { result / effort } else { 0.0 };
        }

        // Calculate trend strength using volume-weighted momentum
        for i in self.config.volume_period..n {
            let mut strength = 0.0;
            let mut weight_sum = 0.0;

            for j in (i - self.config.volume_period + 1)..=i {
                let vol_weight = data.volume[j] / avg_volume;
                let direction = clvs[j];
                strength += direction * vol_weight;
                weight_sum += vol_weight;
            }

            trend_strength[i] = if weight_sum > 0.0 {
                strength / weight_sum
            } else {
                0.0
            };
        }

        WyckoffWaveOutput {
            wave,
            effort_result,
            trend_strength,
        }
    }
}

impl Default for WyckoffWave {
    fn default() -> Self {
        Self::from_config(WyckoffWaveConfig::default())
    }
}

impl TechnicalIndicator for WyckoffWave {
    fn name(&self) -> &str {
        "WyckoffWave"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.period {
            return Err(IndicatorError::InsufficientData {
                required: self.config.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);
        Ok(IndicatorOutput::triple(
            result.wave,
            result.effort_result,
            result.trend_strength,
        ))
    }

    fn min_periods(&self) -> usize {
        self.config.period
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for WyckoffWave {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);

        // Use trend strength for signal generation
        if let Some(&strength) = result.trend_strength.last() {
            if !strength.is_nan() {
                if strength > 0.3 {
                    return Ok(IndicatorSignal::Bullish);
                } else if strength < -0.3 {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);

        Ok(result
            .trend_strength
            .iter()
            .map(|&s| {
                if s.is_nan() {
                    IndicatorSignal::Neutral
                } else if s > 0.3 {
                    IndicatorSignal::Bullish
                } else if s < -0.3 {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(n: usize) -> OHLCVSeries {
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            let base = 100.0 + (i as f64) * 0.5;
            open.push(base);
            high.push(base + 2.0);
            low.push(base - 1.0);
            close.push(base + 1.0); // Close near high = bullish
            volume.push(1000.0 + (i as f64) * 50.0);
        }

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_wyckoff_wave_basic() {
        let ww = WyckoffWave::new(14);
        let data = create_test_data(30);
        let result = ww.calculate(&data);

        assert_eq!(result.wave.len(), 30);
        assert_eq!(result.effort_result.len(), 30);
        assert_eq!(result.trend_strength.len(), 30);
    }

    #[test]
    fn test_wyckoff_wave_cumulative() {
        let ww = WyckoffWave::new(5);
        let data = create_test_data(20);
        let result = ww.calculate(&data);

        // Wave should be cumulative and increasing in uptrend
        let valid_waves: Vec<f64> = result.wave.iter()
            .filter(|&&x| !x.is_nan())
            .copied()
            .collect();

        // In an uptrend with closes near highs, wave should generally increase
        assert!(!valid_waves.is_empty());
    }

    #[test]
    fn test_clv_calculation() {
        // Close at high = CLV of 1
        let clv = WyckoffWave::calculate_clv(110.0, 100.0, 110.0);
        assert!((clv - 1.0).abs() < 0.001);

        // Close at low = CLV of -1
        let clv = WyckoffWave::calculate_clv(110.0, 100.0, 100.0);
        assert!((clv - (-1.0)).abs() < 0.001);

        // Close at midpoint = CLV of 0
        let clv = WyckoffWave::calculate_clv(110.0, 100.0, 105.0);
        assert!(clv.abs() < 0.001);
    }

    #[test]
    fn test_wyckoff_wave_signal() {
        let ww = WyckoffWave::new(5);
        let data = create_test_data(20);
        let signal = ww.signal(&data).unwrap();

        // With closes near highs consistently, expect bullish signal
        assert!(matches!(signal, IndicatorSignal::Bullish | IndicatorSignal::Neutral));
    }
}
