//! Optimism Pessimism Index - Volume-based sentiment indicator (IND-229)
//!
//! Measures market sentiment by analyzing the relationship between
//! price movement direction and volume to gauge buying/selling pressure.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Optimism Pessimism Index configuration.
#[derive(Debug, Clone)]
pub struct OptimismPessimismConfig {
    /// Smoothing period for the index
    pub period: usize,
    /// Overbought threshold
    pub overbought: f64,
    /// Oversold threshold
    pub oversold: f64,
}

impl Default for OptimismPessimismConfig {
    fn default() -> Self {
        Self {
            period: 14,
            overbought: 70.0,
            oversold: 30.0,
        }
    }
}

/// Optimism Pessimism Index (O/P Index).
///
/// The Optimism/Pessimism Index is a volume-based sentiment indicator
/// developed by Earl Blumenthal. It accumulates volume based on where
/// the close is relative to the day's range.
///
/// Formula:
/// - Daily O/P = Volume * ((Close - Low) - (High - Close)) / (High - Low)
/// - O/P Index = Cumulative sum of Daily O/P
///
/// Interpretation:
/// - Rising O/P Index with rising prices = bullish confirmation
/// - Rising prices with falling O/P = bearish divergence (distribution)
/// - Falling prices with rising O/P = bullish divergence (accumulation)
#[derive(Debug, Clone)]
pub struct OptimismPessimismIndex {
    config: OptimismPessimismConfig,
}

impl OptimismPessimismIndex {
    pub fn new(period: usize) -> Self {
        Self {
            config: OptimismPessimismConfig {
                period,
                ..Default::default()
            },
        }
    }

    pub fn from_config(config: OptimismPessimismConfig) -> Self {
        Self { config }
    }

    /// Calculate the Close Location Value (CLV).
    fn calculate_clv(high: f64, low: f64, close: f64) -> f64 {
        let range = high - low;
        if range <= 0.0 {
            return 0.0;
        }
        ((close - low) - (high - close)) / range
    }

    /// Calculate raw O/P values (not cumulative).
    pub fn calculate_raw(&self, data: &OHLCVSeries) -> Vec<f64> {
        let n = data.close.len();
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let clv = Self::calculate_clv(data.high[i], data.low[i], data.close[i]);
            result.push(clv * data.volume[i]);
        }

        result
    }

    /// Calculate cumulative O/P Index.
    pub fn calculate_cumulative(&self, data: &OHLCVSeries) -> Vec<f64> {
        let raw = self.calculate_raw(data);
        let n = raw.len();
        let mut cumulative = Vec::with_capacity(n);

        let mut sum = 0.0;
        for val in raw {
            sum += val;
            cumulative.push(sum);
        }

        cumulative
    }

    /// Calculate smoothed O/P Index using EMA.
    pub fn calculate(&self, data: &OHLCVSeries) -> Vec<f64> {
        let raw = self.calculate_raw(data);
        let n = raw.len();

        if n < self.config.period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];
        let alpha = 2.0 / (self.config.period as f64 + 1.0);

        // Initialize with SMA
        let first_sum: f64 = raw[0..self.config.period].iter().sum();
        result[self.config.period - 1] = first_sum / self.config.period as f64;

        // Apply EMA smoothing
        for i in self.config.period..n {
            result[i] = alpha * raw[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Calculate normalized O/P Index (0-100 scale).
    pub fn calculate_normalized(&self, data: &OHLCVSeries) -> Vec<f64> {
        let cumulative = self.calculate_cumulative(data);
        let n = cumulative.len();

        if n < self.config.period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in (self.config.period - 1)..n {
            let start = i + 1 - self.config.period;
            let window = &cumulative[start..=i];

            let min = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            if (max - min).abs() > 1e-10 {
                result[i] = ((cumulative[i] - min) / (max - min)) * 100.0;
            } else {
                result[i] = 50.0;
            }
        }

        result
    }
}

impl Default for OptimismPessimismIndex {
    fn default() -> Self {
        Self::from_config(OptimismPessimismConfig::default())
    }
}

impl TechnicalIndicator for OptimismPessimismIndex {
    fn name(&self) -> &str {
        "OptimismPessimismIndex"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.config.period {
            return Err(IndicatorError::InsufficientData {
                required: self.config.period,
                got: data.close.len(),
            });
        }

        let values = self.calculate_normalized(data);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.config.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for OptimismPessimismIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate_normalized(data);

        if let Some(&last) = values.last() {
            if !last.is_nan() {
                if last >= self.config.overbought {
                    return Ok(IndicatorSignal::Bearish); // Overbought = potential reversal
                } else if last <= self.config.oversold {
                    return Ok(IndicatorSignal::Bullish); // Oversold = potential reversal
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate_normalized(data);

        Ok(values
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    IndicatorSignal::Neutral
                } else if v >= self.config.overbought {
                    IndicatorSignal::Bearish
                } else if v <= self.config.oversold {
                    IndicatorSignal::Bullish
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

    fn create_bullish_data(n: usize) -> OHLCVSeries {
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            let base = 100.0 + (i as f64) * 0.5;
            open.push(base);
            high.push(base + 2.0);
            low.push(base - 0.5);
            close.push(base + 1.8); // Close near high
            volume.push(1000.0);
        }

        OHLCVSeries { open, high, low, close, volume }
    }

    fn create_bearish_data(n: usize) -> OHLCVSeries {
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for i in 0..n {
            let base = 100.0 - (i as f64) * 0.5;
            open.push(base);
            high.push(base + 0.5);
            low.push(base - 2.0);
            close.push(base - 1.8); // Close near low
            volume.push(1000.0);
        }

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_optimism_pessimism_basic() {
        let opi = OptimismPessimismIndex::new(14);
        let data = create_bullish_data(30);
        let result = opi.calculate_normalized(&data);

        assert_eq!(result.len(), 30);

        // Valid values should be in 0-100 range
        for &val in result.iter().skip(13) {
            if !val.is_nan() {
                assert!(val >= 0.0 && val <= 100.0);
            }
        }
    }

    #[test]
    fn test_optimism_pessimism_cumulative() {
        let opi = OptimismPessimismIndex::new(5);
        let data = create_bullish_data(20);
        let cumulative = opi.calculate_cumulative(&data);

        assert_eq!(cumulative.len(), 20);

        // In bullish data (closes near highs), cumulative should trend up
        let first_half_avg: f64 = cumulative[0..10].iter().sum::<f64>() / 10.0;
        let second_half_avg: f64 = cumulative[10..20].iter().sum::<f64>() / 10.0;
        assert!(second_half_avg > first_half_avg);
    }

    #[test]
    fn test_clv_values() {
        // Close at high
        let clv = OptimismPessimismIndex::calculate_clv(110.0, 100.0, 110.0);
        assert!((clv - 1.0).abs() < 0.001);

        // Close at low
        let clv = OptimismPessimismIndex::calculate_clv(110.0, 100.0, 100.0);
        assert!((clv - (-1.0)).abs() < 0.001);

        // Close at midpoint
        let clv = OptimismPessimismIndex::calculate_clv(110.0, 100.0, 105.0);
        assert!(clv.abs() < 0.001);
    }

    #[test]
    fn test_optimism_pessimism_bearish() {
        let opi = OptimismPessimismIndex::new(10);
        let data = create_bearish_data(25);
        let cumulative = opi.calculate_cumulative(&data);

        // In bearish data (closes near lows), cumulative should trend down
        assert!(cumulative.last().unwrap() < &0.0);
    }

    #[test]
    fn test_signal_generation() {
        let opi = OptimismPessimismIndex::new(5);
        let data = create_bullish_data(15);
        let signal = opi.signal(&data).unwrap();

        // Should not crash and return valid signal
        assert!(matches!(
            signal,
            IndicatorSignal::Bullish | IndicatorSignal::Bearish | IndicatorSignal::Neutral
        ));
    }
}
