//! Mass Index implementation.
//!
//! Identifies trend reversals based on range expansion.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Mass Index.
///
/// Identifies trend reversals by measuring the narrowing and widening of
/// the trading range (high - low). A "reversal bulge" occurs when Mass Index
/// rises above 27 and then falls below 26.5.
///
/// Formula:
/// 1. Calculate single EMA of (high - low)
/// 2. Calculate double EMA (EMA of EMA)
/// 3. Calculate EMA ratio = single EMA / double EMA
/// 4. Mass Index = sum of EMA ratios over period
#[derive(Debug, Clone)]
pub struct MassIndex {
    /// EMA period (typically 9).
    ema_period: usize,
    /// Sum period (typically 25).
    sum_period: usize,
    /// Bulge threshold (typically 27).
    bulge_threshold: f64,
    /// Trigger threshold (typically 26.5).
    trigger_threshold: f64,
}

impl MassIndex {
    /// Create a new Mass Index indicator.
    ///
    /// # Arguments
    /// * `ema_period` - Period for EMA calculations (default: 9)
    /// * `sum_period` - Period for summing ratios (default: 25)
    pub fn new(ema_period: usize, sum_period: usize) -> Self {
        Self {
            ema_period,
            sum_period,
            bulge_threshold: 27.0,
            trigger_threshold: 26.5,
        }
    }

    /// Create with default parameters (9, 25).
    pub fn default_params() -> Self {
        Self::new(9, 25)
    }

    /// Set custom bulge and trigger thresholds.
    pub fn with_thresholds(mut self, bulge: f64, trigger: f64) -> Self {
        self.bulge_threshold = bulge;
        self.trigger_threshold = trigger;
        self
    }

    /// Calculate EMA of a series, properly handling leading NaN values.
    fn ema(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period || period == 0 {
            return vec![f64::NAN; n];
        }

        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut result = vec![f64::NAN; n];

        // Find first valid index
        let first_valid = data.iter().position(|x| !x.is_nan());
        if first_valid.is_none() {
            return result;
        }
        let start = first_valid.unwrap();

        if start + period > n {
            return result;
        }

        // Initial EMA is SMA of first `period` valid values
        let initial_sma: f64 = data[start..(start + period)].iter().sum::<f64>() / period as f64;
        result[start + period - 1] = initial_sma;

        let mut prev_ema = initial_sma;
        for i in (start + period)..n {
            let ema = (data[i] - prev_ema) * multiplier + prev_ema;
            result[i] = ema;
            prev_ema = ema;
        }

        result
    }

    /// Calculate Mass Index values.
    pub fn calculate(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        // Need enough data for: 2 EMAs + sum period
        // First valid EMA ratio at index 2*(ema_period-1), then need sum_period values
        let min_len = 2 * (self.ema_period - 1) + self.sum_period;

        if n < min_len || self.ema_period == 0 || self.sum_period == 0 {
            return vec![f64::NAN; n];
        }

        // Calculate H-L spread
        let hl_spread: Vec<f64> = high.iter()
            .zip(low.iter())
            .map(|(&h, &l)| h - l)
            .collect();

        // Calculate single EMA
        let single_ema = Self::ema(&hl_spread, self.ema_period);

        // Calculate double EMA (filter out NaN for second EMA)
        let double_ema = Self::ema(&single_ema, self.ema_period);

        // Calculate EMA ratio
        let ema_ratio: Vec<f64> = single_ema.iter()
            .zip(double_ema.iter())
            .map(|(&single, &double)| {
                if single.is_nan() || double.is_nan() || double.abs() < 1e-10 {
                    f64::NAN
                } else {
                    single / double
                }
            })
            .collect();

        // Calculate Mass Index as sum of ratios
        // First valid EMA ratio is at index 2*(ema_period-1) = 2*ema_period - 2
        // Need sum_period valid ratios, so first valid result at 2*ema_period - 2 + sum_period - 1
        let first_valid_ratio = 2 * (self.ema_period - 1);
        let warmup = first_valid_ratio + self.sum_period - 1;
        let mut result = vec![f64::NAN; warmup];

        for i in warmup..n {
            let start = i + 1 - self.sum_period;
            let window = &ema_ratio[start..=i];

            // Check if we have enough valid values
            let valid_values: Vec<f64> = window.iter()
                .filter(|x| !x.is_nan())
                .copied()
                .collect();

            if valid_values.len() >= self.sum_period {
                let sum: f64 = valid_values.iter().sum();
                result.push(sum);
            } else {
                result.push(f64::NAN);
            }
        }

        result
    }
}

impl TechnicalIndicator for MassIndex {
    fn name(&self) -> &str {
        "MassIndex"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_len = 2 * (self.ema_period - 1) + self.sum_period;
        if data.high.len() < min_len {
            return Err(IndicatorError::InsufficientData {
                required: min_len,
                got: data.high.len(),
            });
        }

        let values = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        2 * (self.ema_period - 1) + self.sum_period
    }
}

impl SignalIndicator for MassIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.high, &data.low);
        let n = values.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let current = values[n - 1];
        let prev = values[n - 2];

        if current.is_nan() || prev.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Reversal bulge: crossed above bulge threshold, now below trigger
        if prev >= self.bulge_threshold && current < self.trigger_threshold {
            // This is a reversal signal - direction depends on prior trend
            // For simplicity, we return Bullish as a generic reversal signal
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..values.len() {
            let current = values[i];
            let prev = values[i - 1];

            if current.is_nan() || prev.is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if prev >= self.bulge_threshold && current < self.trigger_threshold {
                signals.push(IndicatorSignal::Bullish);
            } else {
                signals.push(IndicatorSignal::Neutral);
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_index() {
        let mi = MassIndex::new(9, 25);

        // Generate sample data
        let high: Vec<f64> = (0..100)
            .map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();
        let low: Vec<f64> = (0..100)
            .map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();

        let result = mi.calculate(&high, &low);

        assert_eq!(result.len(), 100);

        // Check warmup period has NaN
        // First valid EMA ratio at 2*(9-1) = 16, then need 25 more for sum = 40
        let warmup = 2 * (9 - 1) + 25 - 1; // = 40
        for i in 0..warmup {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        }

        // Valid values should be around 25 (sum of ~1.0 ratios over 25 periods)
        for i in warmup..100 {
            assert!(!result[i].is_nan(), "Expected valid value at index {}", i);
            assert!(result[i] > 20.0 && result[i] < 35.0);
        }
    }

    #[test]
    fn test_mass_index_default() {
        let mi = MassIndex::default_params();
        assert_eq!(mi.ema_period, 9);
        assert_eq!(mi.sum_period, 25);
    }
}
