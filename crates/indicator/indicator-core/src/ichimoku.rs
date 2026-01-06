//! Ichimoku Cloud implementation.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::IchimokuConfig;

/// Ichimoku Cloud output.
#[derive(Debug, Clone)]
pub struct IchimokuOutput {
    pub tenkan: Vec<f64>,     // Conversion Line
    pub kijun: Vec<f64>,      // Base Line
    pub senkou_a: Vec<f64>,   // Leading Span A
    pub senkou_b: Vec<f64>,   // Leading Span B
    pub chikou: Vec<f64>,     // Lagging Span
}

/// Ichimoku Cloud indicator.
///
/// Japanese charting technique showing support/resistance, momentum, and trend direction.
/// - Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
/// - Kijun-sen (Base Line): (26-period high + 26-period low) / 2
/// - Senkou Span A: (Tenkan + Kijun) / 2, plotted 26 periods ahead
/// - Senkou Span B: (52-period high + 52-period low) / 2, plotted 26 periods ahead
/// - Chikou Span: Close plotted 26 periods behind
#[derive(Debug, Clone)]
pub struct Ichimoku {
    tenkan_period: usize,
    kijun_period: usize,
    senkou_b_period: usize,
}

impl Ichimoku {
    pub fn new(tenkan: usize, kijun: usize, senkou_b: usize) -> Self {
        Self {
            tenkan_period: tenkan,
            kijun_period: kijun,
            senkou_b_period: senkou_b,
        }
    }

    pub fn from_config(config: IchimokuConfig) -> Self {
        Self {
            tenkan_period: config.tenkan_period,
            kijun_period: config.kijun_period,
            senkou_b_period: config.senkou_b_period,
        }
    }

    /// Calculate midpoint (highest high + lowest low) / 2 over period.
    fn calc_midpoint(high: &[f64], low: &[f64], start: usize, end: usize) -> f64 {
        let highest = high[start..=end].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lowest = low[start..=end].iter().cloned().fold(f64::INFINITY, f64::min);
        (highest + lowest) / 2.0
    }

    /// Calculate all Ichimoku components.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> IchimokuOutput {
        let n = high.len();

        let mut tenkan = vec![f64::NAN; n];
        let mut kijun = vec![f64::NAN; n];
        let mut senkou_a = vec![f64::NAN; n];
        let mut senkou_b = vec![f64::NAN; n];
        let mut chikou = vec![f64::NAN; n];

        // Tenkan-sen (Conversion Line)
        for i in (self.tenkan_period - 1)..n {
            tenkan[i] = Self::calc_midpoint(high, low, i + 1 - self.tenkan_period, i);
        }

        // Kijun-sen (Base Line)
        for i in (self.kijun_period - 1)..n {
            kijun[i] = Self::calc_midpoint(high, low, i + 1 - self.kijun_period, i);
        }

        // Senkou Span A = (Tenkan + Kijun) / 2
        for i in 0..n {
            if !tenkan[i].is_nan() && !kijun[i].is_nan() {
                senkou_a[i] = (tenkan[i] + kijun[i]) / 2.0;
            }
        }

        // Senkou Span B
        for i in (self.senkou_b_period - 1)..n {
            senkou_b[i] = Self::calc_midpoint(high, low, i + 1 - self.senkou_b_period, i);
        }

        // Chikou Span (Close shifted back by kijun_period)
        for i in 0..n {
            if i + self.kijun_period < n {
                chikou[i + self.kijun_period] = close[i];
            }
        }

        IchimokuOutput { tenkan, kijun, senkou_a, senkou_b, chikou }
    }
}

impl Default for Ichimoku {
    fn default() -> Self {
        Self::from_config(IchimokuConfig::default())
    }
}

impl TechnicalIndicator for Ichimoku {
    fn name(&self) -> &str {
        "Ichimoku"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.senkou_b_period;
        if data.high.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.high.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        // Return tenkan as primary, kijun as secondary, senkou_a as tertiary
        // (Full output available via calculate())
        Ok(IndicatorOutput::triple(result.tenkan, result.kijun, result.senkou_a))
    }

    fn min_periods(&self) -> usize {
        self.senkou_b_period
    }

    fn output_features(&self) -> usize {
        5 // tenkan, kijun, senkou_a, senkou_b, chikou
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ichimoku() {
        let ichimoku = Ichimoku::default();
        let n = 60;
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let result = ichimoku.calculate(&high, &low, &close);

        // Check tenkan starts at period - 1
        assert!(result.tenkan[7].is_nan());
        assert!(!result.tenkan[8].is_nan());

        // Check kijun starts at period - 1
        assert!(result.kijun[24].is_nan());
        assert!(!result.kijun[25].is_nan());
    }
}
