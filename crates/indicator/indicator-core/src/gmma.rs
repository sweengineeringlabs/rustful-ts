//! Guppy Multiple Moving Average (GMMA) implementation.
//!
//! A set of 12 EMAs used for trend identification and trading signals.

use indicator_spi::{TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries};
use indicator_api::GMMAConfig;

/// Guppy Multiple Moving Average (GMMA).
///
/// GMMA consists of two groups of exponential moving averages:
/// - Short-term group (traders): 3, 5, 8, 10, 12, 15 periods
/// - Long-term group (investors): 30, 35, 40, 45, 50, 60 periods
///
/// The relationship and separation between these groups indicates
/// trend strength and potential trend changes.
#[derive(Debug, Clone)]
pub struct GMMA {
    /// Short-term EMA periods (default: 3, 5, 8, 10, 12, 15)
    short_periods: Vec<usize>,
    /// Long-term EMA periods (default: 30, 35, 40, 45, 50, 60)
    long_periods: Vec<usize>,
}

/// Output structure for GMMA containing all 12 EMA series.
#[derive(Debug, Clone)]
pub struct GMMAOutput {
    /// Short-term EMAs (6 series)
    pub short_emas: Vec<Vec<f64>>,
    /// Long-term EMAs (6 series)
    pub long_emas: Vec<Vec<f64>>,
}

impl GMMA {
    pub fn new(short_periods: Vec<usize>, long_periods: Vec<usize>) -> Self {
        Self { short_periods, long_periods }
    }

    pub fn from_config(config: GMMAConfig) -> Self {
        Self {
            short_periods: config.short_periods,
            long_periods: config.long_periods,
        }
    }

    /// Calculate all GMMA EMAs.
    pub fn calculate(&self, data: &[f64]) -> GMMAOutput {
        let short_emas: Vec<Vec<f64>> = self.short_periods
            .iter()
            .map(|&period| self.ema(data, period))
            .collect();

        let long_emas: Vec<Vec<f64>> = self.long_periods
            .iter()
            .map(|&period| self.ema(data, period))
            .collect();

        GMMAOutput { short_emas, long_emas }
    }

    /// Calculate primary output (average of short-term EMAs).
    pub fn calculate_primary(&self, data: &[f64]) -> Vec<f64> {
        let output = self.calculate(data);
        self.average_emas(&output.short_emas)
    }

    /// Calculate average of a set of EMAs.
    fn average_emas(&self, emas: &[Vec<f64>]) -> Vec<f64> {
        if emas.is_empty() || emas[0].is_empty() {
            return Vec::new();
        }

        let len = emas[0].len();

        (0..len)
            .map(|i| {
                let sum: f64 = emas.iter()
                    .map(|ema| if ema[i].is_nan() { 0.0 } else { ema[i] })
                    .sum();
                let valid_count = emas.iter()
                    .filter(|ema| !ema[i].is_nan())
                    .count() as f64;
                if valid_count > 0.0 {
                    sum / valid_count
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// Calculate EMA for a given period.
    fn ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        if data.len() < period || period == 0 {
            return vec![f64::NAN; data.len()];
        }

        let alpha = 2.0 / (period as f64 + 1.0);
        let mut result = vec![f64::NAN; period - 1];

        // Initial SMA as seed
        let initial_sma: f64 = data[0..period].iter().sum::<f64>() / period as f64;
        result.push(initial_sma);

        let mut ema = initial_sma;
        for i in period..data.len() {
            if data[i].is_nan() {
                result.push(f64::NAN);
            } else {
                ema = alpha * data[i] + (1.0 - alpha) * ema;
                result.push(ema);
            }
        }

        result
    }

    /// Get maximum period (for minimum data requirement).
    pub fn max_period(&self) -> usize {
        self.short_periods.iter().chain(self.long_periods.iter())
            .cloned()
            .max()
            .unwrap_or(60)
    }

    /// Calculate trend strength based on EMA separation.
    pub fn trend_strength(&self, data: &[f64]) -> Vec<f64> {
        let output = self.calculate(data);
        let short_avg = self.average_emas(&output.short_emas);
        let long_avg = self.average_emas(&output.long_emas);

        short_avg.iter()
            .zip(long_avg.iter())
            .map(|(&s, &l)| {
                if s.is_nan() || l.is_nan() || l == 0.0 {
                    f64::NAN
                } else {
                    ((s - l) / l) * 100.0
                }
            })
            .collect()
    }
}

impl Default for GMMA {
    fn default() -> Self {
        Self::from_config(GMMAConfig::default())
    }
}

impl TechnicalIndicator for GMMA {
    fn name(&self) -> &str {
        "GMMA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let max_period = self.max_period();
        if data.close.len() < max_period {
            return Err(IndicatorError::InsufficientData {
                required: max_period,
                got: data.close.len(),
            });
        }

        // Return average of short-term EMAs as primary, long-term as secondary
        let output = self.calculate(&data.close);
        let short_avg = self.average_emas(&output.short_emas);
        let long_avg = self.average_emas(&output.long_emas);

        Ok(IndicatorOutput::dual(short_avg, long_avg))
    }

    fn min_periods(&self) -> usize {
        self.max_period()
    }

    fn output_features(&self) -> usize {
        2 // Short and long averages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gmma() {
        let gmma = GMMA::default();
        let data: Vec<f64> = (0..100).map(|i| 100.0 + i as f64).collect();
        let output = gmma.calculate(&data);

        // Should have 6 short-term EMAs
        assert_eq!(output.short_emas.len(), 6);
        // Should have 6 long-term EMAs
        assert_eq!(output.long_emas.len(), 6);

        // Later values should be valid
        assert!(!output.short_emas[0][99].is_nan());
        assert!(!output.long_emas[5][99].is_nan());
    }

    #[test]
    fn test_gmma_trend_strength() {
        let gmma = GMMA::default();
        // Strong uptrend
        let data: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 2.0).collect();
        let trend = gmma.trend_strength(&data);

        // In uptrend, short EMAs should be above long EMAs (positive trend)
        assert!(trend[99] > 0.0);
    }

    #[test]
    fn test_gmma_periods() {
        let gmma = GMMA::default();
        assert_eq!(gmma.short_periods, vec![3, 5, 8, 10, 12, 15]);
        assert_eq!(gmma.long_periods, vec![30, 35, 40, 45, 50, 60]);
        assert_eq!(gmma.max_period(), 60);
    }

    #[test]
    fn test_gmma_insufficient_data() {
        let gmma = GMMA::default();
        let data = vec![1.0, 2.0, 3.0];
        let output = gmma.calculate(&data);

        assert_eq!(output.short_emas[0].len(), 3);
        // All values should be NaN for long-term EMAs
        assert!(output.long_emas[5].iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_gmma_technical_indicator_trait() {
        let gmma = GMMA::default();
        assert_eq!(gmma.name(), "GMMA");
        assert_eq!(gmma.min_periods(), 60);
        assert_eq!(gmma.output_features(), 2);
    }

    #[test]
    fn test_gmma_custom_periods() {
        let gmma = GMMA::new(
            vec![2, 4, 6],
            vec![10, 15, 20],
        );

        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let output = gmma.calculate(&data);

        assert_eq!(output.short_emas.len(), 3);
        assert_eq!(output.long_emas.len(), 3);
        assert_eq!(gmma.max_period(), 20);
    }
}
