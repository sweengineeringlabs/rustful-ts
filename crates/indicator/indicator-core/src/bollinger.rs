//! Bollinger Bands implementation.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_api::BollingerConfig;

/// Bollinger Bands.
///
/// Volatility indicator consisting of:
/// - Middle band: SMA of price
/// - Upper band: SMA + (std_dev * multiplier)
/// - Lower band: SMA - (std_dev * multiplier)
#[derive(Debug, Clone)]
pub struct BollingerBands {
    period: usize,
    std_dev: f64,
}

impl BollingerBands {
    pub fn new(period: usize, std_dev: f64) -> Self {
        Self { period, std_dev }
    }

    pub fn from_config(config: BollingerConfig) -> Self {
        Self {
            period: config.period,
            std_dev: config.std_dev,
        }
    }

    /// Calculate Bollinger Bands (middle, upper, lower).
    pub fn calculate(&self, data: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n < self.period || self.period == 0 {
            return (
                vec![f64::NAN; n],
                vec![f64::NAN; n],
                vec![f64::NAN; n],
            );
        }

        let mut middle = vec![f64::NAN; self.period - 1];
        let mut upper = vec![f64::NAN; self.period - 1];
        let mut lower = vec![f64::NAN; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let window = &data[start..=i];

            // Calculate SMA (middle band)
            let sma: f64 = window.iter().sum::<f64>() / self.period as f64;

            // Calculate standard deviation
            let variance: f64 = window.iter()
                .map(|x| (x - sma).powi(2))
                .sum::<f64>() / self.period as f64;
            let std = variance.sqrt();

            middle.push(sma);
            upper.push(sma + self.std_dev * std);
            lower.push(sma - self.std_dev * std);
        }

        (middle, upper, lower)
    }

    /// Calculate %B (position within bands).
    pub fn percent_b(&self, data: &[f64]) -> Vec<f64> {
        let (_middle, upper, lower) = self.calculate(data);
        data.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if u.is_nan() || l.is_nan() || (u - l).abs() < 1e-10 {
                    f64::NAN
                } else {
                    (price - l) / (u - l)
                }
            })
            .collect()
    }

    /// Calculate bandwidth (volatility measure).
    pub fn bandwidth(&self, data: &[f64]) -> Vec<f64> {
        let (middle, upper, lower) = self.calculate(data);
        middle.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&m, (&u, &l))| {
                if m.is_nan() || u.is_nan() || l.is_nan() || m.abs() < 1e-10 {
                    f64::NAN
                } else {
                    (u - l) / m
                }
            })
            .collect()
    }
}

impl TechnicalIndicator for BollingerBands {
    fn name(&self) -> &str {
        "BollingerBands"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let (middle, upper, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for BollingerBands {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (_, upper, lower) = self.calculate(&data.close);

        let n = data.close.len();
        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let price = data.close[n - 1];
        let u = upper[n - 1];
        let l = lower[n - 1];

        if u.is_nan() || l.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Price at/below lower band = potential buy (oversold)
        if price <= l {
            Ok(IndicatorSignal::Bullish)
        }
        // Price at/above upper band = potential sell (overbought)
        else if price >= u {
            Ok(IndicatorSignal::Bearish)
        }
        else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (_, upper, lower) = self.calculate(&data.close);

        let signals: Vec<_> = data.close.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if u.is_nan() || l.is_nan() {
                    IndicatorSignal::Neutral
                } else if price <= l {
                    IndicatorSignal::Bullish
                } else if price >= u {
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

    #[test]
    fn test_bollinger_bands() {
        let bb = BollingerBands::new(20, 2.0);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64).sin() * 5.0).collect();
        let (middle, upper, lower) = bb.calculate(&data);

        // Check bands exist after warmup
        for i in 19..50 {
            assert!(!middle[i].is_nan());
            assert!(upper[i] > middle[i]);
            assert!(lower[i] < middle[i]);
        }
    }
}
