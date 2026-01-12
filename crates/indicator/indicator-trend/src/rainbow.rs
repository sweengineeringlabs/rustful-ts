//! Rainbow Moving Average.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Rainbow Moving Average - IND-077, IND-168
///
/// 10 recursive SMAs for trend visualization.
/// Each level is SMA of the previous level.
#[derive(Debug, Clone)]
pub struct RainbowMA {
    period: usize,
    levels: usize,
}

impl RainbowMA {
    pub fn new(period: usize, levels: usize) -> Self {
        Self { period, levels }
    }

    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().filter(|x| !x.is_nan()).sum();
        let count = data[..period].iter().filter(|x| !x.is_nan()).count();

        if count > 0 {
            result.push(sum / count as f64);
        } else {
            result.push(f64::NAN);
        }

        for i in period..n {
            let old = if data[i - period].is_nan() { 0.0 } else { data[i - period] };
            let new = if data[i].is_nan() { 0.0 } else { data[i] };
            sum = sum - old + new;
            result.push(sum / period as f64);
        }

        result
    }

    pub fn calculate(&self, data: &[f64]) -> Vec<Vec<f64>> {
        let n = data.len();
        let mut levels = Vec::with_capacity(self.levels);

        // First level is SMA of price
        let mut current = Self::sma(data, self.period);
        levels.push(current.clone());

        // Each subsequent level is SMA of previous level
        for _ in 1..self.levels {
            current = Self::sma(&current, self.period);
            levels.push(current.clone());
        }

        levels
    }

    /// Calculate average of all rainbow levels.
    pub fn average(&self, data: &[f64]) -> Vec<f64> {
        let levels = self.calculate(data);
        let n = data.len();

        (0..n)
            .map(|i| {
                let valid: Vec<f64> = levels.iter()
                    .filter_map(|level| {
                        if level[i].is_nan() { None } else { Some(level[i]) }
                    })
                    .collect();

                if valid.is_empty() {
                    f64::NAN
                } else {
                    valid.iter().sum::<f64>() / valid.len() as f64
                }
            })
            .collect()
    }
}

impl Default for RainbowMA {
    fn default() -> Self {
        Self::new(2, 10)
    }
}

impl TechnicalIndicator for RainbowMA {
    fn name(&self) -> &str {
        "RainbowMA"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period * self.levels {
            return Err(IndicatorError::InsufficientData {
                required: self.period * self.levels,
                got: data.close.len(),
            });
        }

        // Return the average as primary output
        let avg = self.average(&data.close);
        Ok(IndicatorOutput::single(avg))
    }

    fn min_periods(&self) -> usize {
        self.period * self.levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rainbow_basic() {
        let rainbow = RainbowMA::new(2, 10);
        let data: Vec<f64> = (0..50).map(|i| 100.0 + i as f64).collect();
        let levels = rainbow.calculate(&data);

        assert_eq!(levels.len(), 10);

        // Each level should have same length as input
        for level in levels.iter() {
            assert_eq!(level.len(), 50);
        }
    }

    #[test]
    fn test_rainbow_average() {
        let rainbow = RainbowMA::new(2, 5);
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        let avg = rainbow.average(&data);

        assert_eq!(avg.len(), 30);
    }
}
