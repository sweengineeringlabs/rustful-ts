//! Smart Money Sentiment Indicators
//!
//! Indicators for measuring institutional and smart money behavior.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Smart Money Index Sentiment (IND-283) - Late day vs early day analysis
///
/// Based on the concept that smart money trades late in the day while
/// retail investors trade early. The Smart Money Index compares the
/// first 30 minutes of trading to the last hour.
#[derive(Debug, Clone)]
pub struct SmartMoneyIndexSentiment {
    period: usize,
    smoothing: usize,
}

/// Configuration for SmartMoneyIndexSentiment
#[derive(Debug, Clone)]
pub struct SmartMoneyIndexSentimentConfig {
    pub period: usize,
    pub smoothing: usize,
}

impl Default for SmartMoneyIndexSentimentConfig {
    fn default() -> Self {
        Self {
            period: 14,
            smoothing: 5,
        }
    }
}

impl SmartMoneyIndexSentiment {
    pub fn new(period: usize) -> Result<Self> {
        Self::with_config(SmartMoneyIndexSentimentConfig {
            period,
            ..Default::default()
        })
    }

    pub fn with_config(config: SmartMoneyIndexSentimentConfig) -> Result<Self> {
        if config.period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if config.smoothing < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "smoothing".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self {
            period: config.period,
            smoothing: config.smoothing,
        })
    }

    /// Calculate smart money index sentiment (-100 to 100)
    ///
    /// Uses intraday proxy: opening gap vs closing behavior
    /// Positive = smart money accumulating, Negative = smart money distributing
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());
        let mut raw_smi = vec![0.0; n];
        let mut result = vec![0.0; n];

        for i in 1..n {
            // Early day proxy: gap from previous close to open (retail reaction)
            let gap = if close[i - 1] > 0.0 {
                (open[i] - close[i - 1]) / close[i - 1] * 100.0
            } else {
                0.0
            };

            // Late day proxy: close position relative to day's range
            let range = high[i] - low[i];
            let close_position = if range > 0.0 {
                (close[i] - low[i]) / range * 2.0 - 1.0  // -1 to 1
            } else {
                0.0
            };

            // Smart money tends to:
            // 1. Not chase gaps (fade retail euphoria/panic)
            // 2. Accumulate late in day on down days
            // 3. Distribute late in day on up days

            // If gap is positive but close weak = smart money selling into retail buying
            // If gap is negative but close strong = smart money buying retail panic
            let smart_money_action = close_position - (gap * 0.1).clamp(-1.0, 1.0);

            // Volume confirmation
            let vol_factor = if i >= self.period {
                let avg_vol: f64 = volume[(i - self.period)..i].iter().sum::<f64>() / self.period as f64;
                if avg_vol > 0.0 {
                    (volume[i] / avg_vol).min(2.0).max(0.5)
                } else {
                    1.0
                }
            } else {
                1.0
            };

            raw_smi[i] = smart_money_action * vol_factor * 50.0;
        }

        // Apply smoothing
        for i in self.smoothing..n {
            let sum: f64 = raw_smi[(i - self.smoothing + 1)..=i].iter().sum();
            result[i] = (sum / self.smoothing as f64).clamp(-100.0, 100.0);
        }

        result
    }

    /// Calculate cumulative smart money flow
    pub fn calculate_cumulative(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let smi = self.calculate(open, high, low, close, volume);
        let mut cumulative = vec![0.0; smi.len()];

        for i in 1..smi.len() {
            cumulative[i] = cumulative[i - 1] + smi[i];
        }

        cumulative
    }
}

impl TechnicalIndicator for SmartMoneyIndexSentiment {
    fn name(&self) -> &str {
        "Smart Money Index Sentiment"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smoothing
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let smi = self.calculate(&data.open, &data.high, &data.low, &data.close, &data.volume);
        let cumulative = self.calculate_cumulative(&data.open, &data.high, &data.low, &data.close, &data.volume);

        Ok(IndicatorOutput::dual(smi, cumulative))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let open = vec![100.0, 104.5, 105.5, 106.5, 105.5, 107.5, 108.5, 107.5, 109.5, 108.5,
                       110.5, 109.5, 111.5, 110.5, 112.5, 111.5, 113.5, 112.5, 114.5, 113.5];
        let high = vec![105.0, 106.0, 107.0, 106.5, 108.0, 109.0, 108.5, 110.0, 109.5, 111.0,
                       110.5, 112.0, 111.5, 113.0, 112.5, 114.0, 113.5, 115.0, 114.5, 116.0];
        let low = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 104.5, 106.0,
                      105.5, 107.0, 106.5, 108.0, 107.5, 109.0, 108.5, 110.0, 109.5, 111.0];
        let close = vec![104.0, 105.0, 106.0, 105.0, 107.0, 108.0, 107.0, 109.0, 108.0, 110.0,
                        109.0, 111.0, 110.0, 112.0, 111.0, 113.0, 112.0, 114.0, 113.0, 115.0];
        let volume = vec![1000.0, 1200.0, 1100.0, 1300.0, 1500.0, 1400.0, 1600.0, 1800.0, 1700.0, 1900.0,
                         2000.0, 1800.0, 2100.0, 2200.0, 2000.0, 2300.0, 2100.0, 2400.0, 2200.0, 2500.0];
        (open, high, low, close, volume)
    }

    #[test]
    fn test_smart_money_index_sentiment_creation() {
        let indicator = SmartMoneyIndexSentiment::new(14);
        assert!(indicator.is_ok());

        let indicator = SmartMoneyIndexSentiment::new(3);
        assert!(indicator.is_err());
    }

    #[test]
    fn test_smart_money_index_sentiment_calculation() {
        let (open, high, low, close, volume) = make_test_data();
        let indicator = SmartMoneyIndexSentiment::new(10).unwrap();
        let result = indicator.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for i in 15..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_smart_money_index_cumulative() {
        let (open, high, low, close, volume) = make_test_data();
        let indicator = SmartMoneyIndexSentiment::new(10).unwrap();
        let result = indicator.calculate_cumulative(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        // Cumulative should be monotonic within periods
    }

    #[test]
    fn test_smart_money_index_with_config() {
        let config = SmartMoneyIndexSentimentConfig {
            period: 14,
            smoothing: 3,
        };
        let indicator = SmartMoneyIndexSentiment::with_config(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_smart_money_index_min_periods() {
        let indicator = SmartMoneyIndexSentiment::new(14).unwrap();
        assert_eq!(indicator.min_periods(), 19);
    }
}
