//! Insider Trading Indicators
//!
//! Indicators for analyzing insider trading behavior proxies.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Insider Trading Ratio (IND-286) - Buy/sell ratio proxy
///
/// This indicator creates a proxy for insider trading patterns using
/// volume and price action that often correlate with informed trading.
#[derive(Debug, Clone)]
pub struct InsiderTradingRatio {
    period: usize,
    lookback: usize,
}

/// Configuration for InsiderTradingRatio
#[derive(Debug, Clone)]
pub struct InsiderTradingRatioConfig {
    pub period: usize,
    pub lookback: usize,
}

impl Default for InsiderTradingRatioConfig {
    fn default() -> Self {
        Self {
            period: 14,
            lookback: 90,  // Quarterly lookback for insider patterns
        }
    }
}

impl InsiderTradingRatio {
    pub fn new(period: usize, lookback: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if lookback < period {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least equal to period".to_string(),
            });
        }
        Ok(Self { period, lookback })
    }

    pub fn with_config(config: InsiderTradingRatioConfig) -> Result<Self> {
        Self::new(config.period, config.lookback)
    }

    /// Detect accumulation patterns (insider buying proxy)
    ///
    /// Returns score 0-100, higher = more accumulation signals
    pub fn calculate_accumulation(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);
            let period_start = i.saturating_sub(self.period);
            let mut accumulation_score = 0.0;

            // Pattern 1: Quiet accumulation (price stable, low volume upticks)
            let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / (i - start).max(1) as f64;
            let recent_avg_volume: f64 = volume[period_start..=i].iter().sum::<f64>() / (i - period_start + 1) as f64;

            // Insiders often accumulate on below-average volume
            if avg_volume > 0.0 && recent_avg_volume < avg_volume * 0.8 {
                // Check for subtle price support
                let min_low = low[period_start..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_low = low[period_start..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let low_range = max_low - min_low;
                let avg_price = close[period_start..=i].iter().sum::<f64>() / (i - period_start + 1) as f64;

                // Tight lows with quiet volume = accumulation
                if low_range > 0.0 && avg_price > 0.0 && (low_range / avg_price) < 0.03 {
                    accumulation_score += 30.0;
                }
            }

            // Pattern 2: Buying on weakness (insider confidence)
            let mut weak_day_buys = 0;
            for j in period_start..=i {
                let is_down_day = close[j] < open[j];
                let range = high[j] - low[j];
                if range > 0.0 && is_down_day {
                    // Strong close on down day (buying the dip)
                    let close_position = (close[j] - low[j]) / range;
                    if close_position > 0.6 {
                        weak_day_buys += 1;
                    }
                }
            }
            accumulation_score += (weak_day_buys as f64 / (i - period_start + 1) as f64) * 35.0;

            // Pattern 3: Pre-breakout accumulation
            let max_high = high[start..period_start].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let recent_high = high[period_start..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            // Close approaching but not breaking resistance with volume dropping
            if max_high > 0.0 && close[i] > max_high * 0.95 && close[i] < max_high * 1.02 {
                if avg_volume > 0.0 && volume[i] < avg_volume {
                    accumulation_score += 35.0;
                }
            }

            result[i] = accumulation_score.clamp(0.0, 100.0);
        }
        result
    }

    /// Detect distribution patterns (insider selling proxy)
    ///
    /// Returns score 0-100, higher = more distribution signals
    pub fn calculate_distribution(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(open.len()).min(high.len()).min(low.len()).min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);
            let period_start = i.saturating_sub(self.period);
            let mut distribution_score = 0.0;

            let avg_volume: f64 = volume[start..i].iter().sum::<f64>() / (i - start).max(1) as f64;

            // Pattern 1: High volume with no upside progress (selling into strength)
            let recent_avg_volume: f64 = volume[period_start..=i].iter().sum::<f64>() / (i - period_start + 1) as f64;

            if avg_volume > 0.0 && recent_avg_volume > avg_volume * 1.3 {
                // High volume but price going nowhere
                let price_change = if close[period_start] > 0.0 {
                    (close[i] / close[period_start] - 1.0).abs() * 100.0
                } else {
                    0.0
                };

                if price_change < 2.0 {
                    distribution_score += 30.0;
                }
            }

            // Pattern 2: Selling on strength (insider exiting)
            let mut strong_day_sells = 0;
            for j in period_start..=i {
                let is_up_day = close[j] > open[j];
                let range = high[j] - low[j];
                if range > 0.0 && is_up_day {
                    // Weak close on up day (selling the rip)
                    let close_position = (close[j] - low[j]) / range;
                    if close_position < 0.4 {
                        strong_day_sells += 1;
                    }
                }
            }
            distribution_score += (strong_day_sells as f64 / (i - period_start + 1) as f64) * 35.0;

            // Pattern 3: Post-high distribution
            let max_high = high[start..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let days_since_high = {
                let mut days = 0;
                for j in (start..=i).rev() {
                    if (high[j] - max_high).abs() < 0.001 {
                        break;
                    }
                    days += 1;
                }
                days
            };

            // If we're past the high and volume is elevated
            if days_since_high > 5 && days_since_high < self.period {
                if avg_volume > 0.0 && volume[i] > avg_volume * 1.2 {
                    distribution_score += 35.0;
                }
            }

            result[i] = distribution_score.clamp(0.0, 100.0);
        }
        result
    }

    /// Calculate insider trading ratio (-100 to 100)
    ///
    /// Positive = Net buying signals, Negative = Net selling signals
    pub fn calculate(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let accumulation = self.calculate_accumulation(open, high, low, close, volume);
        let distribution = self.calculate_distribution(open, high, low, close, volume);

        let n = accumulation.len().min(distribution.len());
        let mut result = vec![0.0; n];

        for i in 0..n {
            result[i] = (accumulation[i] - distribution[i]).clamp(-100.0, 100.0);
        }
        result
    }

    /// Calculate insider activity level (0-100)
    ///
    /// Higher values indicate more detected insider-like activity
    pub fn calculate_activity(&self, open: &[f64], high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let accumulation = self.calculate_accumulation(open, high, low, close, volume);
        let distribution = self.calculate_distribution(open, high, low, close, volume);

        let n = accumulation.len().min(distribution.len());
        let mut result = vec![0.0; n];

        for i in 0..n {
            result[i] = ((accumulation[i] + distribution[i]) / 2.0).clamp(0.0, 100.0);
        }
        result
    }
}

impl TechnicalIndicator for InsiderTradingRatio {
    fn name(&self) -> &str {
        "Insider Trading Ratio"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let accumulation = self.calculate_accumulation(&data.open, &data.high, &data.low, &data.close, &data.volume);
        let distribution = self.calculate_distribution(&data.open, &data.high, &data.low, &data.close, &data.volume);
        let ratio = self.calculate(&data.open, &data.high, &data.low, &data.close, &data.volume);
        let activity = self.calculate_activity(&data.open, &data.high, &data.low, &data.close, &data.volume);

        Ok(IndicatorOutput::triple(accumulation, distribution, ratio))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create test data long enough for lookback
        let mut open = Vec::with_capacity(100);
        let mut high = Vec::with_capacity(100);
        let mut low = Vec::with_capacity(100);
        let mut close = Vec::with_capacity(100);
        let mut volume = Vec::with_capacity(100);

        for i in 0..100 {
            let base = 100.0 + (i as f64 * 0.3);
            open.push(base);
            high.push(base + 3.0);
            low.push(base - 1.0);
            close.push(base + 1.5);
            volume.push(1000.0 + (i as f64 * 20.0));
        }

        (open, high, low, close, volume)
    }

    #[test]
    fn test_insider_trading_ratio_creation() {
        let indicator = InsiderTradingRatio::new(14, 90);
        assert!(indicator.is_ok());

        let indicator = InsiderTradingRatio::new(3, 90);
        assert!(indicator.is_err());

        let indicator = InsiderTradingRatio::new(100, 50);
        assert!(indicator.is_err());
    }

    #[test]
    fn test_insider_trading_accumulation() {
        let (open, high, low, close, volume) = make_test_data();
        let indicator = InsiderTradingRatio::new(14, 90).unwrap();
        let result = indicator.calculate_accumulation(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for i in 91..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_insider_trading_distribution() {
        let (open, high, low, close, volume) = make_test_data();
        let indicator = InsiderTradingRatio::new(14, 90).unwrap();
        let result = indicator.calculate_distribution(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for i in 91..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_insider_trading_ratio() {
        let (open, high, low, close, volume) = make_test_data();
        let indicator = InsiderTradingRatio::new(14, 90).unwrap();
        let result = indicator.calculate(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for i in 91..result.len() {
            assert!(result[i] >= -100.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_insider_trading_activity() {
        let (open, high, low, close, volume) = make_test_data();
        let indicator = InsiderTradingRatio::new(14, 90).unwrap();
        let result = indicator.calculate_activity(&open, &high, &low, &close, &volume);

        assert_eq!(result.len(), close.len());
        for i in 91..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0, "Value at {} is {}", i, result[i]);
        }
    }

    #[test]
    fn test_insider_trading_with_config() {
        let config = InsiderTradingRatioConfig {
            period: 14,
            lookback: 60,
        };
        let indicator = InsiderTradingRatio::with_config(config);
        assert!(indicator.is_ok());
    }

    #[test]
    fn test_insider_trading_min_periods() {
        let indicator = InsiderTradingRatio::new(14, 90).unwrap();
        assert_eq!(indicator.min_periods(), 91);
    }
}
