//! Intraday Seasonality Indicator (IND-243)
//!
//! Analyzes hour-of-day patterns in market behavior.
//! Markets often exhibit predictable intraday patterns.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Trading hour classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TradingHour {
    /// Pre-market (4:00-9:30 AM)
    PreMarket,
    /// Opening hour (9:30-10:30 AM)
    Opening,
    /// Morning session (10:30 AM-12:00 PM)
    Morning,
    /// Lunch hour (12:00-1:00 PM)
    Lunch,
    /// Afternoon session (1:00-3:00 PM)
    Afternoon,
    /// Power hour (3:00-4:00 PM)
    PowerHour,
    /// After hours (4:00-8:00 PM)
    AfterHours,
}

/// Configuration for the Intraday Seasonality indicator.
#[derive(Debug, Clone)]
pub struct IntradaySeasonalityConfig {
    /// Number of historical days to analyze
    pub lookback_days: usize,
    /// Number of bars per day (assumes regular intervals)
    pub bars_per_day: usize,
    /// Smoothing period for patterns
    pub smoothing: usize,
}

impl Default for IntradaySeasonalityConfig {
    fn default() -> Self {
        Self {
            lookback_days: 20,
            bars_per_day: 78, // 5-minute bars for regular trading hours
            smoothing: 5,
        }
    }
}

/// Intraday Seasonality indicator for hour-of-day pattern analysis.
///
/// This indicator identifies:
/// - Opening range patterns
/// - Lunch hour volatility contraction
/// - Power hour momentum
/// - Overnight gap behavior
#[derive(Debug, Clone)]
pub struct IntradaySeasonality {
    config: IntradaySeasonalityConfig,
}

impl IntradaySeasonality {
    /// Create a new Intraday Seasonality indicator.
    pub fn new() -> Self {
        Self {
            config: IntradaySeasonalityConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: IntradaySeasonalityConfig) -> Self {
        Self { config }
    }

    /// Estimate the trading hour from bar index within a day.
    fn estimate_trading_hour(&self, bar_in_day: usize) -> TradingHour {
        // Assuming regular trading hours (9:30 AM - 4:00 PM = 6.5 hours)
        // With pre-market and after-hours
        let total_bars = self.config.bars_per_day;
        let fraction = bar_in_day as f64 / total_bars as f64;

        // Map fraction to trading hours
        if fraction < 0.1 {
            TradingHour::Opening // First 10% is opening
        } else if fraction < 0.35 {
            TradingHour::Morning // 10-35% is morning
        } else if fraction < 0.5 {
            TradingHour::Lunch // 35-50% is lunch
        } else if fraction < 0.85 {
            TradingHour::Afternoon // 50-85% is afternoon
        } else {
            TradingHour::PowerHour // Last 15% is power hour
        }
    }

    /// Calculate average return by bar position within day.
    fn calculate_bar_returns(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let bars_per_day = self.config.bars_per_day;
        let mut bar_returns = vec![0.0; bars_per_day];
        let mut bar_counts = vec![0usize; bars_per_day];

        // Calculate returns for each bar position
        for i in 1..n {
            let bar_in_day = i % bars_per_day;
            if close[i - 1] > 0.0 {
                let ret = (close[i] - close[i - 1]) / close[i - 1];
                bar_returns[bar_in_day] += ret;
                bar_counts[bar_in_day] += 1;
            }
        }

        // Average returns
        for i in 0..bars_per_day {
            if bar_counts[i] > 0 {
                bar_returns[i] /= bar_counts[i] as f64;
            }
        }

        bar_returns
    }

    /// Calculate volatility by bar position within day.
    fn calculate_bar_volatility(&self, high: &[f64], low: &[f64]) -> Vec<f64> {
        let n = high.len();
        let bars_per_day = self.config.bars_per_day;
        let mut bar_volatility = vec![0.0; bars_per_day];
        let mut bar_counts = vec![0usize; bars_per_day];

        // Calculate range for each bar position
        for i in 0..n {
            let bar_in_day = i % bars_per_day;
            if low[i] > 0.0 {
                let range = (high[i] - low[i]) / low[i];
                bar_volatility[bar_in_day] += range;
                bar_counts[bar_in_day] += 1;
            }
        }

        // Average volatility
        for i in 0..bars_per_day {
            if bar_counts[i] > 0 {
                bar_volatility[i] /= bar_counts[i] as f64;
            }
        }

        bar_volatility
    }

    /// Calculate volume by bar position within day.
    fn calculate_bar_volume(&self, volume: &[f64]) -> Vec<f64> {
        let n = volume.len();
        let bars_per_day = self.config.bars_per_day;
        let mut bar_volume = vec![0.0; bars_per_day];
        let mut bar_counts = vec![0usize; bars_per_day];

        // Calculate average volume for each bar position
        for i in 0..n {
            let bar_in_day = i % bars_per_day;
            bar_volume[bar_in_day] += volume[i];
            bar_counts[bar_in_day] += 1;
        }

        // Average volume
        for i in 0..bars_per_day {
            if bar_counts[i] > 0 {
                bar_volume[i] /= bar_counts[i] as f64;
            }
        }

        bar_volume
    }

    /// Calculate the intraday seasonality indicators.
    pub fn calculate(
        &self,
        close: &[f64],
        high: &[f64],
        low: &[f64],
        volume: &[f64],
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<TradingHour>) {
        let n = close.len();
        let bars_per_day = self.config.bars_per_day;

        let bar_returns = self.calculate_bar_returns(close);
        let bar_volatility = self.calculate_bar_volatility(high, low);
        let bar_volume = self.calculate_bar_volume(volume);

        // Normalize bar volume to get relative volume
        let avg_volume: f64 = bar_volume.iter().sum::<f64>() / bars_per_day as f64;

        let mut return_bias = vec![f64::NAN; n];
        let mut volatility_pattern = vec![f64::NAN; n];
        let mut volume_pattern = vec![f64::NAN; n];
        let mut trading_hour = vec![TradingHour::Opening; n];

        for i in 0..n {
            let bar_in_day = i % bars_per_day;
            trading_hour[i] = self.estimate_trading_hour(bar_in_day);

            if i >= bars_per_day * self.config.lookback_days {
                return_bias[i] = bar_returns[bar_in_day] * 10000.0; // Convert to basis points
                volatility_pattern[i] = bar_volatility[bar_in_day] * 100.0; // Percentage
                volume_pattern[i] = if avg_volume > 0.0 {
                    bar_volume[bar_in_day] / avg_volume
                } else {
                    1.0
                };
            }
        }

        // Apply smoothing
        if self.config.smoothing > 1 {
            return_bias = self.smooth(&return_bias);
            volatility_pattern = self.smooth(&volatility_pattern);
            volume_pattern = self.smooth(&volume_pattern);
        }

        (return_bias, volatility_pattern, volume_pattern, trading_hour)
    }

    /// Simple moving average smoothing.
    fn smooth(&self, values: &[f64]) -> Vec<f64> {
        let n = values.len();
        let period = self.config.smoothing;
        let mut smoothed = vec![f64::NAN; n];

        for i in period - 1..n {
            let sum: f64 = (0..period)
                .map(|j| values[i - j])
                .filter(|v| !v.is_nan())
                .sum();
            let count = (0..period).filter(|&j| !values[i - j].is_nan()).count();
            if count > 0 {
                smoothed[i] = sum / count as f64;
            }
        }

        smoothed
    }

    /// Get the primary intraday signal (return bias).
    pub fn calculate_signal(
        &self,
        close: &[f64],
        high: &[f64],
        low: &[f64],
        volume: &[f64],
    ) -> Vec<f64> {
        self.calculate(close, high, low, volume).0
    }
}

impl Default for IntradaySeasonality {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for IntradaySeasonality {
    fn name(&self) -> &str {
        "IntradaySeasonality"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_periods = self.config.bars_per_day * self.config.lookback_days;
        if data.close.len() < min_periods {
            return Err(IndicatorError::InsufficientData {
                required: min_periods,
                got: data.close.len(),
            });
        }

        let (return_bias, volatility_pattern, volume_pattern, _) =
            self.calculate(&data.close, &data.high, &data.low, &data.volume);

        Ok(IndicatorOutput::triple(
            return_bias,
            volatility_pattern,
            volume_pattern,
        ))
    }

    fn min_periods(&self) -> usize {
        self.config.bars_per_day * self.config.lookback_days
    }
}

impl SignalIndicator for IntradaySeasonality {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let min_periods = self.config.bars_per_day * self.config.lookback_days;
        if data.close.len() < min_periods {
            return Ok(IndicatorSignal::Neutral);
        }

        let (return_bias, _, volume_pattern, trading_hour) =
            self.calculate(&data.close, &data.high, &data.low, &data.volume);
        let n = return_bias.len();

        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let bias = return_bias[n - 1];
        let volume = volume_pattern[n - 1];
        let hour = trading_hour[n - 1];

        // Strong signal when high volume period with positive/negative bias
        if !bias.is_nan() && !volume.is_nan() {
            let is_high_volume = volume > 1.2;

            // Power hour often has strong trends
            if matches!(hour, TradingHour::PowerHour) && is_high_volume {
                if bias > 2.0 {
                    return Ok(IndicatorSignal::Bullish);
                } else if bias < -2.0 {
                    return Ok(IndicatorSignal::Bearish);
                }
            }

            // Opening hour with bias
            if matches!(hour, TradingHour::Opening) && is_high_volume {
                if bias > 1.5 {
                    return Ok(IndicatorSignal::Bullish);
                } else if bias < -1.5 {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (return_bias, _, volume_pattern, trading_hour) =
            self.calculate(&data.close, &data.high, &data.low, &data.volume);

        let signals = return_bias
            .iter()
            .enumerate()
            .map(|(i, &bias)| {
                if bias.is_nan() || volume_pattern[i].is_nan() {
                    return IndicatorSignal::Neutral;
                }

                let is_high_volume = volume_pattern[i] > 1.2;
                let hour = trading_hour[i];

                if matches!(hour, TradingHour::PowerHour | TradingHour::Opening) && is_high_volume {
                    if bias > 2.0 {
                        IndicatorSignal::Bullish
                    } else if bias < -2.0 {
                        IndicatorSignal::Bearish
                    } else {
                        IndicatorSignal::Neutral
                    }
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

    fn create_test_data(n: usize) -> OHLCVSeries {
        OHLCVSeries {
            open: (0..n).map(|i| 100.0 + (i as f64 * 0.01)).collect(),
            high: (0..n).map(|i| 101.0 + (i as f64 * 0.01)).collect(),
            low: (0..n).map(|i| 99.0 + (i as f64 * 0.01)).collect(),
            close: (0..n).map(|i| 100.0 + (i as f64 * 0.02) * (i % 10) as f64 * 0.1).collect(),
            volume: (0..n).map(|i| 1000.0 + (i % 78) as f64 * 100.0).collect(), // Volume pattern
        }
    }

    #[test]
    fn test_intraday_basic() {
        let config = IntradaySeasonalityConfig {
            lookback_days: 5,
            bars_per_day: 20,
            smoothing: 3,
        };
        let indicator = IntradaySeasonality::with_config(config);
        let data = create_test_data(200);

        let (return_bias, vol_pattern, volume_pattern, trading_hour) =
            indicator.calculate(&data.close, &data.high, &data.low, &data.volume);

        assert_eq!(return_bias.len(), 200);
        assert_eq!(vol_pattern.len(), 200);
        assert_eq!(volume_pattern.len(), 200);
        assert_eq!(trading_hour.len(), 200);
    }

    #[test]
    fn test_trading_hours() {
        let config = IntradaySeasonalityConfig {
            lookback_days: 5,
            bars_per_day: 100,
            smoothing: 1,
        };
        let indicator = IntradaySeasonality::with_config(config);

        // Test hour estimation
        assert_eq!(indicator.estimate_trading_hour(5), TradingHour::Opening);
        assert_eq!(indicator.estimate_trading_hour(20), TradingHour::Morning);
        assert_eq!(indicator.estimate_trading_hour(45), TradingHour::Lunch);
        assert_eq!(indicator.estimate_trading_hour(70), TradingHour::Afternoon);
        assert_eq!(indicator.estimate_trading_hour(90), TradingHour::PowerHour);
    }

    #[test]
    fn test_bar_returns_calculation() {
        let config = IntradaySeasonalityConfig {
            lookback_days: 5,
            bars_per_day: 10,
            smoothing: 1,
        };
        let indicator = IntradaySeasonality::with_config(config);

        // Create simple price data
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i % 10) as f64 * 0.1).collect();
        let bar_returns = indicator.calculate_bar_returns(&close);

        assert_eq!(bar_returns.len(), 10);
    }

    #[test]
    fn test_volume_pattern() {
        let config = IntradaySeasonalityConfig {
            lookback_days: 5,
            bars_per_day: 10,
            smoothing: 1,
        };
        let indicator = IntradaySeasonality::with_config(config);

        // Create volume with pattern (higher at open and close)
        let volume: Vec<f64> = (0..50)
            .map(|i| {
                let bar = i % 10;
                if bar < 2 || bar > 7 {
                    2000.0
                } else {
                    1000.0
                }
            })
            .collect();

        let bar_volume = indicator.calculate_bar_volume(&volume);

        // First bars should have higher volume
        assert!(bar_volume[0] > bar_volume[5]);
    }

    #[test]
    fn test_intraday_insufficient_data() {
        let indicator = IntradaySeasonality::new();
        let data = create_test_data(100); // Less than 78 * 20 = 1560

        let result = indicator.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_intraday_technical_indicator() {
        let config = IntradaySeasonalityConfig {
            lookback_days: 3,
            bars_per_day: 20,
            smoothing: 2,
        };
        let indicator = IntradaySeasonality::with_config(config);
        let data = create_test_data(100);

        let result = indicator.compute(&data);
        assert!(result.is_ok());

        assert_eq!(indicator.name(), "IntradaySeasonality");
    }
}
