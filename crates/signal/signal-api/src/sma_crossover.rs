//! SMA Crossover signal generator

use signal_core::{Result, SignalError};
use signal_spi::{Signal, SignalGenerator};
use serde::{Deserialize, Serialize};

/// Simple Moving Average Crossover signal generator
///
/// Generates buy signals when short-term SMA crosses above long-term SMA,
/// and sell signals when it crosses below.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SMACrossover {
    short_period: usize,
    long_period: usize,
}

impl SMACrossover {
    /// Create a new SMA Crossover generator
    ///
    /// # Arguments
    ///
    /// * `short_period` - Short-term moving average period
    /// * `long_period` - Long-term moving average period
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period >= long_period {
            return Err(SignalError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be less than long_period".to_string(),
            });
        }
        if short_period < 1 {
            return Err(SignalError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }

        Ok(Self {
            short_period,
            long_period,
        })
    }

    fn calculate_sma(data: &[f64], period: usize) -> Option<f64> {
        if data.len() < period {
            return None;
        }
        let sum: f64 = data[data.len() - period..].iter().sum();
        Some(sum / period as f64)
    }

    pub fn short_period(&self) -> usize {
        self.short_period
    }

    pub fn long_period(&self) -> usize {
        self.long_period
    }
}

impl SignalGenerator for SMACrossover {
    fn generate(&self, data: &[f64]) -> Signal {
        if data.len() < self.long_period + 1 {
            return Signal::Hold;
        }

        let current_short = Self::calculate_sma(data, self.short_period);
        let current_long = Self::calculate_sma(data, self.long_period);

        let prev_data = &data[..data.len() - 1];
        let prev_short = Self::calculate_sma(prev_data, self.short_period);
        let prev_long = Self::calculate_sma(prev_data, self.long_period);

        match (current_short, current_long, prev_short, prev_long) {
            (Some(cs), Some(cl), Some(ps), Some(pl)) => {
                // Bullish crossover: short crosses above long
                if ps <= pl && cs > cl {
                    Signal::Buy
                }
                // Bearish crossover: short crosses below long
                else if ps >= pl && cs < cl {
                    Signal::Sell
                } else {
                    Signal::Hold
                }
            }
            _ => Signal::Hold,
        }
    }
}
