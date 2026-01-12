//! Trading signals implementation.

use financial_spi::{Signal, SignalGenerator};

/// Simple moving average crossover signal generator.
#[derive(Debug, Clone)]
pub struct SMACrossover {
    fast_period: usize,
    slow_period: usize,
}

impl SMACrossover {
    /// Create a new SMA crossover signal generator.
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        Self { fast_period, slow_period }
    }

    fn sma(data: &[f64], period: usize) -> Option<f64> {
        if data.len() < period {
            return None;
        }
        let sum: f64 = data[data.len() - period..].iter().sum();
        Some(sum / period as f64)
    }
}

impl SignalGenerator for SMACrossover {
    fn generate(&self, data: &[f64]) -> Signal {
        let fast = Self::sma(data, self.fast_period);
        let slow = Self::sma(data, self.slow_period);

        match (fast, slow) {
            (Some(f), Some(s)) if f > s => Signal::Buy,
            (Some(f), Some(s)) if f < s => Signal::Sell,
            _ => Signal::Hold,
        }
    }
}

/// Momentum signal generator.
#[derive(Debug, Clone)]
pub struct MomentumSignal {
    period: usize,
    threshold: f64,
}

impl MomentumSignal {
    /// Create a new momentum signal generator.
    pub fn new(period: usize, threshold: f64) -> Self {
        Self { period, threshold }
    }
}

impl SignalGenerator for MomentumSignal {
    fn generate(&self, data: &[f64]) -> Signal {
        if data.len() < self.period + 1 {
            return Signal::Hold;
        }

        let current = data[data.len() - 1];
        let past = data[data.len() - self.period - 1];
        let momentum = (current - past) / past;

        if momentum > self.threshold {
            Signal::Buy
        } else if momentum < -self.threshold {
            Signal::Sell
        } else {
            Signal::Hold
        }
    }
}

/// RSI-based signal generator.
#[derive(Debug, Clone)]
pub struct RSISignal {
    period: usize,
    overbought: f64,
    oversold: f64,
}

impl RSISignal {
    /// Create a new RSI signal generator.
    pub fn new(period: usize, overbought: f64, oversold: f64) -> Self {
        Self { period, overbought, oversold }
    }

    fn calculate_rsi(data: &[f64], period: usize) -> Option<f64> {
        if data.len() < period + 1 {
            return None;
        }

        let changes: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
        let recent_changes = &changes[changes.len() - period..];

        let gains: f64 = recent_changes.iter().filter(|&&c| c > 0.0).sum();
        let losses: f64 = recent_changes.iter().filter(|&&c| c < 0.0).map(|c| c.abs()).sum();

        if losses < 1e-10 {
            return Some(100.0);
        }

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        if avg_loss < 1e-10 {
            return Some(100.0);
        }

        let rs = avg_gain / avg_loss;
        Some(100.0 - (100.0 / (1.0 + rs)))
    }
}

impl SignalGenerator for RSISignal {
    fn generate(&self, data: &[f64]) -> Signal {
        match Self::calculate_rsi(data, self.period) {
            Some(rsi) if rsi > self.overbought => Signal::Sell,
            Some(rsi) if rsi < self.oversold => Signal::Buy,
            _ => Signal::Hold,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma_crossover() {
        let signal = SMACrossover::new(2, 4);

        // Uptrend data
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(signal.generate(&data), Signal::Buy);

        // Downtrend data
        let data = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(signal.generate(&data), Signal::Sell);
    }

    #[test]
    fn test_momentum_signal() {
        let signal = MomentumSignal::new(2, 0.05);

        // Strong upward momentum
        let data = vec![100.0, 105.0, 112.0];
        assert_eq!(signal.generate(&data), Signal::Buy);

        // Strong downward momentum
        let data = vec![100.0, 95.0, 88.0];
        assert_eq!(signal.generate(&data), Signal::Sell);
    }
}
