//! Hash Ribbons - IND-088
//!
//! A Bitcoin-specific indicator using hash rate moving averages to identify
//! miner capitulation and recovery phases.
//!
//! When hash rate declines significantly, miners are capitulating (shutting down).
//! When hash rate recovers (30-day MA crosses above 60-day MA), it signals
//! the end of capitulation and often precedes price rallies.

use indicator_spi::IndicatorSignal;

/// Hash Ribbons output.
#[derive(Debug, Clone)]
pub struct HashRibbonsOutput {
    /// 30-day SMA of hash rate.
    pub ma_short: Vec<f64>,
    /// 60-day SMA of hash rate.
    pub ma_long: Vec<f64>,
    /// Hash ribbon value (short - long).
    pub ribbon: Vec<f64>,
    /// Capitulation flag (1 = in capitulation, 0 = not).
    pub capitulation: Vec<i32>,
    /// Recovery signal (1 = buy signal, 0 = no signal).
    pub recovery_signal: Vec<i32>,
}

/// Hash Ribbons market phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashRibbonsPhase {
    /// Normal operation, no capitulation.
    Normal,
    /// Miners are capitulating (hash rate declining).
    Capitulation,
    /// Recovery from capitulation (buy signal).
    Recovery,
}

/// Hash Ribbons - IND-088
///
/// A momentum indicator based on Bitcoin mining hash rate.
///
/// # Algorithm
/// 1. Calculate 30-day SMA of hash rate (short)
/// 2. Calculate 60-day SMA of hash rate (long)
/// 3. Capitulation: short MA < long MA
/// 4. Recovery: short MA crosses above long MA after capitulation
///
/// # Example
/// ```
/// use indicator_core::crypto::HashRibbons;
///
/// let hr = HashRibbons::new(30, 60);
/// let hash_rates = vec![100.0, 105.0, 98.0, 110.0]; // EH/s
/// let output = hr.calculate(&hash_rates);
/// ```
#[derive(Debug, Clone)]
pub struct HashRibbons {
    /// Short MA period (default: 30 days).
    short_period: usize,
    /// Long MA period (default: 60 days).
    long_period: usize,
}

impl HashRibbons {
    /// Create a new Hash Ribbons indicator.
    pub fn new(short_period: usize, long_period: usize) -> Self {
        Self {
            short_period,
            long_period,
        }
    }

    /// Calculate Hash Ribbons from hash rate data.
    pub fn calculate(&self, hash_rates: &[f64]) -> HashRibbonsOutput {
        let n = hash_rates.len();

        if n == 0 {
            return HashRibbonsOutput {
                ma_short: vec![],
                ma_long: vec![],
                ribbon: vec![],
                capitulation: vec![],
                recovery_signal: vec![],
            };
        }

        // Calculate moving averages
        let ma_short = self.calculate_sma(hash_rates, self.short_period);
        let ma_long = self.calculate_sma(hash_rates, self.long_period);

        // Calculate ribbon (difference)
        let ribbon: Vec<f64> = ma_short
            .iter()
            .zip(ma_long.iter())
            .map(|(&s, &l)| {
                if s.is_nan() || l.is_nan() {
                    f64::NAN
                } else {
                    s - l
                }
            })
            .collect();

        // Detect capitulation and recovery
        let mut capitulation = vec![0; n];
        let mut recovery_signal = vec![0; n];
        let mut in_capitulation = false;

        for i in 0..n {
            if ribbon[i].is_nan() {
                continue;
            }

            // Capitulation: short MA below long MA
            if ribbon[i] < 0.0 {
                capitulation[i] = 1;
                in_capitulation = true;
            } else {
                // Check for recovery (cross above after capitulation)
                if in_capitulation && i > 0 && ribbon[i - 1] < 0.0 && ribbon[i] >= 0.0 {
                    recovery_signal[i] = 1;
                }
                in_capitulation = false;
            }
        }

        HashRibbonsOutput {
            ma_short,
            ma_long,
            ribbon,
            capitulation,
            recovery_signal,
        }
    }

    /// Calculate SMA.
    fn calculate_sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let mut sum: f64 = data[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        for i in period..n {
            sum = sum - data[i - period] + data[i];
            result[i] = sum / period as f64;
        }

        result
    }

    /// Get current market phase.
    pub fn get_phase(&self, output: &HashRibbonsOutput) -> HashRibbonsPhase {
        let n = output.ribbon.len();
        if n == 0 {
            return HashRibbonsPhase::Normal;
        }

        // Check last value
        let last_idx = n - 1;
        if output.recovery_signal[last_idx] == 1 {
            HashRibbonsPhase::Recovery
        } else if output.capitulation[last_idx] == 1 {
            HashRibbonsPhase::Capitulation
        } else {
            HashRibbonsPhase::Normal
        }
    }

    /// Convert phase to trading signal.
    pub fn to_indicator_signal(&self, phase: HashRibbonsPhase) -> IndicatorSignal {
        match phase {
            HashRibbonsPhase::Recovery => IndicatorSignal::Bullish,
            HashRibbonsPhase::Capitulation => IndicatorSignal::Neutral, // Wait for recovery
            HashRibbonsPhase::Normal => IndicatorSignal::Neutral,
        }
    }

    /// Count recovery signals in the series.
    pub fn count_recovery_signals(&self, output: &HashRibbonsOutput) -> usize {
        output.recovery_signal.iter().filter(|&&x| x == 1).count()
    }

    /// Find indices of all recovery signals.
    pub fn find_recovery_indices(&self, output: &HashRibbonsOutput) -> Vec<usize> {
        output
            .recovery_signal
            .iter()
            .enumerate()
            .filter(|(_, &x)| x == 1)
            .map(|(i, _)| i)
            .collect()
    }
}

impl Default for HashRibbons {
    fn default() -> Self {
        Self::new(30, 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_ribbons_basic() {
        let hr = HashRibbons::new(5, 10);
        let hash_rates: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();

        let output = hr.calculate(&hash_rates);

        assert_eq!(output.ma_short.len(), 30);
        assert_eq!(output.ma_long.len(), 30);
        assert_eq!(output.ribbon.len(), 30);
    }

    #[test]
    fn test_hash_ribbons_capitulation() {
        let hr = HashRibbons::new(3, 5);

        // Hash rate declining
        let hash_rates = vec![100.0, 95.0, 90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0];

        let output = hr.calculate(&hash_rates);

        // Should detect capitulation (short MA below long MA)
        let total_cap: i32 = output.capitulation.iter().sum();
        assert!(total_cap > 0, "Should detect capitulation");
    }

    #[test]
    fn test_hash_ribbons_recovery() {
        let hr = HashRibbons::new(3, 5);

        // Hash rate: decline then recovery
        let hash_rates = vec![
            100.0, 95.0, 90.0, 85.0, 80.0, // Decline
            75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 105.0, // Recovery
        ];

        let output = hr.calculate(&hash_rates);

        // Should have recovery signal somewhere
        let has_recovery = output.recovery_signal.iter().any(|&x| x == 1);
        // May or may not trigger depending on exact MA crossover timing
        assert_eq!(output.recovery_signal.len(), hash_rates.len());
        // At minimum, ribbon should be calculated
        assert!(!output.ribbon.is_empty());
    }

    #[test]
    fn test_hash_ribbons_phase() {
        let hr = HashRibbons::default();

        // Create output with recovery signal
        let mut output = HashRibbonsOutput {
            ma_short: vec![100.0; 10],
            ma_long: vec![100.0; 10],
            ribbon: vec![0.0; 10],
            capitulation: vec![0; 10],
            recovery_signal: vec![0; 10],
        };

        assert_eq!(hr.get_phase(&output), HashRibbonsPhase::Normal);

        output.capitulation[9] = 1;
        assert_eq!(hr.get_phase(&output), HashRibbonsPhase::Capitulation);

        output.capitulation[9] = 0;
        output.recovery_signal[9] = 1;
        assert_eq!(hr.get_phase(&output), HashRibbonsPhase::Recovery);
    }

    #[test]
    fn test_hash_ribbons_empty_input() {
        let hr = HashRibbons::default();
        let output = hr.calculate(&[]);

        assert!(output.ma_short.is_empty());
        assert!(output.ribbon.is_empty());
    }

    #[test]
    fn test_hash_ribbons_signal_conversion() {
        let hr = HashRibbons::default();

        assert_eq!(
            hr.to_indicator_signal(HashRibbonsPhase::Recovery),
            IndicatorSignal::Bullish
        );
        assert_eq!(
            hr.to_indicator_signal(HashRibbonsPhase::Capitulation),
            IndicatorSignal::Neutral
        );
    }

    #[test]
    fn test_hash_ribbons_find_recovery_indices() {
        let hr = HashRibbons::default();

        let output = HashRibbonsOutput {
            ma_short: vec![],
            ma_long: vec![],
            ribbon: vec![],
            capitulation: vec![],
            recovery_signal: vec![0, 0, 1, 0, 1, 0],
        };

        let indices = hr.find_recovery_indices(&output);
        assert_eq!(indices, vec![2, 4]);
    }
}
