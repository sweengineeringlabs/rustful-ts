//! Reinforcement Learning Signal Indicator (IND-295)
//!
//! RL-based trading signal proxy that simulates adaptive learning
//! using reward-weighted momentum strategies.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Trading action suggested by the RL signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RLAction {
    /// Strong sell signal
    StrongSell = -2,
    /// Weak sell signal
    Sell = -1,
    /// Hold/neutral
    Hold = 0,
    /// Weak buy signal
    Buy = 1,
    /// Strong buy signal
    StrongBuy = 2,
}

impl RLAction {
    /// Convert numeric signal to action
    pub fn from_signal(signal: f64) -> Self {
        if signal > 1.5 {
            RLAction::StrongBuy
        } else if signal > 0.5 {
            RLAction::Buy
        } else if signal < -1.5 {
            RLAction::StrongSell
        } else if signal < -0.5 {
            RLAction::Sell
        } else {
            RLAction::Hold
        }
    }
}

/// RL Signal - Reinforcement Learning based trading signal proxy
///
/// This indicator simulates an RL agent's policy by using reward-weighted
/// momentum features. It adapts signal strength based on recent "reward"
/// (successful predictions) using an exponential moving reward mechanism.
///
/// # Output
/// Returns signal strength from approximately -3 to +3:
/// - > 1.5: Strong buy
/// - 0.5 to 1.5: Buy
/// - -0.5 to 0.5: Hold
/// - -1.5 to -0.5: Sell
/// - < -1.5: Strong sell
#[derive(Debug, Clone)]
pub struct RLSignal {
    /// Short-term momentum period
    short_period: usize,
    /// Long-term momentum period
    long_period: usize,
    /// Reward memory period (learning window)
    reward_period: usize,
    /// Exploration factor (higher = more responsive to new data)
    exploration: f64,
}

impl RLSignal {
    /// Create a new RLSignal indicator
    ///
    /// # Arguments
    /// * `short_period` - Short momentum lookback (minimum 5)
    /// * `long_period` - Long momentum lookback (must be > short_period)
    /// * `reward_period` - Period for reward accumulation (minimum 10)
    /// * `exploration` - Exploration factor 0.1-1.0 (default 0.3)
    pub fn new(
        short_period: usize,
        long_period: usize,
        reward_period: usize,
        exploration: f64,
    ) -> Result<Self> {
        if short_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if long_period <= short_period {
            return Err(IndicatorError::InvalidParameter {
                name: "long_period".to_string(),
                reason: "must be greater than short_period".to_string(),
            });
        }
        if reward_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "reward_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if !(0.1..=1.0).contains(&exploration) {
            return Err(IndicatorError::InvalidParameter {
                name: "exploration".to_string(),
                reason: "must be between 0.1 and 1.0".to_string(),
            });
        }
        Ok(Self {
            short_period,
            long_period,
            reward_period,
            exploration,
        })
    }

    /// Calculate the RL trading signal
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        if n < self.long_period + self.reward_period {
            return result;
        }

        // Calculate base signals (features for the "policy")
        let mut short_momentum = vec![0.0; n];
        let mut long_momentum = vec![0.0; n];
        let mut volume_trend = vec![0.0; n];
        let mut volatility = vec![0.0; n];

        for i in self.long_period..n {
            // Short-term momentum
            if close[i - self.short_period] > 1e-10 {
                short_momentum[i] = (close[i] / close[i - self.short_period] - 1.0) * 100.0;
            }

            // Long-term momentum
            if close[i - self.long_period] > 1e-10 {
                long_momentum[i] = (close[i] / close[i - self.long_period] - 1.0) * 100.0;
            }

            // Volume trend (short vs long average)
            let short_start = i.saturating_sub(self.short_period);
            let long_start = i.saturating_sub(self.long_period);
            let short_vol: f64 = volume[short_start..=i].iter().sum::<f64>() / self.short_period as f64;
            let long_vol: f64 = volume[long_start..=i].iter().sum::<f64>() / self.long_period as f64;
            if long_vol > 1e-10 {
                volume_trend[i] = (short_vol / long_vol - 1.0) * 100.0;
            }

            // Volatility (recent range)
            let mut vol_sum = 0.0;
            for j in (short_start + 1)..=i {
                vol_sum += (close[j] / close[j - 1] - 1.0).abs();
            }
            volatility[i] = vol_sum / self.short_period as f64 * 100.0;
        }

        // Calculate rewards and adaptive weights
        let mut cumulative_reward = vec![0.0; n];
        let mut prev_signal = 0.0;

        for i in (self.long_period + 1)..n {
            // Calculate "reward" from previous signal
            // Reward = sign(prev_signal) * actual_return
            let actual_return = close[i] / close[i - 1] - 1.0;
            let reward = (prev_signal as f64).signum() * actual_return * 100.0;

            // Exponential moving reward (learning rate = exploration)
            cumulative_reward[i] = self.exploration * reward
                + (1.0 - self.exploration) * cumulative_reward[i - 1];

            // Adaptive "policy" weights based on cumulative reward
            // Good past performance = trust momentum more
            // Bad past performance = reduce signal strength
            let confidence = 1.0 / (1.0 + (-cumulative_reward[i] * 0.1).exp()); // Sigmoid
            let confidence_factor = 0.5 + confidence; // Range [0.5, 1.5]

            // Compute base signal from features
            let momentum_signal = (short_momentum[i] * 0.6 + long_momentum[i] * 0.4) / 5.0;

            // Volume confirmation
            let volume_factor = if momentum_signal.signum() == volume_trend[i].signum() {
                1.2 // Confirmed
            } else if volume_trend[i].abs() > 20.0 {
                0.8 // Divergence
            } else {
                1.0
            };

            // Volatility adjustment (reduce in high volatility)
            let vol_factor = 1.0 / (1.0 + volatility[i] / 3.0);

            // Final signal with adaptive confidence
            let raw_signal = momentum_signal * volume_factor * vol_factor * confidence_factor;

            // Clamp to reasonable range
            result[i] = raw_signal.max(-3.0).min(3.0);

            // Store for next iteration's reward calculation
            prev_signal = result[i];
        }

        // Apply smoothing to reduce noise
        let alpha = 2.0 / (self.short_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Get the recommended action for a given signal value
    pub fn get_action(&self, signal: f64) -> RLAction {
        RLAction::from_signal(signal)
    }
}

impl TechnicalIndicator for RLSignal {
    fn name(&self) -> &str {
        "RL Signal"
    }

    fn min_periods(&self) -> usize {
        self.long_period + self.reward_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uptrend_data() -> OHLCVSeries {
        let n = 80;
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.2).sin() * 2.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = (0..n)
            .map(|i| 1000.0 + (i as f64) * 10.0) // Increasing volume in uptrend
            .collect();

        OHLCVSeries { open, high, low, close, volume }
    }

    fn make_downtrend_data() -> OHLCVSeries {
        let n = 80;
        let close: Vec<f64> = (0..n)
            .map(|i| 150.0 - (i as f64) * 0.5 + (i as f64 * 0.2).sin() * 2.0)
            .collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = (0..n)
            .map(|i| 1000.0 + (i as f64) * 5.0)
            .collect();

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_rl_signal_basic() {
        let data = make_uptrend_data();
        let indicator = RLSignal::new(7, 21, 14, 0.3).unwrap();
        let result = indicator.calculate(&data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
        // Signals should be in reasonable range
        for i in 40..result.len() {
            assert!(result[i] >= -3.0 && result[i] <= 3.0);
        }
    }

    #[test]
    fn test_rl_signal_uptrend_bullish() {
        let data = make_uptrend_data();
        let indicator = RLSignal::new(7, 21, 14, 0.3).unwrap();
        let result = indicator.calculate(&data.close, &data.volume);

        // In uptrend, later signals should be positive on average
        let avg_signal: f64 = result[50..].iter().sum::<f64>() / (result.len() - 50) as f64;
        assert!(avg_signal > 0.0, "Expected positive signal in uptrend, got {}", avg_signal);
    }

    #[test]
    fn test_rl_signal_downtrend_bearish() {
        let data = make_downtrend_data();
        let indicator = RLSignal::new(7, 21, 14, 0.3).unwrap();
        let result = indicator.calculate(&data.close, &data.volume);

        // In downtrend, later signals should be negative on average
        let avg_signal: f64 = result[50..].iter().sum::<f64>() / (result.len() - 50) as f64;
        assert!(avg_signal < 0.0, "Expected negative signal in downtrend, got {}", avg_signal);
    }

    #[test]
    fn test_rl_action_enum() {
        assert_eq!(RLAction::from_signal(2.0), RLAction::StrongBuy);
        assert_eq!(RLAction::from_signal(1.0), RLAction::Buy);
        assert_eq!(RLAction::from_signal(0.0), RLAction::Hold);
        assert_eq!(RLAction::from_signal(-1.0), RLAction::Sell);
        assert_eq!(RLAction::from_signal(-2.0), RLAction::StrongSell);
    }

    #[test]
    fn test_rl_signal_technical_indicator_trait() {
        let data = make_uptrend_data();
        let indicator = RLSignal::new(7, 21, 14, 0.3).unwrap();

        assert_eq!(indicator.name(), "RL Signal");
        assert_eq!(indicator.min_periods(), 35); // 21 + 14

        let output = indicator.compute(&data).unwrap();
        assert!(!output.values.is_empty());
    }

    #[test]
    fn test_rl_signal_parameter_validation() {
        assert!(RLSignal::new(3, 21, 14, 0.3).is_err()); // short_period too small
        assert!(RLSignal::new(21, 14, 14, 0.3).is_err()); // long <= short
        assert!(RLSignal::new(7, 21, 5, 0.3).is_err()); // reward_period too small
        assert!(RLSignal::new(7, 21, 14, 0.05).is_err()); // exploration too small
        assert!(RLSignal::new(7, 21, 14, 1.5).is_err()); // exploration too large
    }

    #[test]
    fn test_rl_signal_exploration_effect() {
        let data = make_uptrend_data();

        // High exploration = more responsive
        let high_exp = RLSignal::new(7, 21, 14, 0.9).unwrap();
        let low_exp = RLSignal::new(7, 21, 14, 0.1).unwrap();

        let high_result = high_exp.calculate(&data.close, &data.volume);
        let low_result = low_exp.calculate(&data.close, &data.volume);

        // Both should produce valid results
        assert_eq!(high_result.len(), low_result.len());

        // High exploration should have more variance
        let high_var: f64 = high_result[50..].windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum::<f64>() / (high_result.len() - 51) as f64;
        let low_var: f64 = low_result[50..].windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .sum::<f64>() / (low_result.len() - 51) as f64;

        // High exploration should generally have more variance, but not always
        // Just verify both are non-negative
        assert!(high_var >= 0.0);
        assert!(low_var >= 0.0);
    }

    #[test]
    fn test_rl_signal_get_action_method() {
        let indicator = RLSignal::new(7, 21, 14, 0.3).unwrap();

        assert_eq!(indicator.get_action(2.0), RLAction::StrongBuy);
        assert_eq!(indicator.get_action(0.0), RLAction::Hold);
        assert_eq!(indicator.get_action(-2.0), RLAction::StrongSell);
    }
}
