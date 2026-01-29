//! Exchange Inflow/Outflow (IND-268)
//!
//! Exchange balance change indicator for cryptocurrency networks.
//! Tracks the net flow of coins into and out of exchange wallets.
//!
//! # Concept
//! Exchange flow is a critical on-chain metric for understanding market sentiment:
//! - Net inflow (positive): Coins moving to exchanges, potential selling pressure
//! - Net outflow (negative): Coins leaving exchanges, potential accumulation/HODLing
//!
//! Large inflows often precede sell-offs, while sustained outflows indicate accumulation.
//!
//! # Data Requirements
//! This indicator requires on-chain data tracking exchange wallet balances.
//! When using OHLCV data, it provides a proxy based on price-volume dynamics.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Output from the Exchange Flow calculation.
#[derive(Debug, Clone)]
pub struct ExchangeFlowOutput {
    /// Net exchange flow (positive = inflow, negative = outflow).
    pub net_flow: Vec<f64>,
    /// Moving average of net flow.
    pub flow_ma: Vec<f64>,
    /// Cumulative exchange balance change.
    pub cumulative_flow: Vec<f64>,
    /// Flow momentum (rate of change in flow).
    pub momentum: Vec<f64>,
    /// Flow signal (-100 to 100 scale).
    pub signal: Vec<f64>,
}

/// Exchange flow classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowSignal {
    /// Strong inflow - heavy selling pressure expected.
    StrongInflow,
    /// Moderate inflow - some selling pressure.
    Inflow,
    /// Balanced - no clear direction.
    Neutral,
    /// Moderate outflow - some accumulation.
    Outflow,
    /// Strong outflow - heavy accumulation.
    StrongOutflow,
}

/// Exchange Inflow/Outflow (IND-268)
///
/// Measures the net flow of cryptocurrency into and out of exchanges.
///
/// # Calculation (with on-chain data)
/// ```text
/// NetFlow = ExchangeBalance[t] - ExchangeBalance[t-1]
/// FlowMA = SMA(NetFlow, ma_period)
/// CumulativeFlow = Sum(NetFlow)
/// Signal = Normalized NetFlow relative to historical range
/// ```
///
/// # Calculation (proxy from OHLCV)
/// ```text
/// FlowProxy = Volume * PriceChange direction
/// Positive price + high volume = selling into strength (inflow proxy)
/// Negative price + high volume = buying weakness (outflow proxy)
/// ```
///
/// # Example
/// ```
/// use indicator_core::crypto::ExchangeInflow;
///
/// let ei = ExchangeInflow::new(20, 10, 50).unwrap();
/// let balances = vec![100000.0, 102000.0, 99000.0, 95000.0];
/// let output = ei.calculate_from_balances(&balances);
/// ```
#[derive(Debug, Clone)]
pub struct ExchangeInflow {
    /// Period for moving average calculation.
    ma_period: usize,
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Lookback period for signal normalization.
    lookback_period: usize,
}

impl ExchangeInflow {
    /// Create a new Exchange Inflow indicator.
    ///
    /// # Arguments
    /// * `ma_period` - Period for moving average (minimum 5)
    /// * `momentum_period` - Period for momentum calculation (minimum 1)
    /// * `lookback_period` - Period for signal normalization (minimum 20)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid.
    pub fn new(ma_period: usize, momentum_period: usize, lookback_period: usize) -> Result<Self> {
        if ma_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "ma_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if momentum_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        if lookback_period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback_period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self {
            ma_period,
            momentum_period,
            lookback_period,
        })
    }

    /// Create with default parameters (20, 10, 50).
    pub fn default_params() -> Result<Self> {
        Self::new(20, 10, 50)
    }

    /// Calculate exchange flow from balance snapshots.
    ///
    /// # Arguments
    /// * `balances` - Slice of daily exchange balance snapshots
    ///
    /// # Returns
    /// ExchangeFlowOutput containing all metrics.
    pub fn calculate_from_balances(&self, balances: &[f64]) -> ExchangeFlowOutput {
        let n = balances.len();
        let mut net_flow = vec![0.0; n];
        let mut flow_ma = vec![0.0; n];
        let mut cumulative_flow = vec![0.0; n];
        let mut momentum = vec![0.0; n];
        let mut signal = vec![0.0; n];

        if n < 2 {
            return ExchangeFlowOutput {
                net_flow,
                flow_ma,
                cumulative_flow,
                momentum,
                signal,
            };
        }

        // Calculate net flow (balance changes)
        for i in 1..n {
            net_flow[i] = balances[i] - balances[i - 1];
        }

        // Calculate cumulative flow
        for i in 1..n {
            cumulative_flow[i] = cumulative_flow[i - 1] + net_flow[i];
        }

        // Calculate moving average
        if n >= self.ma_period {
            for i in (self.ma_period - 1)..n {
                let start = i + 1 - self.ma_period;
                flow_ma[i] = net_flow[start..=i].iter().sum::<f64>() / self.ma_period as f64;
            }
        }

        // Calculate momentum
        for i in self.momentum_period..n {
            if flow_ma[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = flow_ma[i] - flow_ma[i - self.momentum_period];
            }
        }

        // Calculate normalized signal (-100 to 100)
        if n >= self.lookback_period {
            for i in (self.lookback_period - 1)..n {
                let start = i + 1 - self.lookback_period;
                let window = &net_flow[start..=i];

                let max_val = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min_val = window.iter().cloned().fold(f64::INFINITY, f64::min);
                let range = max_val - min_val;

                if range > 1e-10 {
                    // Normalize to -100 to 100
                    signal[i] = (net_flow[i] - min_val) / range * 200.0 - 100.0;
                }
            }
        }

        ExchangeFlowOutput {
            net_flow,
            flow_ma,
            cumulative_flow,
            momentum,
            signal,
        }
    }

    /// Calculate from net flow data directly.
    ///
    /// # Arguments
    /// * `flows` - Slice of daily net flow values (inflow positive, outflow negative)
    pub fn calculate(&self, flows: &[f64]) -> ExchangeFlowOutput {
        let n = flows.len();
        let net_flow = flows.to_vec();
        let mut flow_ma = vec![0.0; n];
        let mut cumulative_flow = vec![0.0; n];
        let mut momentum = vec![0.0; n];
        let mut signal = vec![0.0; n];

        if n < 2 {
            return ExchangeFlowOutput {
                net_flow,
                flow_ma,
                cumulative_flow,
                momentum,
                signal,
            };
        }

        // Calculate cumulative flow
        cumulative_flow[0] = flows[0];
        for i in 1..n {
            cumulative_flow[i] = cumulative_flow[i - 1] + flows[i];
        }

        // Calculate moving average
        if n >= self.ma_period {
            for i in (self.ma_period - 1)..n {
                let start = i + 1 - self.ma_period;
                flow_ma[i] = flows[start..=i].iter().sum::<f64>() / self.ma_period as f64;
            }
        }

        // Calculate momentum
        for i in self.momentum_period..n {
            momentum[i] = flow_ma[i] - flow_ma[i - self.momentum_period];
        }

        // Calculate normalized signal
        if n >= self.lookback_period {
            for i in (self.lookback_period - 1)..n {
                let start = i + 1 - self.lookback_period;
                let window = &flows[start..=i];

                let max_val = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min_val = window.iter().cloned().fold(f64::INFINITY, f64::min);
                let range = max_val - min_val;

                if range > 1e-10 {
                    signal[i] = (flows[i] - min_val) / range * 200.0 - 100.0;
                }
            }
        }

        ExchangeFlowOutput {
            net_flow,
            flow_ma,
            cumulative_flow,
            momentum,
            signal,
        }
    }

    /// Calculate exchange flow proxy from OHLCV data.
    pub fn calculate_proxy(&self, data: &OHLCVSeries) -> ExchangeFlowOutput {
        let n = data.close.len();
        let mut net_flow = vec![0.0; n];
        let mut flow_ma = vec![0.0; n];
        let mut cumulative_flow = vec![0.0; n];
        let mut momentum = vec![0.0; n];
        let mut signal = vec![0.0; n];

        if n < 2 {
            return ExchangeFlowOutput {
                net_flow,
                flow_ma,
                cumulative_flow,
                momentum,
                signal,
            };
        }

        // Calculate flow proxy based on price-volume dynamics
        // High volume + price increase = selling into strength (inflow proxy)
        // High volume + price decrease = buying weakness (outflow proxy)
        for i in 1..n {
            let price_change = data.close[i] - data.close[i - 1];
            let typical_volume = data.volume[i];

            // Calculate volume relative to recent average
            let start = if i > self.ma_period { i - self.ma_period } else { 0 };
            let avg_vol: f64 = data.volume[start..i].iter().sum::<f64>()
                / (i - start).max(1) as f64;

            let vol_ratio = if avg_vol > 1e-10 {
                typical_volume / avg_vol
            } else {
                1.0
            };

            // Flow proxy: volume deviation * price direction
            // Positive = inflow (selling), Negative = outflow (accumulation)
            net_flow[i] = vol_ratio * price_change.signum() * (vol_ratio - 1.0).abs() * 100.0;
        }

        // Calculate cumulative flow
        for i in 1..n {
            cumulative_flow[i] = cumulative_flow[i - 1] + net_flow[i];
        }

        // Calculate moving average
        if n >= self.ma_period {
            for i in (self.ma_period - 1)..n {
                let start = i + 1 - self.ma_period;
                flow_ma[i] = net_flow[start..=i].iter().sum::<f64>() / self.ma_period as f64;
            }
        }

        // Calculate momentum
        for i in self.momentum_period..n {
            momentum[i] = flow_ma[i] - flow_ma[i - self.momentum_period];
        }

        // Calculate normalized signal
        if n >= self.lookback_period {
            for i in (self.lookback_period - 1)..n {
                let start = i + 1 - self.lookback_period;
                let window = &net_flow[start..=i];

                let max_val = window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min_val = window.iter().cloned().fold(f64::INFINITY, f64::min);
                let range = max_val - min_val;

                if range > 1e-10 {
                    signal[i] = (net_flow[i] - min_val) / range * 200.0 - 100.0;
                }
            }
        }

        ExchangeFlowOutput {
            net_flow,
            flow_ma,
            cumulative_flow,
            momentum,
            signal,
        }
    }

    /// Get flow signal interpretation.
    pub fn interpret(&self, signal: f64) -> FlowSignal {
        if signal >= 60.0 {
            FlowSignal::StrongInflow
        } else if signal >= 20.0 {
            FlowSignal::Inflow
        } else if signal <= -60.0 {
            FlowSignal::StrongOutflow
        } else if signal <= -20.0 {
            FlowSignal::Outflow
        } else {
            FlowSignal::Neutral
        }
    }

    /// Get flow signals for all values.
    pub fn flow_signals(&self, output: &ExchangeFlowOutput) -> Vec<FlowSignal> {
        output.signal.iter().map(|&s| self.interpret(s)).collect()
    }

    /// Check if there's a significant flow divergence from price.
    ///
    /// Returns true if price is rising but there's net inflow (bearish divergence)
    /// or if price is falling but there's net outflow (bullish divergence).
    pub fn check_divergence(&self, price_change: f64, flow: f64) -> Option<bool> {
        if price_change.abs() < 1e-10 || flow.abs() < 1e-10 {
            return None;
        }

        // Bullish divergence: price down but accumulation (outflow)
        if price_change < 0.0 && flow < 0.0 {
            return Some(true); // Bullish
        }

        // Bearish divergence: price up but distribution (inflow)
        if price_change > 0.0 && flow > 0.0 {
            return Some(false); // Bearish
        }

        None
    }

    /// Get the MA period.
    pub fn ma_period(&self) -> usize {
        self.ma_period
    }

    /// Get the momentum period.
    pub fn momentum_period(&self) -> usize {
        self.momentum_period
    }

    /// Get the lookback period.
    pub fn lookback_period(&self) -> usize {
        self.lookback_period
    }
}

impl Default for ExchangeInflow {
    fn default() -> Self {
        Self {
            ma_period: 20,
            momentum_period: 10,
            lookback_period: 50,
        }
    }
}

impl TechnicalIndicator for ExchangeInflow {
    fn name(&self) -> &str {
        "Exchange Inflow"
    }

    fn min_periods(&self) -> usize {
        self.lookback_period.max(self.ma_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate_proxy(data);
        Ok(IndicatorOutput::triple(output.net_flow, output.flow_ma, output.cumulative_flow))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_balance_data() -> Vec<f64> {
        // Simulate exchange balances with trend and variation
        let mut balances = Vec::with_capacity(60);
        let mut balance = 100000.0;
        for i in 0..60 {
            // General downtrend (outflow) with noise
            balance += (i as f64 * 0.4).sin() * 500.0 - 100.0;
            balances.push(balance.max(0.0));
        }
        balances
    }

    fn make_ohlcv_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..60)
            .map(|i| 100.0 + i as f64 * 0.3 + (i as f64 * 0.3).sin() * 3.0)
            .collect();
        let n = close.len();
        let high: Vec<f64> = close.iter().map(|c| c + 2.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 2.0).collect();
        let open = close.clone();
        let volume: Vec<f64> = (0..n).map(|i| 10000.0 + (i as f64 * 0.5).sin() * 5000.0).collect();

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_exchange_flow_from_balances() {
        let balances = make_balance_data();
        let ei = ExchangeInflow::new(10, 5, 30).unwrap();
        let output = ei.calculate_from_balances(&balances);

        assert_eq!(output.net_flow.len(), balances.len());
        assert_eq!(output.flow_ma.len(), balances.len());
        assert_eq!(output.cumulative_flow.len(), balances.len());

        // First net flow should be 0 (no previous balance)
        assert_eq!(output.net_flow[0], 0.0);
    }

    #[test]
    fn test_exchange_flow_from_flows() {
        let flows: Vec<f64> = (0..50).map(|i| (i as f64 * 0.4).sin() * 1000.0).collect();
        let ei = ExchangeInflow::new(10, 5, 30).unwrap();
        let output = ei.calculate(&flows);

        assert_eq!(output.net_flow.len(), flows.len());
        assert_eq!(output.net_flow, flows);
    }

    #[test]
    fn test_exchange_flow_cumulative() {
        let flows = vec![100.0, 200.0, -50.0, 150.0, -100.0];
        let ei = ExchangeInflow::new(5, 1, 20).unwrap();
        let output = ei.calculate(&flows);

        // Check cumulative calculation
        assert_eq!(output.cumulative_flow[0], 100.0);
        assert_eq!(output.cumulative_flow[1], 300.0);
        assert_eq!(output.cumulative_flow[2], 250.0);
        assert_eq!(output.cumulative_flow[3], 400.0);
        assert_eq!(output.cumulative_flow[4], 300.0);
    }

    #[test]
    fn test_exchange_flow_ma() {
        let flows = vec![100.0; 30];
        let ei = ExchangeInflow::new(10, 5, 20).unwrap();
        let output = ei.calculate(&flows);

        // MA of constant values should equal that constant
        for i in 9..30 {
            assert!((output.flow_ma[i] - 100.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_exchange_flow_proxy() {
        let data = make_ohlcv_data();
        let ei = ExchangeInflow::new(10, 5, 30).unwrap();
        let output = ei.calculate_proxy(&data);

        assert_eq!(output.net_flow.len(), data.close.len());

        // First flow should be 0 (no previous data)
        assert_eq!(output.net_flow[0], 0.0);
    }

    #[test]
    fn test_exchange_flow_signal_range() {
        let balances = make_balance_data();
        let ei = ExchangeInflow::new(10, 5, 30).unwrap();
        let output = ei.calculate_from_balances(&balances);

        // Signal should be in range -100 to 100
        for i in 29..balances.len() {
            assert!(output.signal[i] >= -100.0 && output.signal[i] <= 100.0);
        }
    }

    #[test]
    fn test_exchange_flow_interpretation() {
        let ei = ExchangeInflow::default();

        assert_eq!(ei.interpret(80.0), FlowSignal::StrongInflow);
        assert_eq!(ei.interpret(40.0), FlowSignal::Inflow);
        assert_eq!(ei.interpret(0.0), FlowSignal::Neutral);
        assert_eq!(ei.interpret(-40.0), FlowSignal::Outflow);
        assert_eq!(ei.interpret(-80.0), FlowSignal::StrongOutflow);
    }

    #[test]
    fn test_exchange_flow_divergence() {
        let ei = ExchangeInflow::default();

        // Bullish divergence: price down, outflow (accumulation)
        assert_eq!(ei.check_divergence(-5.0, -100.0), Some(true));

        // Bearish divergence: price up, inflow (distribution)
        assert_eq!(ei.check_divergence(5.0, 100.0), Some(false));

        // No divergence
        assert_eq!(ei.check_divergence(5.0, -100.0), None);
        assert_eq!(ei.check_divergence(-5.0, 100.0), None);

        // Edge cases
        assert_eq!(ei.check_divergence(0.0, 100.0), None);
        assert_eq!(ei.check_divergence(5.0, 0.0), None);
    }

    #[test]
    fn test_exchange_flow_technical_indicator() {
        let data = make_ohlcv_data();
        let ei = ExchangeInflow::new(10, 5, 30).unwrap();

        assert_eq!(ei.name(), "Exchange Inflow");
        assert_eq!(ei.min_periods(), 30);

        let output = ei.compute(&data).unwrap();
        assert!(output.values.contains_key("net_flow"));
        assert!(output.values.contains_key("flow_ma"));
        assert!(output.values.contains_key("cumulative_flow"));
        assert!(output.values.contains_key("momentum"));
        assert!(output.values.contains_key("signal"));
    }

    #[test]
    fn test_exchange_flow_validation() {
        assert!(ExchangeInflow::new(4, 5, 30).is_err());
        assert!(ExchangeInflow::new(10, 0, 30).is_err());
        assert!(ExchangeInflow::new(10, 5, 19).is_err());
    }

    #[test]
    fn test_exchange_flow_empty_input() {
        let ei = ExchangeInflow::default();
        let output = ei.calculate(&[]);

        assert!(output.net_flow.is_empty());
        assert!(output.flow_ma.is_empty());
    }

    #[test]
    fn test_exchange_flow_single_value() {
        let ei = ExchangeInflow::default();
        let output = ei.calculate(&[100.0]);

        assert_eq!(output.net_flow.len(), 1);
        assert_eq!(output.cumulative_flow[0], 100.0);
    }

    #[test]
    fn test_exchange_flow_default() {
        let ei = ExchangeInflow::default();
        assert_eq!(ei.ma_period(), 20);
        assert_eq!(ei.momentum_period(), 10);
        assert_eq!(ei.lookback_period(), 50);
    }

    #[test]
    fn test_flow_signals_iterator() {
        let balances = make_balance_data();
        let ei = ExchangeInflow::new(10, 5, 30).unwrap();
        let output = ei.calculate_from_balances(&balances);
        let signals = ei.flow_signals(&output);

        assert_eq!(signals.len(), balances.len());
    }
}
