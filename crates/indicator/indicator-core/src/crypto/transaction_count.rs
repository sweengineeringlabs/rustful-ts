//! Transaction Count (IND-266)
//!
//! Daily transaction count indicator for cryptocurrency networks.
//! Measures the number of on-chain transactions processed by the network.
//!
//! # Concept
//! Transaction count is a key on-chain metric that reflects network utilization.
//! It differs from active addresses in that one address can generate multiple transactions.
//! Increasing transaction counts indicate network growth and usage.
//!
//! # Data Requirements
//! This indicator requires on-chain data (transaction counts) for accurate results.
//! When used with OHLCV data, it provides a proxy based on volume patterns.

use indicator_spi::{IndicatorError, IndicatorOutput, OHLCVSeries, Result, TechnicalIndicator};

/// Output from the Transaction Count calculation.
#[derive(Debug, Clone)]
pub struct TransactionCountOutput {
    /// Raw transaction count values (or proxy).
    pub tx_count: Vec<f64>,
    /// Moving average of transaction counts.
    pub ma: Vec<f64>,
    /// Transaction count momentum.
    pub momentum: Vec<f64>,
    /// Transactions per active address proxy (network efficiency).
    pub tx_density: Vec<f64>,
    /// Normalized transaction score (0-100).
    pub score: Vec<f64>,
}

/// Transaction activity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionActivity {
    /// Network congestion / very high usage.
    Congested,
    /// High transaction activity.
    High,
    /// Normal transaction levels.
    Normal,
    /// Low transaction activity.
    Low,
    /// Very low / quiet network.
    Quiet,
}

/// Transaction Count (IND-266)
///
/// Measures network transaction throughput and trends.
///
/// # Calculation (with on-chain data)
/// ```text
/// MA = SMA(TxCount, ma_period)
/// Momentum = (TxCount - TxCount[momentum_period]) / TxCount[momentum_period] * 100
/// TxDensity = TxCount / ActiveAddresses (if available)
/// Score = Percentile rank over lookback
/// ```
///
/// # Calculation (proxy from OHLCV)
/// ```text
/// TxProxy = Volume / TypicalPrice (dollar volume normalized)
/// Adjusted by volatility to approximate transaction patterns
/// ```
///
/// # Example
/// ```
/// use indicator_core::crypto::TransactionCount;
///
/// let tc = TransactionCount::new(20, 10, 50).unwrap();
/// let tx_counts = vec![500000.0, 520000.0, 480000.0, 550000.0];
/// let output = tc.calculate(&tx_counts);
/// ```
#[derive(Debug, Clone)]
pub struct TransactionCount {
    /// Period for moving average calculation.
    ma_period: usize,
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Lookback period for score normalization.
    lookback_period: usize,
}

impl TransactionCount {
    /// Create a new Transaction Count indicator.
    ///
    /// # Arguments
    /// * `ma_period` - Period for moving average (minimum 5)
    /// * `momentum_period` - Period for momentum calculation (minimum 1)
    /// * `lookback_period` - Period for percentile ranking (minimum 20)
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

    /// Calculate transaction count metrics from on-chain data.
    ///
    /// # Arguments
    /// * `tx_counts` - Slice of daily transaction counts
    ///
    /// # Returns
    /// TransactionCountOutput containing all metrics.
    pub fn calculate(&self, tx_counts: &[f64]) -> TransactionCountOutput {
        let n = tx_counts.len();
        let tx_count = tx_counts.to_vec();
        let mut ma = vec![0.0; n];
        let mut momentum = vec![0.0; n];
        let tx_density = vec![0.0; n]; // Requires active addresses data
        let mut score = vec![0.0; n];

        if n < self.ma_period {
            return TransactionCountOutput {
                tx_count,
                ma,
                momentum,
                tx_density,
                score,
            };
        }

        // Calculate moving average
        for i in (self.ma_period - 1)..n {
            let start = i + 1 - self.ma_period;
            ma[i] = tx_counts[start..=i].iter().sum::<f64>() / self.ma_period as f64;
        }

        // Calculate momentum
        for i in self.momentum_period..n {
            if tx_counts[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (tx_counts[i] / tx_counts[i - self.momentum_period] - 1.0) * 100.0;
            }
        }

        // Calculate percentile score
        for i in (self.lookback_period - 1)..n {
            let start = i + 1 - self.lookback_period;
            let window = &tx_counts[start..=i];

            let current = tx_counts[i];
            let count_below = window.iter().filter(|&&x| x < current).count();
            score[i] = count_below as f64 / self.lookback_period as f64 * 100.0;
        }

        TransactionCountOutput {
            tx_count,
            ma,
            momentum,
            tx_density,
            score,
        }
    }

    /// Calculate with active addresses to get transaction density.
    ///
    /// # Arguments
    /// * `tx_counts` - Daily transaction counts
    /// * `active_addresses` - Daily active address counts
    pub fn calculate_with_addresses(
        &self,
        tx_counts: &[f64],
        active_addresses: &[f64],
    ) -> TransactionCountOutput {
        let n = tx_counts.len().min(active_addresses.len());
        let mut output = self.calculate(&tx_counts[..n]);

        // Calculate transaction density (transactions per active address)
        for i in 0..n {
            if active_addresses[i] > 1e-10 {
                output.tx_density[i] = tx_counts[i] / active_addresses[i];
            }
        }

        output
    }

    /// Calculate transaction count proxy from OHLCV data.
    pub fn calculate_proxy(&self, data: &OHLCVSeries) -> TransactionCountOutput {
        let n = data.close.len();
        let mut tx_count = vec![0.0; n];
        let mut ma = vec![0.0; n];
        let mut momentum = vec![0.0; n];
        let mut tx_density = vec![0.0; n];
        let mut score = vec![0.0; n];

        if n < self.ma_period {
            return TransactionCountOutput {
                tx_count,
                ma,
                momentum,
                tx_density,
                score,
            };
        }

        // Create transaction count proxy
        // Higher volume relative to price = more transactions
        for i in 0..n {
            let typical_price = (data.high[i] + data.low[i] + data.close[i]) / 3.0;
            if typical_price > 1e-10 {
                // Dollar volume normalized by price
                tx_count[i] = data.volume[i] / typical_price;
            }
        }

        // Scale to reasonable range
        let max_val = tx_count.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_val > 1e-10 {
            for i in 0..n {
                tx_count[i] = tx_count[i] / max_val * 1000000.0;
            }
        }

        // Calculate moving average
        for i in (self.ma_period - 1)..n {
            let start = i + 1 - self.ma_period;
            ma[i] = tx_count[start..=i].iter().sum::<f64>() / self.ma_period as f64;
        }

        // Calculate momentum
        for i in self.momentum_period..n {
            if tx_count[i - self.momentum_period].abs() > 1e-10 {
                momentum[i] = (tx_count[i] / tx_count[i - self.momentum_period] - 1.0) * 100.0;
            }
        }

        // Calculate transaction density proxy (volume relative to price range)
        for i in 0..n {
            let range = data.high[i] - data.low[i];
            if range > 1e-10 {
                tx_density[i] = data.volume[i] / (range * data.close[i]);
            }
        }

        // Calculate percentile score
        for i in (self.lookback_period - 1)..n {
            let start = i + 1 - self.lookback_period;
            let window = &tx_count[start..=i];

            let current = tx_count[i];
            let count_below = window.iter().filter(|&&x| x < current).count();
            score[i] = count_below as f64 / self.lookback_period as f64 * 100.0;
        }

        TransactionCountOutput {
            tx_count,
            ma,
            momentum,
            tx_density,
            score,
        }
    }

    /// Get activity classification.
    pub fn interpret(&self, score: f64, momentum: f64) -> TransactionActivity {
        if score >= 90.0 || (score >= 75.0 && momentum > 20.0) {
            TransactionActivity::Congested
        } else if score >= 65.0 {
            TransactionActivity::High
        } else if score >= 35.0 {
            TransactionActivity::Normal
        } else if score >= 15.0 {
            TransactionActivity::Low
        } else {
            TransactionActivity::Quiet
        }
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

impl Default for TransactionCount {
    fn default() -> Self {
        Self {
            ma_period: 20,
            momentum_period: 10,
            lookback_period: 50,
        }
    }
}

impl TechnicalIndicator for TransactionCount {
    fn name(&self) -> &str {
        "Transaction Count"
    }

    fn min_periods(&self) -> usize {
        self.lookback_period.max(self.ma_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate_proxy(data);
        Ok(IndicatorOutput::triple(output.tx_count, output.ma, output.momentum))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        // Simulate transaction counts with trend and variation
        (0..60)
            .map(|i| 500000.0 + i as f64 * 2000.0 + (i as f64 * 0.4).sin() * 20000.0)
            .collect()
    }

    fn make_ohlcv_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..60)
            .map(|i| 100.0 + i as f64 * 0.5 + (i as f64 * 0.3).sin() * 2.0)
            .collect();
        let n = close.len();
        let high: Vec<f64> = close.iter().map(|c| c + 2.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 2.0).collect();
        let open = close.clone();
        let volume: Vec<f64> = (0..n).map(|i| 10000.0 + (i as f64 * 0.5).sin() * 5000.0).collect();

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_transaction_count_basic() {
        let data = make_test_data();
        let tc = TransactionCount::new(10, 5, 30).unwrap();
        let output = tc.calculate(&data);

        assert_eq!(output.tx_count.len(), data.len());
        assert_eq!(output.ma.len(), data.len());
        assert_eq!(output.momentum.len(), data.len());
        assert_eq!(output.score.len(), data.len());
    }

    #[test]
    fn test_transaction_count_ma() {
        let data = vec![500000.0; 30];
        let tc = TransactionCount::new(10, 5, 20).unwrap();
        let output = tc.calculate(&data);

        // MA of constant values should equal that constant
        for i in 9..30 {
            assert!((output.ma[i] - 500000.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_transaction_count_momentum() {
        // Increasing values
        let data: Vec<f64> = (0..50).map(|i| 500000.0 + i as f64 * 5000.0).collect();
        let tc = TransactionCount::new(10, 10, 30).unwrap();
        let output = tc.calculate(&data);

        // Momentum should be positive for increasing data
        for i in 15..50 {
            assert!(output.momentum[i] > 0.0);
        }
    }

    #[test]
    fn test_transaction_count_with_addresses() {
        let tx_counts: Vec<f64> = (0..50).map(|i| 500000.0 + i as f64 * 1000.0).collect();
        let active_addresses: Vec<f64> = (0..50).map(|i| 100000.0 + i as f64 * 100.0).collect();

        let tc = TransactionCount::new(10, 5, 30).unwrap();
        let output = tc.calculate_with_addresses(&tx_counts, &active_addresses);

        // Transaction density should be approximately 5 (500000 / 100000)
        assert!(output.tx_density[0] > 4.0 && output.tx_density[0] < 6.0);
    }

    #[test]
    fn test_transaction_count_proxy() {
        let data = make_ohlcv_data();
        let tc = TransactionCount::new(10, 5, 30).unwrap();
        let output = tc.calculate_proxy(&data);

        assert_eq!(output.tx_count.len(), data.close.len());

        // Proxy values should be positive
        for i in 1..output.tx_count.len() {
            assert!(output.tx_count[i] >= 0.0);
        }
    }

    #[test]
    fn test_transaction_count_interpretation() {
        let tc = TransactionCount::default();

        assert_eq!(tc.interpret(95.0, 0.0), TransactionActivity::Congested);
        assert_eq!(tc.interpret(80.0, 25.0), TransactionActivity::Congested);
        assert_eq!(tc.interpret(70.0, 0.0), TransactionActivity::High);
        assert_eq!(tc.interpret(50.0, 0.0), TransactionActivity::Normal);
        assert_eq!(tc.interpret(25.0, 0.0), TransactionActivity::Low);
        assert_eq!(tc.interpret(10.0, 0.0), TransactionActivity::Quiet);
    }

    #[test]
    fn test_transaction_count_score_bounded() {
        let data = make_test_data();
        let tc = TransactionCount::new(10, 5, 30).unwrap();
        let output = tc.calculate(&data);

        for i in 29..data.len() {
            assert!(output.score[i] >= 0.0 && output.score[i] <= 100.0);
        }
    }

    #[test]
    fn test_transaction_count_technical_indicator() {
        let data = make_ohlcv_data();
        let tc = TransactionCount::new(10, 5, 30).unwrap();

        assert_eq!(tc.name(), "Transaction Count");
        assert_eq!(tc.min_periods(), 30);

        let output = tc.compute(&data).unwrap();
        assert!(output.values.contains_key("tx_count"));
        assert!(output.values.contains_key("ma"));
        assert!(output.values.contains_key("momentum"));
        assert!(output.values.contains_key("tx_density"));
        assert!(output.values.contains_key("score"));
    }

    #[test]
    fn test_transaction_count_validation() {
        assert!(TransactionCount::new(4, 5, 30).is_err());
        assert!(TransactionCount::new(10, 0, 30).is_err());
        assert!(TransactionCount::new(10, 5, 19).is_err());
    }

    #[test]
    fn test_transaction_count_empty_input() {
        let tc = TransactionCount::default();
        let output = tc.calculate(&[]);

        assert!(output.tx_count.is_empty());
        assert!(output.ma.is_empty());
    }

    #[test]
    fn test_transaction_count_default() {
        let tc = TransactionCount::default();
        assert_eq!(tc.ma_period(), 20);
        assert_eq!(tc.momentum_period(), 10);
        assert_eq!(tc.lookback_period(), 50);
    }
}
