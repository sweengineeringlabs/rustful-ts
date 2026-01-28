//! Extended Crypto/On-Chain Style Indicators
//!
//! Additional cryptocurrency-specific analysis indicators using price/volume proxies.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Active Addresses Proxy - Network activity indicator using volume momentum
#[derive(Debug, Clone)]
pub struct ActiveAddressesProxy {
    period: usize,
    smooth_period: usize,
}

impl ActiveAddressesProxy {
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, smooth_period })
    }

    /// Calculate proxy for active addresses using volume patterns
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Volume weighted price activity
            let mut activity_sum = 0.0;
            let mut vol_sum = 0.0;

            for j in start..=i {
                // Activity proxy: volume * absolute return
                let ret = if j > 0 { (close[j] / close[j - 1] - 1.0).abs() } else { 0.0 };
                activity_sum += volume[j] * ret;
                vol_sum += volume[j];
            }

            // Normalize by average volume
            if vol_sum > 1e-10 {
                result[i] = activity_sum / (vol_sum / self.period as f64);
            }
        }

        // Apply smoothing
        if self.smooth_period > 1 {
            let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
            for i in 1..n {
                result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
            }
        }

        result
    }
}

impl TechnicalIndicator for ActiveAddressesProxy {
    fn name(&self) -> &str {
        "Active Addresses Proxy"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Exchange Flow Proxy - Simulates exchange inflow/outflow using price-volume dynamics
#[derive(Debug, Clone)]
pub struct ExchangeFlowProxy {
    period: usize,
}

impl ExchangeFlowProxy {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate exchange flow proxy: positive = outflow (bullish), negative = inflow (bearish)
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Exchange flow proxy based on price-volume divergence
            // High volume with price increase = outflow (supply absorption)
            // High volume with price decrease = inflow (selling pressure)
            let mut flow_sum = 0.0;
            let mut vol_baseline = 0.0;

            for j in start..=i {
                vol_baseline += volume[j];
            }
            vol_baseline /= self.period as f64;

            for j in (start + 1)..=i {
                let price_change = close[j] - close[j - 1];
                let vol_deviation = volume[j] / vol_baseline - 1.0;

                // Positive flow = outflow (price up with high volume)
                // Negative flow = inflow (price down with high volume)
                flow_sum += price_change.signum() * vol_deviation;
            }

            result[i] = flow_sum / self.period as f64 * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for ExchangeFlowProxy {
    fn name(&self) -> &str {
        "Exchange Flow Proxy"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// HODL Behavior Proxy - Estimates holding behavior from price volatility patterns
#[derive(Debug, Clone)]
pub struct HODLBehaviorProxy {
    short_period: usize,
    long_period: usize,
}

impl HODLBehaviorProxy {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
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
        Ok(Self { short_period, long_period })
    }

    /// Calculate HODL score: higher = more HODLing behavior
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            // Short-term volatility
            let short_start = i.saturating_sub(self.short_period);
            let mut short_vol_sum = 0.0;
            for j in (short_start + 1)..=i {
                let ret = (close[j] / close[j - 1] - 1.0).abs();
                short_vol_sum += ret;
            }
            let short_volatility = short_vol_sum / self.short_period as f64;

            // Long-term volatility
            let long_start = i.saturating_sub(self.long_period);
            let mut long_vol_sum = 0.0;
            for j in (long_start + 1)..=i {
                let ret = (close[j] / close[j - 1] - 1.0).abs();
                long_vol_sum += ret;
            }
            let long_volatility = long_vol_sum / self.long_period as f64;

            // Volume trend
            let short_vol: f64 = volume[short_start..=i].iter().sum::<f64>() / self.short_period as f64;
            let long_vol: f64 = volume[long_start..=i].iter().sum::<f64>() / self.long_period as f64;

            // HODL score: low volatility + low volume = high HODLing
            // Inverted so higher = more HODLing
            let vol_ratio = if long_volatility > 1e-10 { short_volatility / long_volatility } else { 1.0 };
            let volume_ratio = if long_vol > 1e-10 { short_vol / long_vol } else { 1.0 };

            // Score inversely proportional to activity
            result[i] = 100.0 / (1.0 + vol_ratio * volume_ratio);
        }

        result
    }
}

impl TechnicalIndicator for HODLBehaviorProxy {
    fn name(&self) -> &str {
        "HODL Behavior Proxy"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Network Value Momentum - Rate of change in market cap proxy
#[derive(Debug, Clone)]
pub struct NetworkValueMomentum {
    period: usize,
    smooth_period: usize,
}

impl NetworkValueMomentum {
    pub fn new(period: usize, smooth_period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period, smooth_period })
    }

    /// Calculate network value momentum
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Create network value proxy (price * cumulative volume as circulation proxy)
        let mut network_value = vec![0.0; n];
        let mut cum_vol = 0.0;
        for i in 0..n {
            cum_vol += volume[i];
            network_value[i] = close[i] * (cum_vol / (i + 1) as f64); // Normalized
        }

        // Calculate momentum
        for i in self.period..n {
            if network_value[i - self.period] > 1e-10 {
                result[i] = (network_value[i] / network_value[i - self.period] - 1.0) * 100.0;
            }
        }

        // Apply EMA smoothing
        let alpha = 2.0 / (self.smooth_period as f64 + 1.0);
        for i in 1..n {
            result[i] = alpha * result[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }
}

impl TechnicalIndicator for NetworkValueMomentum {
    fn name(&self) -> &str {
        "Network Value Momentum"
    }

    fn min_periods(&self) -> usize {
        self.period + self.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Transaction Velocity Proxy - Estimates transaction throughput
#[derive(Debug, Clone)]
pub struct TransactionVelocityProxy {
    period: usize,
}

impl TransactionVelocityProxy {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate transaction velocity proxy
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Transaction velocity = Volume turnover relative to "market cap"
            let avg_price: f64 = close[start..=i].iter().sum::<f64>() / self.period as f64;
            let total_volume: f64 = volume[start..=i].iter().sum();

            // Velocity ratio
            if avg_price > 1e-10 {
                result[i] = total_volume / (avg_price * self.period as f64);
            }
        }

        // Normalize to z-score style
        let mut mean = 0.0;
        let mut count = 0;
        for i in self.period..n {
            if result[i] > 0.0 {
                mean += result[i];
                count += 1;
            }
        }
        if count > 0 {
            mean /= count as f64;
            for i in self.period..n {
                result[i] = (result[i] / mean - 1.0) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for TransactionVelocityProxy {
    fn name(&self) -> &str {
        "Transaction Velocity Proxy"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Crypto Momentum Score - Composite momentum for crypto assets
#[derive(Debug, Clone)]
pub struct CryptoMomentumScore {
    short_period: usize,
    long_period: usize,
}

impl CryptoMomentumScore {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
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
        Ok(Self { short_period, long_period })
    }

    /// Calculate crypto momentum score combining price, volume, and volatility
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            // Price momentum (short-term)
            let price_mom_short = if close[i - self.short_period] > 1e-10 {
                (close[i] / close[i - self.short_period] - 1.0) * 100.0
            } else {
                0.0
            };

            // Price momentum (long-term)
            let price_mom_long = if close[i - self.long_period] > 1e-10 {
                (close[i] / close[i - self.long_period] - 1.0) * 100.0
            } else {
                0.0
            };

            // Volume momentum
            let short_start = i.saturating_sub(self.short_period);
            let long_start = i.saturating_sub(self.long_period);
            let short_vol: f64 = volume[short_start..=i].iter().sum::<f64>() / self.short_period as f64;
            let long_vol: f64 = volume[long_start..=i].iter().sum::<f64>() / self.long_period as f64;
            let vol_mom = if long_vol > 1e-10 { (short_vol / long_vol - 1.0) * 100.0 } else { 0.0 };

            // Volatility (using recent range)
            let mut vol_sum = 0.0;
            for j in (short_start + 1)..=i {
                vol_sum += (close[j] / close[j - 1] - 1.0).abs();
            }
            let volatility = vol_sum / self.short_period as f64 * 100.0;

            // Composite score: momentum + volume confirmation - volatility penalty
            let momentum_score = price_mom_short * 0.4 + price_mom_long * 0.3;
            let volume_score = vol_mom * 0.2;
            let vol_penalty = volatility * 0.1;

            result[i] = momentum_score + volume_score - vol_penalty;
        }

        result
    }
}

impl TechnicalIndicator for CryptoMomentumScore {
    fn name(&self) -> &str {
        "Crypto Momentum Score"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..40).map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.3).sin() * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = (0..40).map(|i| 1000.0 + (i as f64 * 0.5).sin() * 500.0).collect();

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_active_addresses_proxy() {
        let data = make_test_data();
        let aap = ActiveAddressesProxy::new(10, 5).unwrap();
        let result = aap.calculate(&data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
        assert!(result[20] >= 0.0); // Should be non-negative
    }

    #[test]
    fn test_exchange_flow_proxy() {
        let data = make_test_data();
        let efp = ExchangeFlowProxy::new(10).unwrap();
        let result = efp.calculate(&data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_hodl_behavior_proxy() {
        let data = make_test_data();
        let hodl = HODLBehaviorProxy::new(7, 21).unwrap();
        let result = hodl.calculate(&data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
        // HODL score should be bounded
        for i in 25..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_network_value_momentum() {
        let data = make_test_data();
        let nvm = NetworkValueMomentum::new(10, 5).unwrap();
        let result = nvm.calculate(&data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_transaction_velocity_proxy() {
        let data = make_test_data();
        let tvp = TransactionVelocityProxy::new(10).unwrap();
        let result = tvp.calculate(&data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_crypto_momentum_score() {
        let data = make_test_data();
        let cms = CryptoMomentumScore::new(7, 21).unwrap();
        let result = cms.calculate(&data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_validation() {
        assert!(ActiveAddressesProxy::new(2, 5).is_err());
        assert!(ExchangeFlowProxy::new(2).is_err());
        assert!(HODLBehaviorProxy::new(10, 5).is_err()); // long <= short
        assert!(NetworkValueMomentum::new(2, 5).is_err());
        assert!(TransactionVelocityProxy::new(2).is_err());
        assert!(CryptoMomentumScore::new(10, 5).is_err()); // long <= short
    }
}
