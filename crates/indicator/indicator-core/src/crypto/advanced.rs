//! Advanced Crypto & On-Chain Indicators
//!
//! Additional cryptocurrency-specific indicators for valuation and analysis.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// PUELL Multiple - Miner profitability indicator
#[derive(Debug, Clone)]
pub struct PuellMultiple {
    period: usize,
}

impl PuellMultiple {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate PUELL Multiple using daily issuance proxy (volume as stand-in)
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        // Calculate daily issuance value (volume * price as proxy)
        let mut daily_value: Vec<f64> = close.iter()
            .zip(volume.iter())
            .map(|(c, v)| c * v)
            .collect();

        // Calculate 365-day MA of daily issuance (use period for shorter term)
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let ma: f64 = daily_value[start..=i].iter().sum::<f64>() / self.period as f64;

            if ma > 0.0 {
                result[i] = daily_value[i] / ma;
            }
        }
        result
    }
}

impl TechnicalIndicator for PuellMultiple {
    fn name(&self) -> &str {
        "PUELL Multiple"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Reserve Risk - Long-term holder confidence indicator
#[derive(Debug, Clone)]
pub struct ReserveRisk {
    period: usize,
}

impl ReserveRisk {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Reserve Risk proxy using price and holding behavior
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        // HODL Bank proxy (cumulative opportunity cost)
        let mut hodl_bank = vec![0.0; n];
        for i in 1..n {
            // Simulate HODL bank accumulation
            hodl_bank[i] = hodl_bank[i - 1] + close[i] / 1000.0;
        }

        for i in self.period..n {
            if hodl_bank[i] > 0.0 {
                // Reserve Risk = Price / HODL Bank
                result[i] = close[i] / hodl_bank[i];
            }
        }
        result
    }
}

impl TechnicalIndicator for ReserveRisk {
    fn name(&self) -> &str {
        "Reserve Risk"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Stock-to-Flow Ratio - Scarcity model indicator
#[derive(Debug, Clone)]
pub struct StockToFlow {
    period: usize,
}

impl StockToFlow {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Stock-to-Flow using volume as flow proxy
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        // Cumulative stock
        let mut stock = 0.0;

        for i in self.period..n {
            stock += volume[i];

            // Average annual flow (using period as proxy)
            let start = i.saturating_sub(self.period);
            let flow: f64 = volume[start..=i].iter().sum::<f64>();

            if flow > 0.0 {
                // Stock-to-Flow = Stock / Annual Flow
                result[i] = stock / flow;
            }
        }
        result
    }
}

impl TechnicalIndicator for StockToFlow {
    fn name(&self) -> &str {
        "Stock to Flow"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Thermocap Multiple - Valuation against cumulative security spend
#[derive(Debug, Clone)]
pub struct ThermocapMultiple {
    period: usize,
}

impl ThermocapMultiple {
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Thermocap Multiple
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        // Cumulative thermocap (security spend proxy)
        let mut thermocap = 0.0;

        for i in 1..n {
            // Thermocap accumulates block rewards (proxy using volume fraction)
            thermocap += volume[i] * 0.001;

            if i >= self.period && thermocap > 0.0 {
                // Market Cap / Thermocap
                let market_cap = close[i] * volume[i];
                result[i] = market_cap / thermocap;
            }
        }
        result
    }
}

impl TechnicalIndicator for ThermocapMultiple {
    fn name(&self) -> &str {
        "Thermocap Multiple"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Coin Days Destroyed - Long-term holder activity indicator
#[derive(Debug, Clone)]
pub struct CoinDaysDestroyed {
    period: usize,
}

impl CoinDaysDestroyed {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate CDD using volume and holding time proxy
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            // CDD proxy: volume weighted by price change magnitude
            let price_age: f64 = (close[i] - close[i - self.period]).abs() / close[i - self.period];
            result[i] = volume[i] * price_age * self.period as f64;
        }
        result
    }

    /// Calculate CDD normalized by supply
    pub fn calculate_normalized(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let raw = self.calculate(close, volume);
        let n = raw.len();
        let mut result = vec![0.0; n];

        // Normalize by moving average
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let avg: f64 = raw[start..=i].iter().sum::<f64>() / self.period as f64;
            if avg > 0.0 {
                result[i] = raw[i] / avg;
            }
        }
        result
    }
}

impl TechnicalIndicator for CoinDaysDestroyed {
    fn name(&self) -> &str {
        "Coin Days Destroyed"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Realized Cap HODL Waves - Distribution of realized cap by age
#[derive(Debug, Clone)]
pub struct RealizedCapAge {
    short_period: usize,
    long_period: usize,
}

impl RealizedCapAge {
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

    /// Calculate ratio of short-term to long-term realized cap
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            // Short-term realized cap (recent purchases)
            let short_start = i.saturating_sub(self.short_period);
            let short_cap: f64 = (short_start..=i)
                .map(|j| close[j] * volume[j])
                .sum();

            // Long-term realized cap
            let long_start = i.saturating_sub(self.long_period);
            let long_cap: f64 = (long_start..=i)
                .map(|j| close[j] * volume[j])
                .sum();

            if long_cap > 0.0 {
                result[i] = short_cap / long_cap;
            }
        }
        result
    }
}

impl TechnicalIndicator for RealizedCapAge {
    fn name(&self) -> &str {
        "Realized Cap Age Distribution"
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

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        let close = vec![100.0, 102.0, 101.0, 105.0, 103.0, 108.0, 106.0, 112.0, 110.0, 115.0,
                        118.0, 116.0, 120.0, 122.0, 119.0, 125.0, 123.0, 128.0, 126.0, 130.0,
                        132.0, 129.0, 135.0, 133.0, 138.0, 140.0, 137.0, 142.0, 145.0, 148.0];
        let volume = vec![1000.0, 1200.0, 1100.0, 1500.0, 1300.0, 1600.0, 1400.0, 1800.0, 1700.0, 2000.0,
                         2200.0, 2100.0, 2400.0, 2500.0, 2300.0, 2600.0, 2400.0, 2800.0, 2700.0, 3000.0,
                         3200.0, 3100.0, 3400.0, 3300.0, 3600.0, 3800.0, 3500.0, 4000.0, 4200.0, 4500.0];
        (close, volume)
    }

    #[test]
    fn test_puell_multiple() {
        let (close, volume) = make_test_data();
        let puell = PuellMultiple::new(10).unwrap();
        let result = puell.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_reserve_risk() {
        let (close, volume) = make_test_data();
        let rr = ReserveRisk::new(10).unwrap();
        let result = rr.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_stock_to_flow() {
        let (close, volume) = make_test_data();
        let stf = StockToFlow::new(10).unwrap();
        let result = stf.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_thermocap_multiple() {
        let (close, volume) = make_test_data();
        let tc = ThermocapMultiple::new(10).unwrap();
        let result = tc.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        assert!(result[15] > 0.0);
    }

    #[test]
    fn test_coin_days_destroyed() {
        let (close, volume) = make_test_data();
        let cdd = CoinDaysDestroyed::new(5).unwrap();
        let result = cdd.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        assert!(result[10] >= 0.0);
    }

    #[test]
    fn test_realized_cap_age() {
        let (close, volume) = make_test_data();
        let rca = RealizedCapAge::new(5, 15).unwrap();
        let result = rca.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        assert!(result[20] > 0.0 && result[20] < 1.0);
    }
}
