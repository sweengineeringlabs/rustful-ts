//! Forex/Currency-Specific Indicators
//!
//! Indicators for analyzing currency and foreign exchange markets.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Carry Trade Index - Yield differential proxy
#[derive(Debug, Clone)]
pub struct CarryTradeIndex {
    period: usize,
}

impl CarryTradeIndex {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate carry trade index using trend and volatility
    /// Positive = carry-favorable, Negative = unfavorable
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let returns: Vec<f64> = ((start + 1)..=i)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();

            if returns.is_empty() {
                continue;
            }

            let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean_ret).powi(2))
                .sum::<f64>() / returns.len() as f64;
            let vol = variance.sqrt();

            // Carry index: return / volatility (Sharpe-like)
            if vol > 0.0 {
                result[i] = mean_ret / vol * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for CarryTradeIndex {
    fn name(&self) -> &str {
        "Carry Trade Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// FX Volatility Term Structure - Vol term structure slope
#[derive(Debug, Clone)]
pub struct FXVolatilityTerm {
    short_period: usize,
    long_period: usize,
}

impl FXVolatilityTerm {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period < 2 || long_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "periods must be at least 2".to_string(),
            });
        }
        if short_period >= long_period {
            return Err(IndicatorError::InvalidParameter {
                name: "short_period".to_string(),
                reason: "must be less than long_period".to_string(),
            });
        }
        Ok(Self { short_period, long_period })
    }

    /// Calculate FX volatility term structure
    /// Positive = long-term vol > short-term, Negative = inverted
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.long_period..n {
            // Short-term volatility
            let short_start = i.saturating_sub(self.short_period);
            let short_returns: Vec<f64> = ((short_start + 1)..=i)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();
            let short_vol = if !short_returns.is_empty() {
                let mean = short_returns.iter().sum::<f64>() / short_returns.len() as f64;
                let var = short_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                    / short_returns.len() as f64;
                var.sqrt()
            } else {
                0.0
            };

            // Long-term volatility
            let long_start = i.saturating_sub(self.long_period);
            let long_returns: Vec<f64> = ((long_start + 1)..=i)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();
            let long_vol = if !long_returns.is_empty() {
                let mean = long_returns.iter().sum::<f64>() / long_returns.len() as f64;
                let var = long_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                    / long_returns.len() as f64;
                var.sqrt()
            } else {
                0.0
            };

            // Term structure = (long vol - short vol) / short vol
            if short_vol > 0.0 {
                result[i] = (long_vol - short_vol) / short_vol * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for FXVolatilityTerm {
    fn name(&self) -> &str {
        "FX Volatility Term"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Risk Reversal 25D - Skew measure (call vs put volatility)
#[derive(Debug, Clone)]
pub struct RiskReversal25D {
    period: usize,
}

impl RiskReversal25D {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate 25-delta risk reversal proxy
    /// Positive = calls expensive, Negative = puts expensive
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            let mut up_squared = 0.0;
            let mut down_squared = 0.0;
            let mut up_count = 0;
            let mut down_count = 0;

            for j in (start + 1)..=i {
                let ret = (close[j] / close[j - 1]).ln();
                if ret > 0.0 {
                    up_squared += ret.powi(2);
                    up_count += 1;
                } else {
                    down_squared += ret.powi(2);
                    down_count += 1;
                }
            }

            let up_vol = if up_count > 0 {
                (up_squared / up_count as f64).sqrt()
            } else {
                0.0
            };

            let down_vol = if down_count > 0 {
                (down_squared / down_count as f64).sqrt()
            } else {
                0.0
            };

            // Risk reversal = call vol - put vol
            result[i] = (up_vol - down_vol) * 10000.0;
        }
        result
    }
}

impl TechnicalIndicator for RiskReversal25D {
    fn name(&self) -> &str {
        "Risk Reversal 25D"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Butterfly 25D - Smile measure (wings vs ATM)
#[derive(Debug, Clone)]
pub struct Butterfly25D {
    period: usize,
}

impl Butterfly25D {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate 25-delta butterfly proxy
    /// Positive = fat tails (high kurtosis)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            let returns: Vec<f64> = ((start + 1)..=i)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();

            if returns.is_empty() {
                continue;
            }

            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / returns.len() as f64;
            let std = variance.sqrt();

            if std > 0.0 {
                // Kurtosis as butterfly proxy
                let kurtosis = returns.iter()
                    .map(|r| ((r - mean) / std).powi(4))
                    .sum::<f64>() / returns.len() as f64;
                // Excess kurtosis (subtract 3 for normal distribution)
                result[i] = (kurtosis - 3.0) * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for Butterfly25D {
    fn name(&self) -> &str {
        "Butterfly 25D"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// FX Positioning - Sentiment/positioning proxy from volume
#[derive(Debug, Clone)]
pub struct FXPositioning {
    period: usize,
}

impl FXPositioning {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate FX positioning proxy
    /// Uses volume-weighted price momentum
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            let mut weighted_ret = 0.0;
            let mut total_vol = 0.0;

            for j in (start + 1)..=i {
                let ret = (close[j] / close[j - 1]).ln();
                weighted_ret += ret * volume[j];
                total_vol += volume[j];
            }

            if total_vol > 0.0 {
                result[i] = weighted_ret / total_vol * 10000.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for FXPositioning {
    fn name(&self) -> &str {
        "FX Positioning"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Dollar Smile - USD behavior in different regimes
#[derive(Debug, Clone)]
pub struct DollarSmile {
    period: usize,
}

impl DollarSmile {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate dollar smile regime
    /// -1 = risk-off dollar strength, 0 = neutral, 1 = risk-on dollar weakness
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns and volatility
            let returns: Vec<f64> = ((start + 1)..=i)
                .map(|j| (close[j] / close[j - 1]).ln())
                .collect();

            if returns.is_empty() {
                continue;
            }

            let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean_ret).powi(2))
                .sum::<f64>() / returns.len() as f64;
            let vol = variance.sqrt();

            // Smile regime based on volatility and direction
            // High vol + appreciation = risk-off (-1)
            // Low vol + depreciation = risk-on (1)
            let vol_z = vol * (252.0_f64).sqrt() * 100.0; // Annualized vol %
            let ret_z = mean_ret * 252.0 * 100.0; // Annualized return %

            if vol_z > 15.0 && ret_z > 0.0 {
                result[i] = -1.0; // Risk-off dollar strength
            } else if vol_z < 10.0 && ret_z < 0.0 {
                result[i] = 1.0; // Risk-on dollar weakness
            } else {
                result[i] = 0.0; // Neutral
            }
        }
        result
    }
}

impl TechnicalIndicator for DollarSmile {
    fn name(&self) -> &str {
        "Dollar Smile"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// PPP Deviation - Purchasing Power Parity deviation proxy
#[derive(Debug, Clone)]
pub struct PPPDeviation {
    period: usize,
}

impl PPPDeviation {
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20 for meaningful PPP analysis".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate PPP deviation proxy
    /// Uses mean reversion relative to long-term average
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let avg = close[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;

            if avg > 0.0 {
                // Deviation from "fair value" (long-term average)
                result[i] = (close[i] - avg) / avg * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for PPPDeviation {
    fn name(&self) -> &str {
        "PPP Deviation"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// BEER - Behavioral Equilibrium Exchange Rate proxy
#[derive(Debug, Clone)]
pub struct BEER {
    period: usize,
}

impl BEER {
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate BEER proxy
    /// Uses trend-adjusted deviation from equilibrium
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let window = &close[start..=i];

            // Calculate trend using linear regression
            let n_points = window.len() as f64;
            let sum_x: f64 = (0..window.len()).map(|x| x as f64).sum();
            let sum_y: f64 = window.iter().sum();
            let sum_xy: f64 = window.iter().enumerate().map(|(x, &y)| x as f64 * y).sum();
            let sum_xx: f64 = (0..window.len()).map(|x| (x as f64).powi(2)).sum();

            let slope = if n_points * sum_xx - sum_x * sum_x != 0.0 {
                (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_xx - sum_x * sum_x)
            } else {
                0.0
            };
            let intercept = (sum_y - slope * sum_x) / n_points;

            // Equilibrium value at current point
            let equilibrium = intercept + slope * (window.len() - 1) as f64;

            // Deviation from equilibrium
            if equilibrium > 0.0 {
                result[i] = (close[i] - equilibrium) / equilibrium * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for BEER {
    fn name(&self) -> &str {
        "BEER"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        let close = vec![1.100, 1.102, 1.098, 1.105, 1.103, 1.108, 1.106, 1.110, 1.108, 1.112,
                        1.110, 1.115, 1.113, 1.118, 1.116, 1.120, 1.118, 1.122, 1.120, 1.125,
                        1.123, 1.128, 1.126, 1.130, 1.128, 1.133, 1.131, 1.135, 1.133, 1.138];
        let volume = vec![1000.0, 1200.0, 1100.0, 1300.0, 1500.0, 1400.0, 1600.0, 1800.0, 1700.0, 1900.0,
                         2000.0, 1800.0, 2100.0, 2200.0, 2000.0, 2300.0, 2100.0, 2400.0, 2200.0, 2500.0,
                         2300.0, 2600.0, 2400.0, 2700.0, 2500.0, 2800.0, 2600.0, 2900.0, 2700.0, 3000.0];
        (close, volume)
    }

    #[test]
    fn test_carry_trade_index() {
        let (close, _) = make_test_data();
        let cti = CarryTradeIndex::new(10).unwrap();
        let result = cti.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Uptrend should show positive carry
        assert!(result[20] > 0.0);
    }

    #[test]
    fn test_fx_volatility_term() {
        let (close, _) = make_test_data();
        let fvt = FXVolatilityTerm::new(5, 15).unwrap();
        let result = fvt.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[20].abs() < 200.0);
    }

    #[test]
    fn test_risk_reversal_25d() {
        let (close, _) = make_test_data();
        let rr = RiskReversal25D::new(10).unwrap();
        let result = rr.calculate(&close);

        assert_eq!(result.len(), close.len());
        assert!(result[20].abs() < 1000.0);
    }

    #[test]
    fn test_butterfly_25d() {
        let (close, _) = make_test_data();
        let bf = Butterfly25D::new(10).unwrap();
        let result = bf.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Check kurtosis is reasonable
        assert!(result[20].abs() < 500.0);
    }

    #[test]
    fn test_fx_positioning() {
        let (close, volume) = make_test_data();
        let fxp = FXPositioning::new(10).unwrap();
        let result = fxp.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
        // Uptrend should show positive positioning
        assert!(result[20] > 0.0);
    }

    #[test]
    fn test_dollar_smile() {
        let (close, _) = make_test_data();
        let ds = DollarSmile::new(10).unwrap();
        let result = ds.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Should be -1, 0, or 1
        assert!(result[20] >= -1.0 && result[20] <= 1.0);
    }

    #[test]
    fn test_ppp_deviation() {
        let (close, _) = make_test_data();
        let ppp = PPPDeviation::new(20).unwrap();
        let result = ppp.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Deviation should be reasonable percentage
        assert!(result[25].abs() < 20.0);
    }

    #[test]
    fn test_beer() {
        let (close, _) = make_test_data();
        let beer = BEER::new(20).unwrap();
        let result = beer.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Deviation should be reasonable percentage
        assert!(result[25].abs() < 20.0);
    }
}
