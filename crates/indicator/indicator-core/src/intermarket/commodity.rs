//! Commodity-Specific Indicators
//!
//! Indicators for analyzing commodity market dynamics.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Contango/Backwardation Indicator
/// Measures term structure shape using price momentum
#[derive(Debug, Clone)]
pub struct ContangoBackwardation {
    period: usize,
}

impl ContangoBackwardation {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Positive = Contango (futures > spot), Negative = Backwardation
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let past_avg = close[start..i].iter().sum::<f64>() / (i - start) as f64;

            // Contango approximation: current price vs recent average
            if past_avg > 0.0 {
                result[i] = (close[i] - past_avg) / past_avg * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for ContangoBackwardation {
    fn name(&self) -> &str {
        "Contango/Backwardation"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Roll Yield - Calendar spread approximation
#[derive(Debug, Clone)]
pub struct RollYield {
    period: usize,
}

impl RollYield {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate roll yield approximation using price momentum
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            // Roll yield approximation from price changes
            let ret = (close[i] / close[i - self.period]).ln();
            // Annualized
            result[i] = ret * (252.0 / self.period as f64) * 100.0;
        }
        result
    }
}

impl TechnicalIndicator for RollYield {
    fn name(&self) -> &str {
        "Roll Yield"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Basis - Cash-futures spread proxy
/// Uses volatility-adjusted price difference
#[derive(Debug, Clone)]
pub struct Basis {
    period: usize,
}

impl Basis {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate basis using moving average as spot proxy
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let ma = close[start..=i].iter().sum::<f64>() / (i - start + 1) as f64;

            // Basis = Futures - Spot (approximated)
            if ma > 0.0 {
                result[i] = (close[i] - ma) / ma * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for Basis {
    fn name(&self) -> &str {
        "Basis"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Convenience Yield - Storage value approximation
/// Uses volatility and price momentum
#[derive(Debug, Clone)]
pub struct ConvenienceYield {
    period: usize,
}

impl ConvenienceYield {
    pub fn new(period: usize) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Convenience yield approximation
    /// When backwardation exists, convenience yield is positive
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns volatility
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
            let vol = variance.sqrt() * (252.0_f64).sqrt();

            // Convenience yield proxy: backwardation + volatility component
            let trend = (close[i] / close[start]).ln() * (252.0 / self.period as f64);
            result[i] = (-trend + vol * 0.5) * 100.0; // Higher in backwardation
        }
        result
    }
}

impl TechnicalIndicator for ConvenienceYield {
    fn name(&self) -> &str {
        "Convenience Yield"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Inventory Surprise - Volume deviation as inventory proxy
#[derive(Debug, Clone)]
pub struct InventorySurprise {
    period: usize,
}

impl InventorySurprise {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate inventory surprise using volume changes
    /// Positive = higher than expected inventory/supply, Negative = lower
    pub fn calculate(&self, volume: &[f64]) -> Vec<f64> {
        let n = volume.len();
        let mut result = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let window = &volume[start..i];

            if window.is_empty() {
                continue;
            }

            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance = window.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / window.len() as f64;
            let std = variance.sqrt();

            // Z-score of current volume vs expectation
            if std > 0.0 {
                result[i] = (volume[i] - mean) / std;
            }
        }
        result
    }
}

impl TechnicalIndicator for InventorySurprise {
    fn name(&self) -> &str {
        "Inventory Surprise"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.volume)))
    }
}

/// Crack Spread - Refining margin proxy
/// Uses relative strength between series
#[derive(Debug, Clone)]
pub struct CrackSpread {
    period: usize,
}

impl CrackSpread {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate crack spread approximation using price range
    /// Uses high-low range as a proxy for spread dynamics
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = close.len().min(high.len()).min(low.len());
        let mut result = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);

            // Calculate average true range as spread proxy
            let mut sum_range = 0.0;
            for j in start..=i {
                sum_range += high[j] - low[j];
            }
            let avg_range = sum_range / (i - start + 1) as f64;

            // Normalize by close price
            if close[i] > 0.0 {
                result[i] = avg_range / close[i] * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for CrackSpread {
    fn name(&self) -> &str {
        "Crack Spread"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.high, &data.low, &data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let high = vec![105.0, 106.0, 107.0, 106.5, 108.0, 109.0, 108.5, 110.0, 109.5, 111.0,
                       110.5, 112.0, 111.5, 113.0, 112.5, 114.0, 113.5, 115.0, 114.5, 116.0];
        let low = vec![100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 104.5, 106.0,
                      105.5, 107.0, 106.5, 108.0, 107.5, 109.0, 108.5, 110.0, 109.5, 111.0];
        let close = vec![102.0, 104.0, 105.0, 104.0, 106.0, 107.0, 106.0, 108.0, 107.0, 109.0,
                        108.0, 110.0, 109.0, 111.0, 110.0, 112.0, 111.0, 113.0, 112.0, 114.0];
        let volume = vec![1000.0, 1200.0, 1100.0, 1300.0, 1500.0, 1400.0, 1600.0, 1800.0, 1700.0, 1900.0,
                         2000.0, 1800.0, 2100.0, 2200.0, 2000.0, 2300.0, 2100.0, 2400.0, 2200.0, 2500.0];
        (high, low, close, volume)
    }

    #[test]
    fn test_contango_backwardation() {
        let (_, _, close, _) = make_test_data();
        let cb = ContangoBackwardation::new(5).unwrap();
        let result = cb.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Uptrend should show contango (positive)
        assert!(result[10] > 0.0);
    }

    #[test]
    fn test_roll_yield() {
        let (_, _, close, _) = make_test_data();
        let ry = RollYield::new(5).unwrap();
        let result = ry.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Uptrend should have positive roll yield
        assert!(result[10] > 0.0);
    }

    #[test]
    fn test_basis() {
        let (_, _, close, _) = make_test_data();
        let basis = Basis::new(5).unwrap();
        let result = basis.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Values should be reasonable percentages
        assert!(result[10].abs() < 20.0);
    }

    #[test]
    fn test_convenience_yield() {
        let (_, _, close, _) = make_test_data();
        let cy = ConvenienceYield::new(5).unwrap();
        let result = cy.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Check values exist after warmup
        assert!(result[10].abs() < 200.0);
    }

    #[test]
    fn test_inventory_surprise() {
        let (_, _, _, volume) = make_test_data();
        let inv = InventorySurprise::new(5).unwrap();
        let result = inv.calculate(&volume);

        assert_eq!(result.len(), volume.len());
        // Z-scores should be reasonable
        assert!(result[10].abs() < 5.0);
    }

    #[test]
    fn test_crack_spread() {
        let (high, low, close, _) = make_test_data();
        let cs = CrackSpread::new(5).unwrap();
        let result = cs.calculate(&high, &low, &close);

        assert_eq!(result.len(), close.len());
        // Spread percentage should be positive and reasonable
        assert!(result[10] > 0.0 && result[10] < 20.0);
    }
}
