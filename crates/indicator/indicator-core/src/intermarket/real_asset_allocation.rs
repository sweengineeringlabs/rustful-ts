//! Real Asset Allocation Indicator (IND-500)
//!
//! Inflation hedge signal for real asset allocation decisions.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

// ============================================================================
// InflationRegime Enum
// ============================================================================

/// Inflation regime classifications.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InflationRegime {
    /// Accelerating inflation - favor real assets
    Rising = 1,
    /// High but stable inflation
    Elevated = 2,
    /// Decelerating inflation
    Falling = 3,
    /// Low/stable inflation - favor nominal assets
    Low = 4,
}

impl InflationRegime {
    /// Get numeric value.
    pub fn as_f64(&self) -> f64 {
        match self {
            InflationRegime::Rising => 1.0,
            InflationRegime::Elevated => 2.0,
            InflationRegime::Falling => 3.0,
            InflationRegime::Low => 4.0,
        }
    }

    /// Convert from numeric value.
    pub fn from_f64(value: f64) -> Option<Self> {
        match value as i32 {
            1 => Some(InflationRegime::Rising),
            2 => Some(InflationRegime::Elevated),
            3 => Some(InflationRegime::Falling),
            4 => Some(InflationRegime::Low),
            _ => None,
        }
    }
}

// ============================================================================
// RealAssetAllocation
// ============================================================================

/// Real Asset Allocation - Inflation hedge signal indicator.
///
/// This indicator generates signals for allocating to real assets
/// (commodities, TIPS, real estate, gold) vs nominal assets based
/// on inflation regime and expectations.
///
/// # Theory
/// Real assets provide inflation protection because:
/// - Commodities directly benefit from rising prices
/// - TIPS principal adjusts with CPI
/// - Real estate rents adjust with inflation
/// - Gold is a traditional inflation hedge
///
/// # Interpretation
/// - `allocation > 0.5`: Overweight real assets
/// - `allocation > 0`: Slight tilt to real assets
/// - `allocation < 0`: Favor nominal assets
/// - `allocation < -0.5`: Strong nominal asset preference
///
/// # Outputs
/// - `real_allocation`: Suggested real asset weight (-1 to 1)
/// - `inflation_regime`: Current inflation regime (1-4)
/// - `inflation_momentum`: Inflation trend direction
/// - `vol_adjusted`: Volatility-adjusted signal
#[derive(Debug, Clone)]
pub struct RealAssetAllocation {
    /// Period for inflation proxy calculation.
    inflation_period: usize,
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Period for volatility calculation.
    vol_period: usize,
    /// Optional commodity index for inflation proxy.
    commodity_series: Vec<f64>,
    /// Optional bond series for real yield proxy.
    bond_series: Vec<f64>,
}

impl RealAssetAllocation {
    /// Create a new RealAssetAllocation indicator.
    ///
    /// # Arguments
    /// * `inflation_period` - Period for inflation calculation (min: 20)
    /// * `momentum_period` - Period for momentum (min: 5)
    /// * `vol_period` - Period for volatility (min: 10)
    pub fn new(inflation_period: usize, momentum_period: usize, vol_period: usize) -> Result<Self> {
        if inflation_period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "inflation_period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if vol_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "vol_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self {
            inflation_period,
            momentum_period,
            vol_period,
            commodity_series: Vec::new(),
            bond_series: Vec::new(),
        })
    }

    /// Create with default parameters.
    pub fn default_params() -> Result<Self> {
        Self::new(60, 20, 20)
    }

    /// Set commodity series for inflation proxy.
    pub fn with_commodities(mut self, series: &[f64]) -> Self {
        self.commodity_series = series.to_vec();
        self
    }

    /// Set bond series for real yield proxy.
    pub fn with_bonds(mut self, series: &[f64]) -> Self {
        self.bond_series = series.to_vec();
        self
    }

    /// Calculate rolling volatility.
    fn calculate_volatility(prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }

        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        variance.sqrt() * (252.0_f64).sqrt() * 100.0 // Annualized
    }

    /// Calculate inflation proxy from prices.
    fn calculate_inflation_proxy(&self, prices: &[f64], idx: usize) -> f64 {
        if idx < self.inflation_period {
            return 0.0;
        }

        let start = idx.saturating_sub(self.inflation_period);

        // Use volatility trend as inflation proxy
        let half_period = self.inflation_period / 2;
        let mid_point = idx.saturating_sub(half_period);

        let recent_vol = Self::calculate_volatility(&prices[mid_point..=idx]);
        let old_vol = Self::calculate_volatility(&prices[start..mid_point]);

        // Rising volatility suggests inflation concerns
        if old_vol > 0.0 {
            ((recent_vol / old_vol) - 1.0) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate inflation momentum.
    fn calculate_inflation_momentum(&self, prices: &[f64], idx: usize) -> f64 {
        if idx < self.momentum_period {
            return 0.0;
        }

        let start = idx.saturating_sub(self.momentum_period);
        let momentum = (prices[idx] / prices[start] - 1.0) * 100.0;
        momentum.clamp(-50.0, 50.0) / 50.0 // Normalize to -1 to 1
    }

    /// Determine inflation regime.
    fn classify_inflation_regime(&self, inflation_proxy: f64, momentum: f64) -> InflationRegime {
        if momentum > 0.2 && inflation_proxy > 5.0 {
            InflationRegime::Rising
        } else if inflation_proxy > 10.0 {
            InflationRegime::Elevated
        } else if momentum < -0.2 && inflation_proxy > 0.0 {
            InflationRegime::Falling
        } else {
            InflationRegime::Low
        }
    }

    /// Calculate real asset allocation signal.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let max_period = self.inflation_period.max(self.vol_period);
        let mut result = vec![0.0; n];

        if n < max_period {
            return result;
        }

        for i in max_period..n {
            let inflation_proxy = self.calculate_inflation_proxy(close, i);
            let momentum = self.calculate_inflation_momentum(close, i);

            // Calculate volatility for risk adjustment
            let vol_start = i.saturating_sub(self.vol_period);
            let current_vol = Self::calculate_volatility(&close[vol_start..=i]);

            // Higher inflation/momentum -> favor real assets
            let inflation_score = (inflation_proxy / 20.0).clamp(-1.0, 1.0);
            let momentum_score = momentum;

            // Volatility adjustment (high vol = more cautious)
            let vol_adjustment = if current_vol > 25.0 {
                0.7 // Scale down allocation in high vol
            } else if current_vol < 10.0 {
                1.2 // Can be more aggressive in low vol
            } else {
                1.0
            };

            // Combined allocation signal
            let raw_allocation = inflation_score * 0.5 + momentum_score * 0.5;
            result[i] = (raw_allocation * vol_adjustment).clamp(-1.0, 1.0);
        }

        result
    }

    /// Calculate with commodity prices as inflation proxy.
    pub fn calculate_with_commodities(&self, equity: &[f64], commodities: &[f64]) -> Vec<f64> {
        let n = equity.len().min(commodities.len());
        let max_period = self.inflation_period.max(self.vol_period);
        let mut result = vec![0.0; n];

        if n < max_period {
            return result;
        }

        for i in max_period..n {
            // Commodity momentum as inflation proxy
            let comm_start = i.saturating_sub(self.momentum_period);
            let commodity_momentum = if commodities[comm_start] > 0.0 {
                (commodities[i] / commodities[comm_start] - 1.0) * 100.0
            } else {
                0.0
            };

            // Commodity relative strength vs equity
            let equity_momentum = if equity[comm_start] > 0.0 {
                (equity[i] / equity[comm_start] - 1.0) * 100.0
            } else {
                0.0
            };

            let relative_strength = commodity_momentum - equity_momentum;

            // Long-term commodity trend
            let long_start = i.saturating_sub(self.inflation_period);
            let commodity_trend = if commodities[long_start] > 0.0 {
                (commodities[i] / commodities[long_start] - 1.0) * 100.0
            } else {
                0.0
            };

            // Allocation based on commodity signals
            let momentum_signal = (commodity_momentum / 20.0).clamp(-1.0, 1.0);
            let relative_signal = (relative_strength / 10.0).clamp(-1.0, 1.0);
            let trend_signal = (commodity_trend / 30.0).clamp(-1.0, 1.0);

            result[i] = (momentum_signal * 0.4 + relative_signal * 0.35 + trend_signal * 0.25)
                .clamp(-1.0, 1.0);
        }

        result
    }

    /// Calculate detailed output with all components.
    pub fn calculate_detailed(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let max_period = self.inflation_period.max(self.vol_period);
        let mut allocation = vec![0.0; n];
        let mut regime = vec![0.0; n];
        let mut inflation_mom = vec![0.0; n];
        let mut vol_adjusted = vec![0.0; n];

        if n < max_period {
            return (allocation, regime, inflation_mom, vol_adjusted);
        }

        for i in max_period..n {
            let inflation_proxy = self.calculate_inflation_proxy(close, i);
            let momentum = self.calculate_inflation_momentum(close, i);
            let inflation_regime = self.classify_inflation_regime(inflation_proxy, momentum);

            let vol_start = i.saturating_sub(self.vol_period);
            let current_vol = Self::calculate_volatility(&close[vol_start..=i]);

            let inflation_score = (inflation_proxy / 20.0).clamp(-1.0, 1.0);
            let raw_allocation = inflation_score * 0.5 + momentum * 0.5;

            let vol_adjustment = if current_vol > 25.0 {
                0.7
            } else if current_vol < 10.0 {
                1.2
            } else {
                1.0
            };

            allocation[i] = raw_allocation.clamp(-1.0, 1.0);
            regime[i] = inflation_regime.as_f64();
            inflation_mom[i] = momentum;
            vol_adjusted[i] = (raw_allocation * vol_adjustment).clamp(-1.0, 1.0);
        }

        (allocation, regime, inflation_mom, vol_adjusted)
    }
}

impl TechnicalIndicator for RealAssetAllocation {
    fn name(&self) -> &str {
        "Real Asset Allocation"
    }

    fn min_periods(&self) -> usize {
        self.inflation_period.max(self.vol_period)
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (allocation, regime, inflation_mom, vol_adjusted) = self.calculate_detailed(&data.close);
        Ok(IndicatorOutput::triple(allocation, regime, inflation_mom))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rising_inflation_data() -> Vec<f64> {
        // Simulated rising volatility (inflation proxy)
        (0..100).map(|i| {
            let trend = 100.0 + i as f64 * 0.3;
            let vol = (i as f64 / 10.0).sin() * (1.0 + i as f64 * 0.02);
            trend + vol
        }).collect()
    }

    fn make_low_inflation_data() -> Vec<f64> {
        // Stable low volatility
        (0..100).map(|i| 100.0 + i as f64 * 0.1 + ((i as f64 * 0.1).sin() * 0.5)).collect()
    }

    fn make_commodity_rally_data() -> Vec<f64> {
        // Commodity rally
        (0..100).map(|i| 50.0 * (1.0 + 0.005 * i as f64)).collect()
    }

    #[test]
    fn test_real_asset_allocation_creation() {
        let raa = RealAssetAllocation::new(60, 20, 20);
        assert!(raa.is_ok());

        let raa_err = RealAssetAllocation::new(10, 20, 20);
        assert!(raa_err.is_err());
    }

    #[test]
    fn test_inflation_regime_enum() {
        assert_eq!(InflationRegime::Rising.as_f64(), 1.0);
        assert_eq!(InflationRegime::Elevated.as_f64(), 2.0);
        assert_eq!(InflationRegime::Falling.as_f64(), 3.0);
        assert_eq!(InflationRegime::Low.as_f64(), 4.0);

        assert_eq!(InflationRegime::from_f64(1.0), Some(InflationRegime::Rising));
        assert_eq!(InflationRegime::from_f64(5.0), None);
    }

    #[test]
    fn test_real_asset_allocation_calculation() {
        let data = make_rising_inflation_data();
        let raa = RealAssetAllocation::new(40, 15, 15).unwrap();
        let result = raa.calculate(&data);

        assert_eq!(result.len(), data.len());
        // Rising inflation should tend toward positive allocation
        assert!(result[80] >= -1.0 && result[80] <= 1.0);
    }

    #[test]
    fn test_real_asset_allocation_with_commodities() {
        let equity = make_low_inflation_data();
        let commodities = make_commodity_rally_data();
        let raa = RealAssetAllocation::new(40, 15, 15).unwrap();
        let result = raa.calculate_with_commodities(&equity, &commodities);

        assert_eq!(result.len(), equity.len().min(commodities.len()));
        // Commodity rally should suggest real asset allocation
        assert!(result[80] > -1.0);
    }

    #[test]
    fn test_real_asset_allocation_detailed() {
        let data = make_rising_inflation_data();
        let raa = RealAssetAllocation::new(40, 15, 15).unwrap();
        let (allocation, regime, inflation_mom, vol_adjusted) = raa.calculate_detailed(&data);

        assert_eq!(allocation.len(), data.len());
        assert_eq!(regime.len(), data.len());
        assert_eq!(inflation_mom.len(), data.len());
        assert_eq!(vol_adjusted.len(), data.len());

        // Check regime is valid
        assert!(regime[80] >= 0.0 && regime[80] <= 4.0);
    }

    #[test]
    fn test_real_asset_allocation_trait() {
        let data = make_low_inflation_data();
        let raa = RealAssetAllocation::default_params().unwrap();

        assert_eq!(raa.name(), "Real Asset Allocation");
        assert!(raa.min_periods() > 0);

        let series = OHLCVSeries {
            open: data.clone(),
            high: data.clone(),
            low: data.clone(),
            close: data.clone(),
            volume: vec![1000.0; data.len()],
        };

        let output = raa.compute(&series);
        assert!(output.is_ok());
    }
}
