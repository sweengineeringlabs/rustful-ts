//! Realized Cap (Cost Basis Market Cap) - IND-271
//!
//! Calculates the realized capitalization, which values each coin at its last movement price.
//!
//! Unlike market cap (all coins at current price), realized cap represents
//! the aggregate cost basis of all holders, providing insight into:
//! - True capital invested in the network
//! - Profit/loss status of holders
//! - Market cycle phases
//!
//! Interpretation:
//! - Market Cap > Realized Cap: Aggregate profit (bull market)
//! - Market Cap < Realized Cap: Aggregate loss (bear market)
//! - Realized Cap growth: New capital entering

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Configuration for Realized Cap calculation.
#[derive(Debug, Clone)]
pub struct RealizedCapConfig {
    /// Period for calculating cost basis proxy.
    pub cost_basis_period: usize,
    /// Smoothing period for trends.
    pub smooth_period: usize,
    /// Weight decay factor for older prices (0-1).
    pub decay_factor: f64,
}

impl Default for RealizedCapConfig {
    fn default() -> Self {
        Self {
            cost_basis_period: 90,
            smooth_period: 14,
            decay_factor: 0.95,
        }
    }
}

/// Realized Cap output.
#[derive(Debug, Clone)]
pub struct RealizedCapOutput {
    /// Realized cap proxy values.
    pub realized_cap: Vec<f64>,
    /// Market cap proxy values.
    pub market_cap: Vec<f64>,
    /// MVRV ratio (Market / Realized).
    pub mvrv: Vec<f64>,
    /// Net unrealized profit/loss percentage.
    pub nupl: Vec<f64>,
    /// Realized cap momentum (rate of change).
    pub realized_momentum: Vec<f64>,
}

/// Realized Cap market phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RealizedCapPhase {
    /// Extreme profit - market top zone.
    ExtremeProfit,
    /// Profit - bull market.
    Profit,
    /// Neutral - transition zone.
    Neutral,
    /// Loss - bear market.
    Loss,
    /// Extreme loss - market bottom zone.
    ExtremeLoss,
}

/// Realized Cap (Cost Basis Market Cap) - IND-271
///
/// Estimates realized capitalization using volume-weighted average price history.
///
/// # Formula
/// ```text
/// Realized Cap Proxy = Σ(VWAP_i * Volume_i) / Σ(Volume_i)
/// MVRV = Market Cap / Realized Cap
/// NUPL = (Market Cap - Realized Cap) / Market Cap
/// ```
///
/// # Example
/// ```
/// use indicator_core::crypto::{RealizedCap, RealizedCapConfig};
///
/// let config = RealizedCapConfig::default();
/// let rcap = RealizedCap::new(config).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RealizedCap {
    config: RealizedCapConfig,
}

impl RealizedCap {
    /// Create a new Realized Cap indicator.
    pub fn new(config: RealizedCapConfig) -> Result<Self> {
        if config.cost_basis_period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "cost_basis_period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if config.smooth_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "smooth_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if config.decay_factor <= 0.0 || config.decay_factor >= 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "decay_factor".to_string(),
                reason: "must be between 0 and 1 (exclusive)".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration.
    pub fn default_config() -> Result<Self> {
        Self::new(RealizedCapConfig::default())
    }

    /// Calculate Realized Cap metrics.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> RealizedCapOutput {
        let n = close.len().min(volume.len());

        if n < self.config.cost_basis_period {
            return RealizedCapOutput {
                realized_cap: vec![0.0; n],
                market_cap: vec![0.0; n],
                mvrv: vec![1.0; n],
                nupl: vec![0.0; n],
                realized_momentum: vec![0.0; n],
            };
        }

        let mut realized_cap = vec![0.0; n];
        let mut market_cap = vec![0.0; n];
        let mut mvrv = vec![1.0; n];
        let mut nupl = vec![0.0; n];
        let mut realized_momentum = vec![0.0; n];

        // Calculate market cap proxy (price * average volume as supply proxy)
        let avg_volume: f64 = volume.iter().sum::<f64>() / n as f64;

        for i in self.config.cost_basis_period..n {
            // Market cap proxy: current price * supply proxy
            market_cap[i] = close[i] * avg_volume;

            // Realized cap proxy: volume-weighted average cost basis
            let realized = self.calculate_realized_cap_at(close, volume, i);
            realized_cap[i] = realized * avg_volume;

            // MVRV ratio
            if realized_cap[i] > 1e-10 {
                mvrv[i] = market_cap[i] / realized_cap[i];
            }

            // NUPL: Net Unrealized Profit/Loss
            if market_cap[i] > 1e-10 {
                nupl[i] = (market_cap[i] - realized_cap[i]) / market_cap[i] * 100.0;
            }

            // Realized cap momentum
            if i >= self.config.cost_basis_period + self.config.smooth_period {
                let prev = realized_cap[i - self.config.smooth_period];
                if prev > 1e-10 {
                    realized_momentum[i] = (realized_cap[i] / prev - 1.0) * 100.0;
                }
            }
        }

        // Apply smoothing to momentum
        realized_momentum = self.apply_ema(&realized_momentum, self.config.smooth_period);

        RealizedCapOutput {
            realized_cap,
            market_cap,
            mvrv,
            nupl,
            realized_momentum,
        }
    }

    /// Calculate realized cap proxy at a specific index.
    fn calculate_realized_cap_at(&self, close: &[f64], volume: &[f64], index: usize) -> f64 {
        let start = index.saturating_sub(self.config.cost_basis_period);

        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;

        for (j, i) in (start..=index).enumerate() {
            // Apply decay: more recent prices get higher weight
            let decay_weight = self.config.decay_factor.powi((index - i) as i32);
            let volume_weight = volume[i];
            let total_weight = decay_weight * volume_weight;

            weighted_sum += close[i] * total_weight;
            weight_total += total_weight;
        }

        if weight_total > 1e-10 {
            weighted_sum / weight_total
        } else {
            close[index]
        }
    }

    /// Apply EMA smoothing.
    fn apply_ema(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = data.to_vec();

        if n == 0 || period <= 1 {
            return result;
        }

        let alpha = 2.0 / (period as f64 + 1.0);

        for i in 1..n {
            result[i] = alpha * data[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }

    /// Interpret market phase based on MVRV and NUPL.
    pub fn interpret(&self, mvrv: f64, nupl: f64) -> RealizedCapPhase {
        if mvrv.is_nan() || nupl.is_nan() {
            return RealizedCapPhase::Neutral;
        }

        if mvrv > 3.0 || nupl > 70.0 {
            RealizedCapPhase::ExtremeProfit
        } else if mvrv > 1.5 || nupl > 30.0 {
            RealizedCapPhase::Profit
        } else if mvrv < 0.7 || nupl < -30.0 {
            RealizedCapPhase::ExtremeLoss
        } else if mvrv < 1.0 || nupl < 0.0 {
            RealizedCapPhase::Loss
        } else {
            RealizedCapPhase::Neutral
        }
    }

    /// Convert phase to indicator signal.
    pub fn to_indicator_signal(&self, phase: RealizedCapPhase) -> IndicatorSignal {
        match phase {
            RealizedCapPhase::ExtremeProfit => IndicatorSignal::Bearish, // Sell signal
            RealizedCapPhase::Profit => IndicatorSignal::Neutral,
            RealizedCapPhase::Neutral => IndicatorSignal::Neutral,
            RealizedCapPhase::Loss => IndicatorSignal::Neutral,
            RealizedCapPhase::ExtremeLoss => IndicatorSignal::Bullish, // Buy signal
        }
    }

    /// Get configuration.
    pub fn config(&self) -> &RealizedCapConfig {
        &self.config
    }
}

impl TechnicalIndicator for RealizedCap {
    fn name(&self) -> &str {
        "Realized Cap"
    }

    fn min_periods(&self) -> usize {
        self.config.cost_basis_period + self.config.smooth_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let output = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::triple(output.realized_cap, output.market_cap, output.mvrv))
    }
}

impl Default for RealizedCap {
    fn default() -> Self {
        Self::new(RealizedCapConfig::default()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        // Simulate bull market data (prices increasing)
        let close: Vec<f64> = (0..150)
            .map(|i| 100.0 + (i as f64) * 0.5)
            .collect();

        let volume: Vec<f64> = (0..150)
            .map(|i| 1000.0 + (i as f64 * 0.1).sin() * 200.0)
            .collect();

        (close, volume)
    }

    fn make_bear_data() -> (Vec<f64>, Vec<f64>) {
        // Simulate bear market data (prices decreasing)
        let close: Vec<f64> = (0..150)
            .map(|i| 200.0 - (i as f64) * 0.5)
            .collect();

        let volume: Vec<f64> = vec![1000.0; 150];

        (close, volume)
    }

    #[test]
    fn test_realized_cap_basic() {
        let rcap = RealizedCap::default();
        let (close, volume) = make_test_data();

        let output = rcap.calculate(&close, &volume);

        assert_eq!(output.realized_cap.len(), close.len());
        assert_eq!(output.market_cap.len(), close.len());
        assert_eq!(output.mvrv.len(), close.len());
        assert_eq!(output.nupl.len(), close.len());
    }

    #[test]
    fn test_realized_cap_bull_market() {
        let rcap = RealizedCap::default();
        let (close, volume) = make_test_data();

        let output = rcap.calculate(&close, &volume);

        // In bull market, MVRV should be > 1 (market cap > realized cap)
        let mvrv_avg: f64 = output.mvrv[100..].iter().sum::<f64>()
            / (output.mvrv.len() - 100) as f64;
        assert!(mvrv_avg > 1.0);

        // NUPL should be positive
        let nupl_avg: f64 = output.nupl[100..].iter().sum::<f64>()
            / (output.nupl.len() - 100) as f64;
        assert!(nupl_avg > 0.0);
    }

    #[test]
    fn test_realized_cap_bear_market() {
        let rcap = RealizedCap::default();
        let (close, volume) = make_bear_data();

        let output = rcap.calculate(&close, &volume);

        // In bear market, MVRV should be < 1 (market cap < realized cap)
        let mvrv_end = output.mvrv[140];
        assert!(mvrv_end < 1.0);

        // NUPL should be negative at the end
        let nupl_end = output.nupl[140];
        assert!(nupl_end < 0.0);
    }

    #[test]
    fn test_realized_cap_phase_interpretation() {
        let rcap = RealizedCap::default();

        assert_eq!(
            rcap.interpret(4.0, 80.0),
            RealizedCapPhase::ExtremeProfit
        );
        assert_eq!(rcap.interpret(2.0, 40.0), RealizedCapPhase::Profit);
        assert_eq!(rcap.interpret(1.2, 15.0), RealizedCapPhase::Neutral);
        assert_eq!(rcap.interpret(0.9, -10.0), RealizedCapPhase::Loss);
        assert_eq!(rcap.interpret(0.5, -50.0), RealizedCapPhase::ExtremeLoss);
    }

    #[test]
    fn test_realized_cap_signal_conversion() {
        let rcap = RealizedCap::default();

        assert_eq!(
            rcap.to_indicator_signal(RealizedCapPhase::ExtremeProfit),
            IndicatorSignal::Bearish
        );
        assert_eq!(
            rcap.to_indicator_signal(RealizedCapPhase::ExtremeLoss),
            IndicatorSignal::Bullish
        );
        assert_eq!(
            rcap.to_indicator_signal(RealizedCapPhase::Neutral),
            IndicatorSignal::Neutral
        );
    }

    #[test]
    fn test_realized_cap_validation() {
        assert!(RealizedCap::new(RealizedCapConfig {
            cost_basis_period: 5, // Invalid
            smooth_period: 14,
            decay_factor: 0.95,
        })
        .is_err());

        assert!(RealizedCap::new(RealizedCapConfig {
            cost_basis_period: 90,
            smooth_period: 14,
            decay_factor: 1.5, // Invalid
        })
        .is_err());
    }

    #[test]
    fn test_realized_cap_empty_input() {
        let rcap = RealizedCap::default();
        let output = rcap.calculate(&[], &[]);

        assert!(output.realized_cap.is_empty());
    }

    #[test]
    fn test_realized_cap_technical_indicator_trait() {
        let rcap = RealizedCap::default();
        assert_eq!(rcap.name(), "Realized Cap");
        assert!(rcap.min_periods() > 0);
    }
}
