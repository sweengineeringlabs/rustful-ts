//! Inventory Levels Indicator (IND-416)
//!
//! Storage data proxy that estimates inventory levels and supply conditions
//! using volume and price behavior. High inventory suggests bearish conditions,
//! while low inventory suggests bullish conditions.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Inventory signal classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InventorySignal {
    /// High inventory (bearish for prices).
    HighInventory,
    /// Low inventory (bullish for prices).
    LowInventory,
    /// Normal inventory levels.
    Normal,
    /// Building inventory (increasing).
    Building,
    /// Drawing inventory (decreasing).
    Drawing,
}

/// Output for InventoryLevels indicator.
#[derive(Debug, Clone, Copy)]
pub struct InventoryLevelsOutput {
    /// Estimated inventory level proxy (z-score).
    pub level: f64,
    /// Rate of change in inventory.
    pub change_rate: f64,
    /// Inventory momentum (smoothed change rate).
    pub momentum: f64,
    /// Current inventory signal.
    pub signal: InventorySignal,
}

/// Inventory Levels Indicator (IND-416)
///
/// Estimates inventory levels using volume and price as proxies for supply/demand
/// balance. This is useful when actual inventory data is unavailable or delayed.
///
/// # Algorithm
/// 1. Calculate volume z-score relative to recent average (high volume = inventory moves)
/// 2. Combine with price direction to determine build vs draw
/// 3. Accumulate inventory proxy over time
/// 4. Calculate momentum of inventory changes
/// 5. Classify current inventory state
///
/// # Interpretation
/// - High level (positive z-score): Above-normal inventory, bearish
/// - Low level (negative z-score): Below-normal inventory, bullish
/// - Building: Inventory increasing, potentially bearish
/// - Drawing: Inventory decreasing, potentially bullish
///
/// # Example
/// ```ignore
/// let inv = InventoryLevels::new(20, 2.0)?;
/// let output = inv.compute(&data)?;
/// ```
#[derive(Debug, Clone)]
pub struct InventoryLevels {
    /// Lookback period for statistics.
    period: usize,
    /// Threshold for high/low classification (z-score units).
    threshold: f64,
    /// Smoothing period for momentum.
    smoothing_period: usize,
}

impl InventoryLevels {
    /// Create a new InventoryLevels indicator.
    ///
    /// # Arguments
    /// * `period` - Lookback period for calculations
    /// * `threshold` - Z-score threshold for high/low classification
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid.
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        if threshold <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "threshold".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            period,
            threshold,
            smoothing_period: period / 2,
        })
    }

    /// Create with default parameters (20, 2.0).
    pub fn default_params() -> Result<Self> {
        Self::new(20, 2.0)
    }

    /// Set custom smoothing period.
    pub fn with_smoothing_period(mut self, period: usize) -> Self {
        self.smoothing_period = period.max(1);
        self
    }

    /// Calculate inventory levels.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<Option<InventoryLevelsOutput>> {
        let n = close.len().min(volume.len());
        let mut result = vec![None; n];

        if n < self.period + 1 {
            return result;
        }

        // Calculate cumulative inventory proxy
        let mut inventory_proxy = vec![0.0; n];
        let mut cumulative = 0.0;

        for i in 1..n {
            // Price direction determines build vs draw
            let price_change = close[i] - close[i - 1];
            // Volume magnitude
            let vol_factor = volume[i];

            // Build on down days (accumulation), draw on up days (distribution)
            // This is a simplified model
            if price_change < 0.0 {
                cumulative += vol_factor * 0.0001; // Building
            } else if price_change > 0.0 {
                cumulative -= vol_factor * 0.0001; // Drawing
            }

            inventory_proxy[i] = cumulative;
        }

        // Calculate z-scores and momentum
        let mut ema_change = f64::NAN;
        let ema_mult = 2.0 / (self.smoothing_period as f64 + 1.0);

        for i in self.period..n {
            let start = i - self.period;

            // Calculate mean and std of inventory proxy
            let window = &inventory_proxy[start..i];
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance = window.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / window.len() as f64;
            let std = variance.sqrt();

            // Z-score of current level
            let level = if std > 0.0 {
                (inventory_proxy[i] - mean) / std
            } else {
                0.0
            };

            // Change rate
            let change_rate = inventory_proxy[i] - inventory_proxy[i - 1];

            // Smoothed momentum
            if ema_change.is_nan() {
                ema_change = change_rate;
            } else {
                ema_change = (change_rate - ema_change) * ema_mult + ema_change;
            }

            // Classify signal
            let signal = self.classify_signal(level, ema_change);

            result[i] = Some(InventoryLevelsOutput {
                level,
                change_rate,
                momentum: ema_change,
                signal,
            });
        }

        result
    }

    /// Classify the inventory signal.
    fn classify_signal(&self, level: f64, momentum: f64) -> InventorySignal {
        // Level-based classification
        if level > self.threshold {
            if momentum > 0.0 {
                InventorySignal::Building
            } else {
                InventorySignal::HighInventory
            }
        } else if level < -self.threshold {
            if momentum < 0.0 {
                InventorySignal::Drawing
            } else {
                InventorySignal::LowInventory
            }
        } else {
            // Normal range, check momentum
            if momentum > 0.0 {
                InventorySignal::Building
            } else if momentum < 0.0 {
                InventorySignal::Drawing
            } else {
                InventorySignal::Normal
            }
        }
    }

    /// Get the level series.
    pub fn level_series(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        self.calculate(close, volume)
            .iter()
            .map(|o| o.map(|v| v.level).unwrap_or(f64::NAN))
            .collect()
    }

    /// Get the momentum series.
    pub fn momentum_series(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        self.calculate(close, volume)
            .iter()
            .map(|o| o.map(|v| v.momentum).unwrap_or(f64::NAN))
            .collect()
    }
}

impl Default for InventoryLevels {
    fn default() -> Self {
        Self::default_params().unwrap()
    }
}

impl TechnicalIndicator for InventoryLevels {
    fn name(&self) -> &str {
        "InventoryLevels"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        2 // level, momentum
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }

        let outputs = self.calculate(&data.close, &data.volume);

        let primary: Vec<f64> = outputs
            .iter()
            .map(|o| o.map(|v| v.level).unwrap_or(f64::NAN))
            .collect();

        let secondary: Vec<f64> = outputs
            .iter()
            .map(|o| o.map(|v| v.momentum).unwrap_or(f64::NAN))
            .collect();

        Ok(IndicatorOutput::dual(primary, secondary))
    }
}

impl SignalIndicator for InventoryLevels {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let outputs = self.calculate(&data.close, &data.volume);

        match outputs.last().and_then(|o| *o) {
            Some(out) => match out.signal {
                InventorySignal::HighInventory | InventorySignal::Building => {
                    Ok(IndicatorSignal::Bearish)
                }
                InventorySignal::LowInventory | InventorySignal::Drawing => {
                    Ok(IndicatorSignal::Bullish)
                }
                InventorySignal::Normal => Ok(IndicatorSignal::Neutral),
            },
            None => Ok(IndicatorSignal::Neutral),
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let outputs = self.calculate(&data.close, &data.volume);

        let signals = outputs
            .iter()
            .map(|o| match o {
                Some(out) => match out.signal {
                    InventorySignal::HighInventory | InventorySignal::Building => {
                        IndicatorSignal::Bearish
                    }
                    InventorySignal::LowInventory | InventorySignal::Drawing => {
                        IndicatorSignal::Bullish
                    }
                    InventorySignal::Normal => IndicatorSignal::Neutral,
                },
                None => IndicatorSignal::Neutral,
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_accumulation_data(n: usize) -> (Vec<f64>, Vec<f64>) {
        // Declining prices with high volume (accumulation)
        let close: Vec<f64> = (0..n).map(|i| 100.0 - i as f64 * 0.1).collect();
        let volume: Vec<f64> = (0..n).map(|i| 1000.0 + i as f64 * 50.0).collect();
        (close, volume)
    }

    fn make_distribution_data(n: usize) -> (Vec<f64>, Vec<f64>) {
        // Rising prices with high volume (distribution)
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.1).collect();
        let volume: Vec<f64> = (0..n).map(|i| 1000.0 + i as f64 * 50.0).collect();
        (close, volume)
    }

    fn make_normal_data(n: usize) -> (Vec<f64>, Vec<f64>) {
        // Stable prices with normal volume
        let close: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64 * 0.1).sin())
            .collect();
        let volume: Vec<f64> = vec![1000.0; n];
        (close, volume)
    }

    #[test]
    fn test_new_valid_params() {
        let inv = InventoryLevels::new(20, 2.0);
        assert!(inv.is_ok());
    }

    #[test]
    fn test_new_invalid_period() {
        let inv = InventoryLevels::new(2, 2.0);
        assert!(inv.is_err());
    }

    #[test]
    fn test_new_invalid_threshold() {
        let inv = InventoryLevels::new(20, 0.0);
        assert!(inv.is_err());
    }

    #[test]
    fn test_accumulation_builds_inventory() {
        let (close, volume) = make_accumulation_data(50);
        let inv = InventoryLevels::new(10, 1.5).unwrap();
        let outputs = inv.calculate(&close, &volume);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Later values should show building/high inventory
        let last_outputs: Vec<_> = valid_outputs.iter().rev().take(10).collect();
        let building_count = last_outputs
            .iter()
            .filter(|o| matches!(o.signal, InventorySignal::Building | InventorySignal::HighInventory))
            .count();
        assert!(building_count > 0);
    }

    #[test]
    fn test_distribution_draws_inventory() {
        let (close, volume) = make_distribution_data(50);
        let inv = InventoryLevels::new(10, 1.5).unwrap();
        let outputs = inv.calculate(&close, &volume);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();
        assert!(!valid_outputs.is_empty());

        // Later values should show drawing/low inventory
        let last_outputs: Vec<_> = valid_outputs.iter().rev().take(10).collect();
        let drawing_count = last_outputs
            .iter()
            .filter(|o| matches!(o.signal, InventorySignal::Drawing | InventorySignal::LowInventory))
            .count();
        assert!(drawing_count > 0);
    }

    #[test]
    fn test_normal_data_normal_signal() {
        let (close, volume) = make_normal_data(50);
        let inv = InventoryLevels::new(10, 2.0).unwrap();
        let outputs = inv.calculate(&close, &volume);

        let valid_outputs: Vec<_> = outputs.iter().filter_map(|o| *o).collect();

        // Most should be within normal range
        let normal_count = valid_outputs
            .iter()
            .filter(|o| o.level.abs() < 2.0)
            .count();
        assert!(normal_count > valid_outputs.len() / 2);
    }

    #[test]
    fn test_technical_indicator_impl() {
        let (close, volume) = make_accumulation_data(50);
        let inv = InventoryLevels::default();
        let mut ohlcv = OHLCVSeries::from_close(close);
        ohlcv.volume = volume;

        let result = inv.compute(&ohlcv);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.primary.len(), 50);
        assert!(output.secondary.is_some());
    }

    #[test]
    fn test_signal_indicator_impl() {
        let (close, volume) = make_accumulation_data(50);
        let inv = InventoryLevels::default();
        let mut ohlcv = OHLCVSeries::from_close(close);
        ohlcv.volume = volume;

        let signal = inv.signal(&ohlcv);
        assert!(signal.is_ok());
    }

    #[test]
    fn test_insufficient_data() {
        let (close, volume) = make_accumulation_data(10);
        let inv = InventoryLevels::new(20, 2.0).unwrap();
        let mut ohlcv = OHLCVSeries::from_close(close);
        ohlcv.volume = volume;

        let result = inv.compute(&ohlcv);
        assert!(result.is_err());
    }

    #[test]
    fn test_level_series() {
        let (close, volume) = make_accumulation_data(50);
        let inv = InventoryLevels::new(10, 2.0).unwrap();
        let series = inv.level_series(&close, &volume);

        assert_eq!(series.len(), 50);
        assert!(series[0].is_nan());
    }

    #[test]
    fn test_momentum_series() {
        let (close, volume) = make_accumulation_data(50);
        let inv = InventoryLevels::new(10, 2.0).unwrap();
        let series = inv.momentum_series(&close, &volume);

        assert_eq!(series.len(), 50);
        assert!(series[0].is_nan());
    }

    #[test]
    fn test_with_smoothing_period() {
        let inv = InventoryLevels::new(20, 2.0).unwrap().with_smoothing_period(5);
        assert_eq!(inv.smoothing_period, 5);
    }

    #[test]
    fn test_default_impl() {
        let inv = InventoryLevels::default();
        assert_eq!(inv.period, 20);
        assert!((inv.threshold - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_signal_classification() {
        let inv = InventoryLevels::new(10, 2.0).unwrap();

        // High level + positive momentum = Building
        assert_eq!(inv.classify_signal(2.5, 0.1), InventorySignal::Building);

        // High level + negative momentum = HighInventory
        assert_eq!(inv.classify_signal(2.5, -0.1), InventorySignal::HighInventory);

        // Low level + negative momentum = Drawing
        assert_eq!(inv.classify_signal(-2.5, -0.1), InventorySignal::Drawing);

        // Low level + positive momentum = LowInventory
        assert_eq!(inv.classify_signal(-2.5, 0.1), InventorySignal::LowInventory);

        // Normal level
        assert_eq!(inv.classify_signal(0.5, 0.0), InventorySignal::Normal);
    }
}
