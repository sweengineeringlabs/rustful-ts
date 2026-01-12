//! Commodity Selection Index (CSI) implementation.
//!
//! Combines ADXR and ATR to identify trending commodities with volatility.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use indicator_core::{ADX, ATR};

/// Commodity Selection Index output.
#[derive(Debug, Clone)]
pub struct CommoditySelectionOutput {
    /// CSI values.
    pub csi: Vec<f64>,
    /// ADXR component values.
    pub adxr: Vec<f64>,
    /// ATR component values.
    pub atr: Vec<f64>,
}

/// Commodity Selection Index configuration.
#[derive(Debug, Clone)]
pub struct CommoditySelectionConfig {
    /// ADX period (default: 14).
    pub adx_period: usize,
    /// ATR period (default: 14).
    pub atr_period: usize,
    /// Commission per trade for normalization (default: 10.0).
    pub commission: f64,
    /// Margin requirement for normalization (default: 1000.0).
    pub margin: f64,
    /// Strong trend threshold (default: 25.0).
    pub trend_threshold: f64,
}

impl Default for CommoditySelectionConfig {
    fn default() -> Self {
        Self {
            adx_period: 14,
            atr_period: 14,
            commission: 10.0,
            margin: 1000.0,
            trend_threshold: 25.0,
        }
    }
}

/// Commodity Selection Index (CSI).
///
/// Developed by Welles Wilder, the CSI ranks commodities/securities based on:
/// - ADXR (Average Directional Index Rating): Measures trend strength
/// - ATR (Average True Range): Measures volatility
///
/// Formula: CSI = ADXR * ATR * (100 / sqrt(Margin)) * (1 / (150 + Commission))
///
/// Higher CSI values indicate better trending conditions with sufficient volatility.
/// Use this to select which instruments to trade.
#[derive(Debug, Clone)]
pub struct CommoditySelectionIndex {
    adx: ADX,
    atr: ATR,
    commission: f64,
    margin: f64,
    trend_threshold: f64,
}

impl CommoditySelectionIndex {
    pub fn new(config: CommoditySelectionConfig) -> Self {
        Self {
            adx: ADX::new(config.adx_period),
            atr: ATR::new(config.atr_period),
            commission: config.commission,
            margin: config.margin,
            trend_threshold: config.trend_threshold,
        }
    }

    /// Calculate CSI values.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> CommoditySelectionOutput {
        let n = close.len();

        // Calculate ADX for trend strength
        let adx_output = self.adx.calculate(high, low, close);
        let adx_values = &adx_output.adx;

        // Calculate ADXR (smoothed ADX)
        let adx_period = 14; // Standard period
        let mut adxr = vec![f64::NAN; n];

        for i in adx_period..n {
            if !adx_values[i].is_nan() && !adx_values[i - adx_period].is_nan() {
                adxr[i] = (adx_values[i] + adx_values[i - adx_period]) / 2.0;
            }
        }

        // Calculate ATR for volatility
        let atr_values = self.atr.calculate(high, low, close);

        // Calculate CSI
        let mut csi = vec![f64::NAN; n];
        let margin_factor = 100.0 / self.margin.sqrt();
        let commission_factor = 1.0 / (150.0 + self.commission);

        for i in 0..n {
            if !adxr[i].is_nan() && !atr_values[i].is_nan() {
                csi[i] = adxr[i] * atr_values[i] * margin_factor * commission_factor;
            }
        }

        CommoditySelectionOutput {
            csi,
            adxr,
            atr: atr_values,
        }
    }

    /// Calculate just the CSI values.
    pub fn calculate_csi(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        self.calculate(high, low, close).csi
    }
}

impl Default for CommoditySelectionIndex {
    fn default() -> Self {
        Self::new(CommoditySelectionConfig::default())
    }
}

impl TechnicalIndicator for CommoditySelectionIndex {
    fn name(&self) -> &str {
        "CommoditySelectionIndex"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = 42; // ADX period * 2 + ADX period for ADXR
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(result.csi, result.adxr, result.atr))
    }

    fn min_periods(&self) -> usize {
        42
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for CommoditySelectionIndex {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(&data.high, &data.low, &data.close);
        let n = result.csi.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let curr_csi = result.csi[n - 1];
        let prev_csi = result.csi[n - 2];
        let curr_adxr = result.adxr[n - 1];

        if curr_csi.is_nan() || prev_csi.is_nan() || curr_adxr.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // High CSI and rising indicates good trading opportunity
        // Use ADXR direction to determine bullish/bearish
        if curr_csi > prev_csi && curr_adxr > self.trend_threshold {
            // Check ADX +DI vs -DI for direction (simplified: assume trend continuation)
            let adx_output = self.adx.calculate(&data.high, &data.low, &data.close);
            let plus_di = adx_output.plus_di[n - 1];
            let minus_di = adx_output.minus_di[n - 1];

            if !plus_di.is_nan() && !minus_di.is_nan() {
                if plus_di > minus_di {
                    return Ok(IndicatorSignal::Bullish);
                } else {
                    return Ok(IndicatorSignal::Bearish);
                }
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(&data.high, &data.low, &data.close);
        let adx_output = self.adx.calculate(&data.high, &data.low, &data.close);
        let n = result.csi.len();

        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..n {
            let curr_csi = result.csi[i];
            let prev_csi = result.csi[i - 1];
            let curr_adxr = result.adxr[i];

            if curr_csi.is_nan() || prev_csi.is_nan() || curr_adxr.is_nan() {
                signals.push(IndicatorSignal::Neutral);
                continue;
            }

            if curr_csi > prev_csi && curr_adxr > self.trend_threshold {
                let plus_di = adx_output.plus_di[i];
                let minus_di = adx_output.minus_di[i];

                if !plus_di.is_nan() && !minus_di.is_nan() {
                    if plus_di > minus_di {
                        signals.push(IndicatorSignal::Bullish);
                    } else {
                        signals.push(IndicatorSignal::Bearish);
                    }
                } else {
                    signals.push(IndicatorSignal::Neutral);
                }
            } else {
                signals.push(IndicatorSignal::Neutral);
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..n).map(|i| 105.0 + i as f64 * 0.5).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + i as f64 * 0.5).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.5).collect();
        (high, low, close)
    }

    #[test]
    fn test_commodity_selection_basic() {
        let csi = CommoditySelectionIndex::default();
        let (high, low, close) = generate_test_data(60);

        let result = csi.calculate(&high, &low, &close);

        assert_eq!(result.csi.len(), 60);
        assert_eq!(result.adxr.len(), 60);
        assert_eq!(result.atr.len(), 60);
    }

    #[test]
    fn test_commodity_selection_positive() {
        let csi = CommoditySelectionIndex::default();
        let (high, low, close) = generate_test_data(60);

        let result = csi.calculate(&high, &low, &close);

        // CSI should be positive when we have trend and volatility
        for i in 45..60 {
            if !result.csi[i].is_nan() {
                assert!(result.csi[i] >= 0.0, "CSI should be non-negative");
            }
        }
    }

    #[test]
    fn test_commodity_selection_compute() {
        let csi = CommoditySelectionIndex::default();
        let (high, low, close) = generate_test_data(60);

        let series = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 60],
        };

        let output = csi.compute(&series).unwrap();
        assert_eq!(output.primary.len(), 60);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_commodity_selection_config() {
        let config = CommoditySelectionConfig {
            adx_period: 10,
            atr_period: 10,
            commission: 5.0,
            margin: 500.0,
            trend_threshold: 20.0,
        };

        let csi = CommoditySelectionIndex::new(config);
        assert_eq!(csi.name(), "CommoditySelectionIndex");
    }
}
