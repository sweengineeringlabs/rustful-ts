//! VIX and Volatility Index Derived Indicators
//!
//! Indicators based on VIX methodology and volatility indices.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use crate::SMA;

/// VIX Term Structure - Compares short-term vs long-term volatility.
///
/// Contango: Long-term VIX > Short-term VIX (normal market)
/// Backwardation: Short-term VIX > Long-term VIX (fear/stress)
#[derive(Debug, Clone)]
pub struct VIXTermStructure {
    short_period: usize,
    long_period: usize,
}

impl VIXTermStructure {
    pub fn new() -> Self {
        Self { short_period: 9, long_period: 30 }
    }

    pub fn with_periods(short: usize, long: usize) -> Self {
        Self { short_period: short, long_period: long }
    }

    /// Calculate term structure ratio.
    /// > 1: Backwardation (fear), < 1: Contango (complacency)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        // Use rolling volatility as VIX proxy
        let n = close.len();
        if n < self.long_period + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in self.long_period..n {
            // Short-term realized volatility
            let short_start = i - self.short_period;
            let short_returns: Vec<f64> = (short_start + 1..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            if short_returns.is_empty() {
                continue;
            }

            let short_mean: f64 = short_returns.iter().sum::<f64>() / short_returns.len() as f64;
            let short_var: f64 = short_returns.iter()
                .map(|r| (r - short_mean).powi(2))
                .sum::<f64>() / short_returns.len() as f64;
            let short_vol = (short_var * 252.0).sqrt() * 100.0;

            // Long-term realized volatility
            let long_start = i - self.long_period;
            let long_returns: Vec<f64> = (long_start + 1..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            if long_returns.is_empty() {
                continue;
            }

            let long_mean: f64 = long_returns.iter().sum::<f64>() / long_returns.len() as f64;
            let long_var: f64 = long_returns.iter()
                .map(|r| (r - long_mean).powi(2))
                .sum::<f64>() / long_returns.len() as f64;
            let long_vol = (long_var * 252.0).sqrt() * 100.0;

            if long_vol > 0.0 {
                result[i] = short_vol / long_vol;
            }
        }

        result
    }
}

impl Default for VIXTermStructure {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for VIXTermStructure {
    fn name(&self) -> &str {
        "VIX Term Structure"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.long_period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.long_period + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for VIXTermStructure {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);

        if values.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let ratio = *values.last().unwrap();
        if ratio.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Backwardation (fear) can be a contrarian buy signal
        if ratio > 1.2 {
            Ok(IndicatorSignal::Bullish) // Contrarian
        } else if ratio < 0.8 {
            Ok(IndicatorSignal::Bearish) // Complacency warning
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);

        Ok(values.iter().map(|&r| {
            if r.is_nan() {
                IndicatorSignal::Neutral
            } else if r > 1.2 {
                IndicatorSignal::Bullish
            } else if r < 0.8 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

/// Volatility of Volatility (VVIX proxy) - Measures volatility clustering.
///
/// High values indicate unstable volatility regime.
#[derive(Debug, Clone)]
pub struct VolatilityOfVolatility {
    vol_period: usize,
    vov_period: usize,
}

impl VolatilityOfVolatility {
    pub fn new() -> Self {
        Self { vol_period: 20, vov_period: 20 }
    }

    pub fn with_periods(vol: usize, vov: usize) -> Self {
        Self { vol_period: vol, vov_period: vov }
    }

    /// Calculate volatility of volatility.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let total_lookback = self.vol_period + self.vov_period;

        if n < total_lookback + 1 {
            return vec![f64::NAN; n];
        }

        // First calculate rolling volatility
        let mut volatility = vec![f64::NAN; n];

        for i in self.vol_period..n {
            let returns: Vec<f64> = ((i - self.vol_period + 1)..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() >= 2 {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let var: f64 = returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / (returns.len() - 1) as f64;
                volatility[i] = (var * 252.0).sqrt() * 100.0;
            }
        }

        // Then calculate std dev of volatility
        let mut result = vec![f64::NAN; n];

        for i in total_lookback..n {
            let vol_window: Vec<f64> = volatility[(i - self.vov_period + 1)..=i]
                .iter()
                .filter(|v| !v.is_nan())
                .cloned()
                .collect();

            if vol_window.len() >= 2 {
                let mean: f64 = vol_window.iter().sum::<f64>() / vol_window.len() as f64;
                let var: f64 = vol_window.iter()
                    .map(|v| (v - mean).powi(2))
                    .sum::<f64>() / (vol_window.len() - 1) as f64;
                result[i] = var.sqrt();
            }
        }

        result
    }
}

impl Default for VolatilityOfVolatility {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for VolatilityOfVolatility {
    fn name(&self) -> &str {
        "Volatility of Volatility"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let total_lookback = self.vol_period + self.vov_period;
        if data.close.len() < total_lookback + 1 {
            return Err(IndicatorError::InsufficientData {
                required: total_lookback + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.vol_period + self.vov_period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Volatility Skew - Approximates options skew from price action.
///
/// Measures asymmetry in volatility (puts vs calls proxy).
#[derive(Debug, Clone)]
pub struct VolatilitySkew {
    period: usize,
}

impl VolatilitySkew {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate volatility skew approximation.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in self.period..n {
            let returns: Vec<f64> = ((i - self.period + 1)..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() < 3 {
                continue;
            }

            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let n_f = returns.len() as f64;

            // Calculate moments
            let m2: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n_f;
            let m3: f64 = returns.iter().map(|r| (r - mean).powi(3)).sum::<f64>() / n_f;

            if m2.abs() < 1e-10 {
                continue;
            }

            // Skewness
            let skew = m3 / m2.powf(1.5);
            result[i] = skew;
        }

        result
    }
}

impl TechnicalIndicator for VolatilitySkew {
    fn name(&self) -> &str {
        "Volatility Skew"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Put/Call Ratio Proxy - Approximates sentiment from price action.
///
/// Uses down-volume vs up-volume as a proxy for put/call activity.
#[derive(Debug, Clone)]
pub struct PutCallProxy {
    period: usize,
}

impl PutCallProxy {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    /// Calculate put/call ratio proxy.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.period + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in self.period..n {
            let mut up_volume = 0.0;
            let mut down_volume = 0.0;

            for j in (i - self.period + 1)..=i {
                if close[j] > close[j - 1] {
                    up_volume += volume[j];
                } else if close[j] < close[j - 1] {
                    down_volume += volume[j];
                }
            }

            if up_volume > 0.0 {
                result[i] = down_volume / up_volume;
            }
        }

        result
    }
}

impl TechnicalIndicator for PutCallProxy {
    fn name(&self) -> &str {
        "Put/Call Proxy"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for PutCallProxy {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close, &data.volume);

        if values.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let ratio = *values.last().unwrap();
        if ratio.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // High put/call (fear) = contrarian bullish
        if ratio > 1.2 {
            Ok(IndicatorSignal::Bullish)
        } else if ratio < 0.7 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close, &data.volume);

        Ok(values.iter().map(|&r| {
            if r.is_nan() {
                IndicatorSignal::Neutral
            } else if r > 1.2 {
                IndicatorSignal::Bullish
            } else if r < 0.7 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

/// Volatility Percentile - Current vol relative to historical range.
///
/// Shows where current volatility ranks over history.
#[derive(Debug, Clone)]
pub struct VolatilityPercentile {
    vol_period: usize,
    rank_period: usize,
}

impl VolatilityPercentile {
    pub fn new() -> Self {
        Self { vol_period: 20, rank_period: 252 }
    }

    pub fn with_periods(vol: usize, rank: usize) -> Self {
        Self { vol_period: vol, rank_period: rank }
    }

    /// Calculate volatility percentile (0-100).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let total = self.vol_period + self.rank_period;

        if n < total + 1 {
            return vec![f64::NAN; n];
        }

        // Calculate rolling volatility
        let mut volatility = vec![f64::NAN; n];

        for i in self.vol_period..n {
            let returns: Vec<f64> = ((i - self.vol_period + 1)..=i)
                .filter_map(|j| {
                    if close[j - 1] > 0.0 {
                        Some((close[j] / close[j - 1]).ln())
                    } else {
                        None
                    }
                })
                .collect();

            if returns.len() >= 2 {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let var: f64 = returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / (returns.len() - 1) as f64;
                volatility[i] = (var * 252.0).sqrt() * 100.0;
            }
        }

        // Calculate percentile rank
        let mut result = vec![f64::NAN; n];

        for i in total..n {
            let current_vol = volatility[i];
            if current_vol.is_nan() {
                continue;
            }

            let historical: Vec<f64> = volatility[(i - self.rank_period)..i]
                .iter()
                .filter(|v| !v.is_nan())
                .cloned()
                .collect();

            if historical.is_empty() {
                continue;
            }

            let count_below = historical.iter().filter(|&&v| v < current_vol).count();
            result[i] = (count_below as f64 / historical.len() as f64) * 100.0;
        }

        result
    }
}

impl Default for VolatilityPercentile {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for VolatilityPercentile {
    fn name(&self) -> &str {
        "Volatility Percentile"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let total = self.vol_period + self.rank_period;
        if data.close.len() < total + 1 {
            return Err(IndicatorError::InsufficientData {
                required: total + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.vol_period + self.rank_period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Volatility Regime - Classifies market volatility state.
///
/// Low: Vol below 25th percentile
/// Normal: Vol between 25th-75th percentile
/// High: Vol above 75th percentile
/// Extreme: Vol above 90th percentile
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolRegime {
    Low,
    Normal,
    High,
    Extreme,
}

#[derive(Debug, Clone)]
pub struct VolatilityRegime {
    vol_period: usize,
    rank_period: usize,
}

impl VolatilityRegime {
    pub fn new() -> Self {
        Self { vol_period: 20, rank_period: 252 }
    }

    /// Calculate volatility regime.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let vp = VolatilityPercentile::with_periods(self.vol_period, self.rank_period);
        let percentiles = vp.calculate(close);

        percentiles.iter().map(|&p| {
            if p.is_nan() {
                f64::NAN
            } else if p >= 90.0 {
                4.0 // Extreme
            } else if p >= 75.0 {
                3.0 // High
            } else if p >= 25.0 {
                2.0 // Normal
            } else {
                1.0 // Low
            }
        }).collect()
    }
}

impl Default for VolatilityRegime {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for VolatilityRegime {
    fn name(&self) -> &str {
        "Volatility Regime"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let total = self.vol_period + self.rank_period;
        if data.close.len() < total + 1 {
            return Err(IndicatorError::InsufficientData {
                required: total + 1,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.vol_period + self.rank_period + 1
    }

    fn output_features(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..300).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0 + i as f64 * 0.02).collect();
        let volume: Vec<f64> = (0..300).map(|i| 1000000.0 + (i as f64 * 0.2).sin() * 200000.0).collect();
        (close, volume)
    }

    #[test]
    fn test_vix_term_structure() {
        let (close, _) = make_test_data();
        let vts = VIXTermStructure::new();
        let result = vts.calculate(&close);

        assert_eq!(result.len(), 300);
    }

    #[test]
    fn test_volatility_of_volatility() {
        let (close, _) = make_test_data();
        let vov = VolatilityOfVolatility::new();
        let result = vov.calculate(&close);

        assert_eq!(result.len(), 300);
    }

    #[test]
    fn test_volatility_skew() {
        let (close, _) = make_test_data();
        let skew = VolatilitySkew::new(20);
        let result = skew.calculate(&close);

        assert_eq!(result.len(), 300);
    }

    #[test]
    fn test_put_call_proxy() {
        let (close, volume) = make_test_data();
        let pcp = PutCallProxy::new(20);
        let result = pcp.calculate(&close, &volume);

        assert_eq!(result.len(), 300);
    }

    #[test]
    fn test_volatility_percentile() {
        let (close, _) = make_test_data();
        let vp = VolatilityPercentile::new();
        let result = vp.calculate(&close);

        assert_eq!(result.len(), 300);

        // Percentile should be 0-100
        for i in 272..300 {
            if !result[i].is_nan() {
                assert!(result[i] >= 0.0 && result[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_volatility_regime() {
        let (close, _) = make_test_data();
        let vr = VolatilityRegime::new();
        let result = vr.calculate(&close);

        assert_eq!(result.len(), 300);

        // Regime should be 1-4
        for i in 272..300 {
            if !result[i].is_nan() {
                assert!(result[i] >= 1.0 && result[i] <= 4.0);
            }
        }
    }
}
