//! Implied Volatility Indicators
//!
//! Indicators for analyzing implied volatility metrics.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// IV Rank - Percentile ranking of current IV vs historical IV
#[derive(Debug, Clone)]
pub struct IVRank {
    period: usize,
}

impl IVRank {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate IV Rank from historical volatility as proxy
    /// IV Rank = (Current IV - 52-week Low IV) / (52-week High IV - 52-week Low IV) * 100
    pub fn calculate(&self, iv: &[f64]) -> Vec<f64> {
        let n = iv.len();
        let mut result = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let window = &iv[start..=i];

            let min_iv = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_iv = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            if max_iv > min_iv {
                result[i] = (iv[i] - min_iv) / (max_iv - min_iv) * 100.0;
            } else {
                result[i] = 50.0; // Neutral when no range
            }
        }
        result
    }
}

impl TechnicalIndicator for IVRank {
    fn name(&self) -> &str {
        "IV Rank"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        // Use historical volatility as IV proxy
        let hv = calculate_hv(&data.close, 20);
        Ok(IndicatorOutput::single(self.calculate(&hv)))
    }
}

/// IV Percentile - Historical percentile of current IV
#[derive(Debug, Clone)]
pub struct IVPercentile {
    period: usize,
}

impl IVPercentile {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate IV Percentile - percentage of days IV was lower than current
    pub fn calculate(&self, iv: &[f64]) -> Vec<f64> {
        let n = iv.len();
        let mut result = vec![0.0; n];

        for i in (self.period - 1)..n {
            let start = i.saturating_sub(self.period - 1);
            let current = iv[i];
            let count_below = iv[start..i].iter().filter(|&&v| v < current).count();
            result[i] = count_below as f64 / (i - start) as f64 * 100.0;
        }
        result
    }
}

impl TechnicalIndicator for IVPercentile {
    fn name(&self) -> &str {
        "IV Percentile"
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let hv = calculate_hv(&data.close, 20);
        Ok(IndicatorOutput::single(self.calculate(&hv)))
    }
}

/// IV Skew Slope - Put-call skew measure
/// Approximated using price-based skewness
#[derive(Debug, Clone)]
pub struct IVSkewSlope {
    period: usize,
}

impl IVSkewSlope {
    pub fn new(period: usize) -> Result<Self> {
        if period < 3 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 3".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate skew based on returns distribution
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let returns: Vec<f64> = (start + 1..=i)
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
                // Skewness
                let skew = returns.iter()
                    .map(|r| ((r - mean) / std).powi(3))
                    .sum::<f64>() / returns.len() as f64;
                result[i] = skew;
            }
        }
        result
    }
}

impl TechnicalIndicator for IVSkewSlope {
    fn name(&self) -> &str {
        "IV Skew Slope"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Term Structure Slope - IV term structure analysis
/// Uses short vs long term volatility comparison
#[derive(Debug, Clone)]
pub struct TermStructureSlope {
    short_period: usize,
    long_period: usize,
}

impl TermStructureSlope {
    pub fn new(short_period: usize, long_period: usize) -> Result<Self> {
        if short_period == 0 || long_period == 0 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "periods must be positive".to_string(),
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

    /// Calculate term structure slope
    /// Positive = Contango (normal), Negative = Backwardation
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let short_vol = calculate_hv(close, self.short_period);
        let long_vol = calculate_hv(close, self.long_period);

        let mut result = vec![0.0; n];
        for i in 0..n {
            if long_vol[i] > 0.0 {
                // Slope = (Long Term IV - Short Term IV) / Long Term IV
                result[i] = (long_vol[i] - short_vol[i]) / long_vol[i] * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for TermStructureSlope {
    fn name(&self) -> &str {
        "Term Structure Slope"
    }

    fn min_periods(&self) -> usize {
        self.long_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Vol of Vol - Volatility of volatility
#[derive(Debug, Clone)]
pub struct VolOfVol {
    vol_period: usize,
    vov_period: usize,
}

impl VolOfVol {
    pub fn new(vol_period: usize, vov_period: usize) -> Result<Self> {
        if vol_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "vol_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if vov_period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "vov_period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { vol_period, vov_period })
    }

    /// Calculate volatility of volatility
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let hv = calculate_hv(close, self.vol_period);

        // Now calculate volatility of the HV series
        let mut result = vec![0.0; n];

        for i in (self.vol_period + self.vov_period - 1)..n {
            let start = i.saturating_sub(self.vov_period - 1);
            let window = &hv[start..=i];

            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance = window.iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>() / window.len() as f64;
            result[i] = variance.sqrt();
        }
        result
    }
}

impl TechnicalIndicator for VolOfVol {
    fn name(&self) -> &str {
        "Vol of Vol"
    }

    fn min_periods(&self) -> usize {
        self.vol_period + self.vov_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Risk Reversal - 25-delta skew measure
/// Approximated using upside vs downside volatility
#[derive(Debug, Clone)]
pub struct RiskReversal {
    period: usize,
}

impl RiskReversal {
    pub fn new(period: usize) -> Result<Self> {
        if period < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate risk reversal proxy
    /// Positive = calls more expensive (bullish), Negative = puts more expensive (bearish)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            let mut up_returns: Vec<f64> = Vec::new();
            let mut down_returns: Vec<f64> = Vec::new();

            for j in (start + 1)..=i {
                let ret = (close[j] / close[j - 1]).ln();
                if ret > 0.0 {
                    up_returns.push(ret);
                } else {
                    down_returns.push(ret);
                }
            }

            let up_vol = if !up_returns.is_empty() {
                let mean = up_returns.iter().sum::<f64>() / up_returns.len() as f64;
                let var = up_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                    / up_returns.len() as f64;
                var.sqrt()
            } else {
                0.0
            };

            let down_vol = if !down_returns.is_empty() {
                let mean = down_returns.iter().sum::<f64>() / down_returns.len() as f64;
                let var = down_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                    / down_returns.len() as f64;
                var.sqrt()
            } else {
                0.0
            };

            // Risk reversal = upside vol - downside vol (normalized)
            let total_vol = up_vol + down_vol;
            if total_vol > 0.0 {
                result[i] = (up_vol - down_vol) / total_vol * 100.0;
            }
        }
        result
    }
}

impl TechnicalIndicator for RiskReversal {
    fn name(&self) -> &str {
        "Risk Reversal"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Helper function to calculate historical volatility
fn calculate_hv(close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![0.0; n];

    if n < 2 || period < 2 {
        return result;
    }

    for i in period..n {
        let start = i.saturating_sub(period);

        // Calculate log returns
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

        // Annualized volatility
        result[i] = variance.sqrt() * (252.0_f64).sqrt() * 100.0;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> Vec<f64> {
        vec![100.0, 101.0, 99.5, 102.0, 101.5, 103.0, 102.5, 104.0, 103.5, 105.0,
             104.5, 106.0, 105.5, 107.0, 106.5, 108.0, 107.5, 109.0, 108.5, 110.0,
             109.5, 111.0, 110.5, 112.0, 111.5, 113.0, 112.5, 114.0, 113.5, 115.0]
    }

    #[test]
    fn test_iv_rank() {
        let close = make_test_data();
        let hv = calculate_hv(&close, 10);
        let iv_rank = IVRank::new(20).unwrap();
        let result = iv_rank.calculate(&hv);

        assert_eq!(result.len(), close.len());
        // IV Rank should be between 0 and 100
        for &v in result.iter().skip(20) {
            assert!(v >= 0.0 && v <= 100.0, "IV Rank {} out of range", v);
        }
    }

    #[test]
    fn test_iv_percentile() {
        let close = make_test_data();
        let hv = calculate_hv(&close, 10);
        let iv_pct = IVPercentile::new(20).unwrap();
        let result = iv_pct.calculate(&hv);

        assert_eq!(result.len(), close.len());
        // IV Percentile should be between 0 and 100
        for &v in result.iter().skip(20) {
            assert!(v >= 0.0 && v <= 100.0, "IV Percentile {} out of range", v);
        }
    }

    #[test]
    fn test_iv_skew_slope() {
        let close = make_test_data();
        let skew = IVSkewSlope::new(10).unwrap();
        let result = skew.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Skew values should be reasonable
        for &v in result.iter().skip(15) {
            assert!(v.abs() < 10.0, "Skew {} out of reasonable range", v);
        }
    }

    #[test]
    fn test_term_structure_slope() {
        let close = make_test_data();
        let ts = TermStructureSlope::new(5, 20).unwrap();
        let result = ts.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Check values exist after warmup
        assert!(result[25].abs() < 200.0); // Reasonable range
    }

    #[test]
    fn test_vol_of_vol() {
        let close = make_test_data();
        let vov = VolOfVol::new(10, 5).unwrap();
        let result = vov.calculate(&close);

        assert_eq!(result.len(), close.len());
        // VoV should be non-negative
        for &v in result.iter().skip(15) {
            assert!(v >= 0.0, "VoV {} should be non-negative", v);
        }
    }

    #[test]
    fn test_risk_reversal() {
        let close = make_test_data();
        let rr = RiskReversal::new(10).unwrap();
        let result = rr.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Risk reversal should be between -100 and 100
        for &v in result.iter().skip(15) {
            assert!(v >= -100.0 && v <= 100.0, "Risk Reversal {} out of range", v);
        }
    }
}
