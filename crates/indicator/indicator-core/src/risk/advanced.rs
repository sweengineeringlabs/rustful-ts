//! Advanced Risk Indicators
//!
//! Additional risk metrics and portfolio analysis indicators.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Downside Deviation - Volatility of negative returns only
#[derive(Debug, Clone)]
pub struct DownsideDeviation {
    period: usize,
    threshold: f64,
}

impl DownsideDeviation {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate downside deviation (annualized)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut downside_squared_sum = 0.0;
            let mut count = 0;

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    let ret = close[j] / close[j - 1] - 1.0;
                    if ret < self.threshold {
                        let excess = ret - self.threshold;
                        downside_squared_sum += excess * excess;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                result[i] = (downside_squared_sum / count as f64).sqrt() * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for DownsideDeviation {
    fn name(&self) -> &str {
        "Downside Deviation"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Upside Potential Ratio - Upside vs downside potential
#[derive(Debug, Clone)]
pub struct UpsidePotentialRatio {
    period: usize,
    threshold: f64,
}

impl UpsidePotentialRatio {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate upside potential ratio
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut upside_sum = 0.0;
            let mut downside_squared_sum = 0.0;
            let mut downside_count = 0;

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    let ret = close[j] / close[j - 1] - 1.0;
                    if ret > self.threshold {
                        upside_sum += ret - self.threshold;
                    } else if ret < self.threshold {
                        let excess = ret - self.threshold;
                        downside_squared_sum += excess * excess;
                        downside_count += 1;
                    }
                }
            }

            let period_len = (i - start) as f64;
            let upside_avg = upside_sum / period_len;
            let downside_dev = if downside_count > 0 {
                (downside_squared_sum / downside_count as f64).sqrt()
            } else {
                1e-10
            };

            if downside_dev > 1e-10 {
                result[i] = upside_avg / downside_dev;
            }
        }

        result
    }
}

impl TechnicalIndicator for UpsidePotentialRatio {
    fn name(&self) -> &str {
        "Upside Potential Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Kappa Ratio - Higher moment risk-adjusted return
#[derive(Debug, Clone)]
pub struct KappaRatio {
    period: usize,
    order: usize,
    threshold: f64,
}

impl KappaRatio {
    pub fn new(period: usize, order: usize, threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if order < 1 || order > 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "order".to_string(),
                reason: "must be between 1 and 4".to_string(),
            });
        }
        Ok(Self { period, order, threshold })
    }

    /// Calculate kappa ratio
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let mut returns = Vec::new();
            let mut below_threshold_sum = 0.0;
            let mut below_count = 0;

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    let ret = close[j] / close[j - 1] - 1.0;
                    returns.push(ret);
                    if ret < self.threshold {
                        below_threshold_sum += (self.threshold - ret).powi(self.order as i32);
                        below_count += 1;
                    }
                }
            }

            if returns.is_empty() || below_count == 0 {
                continue;
            }

            let mean_ret: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let lpm = (below_threshold_sum / below_count as f64).powf(1.0 / self.order as f64);

            if lpm > 1e-10 {
                result[i] = (mean_ret - self.threshold) / lpm;
            }
        }

        result
    }
}

impl TechnicalIndicator for KappaRatio {
    fn name(&self) -> &str {
        "Kappa Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Win Rate - Percentage of positive return periods
#[derive(Debug, Clone)]
pub struct WinRate {
    period: usize,
}

impl WinRate {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate win rate (0-100)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut wins = 0;
            let mut total = 0;

            for j in (start + 1)..=i {
                if close[j] > close[j - 1] {
                    wins += 1;
                }
                total += 1;
            }

            if total > 0 {
                result[i] = (wins as f64 / total as f64) * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for WinRate {
    fn name(&self) -> &str {
        "Win Rate"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Profit Factor - Gross profits / Gross losses
#[derive(Debug, Clone)]
pub struct ProfitFactor {
    period: usize,
}

impl ProfitFactor {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate profit factor
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut gross_profits = 0.0;
            let mut gross_losses = 0.0;

            for j in (start + 1)..=i {
                let change = close[j] - close[j - 1];
                if change > 0.0 {
                    gross_profits += change;
                } else {
                    gross_losses += change.abs();
                }
            }

            if gross_losses > 1e-10 {
                result[i] = gross_profits / gross_losses;
            } else if gross_profits > 0.0 {
                result[i] = 10.0; // Cap at 10 if no losses
            }
        }

        result
    }
}

impl TechnicalIndicator for ProfitFactor {
    fn name(&self) -> &str {
        "Profit Factor"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Expectancy - Average profit per trade
#[derive(Debug, Clone)]
pub struct Expectancy {
    period: usize,
}

impl Expectancy {
    pub fn new(period: usize) -> Result<Self> {
        if period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate expectancy (win_rate * avg_win - loss_rate * avg_loss)
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut total_wins = 0.0;
            let mut total_losses = 0.0;
            let mut win_count = 0;
            let mut loss_count = 0;

            for j in (start + 1)..=i {
                let pct_change = if close[j - 1] > 1e-10 {
                    (close[j] / close[j - 1] - 1.0) * 100.0
                } else {
                    0.0
                };

                if pct_change > 0.0 {
                    total_wins += pct_change;
                    win_count += 1;
                } else if pct_change < 0.0 {
                    total_losses += pct_change.abs();
                    loss_count += 1;
                }
            }

            let total_trades = win_count + loss_count;
            if total_trades > 0 {
                let win_rate = win_count as f64 / total_trades as f64;
                let loss_rate = loss_count as f64 / total_trades as f64;
                let avg_win = if win_count > 0 { total_wins / win_count as f64 } else { 0.0 };
                let avg_loss = if loss_count > 0 { total_losses / loss_count as f64 } else { 0.0 };

                result[i] = win_rate * avg_win - loss_rate * avg_loss;
            }
        }

        result
    }
}

impl TechnicalIndicator for Expectancy {
    fn name(&self) -> &str {
        "Expectancy"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> OHLCVSeries {
        let close: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64) * 0.5 + (i as f64 * 0.3).sin() * 2.0).collect();
        let high: Vec<f64> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f64> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f64> = close.clone();
        let volume: Vec<f64> = vec![1000.0; 50];

        OHLCVSeries { open, high, low, close, volume }
    }

    #[test]
    fn test_downside_deviation() {
        let data = make_test_data();
        let dd = DownsideDeviation::new(14, 0.0).unwrap();
        let result = dd.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Downside deviation should be non-negative
        for i in 20..result.len() {
            assert!(result[i] >= 0.0);
        }
    }

    #[test]
    fn test_upside_potential_ratio() {
        let data = make_test_data();
        let upr = UpsidePotentialRatio::new(14, 0.0).unwrap();
        let result = upr.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_kappa_ratio() {
        let data = make_test_data();
        let kr = KappaRatio::new(14, 2, 0.0).unwrap();
        let result = kr.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_win_rate() {
        let data = make_test_data();
        let wr = WinRate::new(14).unwrap();
        let result = wr.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Win rate should be 0-100
        for i in 20..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }

    #[test]
    fn test_profit_factor() {
        let data = make_test_data();
        let pf = ProfitFactor::new(14).unwrap();
        let result = pf.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // Profit factor should be non-negative
        for i in 20..result.len() {
            assert!(result[i] >= 0.0);
        }
    }

    #[test]
    fn test_expectancy() {
        let data = make_test_data();
        let exp = Expectancy::new(14).unwrap();
        let result = exp.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_validation() {
        assert!(DownsideDeviation::new(5, 0.0).is_err());
        assert!(UpsidePotentialRatio::new(5, 0.0).is_err());
        assert!(KappaRatio::new(5, 2, 0.0).is_err());
        assert!(KappaRatio::new(14, 5, 0.0).is_err()); // order > 4
        assert!(WinRate::new(2).is_err());
        assert!(ProfitFactor::new(2).is_err());
        assert!(Expectancy::new(2).is_err());
    }
}
