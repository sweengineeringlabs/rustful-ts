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

/// Conditional Beta - Beta that varies with market conditions
///
/// Calculates beta only when benchmark returns exceed a threshold,
/// capturing how the asset behaves during significant market moves.
#[derive(Debug, Clone)]
pub struct ConditionalBeta {
    period: usize,
    threshold: f64,
}

impl ConditionalBeta {
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate conditional beta based on market condition threshold
    pub fn calculate(&self, close: &[f64], benchmark: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if benchmark.len() != n {
            return result;
        }

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Collect returns where benchmark exceeds threshold
            let mut asset_returns = Vec::new();
            let mut benchmark_returns = Vec::new();

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 && benchmark[j - 1] > 1e-10 {
                    let bench_ret = benchmark[j] / benchmark[j - 1] - 1.0;
                    if bench_ret.abs() > self.threshold {
                        let asset_ret = close[j] / close[j - 1] - 1.0;
                        asset_returns.push(asset_ret);
                        benchmark_returns.push(bench_ret);
                    }
                }
            }

            if asset_returns.len() < 5 {
                continue;
            }

            // Calculate beta using covariance / variance
            let asset_mean: f64 = asset_returns.iter().sum::<f64>() / asset_returns.len() as f64;
            let bench_mean: f64 = benchmark_returns.iter().sum::<f64>() / benchmark_returns.len() as f64;

            let mut covariance = 0.0;
            let mut bench_variance = 0.0;

            for k in 0..asset_returns.len() {
                let asset_dev = asset_returns[k] - asset_mean;
                let bench_dev = benchmark_returns[k] - bench_mean;
                covariance += asset_dev * bench_dev;
                bench_variance += bench_dev * bench_dev;
            }

            if bench_variance > 1e-10 {
                result[i] = covariance / bench_variance;
            }
        }

        result
    }
}

impl TechnicalIndicator for ConditionalBeta {
    fn name(&self) -> &str {
        "Conditional Beta"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        // Use close as benchmark proxy when not provided separately
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.close)))
    }
}

/// Tail VaR - Value at Risk focusing on tail events
///
/// Measures extreme loss potential by focusing on the tail of the return distribution
/// beyond normal VaR thresholds.
#[derive(Debug, Clone)]
pub struct TailVaR {
    period: usize,
    var_level: f64,
    tail_level: f64,
}

impl TailVaR {
    pub fn new(period: usize, var_level: f64, tail_level: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if var_level < 0.9 || var_level > 0.999 {
            return Err(IndicatorError::InvalidParameter {
                name: "var_level".to_string(),
                reason: "must be between 0.9 and 0.999".to_string(),
            });
        }
        if tail_level <= var_level || tail_level > 0.999 {
            return Err(IndicatorError::InvalidParameter {
                name: "tail_level".to_string(),
                reason: "must be greater than var_level and at most 0.999".to_string(),
            });
        }
        Ok(Self { period, var_level, tail_level })
    }

    /// Calculate Tail VaR - average of losses beyond the VaR threshold in the tail
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let mut returns: Vec<f64> = Vec::new();
            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            if returns.len() < 10 {
                continue;
            }

            // Sort returns ascending (losses first)
            returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Find VaR and tail thresholds
            let var_idx = ((1.0 - self.var_level) * returns.len() as f64).floor() as usize;
            let tail_idx = ((1.0 - self.tail_level) * returns.len() as f64).floor() as usize;

            // Calculate average loss in the tail region (beyond VaR but including extreme tail)
            let tail_start = tail_idx.min(returns.len().saturating_sub(1));
            let tail_end = var_idx.min(returns.len());

            if tail_start < tail_end {
                let tail_losses: Vec<f64> = returns[tail_start..tail_end].to_vec();
                if !tail_losses.is_empty() {
                    let avg_tail_loss: f64 = tail_losses.iter().sum::<f64>() / tail_losses.len() as f64;
                    result[i] = avg_tail_loss.abs() * 100.0;
                }
            }
        }

        result
    }
}

impl TechnicalIndicator for TailVaR {
    fn name(&self) -> &str {
        "Tail VaR"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Stress Test Metric - Measures performance under stress scenarios
///
/// Evaluates how the asset performs during periods of elevated volatility,
/// simulating stress conditions.
#[derive(Debug, Clone)]
pub struct StressTestMetric {
    period: usize,
    stress_threshold: f64,
}

impl StressTestMetric {
    pub fn new(period: usize, stress_threshold: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if stress_threshold < 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "stress_threshold".to_string(),
                reason: "must be at least 1.0 (multiplier of normal volatility)".to_string(),
            });
        }
        Ok(Self { period, stress_threshold })
    }

    /// Calculate stress test metric - average return during high volatility periods
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns and volatility
            let mut returns: Vec<f64> = Vec::new();
            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            if returns.len() < 10 {
                continue;
            }

            // Calculate mean and standard deviation
            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns.iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>() / returns.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev < 1e-10 {
                continue;
            }

            // Identify stress periods (returns beyond threshold * std_dev)
            let stress_boundary = self.stress_threshold * std_dev;
            let mut stress_returns: Vec<f64> = Vec::new();

            for ret in &returns {
                if ret.abs() > stress_boundary {
                    stress_returns.push(*ret);
                }
            }

            // Calculate stress performance (average return during stress)
            if !stress_returns.is_empty() {
                let stress_avg: f64 = stress_returns.iter().sum::<f64>() / stress_returns.len() as f64;
                result[i] = stress_avg * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for StressTestMetric {
    fn name(&self) -> &str {
        "Stress Test Metric"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Liquidity Adjusted VaR - VaR adjusted for liquidity risk
///
/// Incorporates liquidity considerations into VaR calculation by adjusting
/// for the potential cost of liquidating positions.
#[derive(Debug, Clone)]
pub struct LiquidityAdjustedVaR {
    period: usize,
    confidence: f64,
    liquidity_factor: f64,
}

impl LiquidityAdjustedVaR {
    pub fn new(period: usize, confidence: f64, liquidity_factor: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if confidence < 0.9 || confidence > 0.999 {
            return Err(IndicatorError::InvalidParameter {
                name: "confidence".to_string(),
                reason: "must be between 0.9 and 0.999".to_string(),
            });
        }
        if liquidity_factor < 0.0 || liquidity_factor > 1.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "liquidity_factor".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(Self { period, confidence, liquidity_factor })
    }

    /// Calculate Liquidity Adjusted VaR
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if volume.len() != n {
            return result;
        }

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let mut returns: Vec<f64> = Vec::new();
            let mut avg_volume = 0.0;

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                    avg_volume += volume[j];
                }
            }

            if returns.len() < 10 {
                continue;
            }

            avg_volume /= returns.len() as f64;

            // Sort returns ascending
            returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Calculate historical VaR
            let var_idx = ((1.0 - self.confidence) * returns.len() as f64).floor() as usize;
            let var_idx = var_idx.min(returns.len().saturating_sub(1));
            let base_var = returns[var_idx].abs();

            // Calculate liquidity adjustment based on relative volume
            let current_volume = volume[i];
            let liquidity_ratio = if avg_volume > 1e-10 {
                (current_volume / avg_volume).min(2.0).max(0.5)
            } else {
                1.0
            };

            // Lower liquidity = higher risk adjustment
            let liquidity_adjustment = 1.0 + self.liquidity_factor * (2.0 - liquidity_ratio);

            result[i] = base_var * liquidity_adjustment * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for LiquidityAdjustedVaR {
    fn name(&self) -> &str {
        "Liquidity Adjusted VaR"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Correlation VaR - VaR considering correlation changes
///
/// Calculates VaR while accounting for correlation dynamics with a benchmark,
/// recognizing that correlations increase during market stress.
#[derive(Debug, Clone)]
pub struct CorrelationVaR {
    period: usize,
    confidence: f64,
}

impl CorrelationVaR {
    pub fn new(period: usize, confidence: f64) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if confidence < 0.9 || confidence > 0.999 {
            return Err(IndicatorError::InvalidParameter {
                name: "confidence".to_string(),
                reason: "must be between 0.9 and 0.999".to_string(),
            });
        }
        Ok(Self { period, confidence })
    }

    /// Calculate Correlation VaR - VaR adjusted by correlation with benchmark
    pub fn calculate(&self, close: &[f64], benchmark: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if benchmark.len() != n {
            return result;
        }

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let mut asset_returns: Vec<f64> = Vec::new();
            let mut bench_returns: Vec<f64> = Vec::new();

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 && benchmark[j - 1] > 1e-10 {
                    asset_returns.push(close[j] / close[j - 1] - 1.0);
                    bench_returns.push(benchmark[j] / benchmark[j - 1] - 1.0);
                }
            }

            if asset_returns.len() < 10 {
                continue;
            }

            // Calculate correlation
            let asset_mean: f64 = asset_returns.iter().sum::<f64>() / asset_returns.len() as f64;
            let bench_mean: f64 = bench_returns.iter().sum::<f64>() / bench_returns.len() as f64;

            let mut covariance = 0.0;
            let mut asset_variance = 0.0;
            let mut bench_variance = 0.0;

            for k in 0..asset_returns.len() {
                let asset_dev = asset_returns[k] - asset_mean;
                let bench_dev = bench_returns[k] - bench_mean;
                covariance += asset_dev * bench_dev;
                asset_variance += asset_dev * asset_dev;
                bench_variance += bench_dev * bench_dev;
            }

            let correlation = if asset_variance > 1e-10 && bench_variance > 1e-10 {
                covariance / (asset_variance.sqrt() * bench_variance.sqrt())
            } else {
                0.0
            };

            // Sort returns for VaR calculation
            let mut sorted_returns = asset_returns.clone();
            sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let var_idx = ((1.0 - self.confidence) * sorted_returns.len() as f64).floor() as usize;
            let var_idx = var_idx.min(sorted_returns.len().saturating_sub(1));
            let base_var = sorted_returns[var_idx].abs();

            // Higher correlation = higher systemic risk adjustment
            let correlation_adjustment = 1.0 + 0.5 * correlation.abs();

            result[i] = base_var * correlation_adjustment * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for CorrelationVaR {
    fn name(&self) -> &str {
        "Correlation VaR"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        // Use close as benchmark proxy when not provided separately
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Regime Aware Risk - Risk measure that adapts to market regime
///
/// Dynamically adjusts risk calculations based on detected market regime
/// (low/medium/high volatility environments).
#[derive(Debug, Clone)]
pub struct RegimeAwareRisk {
    period: usize,
    vol_period: usize,
}

impl RegimeAwareRisk {
    pub fn new(period: usize, vol_period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        if vol_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "vol_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        Ok(Self { period, vol_period })
    }

    /// Calculate regime-aware risk metric
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // First, calculate rolling volatility
        let mut volatilities: Vec<f64> = vec![0.0; n];

        for i in self.vol_period..n {
            let start = i.saturating_sub(self.vol_period);
            let mut returns: Vec<f64> = Vec::new();

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            if returns.len() > 1 {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance: f64 = returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                volatilities[i] = variance.sqrt();
            }
        }

        // Calculate long-term average volatility for regime detection
        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Get recent volatilities for regime detection
            let recent_vols: Vec<f64> = volatilities[(start + self.vol_period)..=i]
                .iter()
                .filter(|&&v| v > 0.0)
                .copied()
                .collect();

            if recent_vols.is_empty() {
                continue;
            }

            let avg_vol: f64 = recent_vols.iter().sum::<f64>() / recent_vols.len() as f64;
            let current_vol = volatilities[i];

            if avg_vol < 1e-10 {
                continue;
            }

            // Detect regime: current vol relative to average
            let vol_ratio = current_vol / avg_vol;

            // Calculate base risk (current volatility)
            let base_risk = current_vol;

            // Apply regime multiplier
            let regime_multiplier = if vol_ratio < 0.5 {
                // Low volatility regime - scale up risk for potential breakout
                1.5
            } else if vol_ratio > 2.0 {
                // High volatility regime - already stressed, use full risk
                1.0
            } else if vol_ratio > 1.5 {
                // Elevated regime - scale up moderately
                1.2
            } else {
                // Normal regime
                1.0
            };

            result[i] = base_risk * regime_multiplier * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for RegimeAwareRisk {
    fn name(&self) -> &str {
        "Regime Aware Risk"
    }

    fn min_periods(&self) -> usize {
        self.period + self.vol_period
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Conditional Drawdown at Risk (CDaR) - Expected drawdown conditional on exceeding a threshold
///
/// Measures the expected drawdown given that the drawdown exceeds a specified confidence level.
/// Similar to CVaR but for drawdowns instead of returns, providing a tail risk measure
/// for drawdown-based risk management.
///
/// # Formula
/// CDaR = E[DD | DD > DD_alpha]
/// where DD_alpha is the drawdown at the alpha confidence level
///
/// # Example
/// ```ignore
/// let cdar = ConditionalDrawdownAtRisk::new(50, 0.95).unwrap();
/// let result = cdar.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct ConditionalDrawdownAtRisk {
    period: usize,
    confidence: f64,
}

impl ConditionalDrawdownAtRisk {
    /// Create a new Conditional Drawdown at Risk indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 20)
    /// * `confidence` - Confidence level for the threshold (0.9 to 0.999)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, confidence: f64) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20 for sufficient drawdown samples".to_string(),
            });
        }
        if confidence < 0.9 || confidence > 0.999 {
            return Err(IndicatorError::InvalidParameter {
                name: "confidence".to_string(),
                reason: "must be between 0.9 and 0.999".to_string(),
            });
        }
        Ok(Self { period, confidence })
    }

    /// Calculate Conditional Drawdown at Risk values
    ///
    /// Returns the expected drawdown conditional on exceeding the confidence threshold,
    /// expressed as a percentage (0-100 scale).
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate drawdowns for each point in the window
            let mut peak = close[start];
            let mut drawdowns: Vec<f64> = Vec::new();

            for j in start..=i {
                if close[j] > peak {
                    peak = close[j];
                }
                let dd = (peak - close[j]) / peak;
                drawdowns.push(dd);
            }

            if drawdowns.len() < 10 {
                continue;
            }

            // Sort drawdowns in descending order (worst first)
            drawdowns.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            // Find the threshold drawdown at the confidence level
            let threshold_idx = ((1.0 - self.confidence) * drawdowns.len() as f64).ceil() as usize;
            let threshold_idx = threshold_idx.max(1).min(drawdowns.len());

            // Calculate expected drawdown for drawdowns exceeding the threshold
            let tail_drawdowns: Vec<f64> = drawdowns[..threshold_idx].to_vec();

            if !tail_drawdowns.is_empty() {
                let cdar: f64 = tail_drawdowns.iter().sum::<f64>() / tail_drawdowns.len() as f64;
                result[i] = cdar * 100.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for ConditionalDrawdownAtRisk {
    fn name(&self) -> &str {
        "Conditional Drawdown at Risk"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Upside Downside Ratio - Ratio of upside to downside volatility
///
/// Measures the asymmetry between upside and downside price movements by comparing
/// the volatility of positive returns to the volatility of negative returns.
/// A ratio > 1 indicates more upside volatility, while < 1 indicates more downside risk.
///
/// # Formula
/// UDR = StdDev(positive returns) / StdDev(negative returns)
///
/// # Example
/// ```ignore
/// let udr = UpsideDownsideRatio::new(20).unwrap();
/// let result = udr.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct UpsideDownsideRatio {
    period: usize,
}

impl UpsideDownsideRatio {
    /// Create a new Upside Downside Ratio indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 10)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10 for meaningful volatility calculation".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Upside Downside Ratio values
    ///
    /// Returns the ratio of upside volatility to downside volatility.
    /// Values > 1 indicate more upside potential, values < 1 indicate more downside risk.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Separate positive and negative returns
            let mut upside_returns: Vec<f64> = Vec::new();
            let mut downside_returns: Vec<f64> = Vec::new();

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    let ret = close[j] / close[j - 1] - 1.0;
                    if ret > 0.0 {
                        upside_returns.push(ret);
                    } else if ret < 0.0 {
                        downside_returns.push(ret.abs());
                    }
                }
            }

            // Need sufficient samples on both sides
            if upside_returns.len() < 3 || downside_returns.len() < 3 {
                continue;
            }

            // Calculate upside volatility (standard deviation of positive returns)
            let upside_mean: f64 = upside_returns.iter().sum::<f64>() / upside_returns.len() as f64;
            let upside_variance: f64 = upside_returns
                .iter()
                .map(|r| (r - upside_mean).powi(2))
                .sum::<f64>() / upside_returns.len() as f64;
            let upside_vol = upside_variance.sqrt();

            // Calculate downside volatility (standard deviation of negative returns)
            let downside_mean: f64 = downside_returns.iter().sum::<f64>() / downside_returns.len() as f64;
            let downside_variance: f64 = downside_returns
                .iter()
                .map(|r| (r - downside_mean).powi(2))
                .sum::<f64>() / downside_returns.len() as f64;
            let downside_vol = downside_variance.sqrt();

            // Calculate ratio
            if downside_vol > 1e-10 {
                result[i] = upside_vol / downside_vol;
            }
        }

        result
    }
}

impl TechnicalIndicator for UpsideDownsideRatio {
    fn name(&self) -> &str {
        "Upside Downside Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Risk Adjusted Return Metric - Return adjusted for multiple risk factors
///
/// A comprehensive risk-adjusted return measure that considers volatility,
/// drawdown risk, and downside deviation to provide a more holistic view
/// of risk-adjusted performance than traditional Sharpe ratio.
///
/// # Formula
/// RARM = (Annualized Return - Risk Free Rate) / Composite Risk Score
/// where Composite Risk Score = sqrt(Vol^2 + Downside_Dev^2 + Max_DD^2)
///
/// # Example
/// ```ignore
/// let rarm = RiskAdjustedReturnMetric::new(50, 0.02).unwrap();
/// let result = rarm.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct RiskAdjustedReturnMetric {
    period: usize,
    risk_free_rate: f64,
}

impl RiskAdjustedReturnMetric {
    /// Create a new Risk Adjusted Return Metric indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 20)
    /// * `risk_free_rate` - Annualized risk-free rate (e.g., 0.02 for 2%)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, risk_free_rate: f64) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20 for robust risk calculation".to_string(),
            });
        }
        if risk_free_rate < 0.0 || risk_free_rate > 0.5 {
            return Err(IndicatorError::InvalidParameter {
                name: "risk_free_rate".to_string(),
                reason: "must be between 0.0 and 0.5 (50%)".to_string(),
            });
        }
        Ok(Self { period, risk_free_rate })
    }

    /// Calculate Risk Adjusted Return Metric values
    ///
    /// Returns a composite risk-adjusted return that accounts for volatility,
    /// downside deviation, and maximum drawdown.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let mut returns: Vec<f64> = Vec::new();
            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            if returns.len() < 10 {
                continue;
            }

            // 1. Calculate average return and volatility
            let avg_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns
                .iter()
                .map(|r| (r - avg_return).powi(2))
                .sum::<f64>() / returns.len() as f64;
            let volatility = variance.sqrt() * (252.0_f64).sqrt();

            // 2. Calculate downside deviation
            let downside_returns: Vec<f64> = returns
                .iter()
                .filter(|&&r| r < 0.0)
                .map(|r| r.powi(2))
                .collect();
            let downside_dev = if !downside_returns.is_empty() {
                (downside_returns.iter().sum::<f64>() / downside_returns.len() as f64).sqrt()
                    * (252.0_f64).sqrt()
            } else {
                0.0
            };

            // 3. Calculate maximum drawdown
            let mut peak = close[start];
            let mut max_dd = 0.0;
            for j in start..=i {
                if close[j] > peak {
                    peak = close[j];
                }
                let dd = (peak - close[j]) / peak;
                if dd > max_dd {
                    max_dd = dd;
                }
            }

            // Annualize return
            let annualized_return = avg_return * 252.0;
            let excess_return = annualized_return - self.risk_free_rate;

            // Composite risk score (geometric combination of risks)
            let composite_risk = (volatility.powi(2) + downside_dev.powi(2) + max_dd.powi(2)).sqrt();

            if composite_risk > 1e-10 {
                result[i] = excess_return / composite_risk;
            }
        }

        result
    }
}

impl TechnicalIndicator for RiskAdjustedReturnMetric {
    fn name(&self) -> &str {
        "Risk Adjusted Return Metric"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Maximum Drawdown Duration - Duration of the maximum drawdown period
///
/// Measures how long the asset remains in its maximum drawdown state,
/// providing insight into the time required to recover from the worst decline.
/// This is crucial for understanding capital lock-up risk.
///
/// # Formula
/// MaxDDDuration = Number of periods from peak to trough during max drawdown
///
/// # Example
/// ```ignore
/// let mdd_dur = MaxDrawdownDuration::new(50).unwrap();
/// let result = mdd_dur.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct MaxDrawdownDuration {
    period: usize,
}

impl MaxDrawdownDuration {
    /// Create a new Maximum Drawdown Duration indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 10)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Maximum Drawdown Duration values
    ///
    /// Returns the number of periods spent in the maximum drawdown within the window.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Track drawdown periods and identify the maximum one
            let mut peak_idx = start;
            let mut max_dd = 0.0;
            let mut max_dd_end = start;

            // First pass: find peak and calculate drawdowns
            let mut current_peak = close[start];
            let mut current_peak_idx = start;

            for j in start..=i {
                if close[j] > current_peak {
                    current_peak = close[j];
                    current_peak_idx = j;
                }

                let dd = (current_peak - close[j]) / current_peak;
                if dd > max_dd {
                    max_dd = dd;
                    peak_idx = current_peak_idx;
                    max_dd_end = j;
                }
            }

            // Duration is from peak to trough of maximum drawdown
            let duration = max_dd_end - peak_idx;
            result[i] = duration as f64;
        }

        result
    }

    /// Calculate the total time underwater (from peak to recovery)
    ///
    /// Returns the total number of periods from peak until price returns to that peak level.
    pub fn calculate_underwater(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        let mut peak = close[0];
        let mut underwater_duration = 0;

        for i in 0..n {
            if close[i] >= peak {
                peak = close[i];
                underwater_duration = 0;
            } else {
                underwater_duration += 1;
            }
            result[i] = underwater_duration as f64;
        }

        result
    }
}

impl TechnicalIndicator for MaxDrawdownDuration {
    fn name(&self) -> &str {
        "Maximum Drawdown Duration"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Drawdown Recovery Factor - Measures the ability to recover from drawdowns
///
/// Calculates the ratio of net profit to maximum drawdown, indicating how well
/// an asset or strategy recovers from its worst losses. Higher values indicate
/// better recovery characteristics.
///
/// # Formula
/// DRF = Net Profit / Maximum Drawdown
///
/// # Example
/// ```ignore
/// let drf = DrawdownRecoveryFactor::new(50).unwrap();
/// let result = drf.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct DrawdownRecoveryFactor {
    period: usize,
}

impl DrawdownRecoveryFactor {
    /// Create a new Drawdown Recovery Factor indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 20)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20 for meaningful recovery analysis".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Drawdown Recovery Factor values
    ///
    /// Returns the ratio of cumulative return to maximum drawdown within the window.
    /// Positive values indicate net profit relative to worst drawdown.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            if close[start] < 1e-10 {
                continue;
            }

            // Calculate net profit (total return over period)
            let net_profit = (close[i] / close[start] - 1.0) * 100.0;

            // Calculate maximum drawdown
            let mut peak = close[start];
            let mut max_dd = 0.0;

            for j in start..=i {
                if close[j] > peak {
                    peak = close[j];
                }
                let dd = (peak - close[j]) / peak;
                if dd > max_dd {
                    max_dd = dd;
                }
            }

            // Recovery factor = net profit / max drawdown
            let max_dd_pct = max_dd * 100.0;
            if max_dd_pct > 1e-10 {
                result[i] = net_profit / max_dd_pct;
            } else if net_profit > 0.0 {
                // No drawdown but positive return - cap at high value
                result[i] = 10.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for DrawdownRecoveryFactor {
    fn name(&self) -> &str {
        "Drawdown Recovery Factor"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Risk Regime Indicator - Detects risk regime changes in the market
///
/// Identifies different risk regimes (low, normal, elevated, high) based on
/// volatility clustering, correlation changes, and tail risk metrics.
/// Useful for adaptive risk management and dynamic position sizing.
///
/// # Regime Levels
/// * 0 = Low risk regime
/// * 1 = Normal risk regime
/// * 2 = Elevated risk regime
/// * 3 = High/Crisis risk regime
///
/// # Example
/// ```ignore
/// let rri = RiskRegimeIndicator::new(50, 20).unwrap();
/// let result = rri.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct RiskRegimeIndicator {
    period: usize,
    vol_period: usize,
}

impl RiskRegimeIndicator {
    /// Create a new Risk Regime Indicator
    ///
    /// # Arguments
    /// * `period` - Long-term window for regime baseline (minimum 30)
    /// * `vol_period` - Short-term window for current volatility (minimum 5)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, vol_period: usize) -> Result<Self> {
        if period < 30 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 30 for robust regime detection".to_string(),
            });
        }
        if vol_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "vol_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if vol_period >= period {
            return Err(IndicatorError::InvalidParameter {
                name: "vol_period".to_string(),
                reason: "must be less than period".to_string(),
            });
        }
        Ok(Self { period, vol_period })
    }

    /// Calculate Risk Regime Indicator values
    ///
    /// Returns a regime score from 0-3 indicating the current risk environment.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Pre-calculate rolling volatilities
        let mut short_vols: Vec<f64> = vec![0.0; n];
        let mut long_vols: Vec<f64> = vec![0.0; n];

        // Calculate short-term volatility
        for i in self.vol_period..n {
            let start = i.saturating_sub(self.vol_period);
            let mut returns: Vec<f64> = Vec::new();

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            if returns.len() > 1 {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance: f64 = returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                short_vols[i] = variance.sqrt();
            }
        }

        // Calculate long-term volatility
        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let mut returns: Vec<f64> = Vec::new();

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            if returns.len() > 1 {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance: f64 = returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                long_vols[i] = variance.sqrt();
            }
        }

        // Determine regime based on volatility ratio and other factors
        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            if long_vols[i] < 1e-10 {
                continue;
            }

            // 1. Volatility ratio (current vs historical)
            let vol_ratio = short_vols[i] / long_vols[i];

            // 2. Calculate recent drawdown
            let mut peak = close[start];
            let mut current_dd = 0.0;
            for j in start..=i {
                if close[j] > peak {
                    peak = close[j];
                }
                let dd = (peak - close[j]) / peak;
                if dd > current_dd {
                    current_dd = dd;
                }
            }

            // 3. Calculate return momentum (negative momentum adds to risk)
            let short_start = i.saturating_sub(self.vol_period);
            let short_return = if close[short_start] > 1e-10 {
                close[i] / close[short_start] - 1.0
            } else {
                0.0
            };

            // 4. Combine factors into regime score
            let mut regime_score = 0.0;

            // Volatility contribution (0-1.5)
            if vol_ratio < 0.7 {
                regime_score += 0.0;
            } else if vol_ratio < 1.0 {
                regime_score += 0.5;
            } else if vol_ratio < 1.5 {
                regime_score += 1.0;
            } else {
                regime_score += 1.5;
            }

            // Drawdown contribution (0-1)
            if current_dd > 0.15 {
                regime_score += 1.0;
            } else if current_dd > 0.10 {
                regime_score += 0.7;
            } else if current_dd > 0.05 {
                regime_score += 0.3;
            }

            // Momentum contribution (0-0.5)
            if short_return < -0.05 {
                regime_score += 0.5;
            } else if short_return < -0.02 {
                regime_score += 0.25;
            }

            // Convert to discrete regime (0-3)
            result[i] = if regime_score < 0.5 {
                0.0 // Low risk
            } else if regime_score < 1.2 {
                1.0 // Normal risk
            } else if regime_score < 2.0 {
                2.0 // Elevated risk
            } else {
                3.0 // High/Crisis risk
            };
        }

        result
    }

    /// Calculate continuous regime score (not discretized)
    ///
    /// Returns a continuous score representing the risk regime intensity.
    pub fn calculate_continuous(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate short-term volatility
            let short_start = i.saturating_sub(self.vol_period);
            let mut short_returns: Vec<f64> = Vec::new();
            for j in (short_start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    short_returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            let short_vol = if short_returns.len() > 1 {
                let mean: f64 = short_returns.iter().sum::<f64>() / short_returns.len() as f64;
                let variance: f64 = short_returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / short_returns.len() as f64;
                variance.sqrt()
            } else {
                0.0
            };

            // Calculate long-term volatility
            let mut long_returns: Vec<f64> = Vec::new();
            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    long_returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            let long_vol = if long_returns.len() > 1 {
                let mean: f64 = long_returns.iter().sum::<f64>() / long_returns.len() as f64;
                let variance: f64 = long_returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / long_returns.len() as f64;
                variance.sqrt()
            } else {
                1e-10
            };

            if long_vol > 1e-10 {
                result[i] = short_vol / long_vol;
            }
        }

        result
    }
}

impl TechnicalIndicator for RiskRegimeIndicator {
    fn name(&self) -> &str {
        "Risk Regime Indicator"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

// ============================================================
// NEW RISK INDICATORS (6 total)
// ============================================================

/// Sortino Ratio (Advanced) - Downside risk-adjusted return ratio
///
/// Measures the risk-adjusted return of an investment, penalizing only
/// downside volatility (returns below a target threshold). Unlike the
/// standard Sharpe ratio which penalizes all volatility equally, the
/// Sortino ratio recognizes that upside volatility is beneficial.
///
/// # Formula
/// Sortino = (R_p - R_f) / Downside_Deviation
///
/// where:
/// - R_p = Average portfolio return
/// - R_f = Target return (often risk-free rate)
/// - Downside_Deviation = sqrt(mean(min(0, R - R_f)^2))
///
/// # Interpretation
/// - Higher values indicate better risk-adjusted performance
/// - Positive values suggest returns exceed the target on a risk-adjusted basis
/// - More appropriate than Sharpe for asymmetric return distributions
///
/// # Example
/// ```ignore
/// let sortino = SortinoRatioAdvanced::new(50, 0.0, 252.0).unwrap();
/// let result = sortino.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct SortinoRatioAdvanced {
    /// Rolling window period for calculation
    period: usize,
    /// Target return (Minimum Acceptable Return), often risk-free rate (annualized)
    target_return: f64,
    /// Annualization factor (252 for daily, 52 for weekly, 12 for monthly)
    annualization_factor: f64,
}

impl SortinoRatioAdvanced {
    /// Create a new Sortino Ratio (Advanced) indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 20)
    /// * `target_return` - Annualized target return (e.g., 0.02 for 2%)
    /// * `annualization_factor` - Trading periods per year (252 for daily)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, target_return: f64, annualization_factor: f64) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20 for robust downside deviation calculation".to_string(),
            });
        }
        if annualization_factor <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "annualization_factor".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            period,
            target_return,
            annualization_factor,
        })
    }

    /// Calculate Sortino Ratio values
    ///
    /// Returns the annualized Sortino ratio for each point in the series.
    /// Higher values indicate better downside risk-adjusted performance.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // Convert annualized target to per-period target
        let per_period_target = self.target_return / self.annualization_factor;

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let mut returns: Vec<f64> = Vec::new();
            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            if returns.len() < 10 {
                continue;
            }

            // Calculate mean return
            let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

            // Calculate downside deviation (only returns below target)
            let downside_squared_sum: f64 = returns
                .iter()
                .map(|&r| {
                    let excess = r - per_period_target;
                    if excess < 0.0 {
                        excess * excess
                    } else {
                        0.0
                    }
                })
                .sum();

            let downside_deviation = (downside_squared_sum / returns.len() as f64).sqrt();

            // Calculate annualized Sortino ratio
            if downside_deviation > 1e-10 {
                let annualized_excess_return = (mean_return - per_period_target) * self.annualization_factor;
                let annualized_downside_dev = downside_deviation * self.annualization_factor.sqrt();
                result[i] = annualized_excess_return / annualized_downside_dev;
            }
        }

        result
    }
}

impl TechnicalIndicator for SortinoRatioAdvanced {
    fn name(&self) -> &str {
        "Sortino Ratio Advanced"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Calmar Ratio (Advanced) - Return to maximum drawdown ratio
///
/// Measures the risk-adjusted return by comparing annualized return to
/// the maximum drawdown. Originally designed for hedge fund evaluation,
/// it provides insight into whether returns are worth the drawdown risk.
///
/// # Formula
/// Calmar = Annualized_Return / |Maximum_Drawdown|
///
/// # Interpretation
/// - Higher values indicate better risk-adjusted performance
/// - Values > 1 suggest annualized return exceeds the worst drawdown
/// - Typically calculated over 3-year periods for hedge funds
/// - More conservative than Sharpe as it uses worst-case drawdown
///
/// # Example
/// ```ignore
/// let calmar = CalmarRatioAdvanced::new(50, 252.0).unwrap();
/// let result = calmar.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct CalmarRatioAdvanced {
    /// Rolling window period for calculation
    period: usize,
    /// Annualization factor (252 for daily, 52 for weekly, 12 for monthly)
    annualization_factor: f64,
}

impl CalmarRatioAdvanced {
    /// Create a new Calmar Ratio (Advanced) indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 20)
    /// * `annualization_factor` - Trading periods per year (252 for daily)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, annualization_factor: f64) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20 for meaningful drawdown measurement".to_string(),
            });
        }
        if annualization_factor <= 0.0 {
            return Err(IndicatorError::InvalidParameter {
                name: "annualization_factor".to_string(),
                reason: "must be positive".to_string(),
            });
        }
        Ok(Self {
            period,
            annualization_factor,
        })
    }

    /// Calculate Calmar Ratio values
    ///
    /// Returns the Calmar ratio (annualized return / max drawdown) for each point.
    /// Higher values indicate better return relative to worst drawdown.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            if close[start] < 1e-10 {
                continue;
            }

            // Calculate total return over the period
            let total_return = close[i] / close[start] - 1.0;

            // Annualize the return
            let periods = (i - start) as f64;
            let annualized_return = (1.0 + total_return).powf(self.annualization_factor / periods) - 1.0;

            // Calculate maximum drawdown
            let mut peak = close[start];
            let mut max_dd = 0.0;

            for j in start..=i {
                if close[j] > peak {
                    peak = close[j];
                }
                let dd = (peak - close[j]) / peak;
                if dd > max_dd {
                    max_dd = dd;
                }
            }

            // Calculate Calmar ratio
            if max_dd > 1e-10 {
                result[i] = annualized_return / max_dd;
            } else if annualized_return > 0.0 {
                // No drawdown but positive return - cap at high value
                result[i] = 10.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for CalmarRatioAdvanced {
    fn name(&self) -> &str {
        "Calmar Ratio Advanced"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Omega Ratio (Advanced) - Probability weighted ratio of gains vs losses
///
/// A comprehensive performance measure that considers the entire return
/// distribution, not just mean and variance. It calculates the probability-
/// weighted ratio of gains (above threshold) to losses (below threshold).
///
/// # Formula
/// Omega = Sum(max(R - threshold, 0)) / Sum(max(threshold - R, 0))
///
/// This is equivalent to:
/// Omega = (1 + (mean - threshold) / LPM_1) where LPM_1 is the first lower partial moment
///
/// # Interpretation
/// - Omega > 1: More probability-weighted gains than losses
/// - Omega = 1: Equal probability-weighted gains and losses
/// - Omega < 1: More probability-weighted losses than gains
/// - No arbitrary assumptions about return distribution shape
///
/// # Example
/// ```ignore
/// let omega = OmegaRatioAdvanced::new(50, 0.0).unwrap();
/// let result = omega.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct OmegaRatioAdvanced {
    /// Rolling window period for calculation
    period: usize,
    /// Threshold return (minimum acceptable return per period)
    threshold: f64,
}

impl OmegaRatioAdvanced {
    /// Create a new Omega Ratio (Advanced) indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 20)
    /// * `threshold` - Per-period threshold return (e.g., 0.0 for zero, or risk-free rate/252)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, threshold: f64) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20 for meaningful probability weighting".to_string(),
            });
        }
        Ok(Self { period, threshold })
    }

    /// Calculate Omega Ratio values
    ///
    /// Returns the Omega ratio for each point in the series.
    /// Values > 1 indicate more gains than losses relative to threshold.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let mut returns: Vec<f64> = Vec::new();
            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            if returns.len() < 10 {
                continue;
            }

            // Calculate gains (returns above threshold)
            let gains_sum: f64 = returns
                .iter()
                .map(|&r| (r - self.threshold).max(0.0))
                .sum();

            // Calculate losses (returns below threshold)
            let losses_sum: f64 = returns
                .iter()
                .map(|&r| (self.threshold - r).max(0.0))
                .sum();

            // Calculate Omega ratio
            if losses_sum > 1e-10 {
                result[i] = gains_sum / losses_sum;
            } else if gains_sum > 0.0 {
                // All gains, no losses - cap at high value
                result[i] = 10.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for OmegaRatioAdvanced {
    fn name(&self) -> &str {
        "Omega Ratio Advanced"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Pain Ratio - Average drawdown-adjusted return measure
///
/// Measures the return per unit of average drawdown (pain). Unlike the
/// Calmar ratio which uses maximum drawdown, the Pain ratio considers
/// the average depth and duration of drawdowns, providing a more
/// comprehensive view of the typical pain experienced.
///
/// # Formula
/// Pain_Ratio = (Annualized_Return - Risk_Free_Rate) / Pain_Index
///
/// where Pain_Index = Average(Drawdown percentages over the period)
///
/// # Interpretation
/// - Higher values indicate better return per unit of average pain
/// - More stable than Calmar as it's not dominated by a single worst drawdown
/// - Better for comparing strategies with different drawdown patterns
///
/// # Example
/// ```ignore
/// let pain_ratio = PainRatio::new(50, 0.02).unwrap();
/// let result = pain_ratio.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct PainRatio {
    /// Rolling window period for calculation
    period: usize,
    /// Annualized risk-free rate
    risk_free_rate: f64,
}

impl PainRatio {
    /// Create a new Pain Ratio indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 20)
    /// * `risk_free_rate` - Annualized risk-free rate (e.g., 0.02 for 2%)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, risk_free_rate: f64) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20 for meaningful pain calculation".to_string(),
            });
        }
        if risk_free_rate < 0.0 || risk_free_rate > 0.5 {
            return Err(IndicatorError::InvalidParameter {
                name: "risk_free_rate".to_string(),
                reason: "must be between 0.0 and 0.5 (50%)".to_string(),
            });
        }
        Ok(Self {
            period,
            risk_free_rate,
        })
    }

    /// Calculate Pain Ratio values
    ///
    /// Returns the excess return per unit of average drawdown for each point.
    /// Higher values indicate better risk-adjusted performance.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            if close[start] < 1e-10 {
                continue;
            }

            // Calculate returns
            let mut returns: Vec<f64> = Vec::new();
            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            if returns.is_empty() {
                continue;
            }

            // Calculate annualized return
            let avg_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let annualized_return = avg_return * 252.0;
            let excess_return = annualized_return - self.risk_free_rate;

            // Calculate Pain Index (average drawdown)
            let mut peak = close[start];
            let mut dd_sum = 0.0;
            let mut dd_count = 0;

            for j in start..=i {
                if close[j] > peak {
                    peak = close[j];
                }
                let dd = (peak - close[j]) / peak;
                dd_sum += dd;
                dd_count += 1;
            }

            let pain_index = if dd_count > 0 {
                dd_sum / dd_count as f64
            } else {
                0.0
            };

            // Calculate Pain Ratio
            if pain_index > 1e-10 {
                result[i] = excess_return / pain_index;
            } else if excess_return > 0.0 {
                // No pain but positive return - cap at high value
                result[i] = 10.0;
            }
        }

        result
    }
}

impl TechnicalIndicator for PainRatio {
    fn name(&self) -> &str {
        "Pain Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Ulcer Index - Quadratic mean of percentage drawdowns
///
/// Developed by Peter Martin in 1987, the Ulcer Index measures downside
/// volatility using the quadratic mean (root mean square) of percentage
/// drawdowns from recent peaks. It focuses on the depth and duration of
/// drawdowns rather than standard deviation, making it particularly
/// relevant for risk-averse investors.
///
/// # Formula
/// UI = sqrt(mean(Drawdown_Pct^2))
///
/// where Drawdown_Pct = 100 * (Peak - Price) / Peak
///
/// # Interpretation
/// - Lower values indicate less severe drawdowns (lower risk)
/// - Zero means no drawdowns (price always at or above previous peak)
/// - Expressed as a percentage
/// - Can be used to calculate Ulcer Performance Index (return / UI)
///
/// # Example
/// ```ignore
/// let ulcer = UlcerIndex::new(14).unwrap();
/// let result = ulcer.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct UlcerIndex {
    /// Rolling window period for calculation
    period: usize,
}

impl UlcerIndex {
    /// Create a new Ulcer Index indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 10)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize) -> Result<Self> {
        if period < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 10 for meaningful ulcer calculation".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Ulcer Index values
    ///
    /// Returns the quadratic mean of percentage drawdowns for each point.
    /// Lower values indicate less severe drawdowns.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Find the peak in the lookback period
            let mut peak = close[start];
            let mut squared_dd_sum = 0.0;
            let mut count = 0;

            for j in start..=i {
                if close[j] > peak {
                    peak = close[j];
                }
                // Calculate percentage drawdown
                let pct_dd = 100.0 * (peak - close[j]) / peak;
                squared_dd_sum += pct_dd * pct_dd;
                count += 1;
            }

            // Calculate Ulcer Index as quadratic mean (RMS) of drawdowns
            if count > 0 {
                result[i] = (squared_dd_sum / count as f64).sqrt();
            }
        }

        result
    }
}

impl TechnicalIndicator for UlcerIndex {
    fn name(&self) -> &str {
        "Ulcer Index"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Kelly Fraction - Optimal betting/position sizing fraction
///
/// Based on the Kelly Criterion, this indicator calculates the optimal
/// fraction of capital to risk on each trade/position to maximize long-term
/// growth. It balances the trade-off between risk and reward based on
/// historical win rates and payoff ratios.
///
/// # Formula
/// Kelly = W - (1-W)/R
///
/// where:
/// - W = Win rate (probability of winning)
/// - R = Win/Loss ratio (average win size / average loss size)
///
/// Alternative form: Kelly = (p*b - q) / b
/// where p = win probability, q = loss probability (1-p), b = odds (win/loss ratio)
///
/// # Interpretation
/// - Positive values: Should take a long position
/// - Negative values: Should take a short position or stay out
/// - Value represents optimal fraction of capital to risk
/// - In practice, traders often use "half-Kelly" or "quarter-Kelly" for safety
/// - Values > 1 are theoretically possible but suggest extreme risk
///
/// # Example
/// ```ignore
/// let kelly = KellyFraction::new(50).unwrap();
/// let result = kelly.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct KellyFraction {
    /// Rolling window period for calculation
    period: usize,
}

impl KellyFraction {
    /// Create a new Kelly Fraction indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 20)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20 for statistically meaningful Kelly calculation".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Kelly Fraction values
    ///
    /// Returns the optimal fraction of capital to risk based on historical
    /// win rate and payoff ratio. Positive values suggest long positions.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns and classify as wins/losses
            let mut wins: Vec<f64> = Vec::new();
            let mut losses: Vec<f64> = Vec::new();

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    let ret = close[j] / close[j - 1] - 1.0;
                    if ret > 0.0 {
                        wins.push(ret);
                    } else if ret < 0.0 {
                        losses.push(ret.abs());
                    }
                }
            }

            let total_trades = wins.len() + losses.len();
            if total_trades < 10 || losses.is_empty() {
                continue;
            }

            // Calculate win rate
            let win_rate = wins.len() as f64 / total_trades as f64;

            // Calculate average win and average loss
            let avg_win: f64 = if !wins.is_empty() {
                wins.iter().sum::<f64>() / wins.len() as f64
            } else {
                0.0
            };

            let avg_loss: f64 = losses.iter().sum::<f64>() / losses.len() as f64;

            // Calculate win/loss ratio
            if avg_loss < 1e-10 {
                continue;
            }
            let win_loss_ratio = avg_win / avg_loss;

            // Calculate Kelly fraction: W - (1-W)/R
            let kelly = win_rate - (1.0 - win_rate) / win_loss_ratio;

            // Clamp to reasonable range (-1 to 1 for practical purposes)
            result[i] = kelly.clamp(-1.0, 1.0);
        }

        result
    }

    /// Calculate Kelly Fraction with custom edge calculation
    ///
    /// Uses a more sophisticated edge calculation that considers
    /// the magnitude of wins and losses, not just the count.
    pub fn calculate_weighted(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let mut returns: Vec<f64> = Vec::new();
            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            if returns.len() < 10 {
                continue;
            }

            // Calculate expected value and variance
            let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / returns.len() as f64;

            // Kelly formula for continuous returns: f = mu / sigma^2
            // This is the optimal fraction when returns are normally distributed
            if variance > 1e-10 {
                let kelly = mean_return / variance;
                // Clamp to reasonable range
                result[i] = kelly.clamp(-2.0, 2.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for KellyFraction {
    fn name(&self) -> &str {
        "Kelly Fraction"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

// ============================================================
// 6 NEWEST RISK INDICATORS
// ============================================================

/// Tail Risk Ratio - Ratio of tail loss to typical loss
///
/// Measures the severity of tail risk by comparing the average loss
/// in the tail of the distribution to the average loss overall.
/// A higher ratio indicates fatter tails and more extreme risk events.
///
/// # Formula
/// TRR = Average(Losses beyond VaR threshold) / Average(All losses)
///
/// # Interpretation
/// - Values close to 1 indicate normal distribution-like behavior
/// - Values > 1 indicate fat tails with more severe tail events
/// - Higher values suggest more tail risk than expected
/// - Useful for detecting non-normal return distributions
///
/// # Example
/// ```ignore
/// let trr = TailRiskRatio::new(50, 0.95).unwrap();
/// let result = trr.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct TailRiskRatio {
    /// Rolling window period for calculation
    period: usize,
    /// Confidence level for tail threshold (e.g., 0.95 for 5% tail)
    confidence: f64,
}

impl TailRiskRatio {
    /// Create a new Tail Risk Ratio indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 20)
    /// * `confidence` - Confidence level for tail threshold (0.9 to 0.999)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, confidence: f64) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20 for meaningful tail analysis".to_string(),
            });
        }
        if confidence < 0.9 || confidence > 0.999 {
            return Err(IndicatorError::InvalidParameter {
                name: "confidence".to_string(),
                reason: "must be between 0.9 and 0.999".to_string(),
            });
        }
        Ok(Self { period, confidence })
    }

    /// Calculate Tail Risk Ratio values
    ///
    /// Returns the ratio of tail losses to average losses for each point.
    /// Higher values indicate more severe tail risk.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let mut returns: Vec<f64> = Vec::new();
            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            if returns.len() < 10 {
                continue;
            }

            // Separate losses (negative returns)
            let mut losses: Vec<f64> = returns
                .iter()
                .filter(|&&r| r < 0.0)
                .map(|r| r.abs())
                .collect();

            if losses.len() < 5 {
                continue;
            }

            // Calculate average loss
            let avg_loss: f64 = losses.iter().sum::<f64>() / losses.len() as f64;

            // Sort losses to find tail threshold
            losses.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

            // Find tail losses (beyond confidence threshold)
            let tail_idx = ((1.0 - self.confidence) * losses.len() as f64).ceil() as usize;
            let tail_idx = tail_idx.max(1).min(losses.len());

            // Calculate average tail loss
            let tail_losses: Vec<f64> = losses[..tail_idx].to_vec();
            let avg_tail_loss: f64 = if !tail_losses.is_empty() {
                tail_losses.iter().sum::<f64>() / tail_losses.len() as f64
            } else {
                avg_loss
            };

            // Calculate ratio
            if avg_loss > 1e-10 {
                result[i] = avg_tail_loss / avg_loss;
            }
        }

        result
    }
}

impl TechnicalIndicator for TailRiskRatio {
    fn name(&self) -> &str {
        "Tail Risk Ratio"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// VaR Breach Rate - Frequency of Value at Risk breaches
///
/// Measures how often actual losses exceed the predicted VaR threshold.
/// This is a key metric for VaR model validation and risk management
/// effectiveness assessment.
///
/// # Formula
/// Breach Rate = (Number of VaR breaches / Total observations) * 100
///
/// # Interpretation
/// - At 95% confidence, expected breach rate is ~5%
/// - Breach rate > expected suggests model underestimates risk
/// - Breach rate < expected suggests model overestimates risk
/// - Used for regulatory backtesting requirements
///
/// # Example
/// ```ignore
/// let vbr = VaRBreachRate::new(50, 0.95).unwrap();
/// let result = vbr.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct VaRBreachRate {
    /// Rolling window period for calculation
    period: usize,
    /// Confidence level for VaR (e.g., 0.95 for 95% VaR)
    confidence: f64,
}

impl VaRBreachRate {
    /// Create a new VaR Breach Rate indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 20)
    /// * `confidence` - Confidence level for VaR (0.9 to 0.999)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, confidence: f64) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20 for meaningful breach analysis".to_string(),
            });
        }
        if confidence < 0.9 || confidence > 0.999 {
            return Err(IndicatorError::InvalidParameter {
                name: "confidence".to_string(),
                reason: "must be between 0.9 and 0.999".to_string(),
            });
        }
        Ok(Self { period, confidence })
    }

    /// Calculate VaR Breach Rate values
    ///
    /// Returns the percentage of observations that breached VaR (0-100 scale).
    /// Compare to expected rate: (1 - confidence) * 100.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate returns
            let mut returns: Vec<f64> = Vec::new();
            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            if returns.len() < 10 {
                continue;
            }

            // Calculate historical VaR
            let mut sorted_returns = returns.clone();
            sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let var_idx = ((1.0 - self.confidence) * sorted_returns.len() as f64).floor() as usize;
            let var_idx = var_idx.min(sorted_returns.len().saturating_sub(1));
            let var_threshold = sorted_returns[var_idx];

            // Count breaches (returns worse than VaR)
            let breach_count = returns.iter().filter(|&&r| r < var_threshold).count();

            // Calculate breach rate as percentage
            result[i] = (breach_count as f64 / returns.len() as f64) * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for VaRBreachRate {
    fn name(&self) -> &str {
        "VaR Breach Rate"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Volatility of Volatility - Measures volatility clustering and instability
///
/// Calculates the standard deviation of rolling volatility values,
/// indicating how stable or unstable volatility is over time.
/// High values suggest volatility clustering and regime changes.
///
/// # Formula
/// VoV = StdDev(Rolling_Volatility over outer_period)
///
/// # Interpretation
/// - Low VoV indicates stable, predictable volatility
/// - High VoV indicates volatility clustering and regime shifts
/// - Useful for options pricing and dynamic hedging strategies
/// - Important for GARCH-type model selection
///
/// # Example
/// ```ignore
/// let vov = VolatilityOfVolatility::new(50, 10).unwrap();
/// let result = vov.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct VolatilityOfVolatility {
    /// Outer window for calculating VoV
    period: usize,
    /// Inner window for calculating base volatility
    vol_period: usize,
}

impl VolatilityOfVolatility {
    /// Create a new Volatility of Volatility indicator
    ///
    /// # Arguments
    /// * `period` - Outer window for VoV calculation (minimum 20)
    /// * `vol_period` - Inner window for volatility calculation (minimum 5, < period)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize, vol_period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if vol_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "vol_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if vol_period >= period {
            return Err(IndicatorError::InvalidParameter {
                name: "vol_period".to_string(),
                reason: "must be less than period".to_string(),
            });
        }
        Ok(Self { period, vol_period })
    }

    /// Calculate Volatility of Volatility values
    ///
    /// Returns the standard deviation of rolling volatility values,
    /// annualized as a percentage.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        // First, calculate rolling volatility
        let mut volatilities: Vec<f64> = vec![0.0; n];

        for i in self.vol_period..n {
            let start = i.saturating_sub(self.vol_period);
            let mut returns: Vec<f64> = Vec::new();

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    returns.push(close[j] / close[j - 1] - 1.0);
                }
            }

            if returns.len() > 1 {
                let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
                let variance: f64 = returns
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64;
                volatilities[i] = variance.sqrt();
            }
        }

        // Calculate volatility of volatility
        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Collect valid volatilities in the window
            let vols: Vec<f64> = volatilities[(start + self.vol_period)..=i]
                .iter()
                .filter(|&&v| v > 0.0)
                .copied()
                .collect();

            if vols.len() < 5 {
                continue;
            }

            // Calculate mean volatility
            let mean_vol: f64 = vols.iter().sum::<f64>() / vols.len() as f64;

            // Calculate standard deviation of volatility
            let vol_variance: f64 = vols
                .iter()
                .map(|v| (v - mean_vol).powi(2))
                .sum::<f64>() / vols.len() as f64;

            // Annualize and convert to percentage
            result[i] = vol_variance.sqrt() * (252.0_f64).sqrt() * 100.0;
        }

        result
    }
}

impl TechnicalIndicator for VolatilityOfVolatility {
    fn name(&self) -> &str {
        "Volatility of Volatility"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Asymmetric Beta - Separate beta for up and down markets
///
/// Calculates different beta values for positive and negative benchmark returns,
/// capturing the asymmetric response of an asset to market movements.
/// Returns the ratio of downside beta to upside beta.
///
/// # Formula
/// Asymmetric Beta Ratio = Downside_Beta / Upside_Beta
///
/// where:
/// - Upside_Beta = Cov(R_asset, R_bench | R_bench > 0) / Var(R_bench | R_bench > 0)
/// - Downside_Beta = Cov(R_asset, R_bench | R_bench < 0) / Var(R_bench | R_bench < 0)
///
/// # Interpretation
/// - Ratio = 1: Symmetric response to market movements
/// - Ratio > 1: More sensitive to down markets (higher risk)
/// - Ratio < 1: More sensitive to up markets (defensive asset)
/// - Important for tail risk hedging and portfolio construction
///
/// # Example
/// ```ignore
/// let ab = AsymmetricBeta::new(50).unwrap();
/// let result = ab.calculate(&close_prices, &benchmark_prices);
/// ```
#[derive(Debug, Clone)]
pub struct AsymmetricBeta {
    /// Rolling window period for calculation
    period: usize,
}

impl AsymmetricBeta {
    /// Create a new Asymmetric Beta indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 20)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20 for robust beta calculation".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Asymmetric Beta ratio values
    ///
    /// Returns the ratio of downside beta to upside beta.
    /// Values > 1 indicate higher sensitivity to down markets.
    pub fn calculate(&self, close: &[f64], benchmark: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if benchmark.len() != n {
            return result;
        }

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Separate up and down market returns
            let mut up_asset: Vec<f64> = Vec::new();
            let mut up_bench: Vec<f64> = Vec::new();
            let mut down_asset: Vec<f64> = Vec::new();
            let mut down_bench: Vec<f64> = Vec::new();

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 && benchmark[j - 1] > 1e-10 {
                    let asset_ret = close[j] / close[j - 1] - 1.0;
                    let bench_ret = benchmark[j] / benchmark[j - 1] - 1.0;

                    if bench_ret > 0.0 {
                        up_asset.push(asset_ret);
                        up_bench.push(bench_ret);
                    } else if bench_ret < 0.0 {
                        down_asset.push(asset_ret);
                        down_bench.push(bench_ret);
                    }
                }
            }

            // Need sufficient samples in both directions
            if up_asset.len() < 5 || down_asset.len() < 5 {
                continue;
            }

            // Calculate upside beta
            let up_asset_mean: f64 = up_asset.iter().sum::<f64>() / up_asset.len() as f64;
            let up_bench_mean: f64 = up_bench.iter().sum::<f64>() / up_bench.len() as f64;

            let mut up_cov = 0.0;
            let mut up_var = 0.0;
            for k in 0..up_asset.len() {
                let asset_dev = up_asset[k] - up_asset_mean;
                let bench_dev = up_bench[k] - up_bench_mean;
                up_cov += asset_dev * bench_dev;
                up_var += bench_dev * bench_dev;
            }

            let upside_beta = if up_var > 1e-10 {
                up_cov / up_var
            } else {
                1.0
            };

            // Calculate downside beta
            let down_asset_mean: f64 = down_asset.iter().sum::<f64>() / down_asset.len() as f64;
            let down_bench_mean: f64 = down_bench.iter().sum::<f64>() / down_bench.len() as f64;

            let mut down_cov = 0.0;
            let mut down_var = 0.0;
            for k in 0..down_asset.len() {
                let asset_dev = down_asset[k] - down_asset_mean;
                let bench_dev = down_bench[k] - down_bench_mean;
                down_cov += asset_dev * bench_dev;
                down_var += bench_dev * bench_dev;
            }

            let downside_beta = if down_var > 1e-10 {
                down_cov / down_var
            } else {
                1.0
            };

            // Calculate ratio (downside / upside)
            if upside_beta.abs() > 1e-10 {
                result[i] = downside_beta / upside_beta;
            }
        }

        result
    }
}

impl TechnicalIndicator for AsymmetricBeta {
    fn name(&self) -> &str {
        "Asymmetric Beta"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        // Use close as benchmark proxy when not provided separately
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Risk Parity Score - Measures risk contribution balance
///
/// Calculates a score indicating how balanced the risk contributions
/// are across time periods in the rolling window. Based on risk parity
/// principles used in portfolio construction.
///
/// # Formula
/// RPS = 1 - Coefficient of Variation of period risk contributions
///
/// # Interpretation
/// - Score close to 1: Well-balanced, stable risk contribution
/// - Score close to 0: Unbalanced, concentrated risk in specific periods
/// - Useful for identifying periods of risk concentration
/// - Helps in dynamic risk allocation decisions
///
/// # Example
/// ```ignore
/// let rps = RiskParityScore::new(50).unwrap();
/// let result = rps.calculate(&close_prices);
/// ```
#[derive(Debug, Clone)]
pub struct RiskParityScore {
    /// Rolling window period for calculation
    period: usize,
}

impl RiskParityScore {
    /// Create a new Risk Parity Score indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 20)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Risk Parity Score values
    ///
    /// Returns a score from 0 to 1 indicating risk balance.
    /// Higher values indicate more balanced risk contribution.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate squared returns (as risk contribution proxy)
            let mut risk_contributions: Vec<f64> = Vec::new();

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 {
                    let ret = close[j] / close[j - 1] - 1.0;
                    risk_contributions.push(ret * ret);
                }
            }

            if risk_contributions.len() < 10 {
                continue;
            }

            // Calculate mean and standard deviation of risk contributions
            let mean_risk: f64 = risk_contributions.iter().sum::<f64>() / risk_contributions.len() as f64;

            if mean_risk < 1e-10 {
                result[i] = 1.0; // Perfect parity if no variance
                continue;
            }

            let risk_variance: f64 = risk_contributions
                .iter()
                .map(|r| (r - mean_risk).powi(2))
                .sum::<f64>() / risk_contributions.len() as f64;
            let risk_std = risk_variance.sqrt();

            // Calculate coefficient of variation
            let cv = risk_std / mean_risk;

            // Convert to score (1 - CV, bounded 0 to 1)
            result[i] = (1.0 - cv).max(0.0).min(1.0);
        }

        result
    }
}

impl TechnicalIndicator for RiskParityScore {
    fn name(&self) -> &str {
        "Risk Parity Score"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Tracking Error Variance - Squared tracking error relative to benchmark
///
/// Measures the variance of return differences between an asset and its
/// benchmark, indicating how closely the asset tracks the benchmark.
/// Used for index fund evaluation and active risk measurement.
///
/// # Formula
/// TEV = Variance(R_asset - R_benchmark) * 252
///
/// # Interpretation
/// - Low TEV indicates close tracking to benchmark
/// - High TEV indicates significant deviation from benchmark
/// - Annualized for comparability across different time frames
/// - Square root gives tracking error (standard deviation)
///
/// # Example
/// ```ignore
/// let tev = TrackingErrorVariance::new(50).unwrap();
/// let result = tev.calculate(&close_prices, &benchmark_prices);
/// ```
#[derive(Debug, Clone)]
pub struct TrackingErrorVariance {
    /// Rolling window period for calculation
    period: usize,
}

impl TrackingErrorVariance {
    /// Create a new Tracking Error Variance indicator
    ///
    /// # Arguments
    /// * `period` - Rolling window period for calculation (minimum 20)
    ///
    /// # Returns
    /// Result containing the indicator or an error if parameters are invalid
    pub fn new(period: usize) -> Result<Self> {
        if period < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self { period })
    }

    /// Calculate Tracking Error Variance values
    ///
    /// Returns the annualized variance of return differences, expressed
    /// as a percentage squared. Take square root for tracking error.
    pub fn calculate(&self, close: &[f64], benchmark: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        if benchmark.len() != n {
            return result;
        }

        for i in self.period..n {
            let start = i.saturating_sub(self.period);

            // Calculate return differences
            let mut diffs: Vec<f64> = Vec::new();

            for j in (start + 1)..=i {
                if close[j - 1] > 1e-10 && benchmark[j - 1] > 1e-10 {
                    let asset_ret = close[j] / close[j - 1] - 1.0;
                    let bench_ret = benchmark[j] / benchmark[j - 1] - 1.0;
                    diffs.push(asset_ret - bench_ret);
                }
            }

            if diffs.len() < 10 {
                continue;
            }

            // Calculate variance of differences
            let mean_diff: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
            let variance: f64 = diffs
                .iter()
                .map(|d| (d - mean_diff).powi(2))
                .sum::<f64>() / diffs.len() as f64;

            // Annualize and express as basis points squared
            result[i] = variance * 252.0 * 10000.0;
        }

        result
    }

    /// Calculate Tracking Error (standard deviation)
    ///
    /// Returns the annualized tracking error as a percentage.
    pub fn calculate_tracking_error(&self, close: &[f64], benchmark: &[f64]) -> Vec<f64> {
        self.calculate(close, benchmark)
            .iter()
            .map(|&v| (v / 10000.0).sqrt() * 100.0)
            .collect()
    }
}

impl TechnicalIndicator for TrackingErrorVariance {
    fn name(&self) -> &str {
        "Tracking Error Variance"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        // Use close as benchmark proxy when not provided separately
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.close)))
    }

    fn output_features(&self) -> usize {
        1
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

    #[test]
    fn test_conditional_beta() {
        let data = make_test_data();
        let cb = ConditionalBeta::new(14, 0.0).unwrap();
        let benchmark: Vec<f64> = data.close.iter().map(|c| c * 0.98).collect();
        let result = cb.calculate(&data.close, &benchmark);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_tail_var() {
        let data = make_test_data();
        let tv = TailVaR::new(20, 0.95, 0.99).unwrap();
        let result = tv.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // TailVaR should be non-negative
        for i in 25..result.len() {
            assert!(result[i] >= 0.0);
        }
    }

    #[test]
    fn test_stress_test_metric() {
        let data = make_test_data();
        let stm = StressTestMetric::new(20, 2.0).unwrap();
        let result = stm.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
    }

    #[test]
    fn test_liquidity_adjusted_var() {
        let data = make_test_data();
        let lav = LiquidityAdjustedVaR::new(20, 0.95, 0.1).unwrap();
        let result = lav.calculate(&data.close, &data.volume);

        assert_eq!(result.len(), data.close.len());
        // LiquidityAdjustedVaR should be non-negative
        for i in 25..result.len() {
            assert!(result[i] >= 0.0);
        }
    }

    #[test]
    fn test_correlation_var() {
        let data = make_test_data();
        let cv = CorrelationVaR::new(20, 0.95).unwrap();
        let benchmark: Vec<f64> = data.close.iter().map(|c| c * 0.98).collect();
        let result = cv.calculate(&data.close, &benchmark);

        assert_eq!(result.len(), data.close.len());
        // CorrelationVaR should be non-negative
        for i in 25..result.len() {
            assert!(result[i] >= 0.0);
        }
    }

    #[test]
    fn test_regime_aware_risk() {
        let data = make_test_data();
        let rar = RegimeAwareRisk::new(20, 10).unwrap();
        let result = rar.calculate(&data.close);

        assert_eq!(result.len(), data.close.len());
        // RegimeAwareRisk should be non-negative
        for i in 25..result.len() {
            assert!(result[i] >= 0.0);
        }
    }

    #[test]
    fn test_new_indicators_validation() {
        assert!(ConditionalBeta::new(5, 0.0).is_err());
        assert!(TailVaR::new(5, 0.95, 0.99).is_err());
        assert!(TailVaR::new(20, 0.5, 0.99).is_err()); // var_level too low
        assert!(TailVaR::new(20, 0.95, 0.8).is_err()); // tail_level < var_level
        assert!(StressTestMetric::new(5, 2.0).is_err());
        assert!(StressTestMetric::new(20, 0.5).is_err()); // stress_threshold < 1.0
        assert!(LiquidityAdjustedVaR::new(5, 0.95, 0.1).is_err());
        assert!(LiquidityAdjustedVaR::new(20, 0.5, 0.1).is_err()); // confidence too low
        assert!(CorrelationVaR::new(5, 0.95).is_err());
        assert!(CorrelationVaR::new(20, 0.5).is_err()); // confidence too low
        assert!(RegimeAwareRisk::new(5, 10).is_err());
        assert!(RegimeAwareRisk::new(20, 3).is_err()); // vol_period too small
    }

    // ============================================================
    // Tests for the 6 NEW risk indicators
    // ============================================================

    fn make_volatile_test_data() -> Vec<f64> {
        // Data with clear ups and downs for testing drawdowns and risk regimes
        vec![
            100.0, 102.0, 99.0, 104.0, 101.0, 106.0, 103.0, 108.0, 105.0, 110.0,
            107.0, 112.0, 109.0, 114.0, 111.0, 116.0, 113.0, 118.0, 115.0, 120.0,
            117.0, 122.0, 119.0, 124.0, 121.0, 126.0, 123.0, 128.0, 125.0, 130.0,
            127.0, 132.0, 129.0, 134.0, 131.0, 136.0, 133.0, 138.0, 135.0, 140.0,
            137.0, 142.0, 139.0, 144.0, 141.0, 146.0, 143.0, 148.0, 145.0, 150.0,
            147.0, 152.0, 149.0, 154.0, 151.0, 156.0, 153.0, 158.0, 155.0, 160.0,
        ]
    }

    fn make_drawdown_test_data() -> Vec<f64> {
        // Data with a clear drawdown pattern for testing drawdown-related indicators
        vec![
            100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, // uptrend
            140.0, 135.0, 130.0, 125.0, 120.0, 115.0, 110.0, 105.0, 100.0, 95.0,  // drawdown
            100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0, 140.0, 145.0, // recovery
            150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, // new highs
            190.0, 185.0, 180.0, 175.0, 170.0, 165.0, 160.0, 155.0, 150.0, 145.0, // another drawdown
            150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 200.0, // recovery
        ]
    }

    // ============================================================
    // 1. ConditionalDrawdownAtRisk Tests
    // ============================================================

    #[test]
    fn test_conditional_drawdown_at_risk_basic() {
        let close = make_drawdown_test_data();
        let cdar = ConditionalDrawdownAtRisk::new(30, 0.95).unwrap();
        let result = cdar.calculate(&close);

        assert_eq!(result.len(), close.len());
        // CDaR should be non-negative (expressed as percentage)
        for i in 35..result.len() {
            assert!(result[i] >= 0.0, "CDaR should be non-negative at index {}", i);
        }
    }

    #[test]
    fn test_conditional_drawdown_at_risk_with_drawdowns() {
        let close = make_drawdown_test_data();
        let cdar = ConditionalDrawdownAtRisk::new(25, 0.9).unwrap();
        let result = cdar.calculate(&close);

        // During drawdown periods, CDaR should be positive
        // The max drawdown occurs around index 19 (from 145 to 95, ~34%)
        let max_cdar = result.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_cdar > 0.0, "CDaR should detect drawdowns");
    }

    #[test]
    fn test_conditional_drawdown_at_risk_validation() {
        // Period too small
        assert!(ConditionalDrawdownAtRisk::new(10, 0.95).is_err());

        // Confidence too low
        assert!(ConditionalDrawdownAtRisk::new(30, 0.5).is_err());

        // Confidence too high
        assert!(ConditionalDrawdownAtRisk::new(30, 1.0).is_err());

        // Valid parameters
        assert!(ConditionalDrawdownAtRisk::new(30, 0.95).is_ok());
        assert!(ConditionalDrawdownAtRisk::new(20, 0.9).is_ok());
        assert!(ConditionalDrawdownAtRisk::new(50, 0.99).is_ok());
    }

    #[test]
    fn test_conditional_drawdown_at_risk_indicator_trait() {
        let data = make_test_data();
        let cdar = ConditionalDrawdownAtRisk::new(25, 0.95).unwrap();

        assert_eq!(cdar.name(), "Conditional Drawdown at Risk");
        assert_eq!(cdar.min_periods(), 26);
        assert_eq!(cdar.output_features(), 1);

        let output = cdar.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // ============================================================
    // 2. UpsideDownsideRatio Tests
    // ============================================================

    #[test]
    fn test_upside_downside_ratio_basic() {
        let close = make_volatile_test_data();
        let udr = UpsideDownsideRatio::new(20).unwrap();
        let result = udr.calculate(&close);

        assert_eq!(result.len(), close.len());
        // UDR should be non-negative
        for i in 25..result.len() {
            assert!(result[i] >= 0.0, "UDR should be non-negative at index {}", i);
        }
    }

    #[test]
    fn test_upside_downside_ratio_symmetric_data() {
        // Symmetric data should give ratio close to 1
        let close: Vec<f64> = (0..50)
            .map(|i| 100.0 + (i as f64 * 0.5).sin() * 5.0)
            .collect();
        let udr = UpsideDownsideRatio::new(20).unwrap();
        let result = udr.calculate(&close);

        // Check that non-zero values are reasonable
        for i in 25..result.len() {
            if result[i] > 0.0 {
                assert!(result[i] > 0.1 && result[i] < 10.0,
                    "UDR should be in reasonable range at index {}: {}", i, result[i]);
            }
        }
    }

    #[test]
    fn test_upside_downside_ratio_validation() {
        // Period too small
        assert!(UpsideDownsideRatio::new(5).is_err());

        // Valid parameters
        assert!(UpsideDownsideRatio::new(10).is_ok());
        assert!(UpsideDownsideRatio::new(20).is_ok());
        assert!(UpsideDownsideRatio::new(50).is_ok());
    }

    #[test]
    fn test_upside_downside_ratio_indicator_trait() {
        let data = make_test_data();
        let udr = UpsideDownsideRatio::new(15).unwrap();

        assert_eq!(udr.name(), "Upside Downside Ratio");
        assert_eq!(udr.min_periods(), 16);
        assert_eq!(udr.output_features(), 1);

        let output = udr.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // ============================================================
    // 3. RiskAdjustedReturnMetric Tests
    // ============================================================

    #[test]
    fn test_risk_adjusted_return_metric_basic() {
        let close = make_volatile_test_data();
        let rarm = RiskAdjustedReturnMetric::new(30, 0.02).unwrap();
        let result = rarm.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_risk_adjusted_return_metric_uptrend() {
        // Uptrend data should give positive risk-adjusted return
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 0.5).collect();
        let rarm = RiskAdjustedReturnMetric::new(30, 0.02).unwrap();
        let result = rarm.calculate(&close);

        // Should have positive values in strong uptrend (after warmup)
        let positive_count = result[35..].iter().filter(|&&x| x > 0.0).count();
        assert!(positive_count > 0, "Should have positive RARM in uptrend");
    }

    #[test]
    fn test_risk_adjusted_return_metric_validation() {
        // Period too small
        assert!(RiskAdjustedReturnMetric::new(10, 0.02).is_err());

        // Negative risk-free rate
        assert!(RiskAdjustedReturnMetric::new(30, -0.01).is_err());

        // Risk-free rate too high
        assert!(RiskAdjustedReturnMetric::new(30, 0.6).is_err());

        // Valid parameters
        assert!(RiskAdjustedReturnMetric::new(20, 0.0).is_ok());
        assert!(RiskAdjustedReturnMetric::new(30, 0.02).is_ok());
        assert!(RiskAdjustedReturnMetric::new(50, 0.05).is_ok());
    }

    #[test]
    fn test_risk_adjusted_return_metric_indicator_trait() {
        let data = make_test_data();
        let rarm = RiskAdjustedReturnMetric::new(25, 0.02).unwrap();

        assert_eq!(rarm.name(), "Risk Adjusted Return Metric");
        assert_eq!(rarm.min_periods(), 26);
        assert_eq!(rarm.output_features(), 1);

        let output = rarm.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // ============================================================
    // 4. MaxDrawdownDuration Tests
    // ============================================================

    #[test]
    fn test_max_drawdown_duration_basic() {
        let close = make_drawdown_test_data();
        let mdd = MaxDrawdownDuration::new(20).unwrap();
        let result = mdd.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Duration should be non-negative
        for i in 25..result.len() {
            assert!(result[i] >= 0.0, "Duration should be non-negative at index {}", i);
        }
    }

    #[test]
    fn test_max_drawdown_duration_detects_drawdown() {
        let close = make_drawdown_test_data();
        let mdd = MaxDrawdownDuration::new(15).unwrap();
        let result = mdd.calculate(&close);

        // Should detect some non-zero duration during drawdown periods
        let max_duration = result.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_duration > 0.0, "Should detect drawdown duration");
    }

    #[test]
    fn test_max_drawdown_duration_underwater() {
        let close = vec![100.0, 110.0, 105.0, 100.0, 95.0, 100.0, 110.0, 115.0];
        let mdd = MaxDrawdownDuration::new(10).unwrap();
        let result = mdd.calculate_underwater(&close);

        assert_eq!(result.len(), close.len());
        // At peak (index 1), underwater = 0
        assert_eq!(result[1], 0.0);
        // After peak, underwater increases
        assert_eq!(result[2], 1.0);
        assert_eq!(result[3], 2.0);
        assert_eq!(result[4], 3.0);
        // After new high (index 7), resets
        assert_eq!(result[7], 0.0);
    }

    #[test]
    fn test_max_drawdown_duration_validation() {
        // Period too small
        assert!(MaxDrawdownDuration::new(5).is_err());

        // Valid parameters
        assert!(MaxDrawdownDuration::new(10).is_ok());
        assert!(MaxDrawdownDuration::new(20).is_ok());
        assert!(MaxDrawdownDuration::new(50).is_ok());
    }

    #[test]
    fn test_max_drawdown_duration_indicator_trait() {
        let data = make_test_data();
        let mdd = MaxDrawdownDuration::new(15).unwrap();

        assert_eq!(mdd.name(), "Maximum Drawdown Duration");
        assert_eq!(mdd.min_periods(), 16);
        assert_eq!(mdd.output_features(), 1);

        let output = mdd.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // ============================================================
    // 5. DrawdownRecoveryFactor Tests
    // ============================================================

    #[test]
    fn test_drawdown_recovery_factor_basic() {
        let close = make_drawdown_test_data();
        let drf = DrawdownRecoveryFactor::new(30).unwrap();
        let result = drf.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_drawdown_recovery_factor_uptrend() {
        // Pure uptrend should give positive recovery factor
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 0.5).collect();
        let drf = DrawdownRecoveryFactor::new(25).unwrap();
        let result = drf.calculate(&close);

        // Should have positive values (net profit positive)
        for i in 30..result.len() {
            assert!(result[i] >= 0.0, "DRF should be positive in uptrend at index {}", i);
        }
    }

    #[test]
    fn test_drawdown_recovery_factor_with_drawdown() {
        let close = make_drawdown_test_data();
        let drf = DrawdownRecoveryFactor::new(25).unwrap();
        let result = drf.calculate(&close);

        // During recovery periods, should have meaningful values
        let nonzero_count = result[30..].iter().filter(|&&x| x != 0.0).count();
        assert!(nonzero_count > 0, "Should have non-zero recovery factors");
    }

    #[test]
    fn test_drawdown_recovery_factor_validation() {
        // Period too small
        assert!(DrawdownRecoveryFactor::new(10).is_err());

        // Valid parameters
        assert!(DrawdownRecoveryFactor::new(20).is_ok());
        assert!(DrawdownRecoveryFactor::new(30).is_ok());
        assert!(DrawdownRecoveryFactor::new(50).is_ok());
    }

    #[test]
    fn test_drawdown_recovery_factor_indicator_trait() {
        let data = make_test_data();
        let drf = DrawdownRecoveryFactor::new(25).unwrap();

        assert_eq!(drf.name(), "Drawdown Recovery Factor");
        assert_eq!(drf.min_periods(), 26);
        assert_eq!(drf.output_features(), 1);

        let output = drf.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // ============================================================
    // 6. RiskRegimeIndicator Tests
    // ============================================================

    #[test]
    fn test_risk_regime_indicator_basic() {
        let close = make_volatile_test_data();
        let rri = RiskRegimeIndicator::new(40, 10).unwrap();
        let result = rri.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Regime should be 0, 1, 2, or 3
        for i in 45..result.len() {
            assert!(
                result[i] >= 0.0 && result[i] <= 3.0,
                "Regime should be 0-3 at index {}: {}", i, result[i]
            );
        }
    }

    #[test]
    fn test_risk_regime_indicator_discrete_values() {
        let close = make_drawdown_test_data();
        let rri = RiskRegimeIndicator::new(35, 10).unwrap();
        let result = rri.calculate(&close);

        // Check that results are discrete (0, 1, 2, or 3)
        for i in 40..result.len() {
            if result[i] > 0.0 {
                assert!(
                    (result[i] - 0.0).abs() < 0.01 ||
                    (result[i] - 1.0).abs() < 0.01 ||
                    (result[i] - 2.0).abs() < 0.01 ||
                    (result[i] - 3.0).abs() < 0.01,
                    "Regime should be discrete at index {}: {}", i, result[i]
                );
            }
        }
    }

    #[test]
    fn test_risk_regime_indicator_continuous() {
        let close = make_volatile_test_data();
        let rri = RiskRegimeIndicator::new(40, 10).unwrap();
        let result = rri.calculate_continuous(&close);

        assert_eq!(result.len(), close.len());
        // Continuous values should be non-negative
        for i in 45..result.len() {
            assert!(result[i] >= 0.0, "Continuous regime should be non-negative at index {}", i);
        }
    }

    #[test]
    fn test_risk_regime_indicator_validation() {
        // Period too small
        assert!(RiskRegimeIndicator::new(20, 10).is_err());

        // Vol period too small
        assert!(RiskRegimeIndicator::new(50, 3).is_err());

        // Vol period >= period
        assert!(RiskRegimeIndicator::new(30, 30).is_err());
        assert!(RiskRegimeIndicator::new(30, 35).is_err());

        // Valid parameters
        assert!(RiskRegimeIndicator::new(30, 5).is_ok());
        assert!(RiskRegimeIndicator::new(50, 10).is_ok());
        assert!(RiskRegimeIndicator::new(100, 20).is_ok());
    }

    #[test]
    fn test_risk_regime_indicator_trait() {
        let data = make_test_data();
        let rri = RiskRegimeIndicator::new(35, 10).unwrap();

        assert_eq!(rri.name(), "Risk Regime Indicator");
        assert_eq!(rri.min_periods(), 36);
        assert_eq!(rri.output_features(), 1);

        let output = rri.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // ============================================================
    // Integration Tests for all 6 new indicators
    // ============================================================

    #[test]
    fn test_all_new_indicators_names() {
        let cdar = ConditionalDrawdownAtRisk::new(25, 0.95).unwrap();
        assert_eq!(cdar.name(), "Conditional Drawdown at Risk");

        let udr = UpsideDownsideRatio::new(15).unwrap();
        assert_eq!(udr.name(), "Upside Downside Ratio");

        let rarm = RiskAdjustedReturnMetric::new(25, 0.02).unwrap();
        assert_eq!(rarm.name(), "Risk Adjusted Return Metric");

        let mdd = MaxDrawdownDuration::new(15).unwrap();
        assert_eq!(mdd.name(), "Maximum Drawdown Duration");

        let drf = DrawdownRecoveryFactor::new(25).unwrap();
        assert_eq!(drf.name(), "Drawdown Recovery Factor");

        let rri = RiskRegimeIndicator::new(35, 10).unwrap();
        assert_eq!(rri.name(), "Risk Regime Indicator");
    }

    #[test]
    fn test_all_new_indicators_empty_data() {
        let empty: Vec<f64> = Vec::new();

        let cdar = ConditionalDrawdownAtRisk::new(25, 0.95).unwrap();
        assert_eq!(cdar.calculate(&empty).len(), 0);

        let udr = UpsideDownsideRatio::new(15).unwrap();
        assert_eq!(udr.calculate(&empty).len(), 0);

        let rarm = RiskAdjustedReturnMetric::new(25, 0.02).unwrap();
        assert_eq!(rarm.calculate(&empty).len(), 0);

        let mdd = MaxDrawdownDuration::new(15).unwrap();
        assert_eq!(mdd.calculate(&empty).len(), 0);

        let drf = DrawdownRecoveryFactor::new(25).unwrap();
        assert_eq!(drf.calculate(&empty).len(), 0);

        let rri = RiskRegimeIndicator::new(35, 10).unwrap();
        assert_eq!(rri.calculate(&empty).len(), 0);
    }

    #[test]
    fn test_all_new_indicators_insufficient_data() {
        // Data shorter than period
        let short_data: Vec<f64> = (0..15).map(|i| 100.0 + i as f64).collect();

        let cdar = ConditionalDrawdownAtRisk::new(25, 0.95).unwrap();
        let result = cdar.calculate(&short_data);
        assert!(result.iter().all(|&x| x == 0.0), "CDaR should be zero for insufficient data");

        let udr = UpsideDownsideRatio::new(20).unwrap();
        let result = udr.calculate(&short_data);
        assert!(result.iter().all(|&x| x == 0.0), "UDR should be zero for insufficient data");

        let rarm = RiskAdjustedReturnMetric::new(25, 0.02).unwrap();
        let result = rarm.calculate(&short_data);
        assert!(result.iter().all(|&x| x == 0.0), "RARM should be zero for insufficient data");

        let mdd = MaxDrawdownDuration::new(20).unwrap();
        let result = mdd.calculate(&short_data);
        assert!(result.iter().all(|&x| x == 0.0), "MDD Duration should be zero for insufficient data");

        let drf = DrawdownRecoveryFactor::new(25).unwrap();
        let result = drf.calculate(&short_data);
        assert!(result.iter().all(|&x| x == 0.0), "DRF should be zero for insufficient data");

        let rri = RiskRegimeIndicator::new(35, 10).unwrap();
        let result = rri.calculate(&short_data);
        assert!(result.iter().all(|&x| x == 0.0), "RRI should be zero for insufficient data");
    }

    // ============================================================
    // Tests for the 6 ADDITIONAL NEW risk indicators
    // SortinoRatioAdvanced, CalmarRatioAdvanced, OmegaRatioAdvanced,
    // PainRatio, UlcerIndex, KellyFraction
    // ============================================================

    // ============================================================
    // 1. SortinoRatioAdvanced Tests
    // ============================================================

    #[test]
    fn test_sortino_ratio_advanced_basic() {
        let close = make_volatile_test_data();
        let sortino = SortinoRatioAdvanced::new(30, 0.0, 252.0).unwrap();
        let result = sortino.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_sortino_ratio_advanced_uptrend() {
        // Uptrend with some volatility (for downside deviation calculation)
        let close = make_volatile_test_data();
        let sortino = SortinoRatioAdvanced::new(25, 0.0, 252.0).unwrap();
        let result = sortino.calculate(&close);

        // Should produce valid non-zero values
        assert_eq!(result.len(), close.len());
        // Check that some values are computed
        let nonzero_count = result[30..].iter().filter(|&&x| x != 0.0).count();
        assert!(nonzero_count > 0, "Should compute Sortino values for volatile data");
    }

    #[test]
    fn test_sortino_ratio_advanced_with_target() {
        let close = make_volatile_test_data();
        let sortino = SortinoRatioAdvanced::new(30, 0.02, 252.0).unwrap();
        let result = sortino.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Should still compute valid values
        for i in 35..result.len() {
            assert!(!result[i].is_nan(), "Sortino should not be NaN at index {}", i);
        }
    }

    #[test]
    fn test_sortino_ratio_advanced_validation() {
        // Period too small
        assert!(SortinoRatioAdvanced::new(10, 0.0, 252.0).is_err());

        // Invalid annualization factor
        assert!(SortinoRatioAdvanced::new(30, 0.0, 0.0).is_err());
        assert!(SortinoRatioAdvanced::new(30, 0.0, -1.0).is_err());

        // Valid parameters
        assert!(SortinoRatioAdvanced::new(20, 0.0, 252.0).is_ok());
        assert!(SortinoRatioAdvanced::new(30, 0.02, 252.0).is_ok());
        assert!(SortinoRatioAdvanced::new(50, 0.05, 52.0).is_ok()); // Weekly
    }

    #[test]
    fn test_sortino_ratio_advanced_indicator_trait() {
        let data = make_test_data();
        let sortino = SortinoRatioAdvanced::new(25, 0.0, 252.0).unwrap();

        assert_eq!(sortino.name(), "Sortino Ratio Advanced");
        assert_eq!(sortino.min_periods(), 26);
        assert_eq!(sortino.output_features(), 1);

        let output = sortino.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // ============================================================
    // 2. CalmarRatioAdvanced Tests
    // ============================================================

    #[test]
    fn test_calmar_ratio_advanced_basic() {
        let close = make_drawdown_test_data();
        let calmar = CalmarRatioAdvanced::new(30, 252.0).unwrap();
        let result = calmar.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_calmar_ratio_advanced_uptrend() {
        // Pure uptrend should give positive Calmar
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 0.5).collect();
        let calmar = CalmarRatioAdvanced::new(25, 252.0).unwrap();
        let result = calmar.calculate(&close);

        // Should have positive values in uptrend (after warmup)
        let positive_count = result[30..].iter().filter(|&&x| x > 0.0).count();
        assert!(positive_count > 0, "Should have positive Calmar in uptrend");
    }

    #[test]
    fn test_calmar_ratio_advanced_with_drawdown() {
        let close = make_drawdown_test_data();
        let calmar = CalmarRatioAdvanced::new(25, 252.0).unwrap();
        let result = calmar.calculate(&close);

        // Should have meaningful values during and after drawdowns
        let nonzero_count = result[30..].iter().filter(|&&x| x != 0.0).count();
        assert!(nonzero_count > 0, "Should have non-zero Calmar values");
    }

    #[test]
    fn test_calmar_ratio_advanced_validation() {
        // Period too small
        assert!(CalmarRatioAdvanced::new(10, 252.0).is_err());

        // Invalid annualization factor
        assert!(CalmarRatioAdvanced::new(30, 0.0).is_err());
        assert!(CalmarRatioAdvanced::new(30, -1.0).is_err());

        // Valid parameters
        assert!(CalmarRatioAdvanced::new(20, 252.0).is_ok());
        assert!(CalmarRatioAdvanced::new(30, 52.0).is_ok()); // Weekly
        assert!(CalmarRatioAdvanced::new(50, 12.0).is_ok()); // Monthly
    }

    #[test]
    fn test_calmar_ratio_advanced_indicator_trait() {
        let data = make_test_data();
        let calmar = CalmarRatioAdvanced::new(25, 252.0).unwrap();

        assert_eq!(calmar.name(), "Calmar Ratio Advanced");
        assert_eq!(calmar.min_periods(), 26);
        assert_eq!(calmar.output_features(), 1);

        let output = calmar.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // ============================================================
    // 3. OmegaRatioAdvanced Tests
    // ============================================================

    #[test]
    fn test_omega_ratio_advanced_basic() {
        let close = make_volatile_test_data();
        let omega = OmegaRatioAdvanced::new(30, 0.0).unwrap();
        let result = omega.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_omega_ratio_advanced_uptrend() {
        // Pure uptrend should give Omega > 1
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 0.5).collect();
        let omega = OmegaRatioAdvanced::new(25, 0.0).unwrap();
        let result = omega.calculate(&close);

        // Should have values >= 1 in uptrend
        let high_omega_count = result[30..].iter().filter(|&&x| x >= 1.0).count();
        assert!(high_omega_count > 0, "Should have Omega >= 1 in uptrend");
    }

    #[test]
    fn test_omega_ratio_advanced_with_threshold() {
        let close = make_volatile_test_data();
        let omega = OmegaRatioAdvanced::new(30, 0.001).unwrap();
        let result = omega.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Non-zero threshold should still produce valid results
        for i in 35..result.len() {
            assert!(result[i] >= 0.0, "Omega should be non-negative at index {}", i);
        }
    }

    #[test]
    fn test_omega_ratio_advanced_validation() {
        // Period too small
        assert!(OmegaRatioAdvanced::new(10, 0.0).is_err());

        // Valid parameters (threshold can be any value)
        assert!(OmegaRatioAdvanced::new(20, 0.0).is_ok());
        assert!(OmegaRatioAdvanced::new(30, 0.001).is_ok());
        assert!(OmegaRatioAdvanced::new(50, -0.001).is_ok()); // Negative threshold is valid
    }

    #[test]
    fn test_omega_ratio_advanced_indicator_trait() {
        let data = make_test_data();
        let omega = OmegaRatioAdvanced::new(25, 0.0).unwrap();

        assert_eq!(omega.name(), "Omega Ratio Advanced");
        assert_eq!(omega.min_periods(), 26);
        assert_eq!(omega.output_features(), 1);

        let output = omega.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // ============================================================
    // 4. PainRatio Tests
    // ============================================================

    #[test]
    fn test_pain_ratio_basic() {
        let close = make_drawdown_test_data();
        let pain_ratio = PainRatio::new(30, 0.02).unwrap();
        let result = pain_ratio.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_pain_ratio_uptrend() {
        // Pure uptrend should give positive Pain Ratio
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 0.5).collect();
        let pain_ratio = PainRatio::new(25, 0.02).unwrap();
        let result = pain_ratio.calculate(&close);

        // Should have positive values in uptrend
        let positive_count = result[30..].iter().filter(|&&x| x > 0.0).count();
        assert!(positive_count > 0, "Should have positive Pain Ratio in uptrend");
    }

    #[test]
    fn test_pain_ratio_with_drawdown() {
        let close = make_drawdown_test_data();
        let pain_ratio = PainRatio::new(25, 0.0).unwrap();
        let result = pain_ratio.calculate(&close);

        // Should have meaningful values
        let nonzero_count = result[30..].iter().filter(|&&x| x != 0.0).count();
        assert!(nonzero_count > 0, "Should have non-zero Pain Ratio values");
    }

    #[test]
    fn test_pain_ratio_validation() {
        // Period too small
        assert!(PainRatio::new(10, 0.02).is_err());

        // Invalid risk-free rate
        assert!(PainRatio::new(30, -0.01).is_err());
        assert!(PainRatio::new(30, 0.6).is_err());

        // Valid parameters
        assert!(PainRatio::new(20, 0.0).is_ok());
        assert!(PainRatio::new(30, 0.02).is_ok());
        assert!(PainRatio::new(50, 0.05).is_ok());
    }

    #[test]
    fn test_pain_ratio_indicator_trait() {
        let data = make_test_data();
        let pain_ratio = PainRatio::new(25, 0.02).unwrap();

        assert_eq!(pain_ratio.name(), "Pain Ratio");
        assert_eq!(pain_ratio.min_periods(), 26);
        assert_eq!(pain_ratio.output_features(), 1);

        let output = pain_ratio.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // ============================================================
    // 5. UlcerIndex Tests
    // ============================================================

    #[test]
    fn test_ulcer_index_basic() {
        let close = make_drawdown_test_data();
        let ulcer = UlcerIndex::new(14).unwrap();
        let result = ulcer.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Ulcer Index should be non-negative
        for i in 20..result.len() {
            assert!(result[i] >= 0.0, "Ulcer Index should be non-negative at index {}", i);
        }
    }

    #[test]
    fn test_ulcer_index_uptrend() {
        // Pure uptrend with minimal drawdowns should have low Ulcer Index
        let close: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 0.5).collect();
        let ulcer = UlcerIndex::new(14).unwrap();
        let result = ulcer.calculate(&close);

        // Should have low values in smooth uptrend
        for i in 20..result.len() {
            assert!(result[i] < 5.0, "Ulcer Index should be low in smooth uptrend at index {}", i);
        }
    }

    #[test]
    fn test_ulcer_index_with_drawdown() {
        let close = make_drawdown_test_data();
        let ulcer = UlcerIndex::new(14).unwrap();
        let result = ulcer.calculate(&close);

        // During drawdown, Ulcer Index should increase
        let max_ulcer = result.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_ulcer > 0.0, "Ulcer Index should detect drawdowns");
    }

    #[test]
    fn test_ulcer_index_validation() {
        // Period too small
        assert!(UlcerIndex::new(5).is_err());

        // Valid parameters
        assert!(UlcerIndex::new(10).is_ok());
        assert!(UlcerIndex::new(14).is_ok());
        assert!(UlcerIndex::new(50).is_ok());
    }

    #[test]
    fn test_ulcer_index_indicator_trait() {
        let data = make_test_data();
        let ulcer = UlcerIndex::new(14).unwrap();

        assert_eq!(ulcer.name(), "Ulcer Index");
        assert_eq!(ulcer.min_periods(), 15);
        assert_eq!(ulcer.output_features(), 1);

        let output = ulcer.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // ============================================================
    // 6. KellyFraction Tests
    // ============================================================

    #[test]
    fn test_kelly_fraction_basic() {
        let close = make_volatile_test_data();
        let kelly = KellyFraction::new(30).unwrap();
        let result = kelly.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_kelly_fraction_uptrend() {
        // Use volatile data which has both ups and downs for Kelly calculation
        let close = make_volatile_test_data();
        let kelly = KellyFraction::new(25).unwrap();
        let result = kelly.calculate(&close);

        // Should produce valid values - volatile data has wins and losses
        assert_eq!(result.len(), close.len());
        // Check that some values are computed (Kelly requires both wins and losses)
        let nonzero_count = result[30..].iter().filter(|&&x| x != 0.0).count();
        assert!(nonzero_count > 0, "Should compute Kelly values for volatile data");
    }

    #[test]
    fn test_kelly_fraction_bounded() {
        let close = make_volatile_test_data();
        let kelly = KellyFraction::new(30).unwrap();
        let result = kelly.calculate(&close);

        // Kelly should be bounded between -1 and 1
        for i in 35..result.len() {
            assert!(
                result[i] >= -1.0 && result[i] <= 1.0,
                "Kelly should be bounded at index {}: {}", i, result[i]
            );
        }
    }

    #[test]
    fn test_kelly_fraction_weighted() {
        let close = make_volatile_test_data();
        let kelly = KellyFraction::new(30).unwrap();
        let result = kelly.calculate_weighted(&close);

        assert_eq!(result.len(), close.len());
        // Weighted Kelly should be bounded between -2 and 2
        for i in 35..result.len() {
            assert!(
                result[i] >= -2.0 && result[i] <= 2.0,
                "Weighted Kelly should be bounded at index {}: {}", i, result[i]
            );
        }
    }

    #[test]
    fn test_kelly_fraction_validation() {
        // Period too small
        assert!(KellyFraction::new(10).is_err());

        // Valid parameters
        assert!(KellyFraction::new(20).is_ok());
        assert!(KellyFraction::new(30).is_ok());
        assert!(KellyFraction::new(50).is_ok());
    }

    #[test]
    fn test_kelly_fraction_indicator_trait() {
        let data = make_test_data();
        let kelly = KellyFraction::new(25).unwrap();

        assert_eq!(kelly.name(), "Kelly Fraction");
        assert_eq!(kelly.min_periods(), 26);
        assert_eq!(kelly.output_features(), 1);

        let output = kelly.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    // ============================================================
    // Integration Tests for the 6 additional new indicators
    // ============================================================

    #[test]
    fn test_additional_new_indicators_names() {
        let sortino = SortinoRatioAdvanced::new(25, 0.0, 252.0).unwrap();
        assert_eq!(sortino.name(), "Sortino Ratio Advanced");

        let calmar = CalmarRatioAdvanced::new(25, 252.0).unwrap();
        assert_eq!(calmar.name(), "Calmar Ratio Advanced");

        let omega = OmegaRatioAdvanced::new(25, 0.0).unwrap();
        assert_eq!(omega.name(), "Omega Ratio Advanced");

        let pain_ratio = PainRatio::new(25, 0.02).unwrap();
        assert_eq!(pain_ratio.name(), "Pain Ratio");

        let ulcer = UlcerIndex::new(14).unwrap();
        assert_eq!(ulcer.name(), "Ulcer Index");

        let kelly = KellyFraction::new(25).unwrap();
        assert_eq!(kelly.name(), "Kelly Fraction");
    }

    #[test]
    fn test_additional_new_indicators_empty_data() {
        let empty: Vec<f64> = Vec::new();

        let sortino = SortinoRatioAdvanced::new(25, 0.0, 252.0).unwrap();
        assert_eq!(sortino.calculate(&empty).len(), 0);

        let calmar = CalmarRatioAdvanced::new(25, 252.0).unwrap();
        assert_eq!(calmar.calculate(&empty).len(), 0);

        let omega = OmegaRatioAdvanced::new(25, 0.0).unwrap();
        assert_eq!(omega.calculate(&empty).len(), 0);

        let pain_ratio = PainRatio::new(25, 0.02).unwrap();
        assert_eq!(pain_ratio.calculate(&empty).len(), 0);

        let ulcer = UlcerIndex::new(14).unwrap();
        assert_eq!(ulcer.calculate(&empty).len(), 0);

        let kelly = KellyFraction::new(25).unwrap();
        assert_eq!(kelly.calculate(&empty).len(), 0);
    }

    #[test]
    fn test_additional_new_indicators_insufficient_data() {
        // Data shorter than period
        let short_data: Vec<f64> = (0..15).map(|i| 100.0 + i as f64).collect();

        let sortino = SortinoRatioAdvanced::new(25, 0.0, 252.0).unwrap();
        let result = sortino.calculate(&short_data);
        assert!(result.iter().all(|&x| x == 0.0), "Sortino should be zero for insufficient data");

        let calmar = CalmarRatioAdvanced::new(25, 252.0).unwrap();
        let result = calmar.calculate(&short_data);
        assert!(result.iter().all(|&x| x == 0.0), "Calmar should be zero for insufficient data");

        let omega = OmegaRatioAdvanced::new(25, 0.0).unwrap();
        let result = omega.calculate(&short_data);
        assert!(result.iter().all(|&x| x == 0.0), "Omega should be zero for insufficient data");

        let pain_ratio = PainRatio::new(25, 0.02).unwrap();
        let result = pain_ratio.calculate(&short_data);
        assert!(result.iter().all(|&x| x == 0.0), "Pain Ratio should be zero for insufficient data");

        let ulcer = UlcerIndex::new(20).unwrap();
        let result = ulcer.calculate(&short_data);
        assert!(result.iter().all(|&x| x == 0.0), "Ulcer Index should be zero for insufficient data");

        let kelly = KellyFraction::new(25).unwrap();
        let result = kelly.calculate(&short_data);
        assert!(result.iter().all(|&x| x == 0.0), "Kelly should be zero for insufficient data");
    }

    #[test]
    fn test_all_six_new_indicators_compute_together() {
        // Test that all six indicators can be computed on the same dataset
        let data = make_test_data();

        let sortino = SortinoRatioAdvanced::new(25, 0.0, 252.0).unwrap();
        let calmar = CalmarRatioAdvanced::new(25, 252.0).unwrap();
        let omega = OmegaRatioAdvanced::new(25, 0.0).unwrap();
        let pain_ratio = PainRatio::new(25, 0.02).unwrap();
        let ulcer = UlcerIndex::new(14).unwrap();
        let kelly = KellyFraction::new(25).unwrap();

        let sortino_result = sortino.compute(&data).unwrap();
        let calmar_result = calmar.compute(&data).unwrap();
        let omega_result = omega.compute(&data).unwrap();
        let pain_ratio_result = pain_ratio.compute(&data).unwrap();
        let ulcer_result = ulcer.compute(&data).unwrap();
        let kelly_result = kelly.compute(&data).unwrap();

        // All should have the same length
        assert_eq!(sortino_result.primary.len(), data.close.len());
        assert_eq!(calmar_result.primary.len(), data.close.len());
        assert_eq!(omega_result.primary.len(), data.close.len());
        assert_eq!(pain_ratio_result.primary.len(), data.close.len());
        assert_eq!(ulcer_result.primary.len(), data.close.len());
        assert_eq!(kelly_result.primary.len(), data.close.len());
    }

    // ============================================================
    // Tests for the 6 NEWEST risk indicators
    // TailRiskRatio, VaRBreachRate, VolatilityOfVolatility,
    // AsymmetricBeta, RiskParityScore, TrackingErrorVariance
    // ============================================================

    #[test]
    fn test_tail_risk_ratio_basic() {
        let close = make_volatile_test_data();
        let trr = TailRiskRatio::new(30, 0.95).unwrap();
        let result = trr.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Tail risk ratio should be non-negative
        for i in 35..result.len() {
            assert!(result[i] >= 0.0, "TRR should be non-negative at index {}", i);
        }
    }

    #[test]
    fn test_tail_risk_ratio_validation() {
        assert!(TailRiskRatio::new(10, 0.95).is_err());
        assert!(TailRiskRatio::new(30, 0.5).is_err());
        assert!(TailRiskRatio::new(30, 0.95).is_ok());
    }

    #[test]
    fn test_tail_risk_ratio_indicator_trait() {
        let data = make_test_data();
        let trr = TailRiskRatio::new(25, 0.95).unwrap();

        assert_eq!(trr.name(), "Tail Risk Ratio");
        assert_eq!(trr.min_periods(), 26);
        assert_eq!(trr.output_features(), 1);

        let output = trr.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_var_breach_rate_basic() {
        let close = make_volatile_test_data();
        let vbr = VaRBreachRate::new(30, 0.95).unwrap();
        let result = vbr.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Breach rate should be 0-100
        for i in 35..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0,
                "Breach rate should be 0-100 at index {}: {}", i, result[i]);
        }
    }

    #[test]
    fn test_var_breach_rate_validation() {
        assert!(VaRBreachRate::new(10, 0.95).is_err());
        assert!(VaRBreachRate::new(30, 0.5).is_err());
        assert!(VaRBreachRate::new(30, 0.95).is_ok());
    }

    #[test]
    fn test_var_breach_rate_indicator_trait() {
        let data = make_test_data();
        let vbr = VaRBreachRate::new(25, 0.95).unwrap();

        assert_eq!(vbr.name(), "VaR Breach Rate");
        assert_eq!(vbr.min_periods(), 26);
        assert_eq!(vbr.output_features(), 1);

        let output = vbr.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_volatility_of_volatility_basic() {
        let close = make_volatile_test_data();
        let vov = VolatilityOfVolatility::new(30, 10).unwrap();
        let result = vov.calculate(&close);

        assert_eq!(result.len(), close.len());
        // VoV should be non-negative
        for i in 35..result.len() {
            assert!(result[i] >= 0.0, "VoV should be non-negative at index {}", i);
        }
    }

    #[test]
    fn test_volatility_of_volatility_validation() {
        assert!(VolatilityOfVolatility::new(10, 5).is_err());
        assert!(VolatilityOfVolatility::new(30, 2).is_err());
        assert!(VolatilityOfVolatility::new(30, 30).is_err());
        assert!(VolatilityOfVolatility::new(30, 10).is_ok());
    }

    #[test]
    fn test_volatility_of_volatility_indicator_trait() {
        let data = make_test_data();
        let vov = VolatilityOfVolatility::new(30, 10).unwrap();

        assert_eq!(vov.name(), "Volatility of Volatility");
        assert_eq!(vov.min_periods(), 31);
        assert_eq!(vov.output_features(), 1);

        let output = vov.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_asymmetric_beta_basic() {
        let close = make_volatile_test_data();
        let ab = AsymmetricBeta::new(30).unwrap();
        let benchmark: Vec<f64> = close.iter().map(|c| c * 0.98).collect();
        let result = ab.calculate(&close, &benchmark);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_asymmetric_beta_validation() {
        assert!(AsymmetricBeta::new(10).is_err());
        assert!(AsymmetricBeta::new(20).is_ok());
    }

    #[test]
    fn test_asymmetric_beta_indicator_trait() {
        let data = make_test_data();
        let ab = AsymmetricBeta::new(25).unwrap();

        assert_eq!(ab.name(), "Asymmetric Beta");
        assert_eq!(ab.min_periods(), 26);
        assert_eq!(ab.output_features(), 1);

        let output = ab.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_risk_parity_score_basic() {
        let close = make_volatile_test_data();
        let rps = RiskParityScore::new(30).unwrap();
        let result = rps.calculate(&close);

        assert_eq!(result.len(), close.len());
        // Score should be non-negative
        for i in 35..result.len() {
            assert!(result[i] >= 0.0, "RPS should be non-negative at index {}", i);
        }
    }

    #[test]
    fn test_risk_parity_score_validation() {
        assert!(RiskParityScore::new(10).is_err());
        assert!(RiskParityScore::new(20).is_ok());
    }

    #[test]
    fn test_risk_parity_score_indicator_trait() {
        let data = make_test_data();
        let rps = RiskParityScore::new(25).unwrap();

        assert_eq!(rps.name(), "Risk Parity Score");
        assert_eq!(rps.min_periods(), 26);
        assert_eq!(rps.output_features(), 1);

        let output = rps.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_tracking_error_variance_basic() {
        let close = make_volatile_test_data();
        let tev = TrackingErrorVariance::new(30).unwrap();
        let benchmark: Vec<f64> = close.iter().map(|c| c * 0.98).collect();
        let result = tev.calculate(&close, &benchmark);

        assert_eq!(result.len(), close.len());
        // Variance should be non-negative
        for i in 35..result.len() {
            assert!(result[i] >= 0.0, "TEV should be non-negative at index {}", i);
        }
    }

    #[test]
    fn test_tracking_error_variance_validation() {
        assert!(TrackingErrorVariance::new(10).is_err());
        assert!(TrackingErrorVariance::new(20).is_ok());
    }

    #[test]
    fn test_tracking_error_variance_indicator_trait() {
        let data = make_test_data();
        let tev = TrackingErrorVariance::new(25).unwrap();

        assert_eq!(tev.name(), "Tracking Error Variance");
        assert_eq!(tev.min_periods(), 26);
        assert_eq!(tev.output_features(), 1);

        let output = tev.compute(&data).unwrap();
        assert_eq!(output.primary.len(), data.close.len());
    }

    #[test]
    fn test_newest_indicators_names() {
        let trr = TailRiskRatio::new(25, 0.95).unwrap();
        assert_eq!(trr.name(), "Tail Risk Ratio");

        let vbr = VaRBreachRate::new(25, 0.95).unwrap();
        assert_eq!(vbr.name(), "VaR Breach Rate");

        let vov = VolatilityOfVolatility::new(30, 10).unwrap();
        assert_eq!(vov.name(), "Volatility of Volatility");

        let ab = AsymmetricBeta::new(25).unwrap();
        assert_eq!(ab.name(), "Asymmetric Beta");

        let rps = RiskParityScore::new(25).unwrap();
        assert_eq!(rps.name(), "Risk Parity Score");

        let tev = TrackingErrorVariance::new(25).unwrap();
        assert_eq!(tev.name(), "Tracking Error Variance");
    }

    #[test]
    fn test_newest_indicators_empty_data() {
        let empty: Vec<f64> = Vec::new();

        let trr = TailRiskRatio::new(25, 0.95).unwrap();
        assert_eq!(trr.calculate(&empty).len(), 0);

        let vbr = VaRBreachRate::new(25, 0.95).unwrap();
        assert_eq!(vbr.calculate(&empty).len(), 0);

        let vov = VolatilityOfVolatility::new(30, 10).unwrap();
        assert_eq!(vov.calculate(&empty).len(), 0);

        let ab = AsymmetricBeta::new(25).unwrap();
        assert_eq!(ab.calculate(&empty, &empty).len(), 0);

        let rps = RiskParityScore::new(25).unwrap();
        assert_eq!(rps.calculate(&empty).len(), 0);

        let tev = TrackingErrorVariance::new(25).unwrap();
        assert_eq!(tev.calculate(&empty, &empty).len(), 0);
    }

    #[test]
    fn test_newest_indicators_compute_together() {
        let data = make_test_data();

        let trr = TailRiskRatio::new(25, 0.95).unwrap();
        let vbr = VaRBreachRate::new(25, 0.95).unwrap();
        let vov = VolatilityOfVolatility::new(30, 10).unwrap();
        let ab = AsymmetricBeta::new(25).unwrap();
        let rps = RiskParityScore::new(25).unwrap();
        let tev = TrackingErrorVariance::new(25).unwrap();

        let trr_result = trr.compute(&data).unwrap();
        let vbr_result = vbr.compute(&data).unwrap();
        let vov_result = vov.compute(&data).unwrap();
        let ab_result = ab.compute(&data).unwrap();
        let rps_result = rps.compute(&data).unwrap();
        let tev_result = tev.compute(&data).unwrap();

        assert_eq!(trr_result.primary.len(), data.close.len());
        assert_eq!(vbr_result.primary.len(), data.close.len());
        assert_eq!(vov_result.primary.len(), data.close.len());
        assert_eq!(ab_result.primary.len(), data.close.len());
        assert_eq!(rps_result.primary.len(), data.close.len());
        assert_eq!(tev_result.primary.len(), data.close.len());
    }
}
