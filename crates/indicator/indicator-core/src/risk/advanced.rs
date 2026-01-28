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
}
