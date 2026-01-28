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
}
