//! Stan Weinstein Stage Analysis Indicators
//!
//! Trend analysis tools based on Weinstein's 4-stage market cycle methodology.

use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};
use crate::{SMA, EMA};

/// Market Stage according to Weinstein methodology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeinStage {
    /// Stage 1: Basing/Accumulation
    Accumulation,
    /// Stage 2: Advancing/Markup
    Markup,
    /// Stage 3: Topping/Distribution
    Distribution,
    /// Stage 4: Declining/Markdown
    Markdown,
    /// Unknown/Transitional
    Unknown,
}

/// Stage Analysis - Weinstein's 4-stage market cycle identification.
///
/// Identifies which of the four stages a stock is currently in:
/// 1. Basing/Accumulation - Price consolidates below flattening MA
/// 2. Advancing/Markup - Price breaks above rising MA
/// 3. Topping/Distribution - Price consolidates around flattening MA
/// 4. Declining/Markdown - Price breaks below falling MA
#[derive(Debug, Clone)]
pub struct StageAnalysis {
    ma_period: usize,
    slope_period: usize,
}

/// Output for Stage Analysis.
#[derive(Debug, Clone)]
pub struct StageAnalysisOutput {
    /// The 30-week (or configured) moving average
    pub ma: Vec<f64>,
    /// MA slope
    pub slope: Vec<f64>,
    /// Detected stage at each point
    pub stage: Vec<WeinStage>,
}

impl StageAnalysis {
    pub fn new() -> Self {
        Self { ma_period: 30, slope_period: 5 }
    }

    pub fn with_periods(ma_period: usize, slope_period: usize) -> Self {
        Self { ma_period, slope_period }
    }

    /// Calculate stage analysis.
    pub fn calculate(&self, close: &[f64]) -> StageAnalysisOutput {
        let n = close.len();
        let sma = SMA::new(self.ma_period);
        let ma = sma.calculate(close);

        // Calculate MA slope
        let mut slope = vec![f64::NAN; n];
        if n >= self.ma_period + self.slope_period {
            for i in (self.ma_period + self.slope_period - 1)..n {
                if !ma[i].is_nan() && !ma[i - self.slope_period].is_nan() {
                    slope[i] = (ma[i] - ma[i - self.slope_period]) / self.slope_period as f64;
                }
            }
        }

        // Determine stage
        let mut stage = vec![WeinStage::Unknown; n];

        for i in (self.ma_period + self.slope_period - 1)..n {
            if ma[i].is_nan() || slope[i].is_nan() {
                continue;
            }

            let price_above_ma = close[i] > ma[i];
            let slope_rising = slope[i] > 0.0;
            let slope_flat = slope[i].abs() < ma[i] * 0.001; // < 0.1% change is "flat"

            stage[i] = if price_above_ma && slope_rising {
                WeinStage::Markup
            } else if price_above_ma && (slope_flat || !slope_rising) {
                WeinStage::Distribution
            } else if !price_above_ma && !slope_rising {
                WeinStage::Markdown
            } else if !price_above_ma && (slope_flat || slope_rising) {
                WeinStage::Accumulation
            } else {
                WeinStage::Unknown
            };
        }

        StageAnalysisOutput { ma, slope, stage }
    }
}

impl Default for StageAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for StageAnalysis {
    fn name(&self) -> &str {
        "Stage Analysis"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.ma_period + self.slope_period;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let output = self.calculate(&data.close);
        let stage_values: Vec<f64> = output.stage.iter().map(|s| match s {
            WeinStage::Accumulation => 1.0,
            WeinStage::Markup => 2.0,
            WeinStage::Distribution => 3.0,
            WeinStage::Markdown => 4.0,
            WeinStage::Unknown => 0.0,
        }).collect();

        Ok(IndicatorOutput::dual(output.ma, stage_values))
    }

    fn min_periods(&self) -> usize {
        self.ma_period + self.slope_period
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for StageAnalysis {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let output = self.calculate(&data.close);

        if output.stage.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        match output.stage.last().unwrap() {
            WeinStage::Markup => Ok(IndicatorSignal::Bullish),
            WeinStage::Markdown => Ok(IndicatorSignal::Bearish),
            _ => Ok(IndicatorSignal::Neutral),
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let output = self.calculate(&data.close);

        Ok(output.stage.iter().map(|s| match s {
            WeinStage::Markup => IndicatorSignal::Bullish,
            WeinStage::Markdown => IndicatorSignal::Bearish,
            _ => IndicatorSignal::Neutral,
        }).collect())
    }
}

/// 30-Week Moving Average - Weinstein's key trend indicator.
///
/// For daily data, this is approximately a 150-day MA.
#[derive(Debug, Clone)]
pub struct WeinsteinMA {
    period: usize,
}

impl WeinsteinMA {
    pub fn new() -> Self {
        Self { period: 150 } // 30 weeks * 5 days
    }

    pub fn weekly() -> Self {
        Self { period: 30 }
    }

    pub fn with_period(period: usize) -> Self {
        Self { period }
    }

    /// Calculate the moving average.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let sma = SMA::new(self.period);
        sma.calculate(close)
    }
}

impl Default for WeinsteinMA {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for WeinsteinMA {
    fn name(&self) -> &str {
        "30-Week Moving Average"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Mansfield Relative Strength - Price relative to market.
///
/// RS = (Stock Price / Index Price) * 100
///
/// Measures how well a stock is performing relative to the overall market.
#[derive(Debug, Clone)]
pub struct MansfieldRS {
    ma_period: usize,
}

impl MansfieldRS {
    pub fn new(ma_period: usize) -> Self {
        Self { ma_period }
    }

    /// Calculate Mansfield RS.
    /// Requires stock price and market index price.
    pub fn calculate(&self, stock: &[f64], market: &[f64]) -> Vec<f64> {
        let n = stock.len().min(market.len());
        if n == 0 {
            return vec![];
        }

        // Calculate ratio
        let mut ratio = Vec::with_capacity(n);
        for i in 0..n {
            if market[i].abs() > 1e-10 {
                ratio.push((stock[i] / market[i]) * 100.0);
            } else {
                ratio.push(f64::NAN);
            }
        }

        // Calculate MA of ratio
        let sma = SMA::new(self.ma_period);
        let ratio_ma = sma.calculate(&ratio);

        // Mansfield RS = ratio / ratio_ma - 1 (as percentage)
        ratio.iter()
            .zip(ratio_ma.iter())
            .map(|(&r, &ma)| {
                if r.is_nan() || ma.is_nan() || ma.abs() < 1e-10 {
                    f64::NAN
                } else {
                    (r / ma - 1.0) * 100.0
                }
            })
            .collect()
    }
}

impl TechnicalIndicator for MansfieldRS {
    fn name(&self) -> &str {
        "Mansfield Relative Strength"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        // Standard compute uses close vs itself - not meaningful
        // This indicator requires external market data
        if data.close.len() < self.ma_period {
            return Err(IndicatorError::InsufficientData {
                required: self.ma_period,
                got: data.close.len(),
            });
        }

        Ok(IndicatorOutput::single(vec![f64::NAN; data.close.len()]))
    }

    fn min_periods(&self) -> usize {
        self.ma_period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Relative Price Strength - IBD-style RS rating.
///
/// Compares price performance over multiple periods.
#[derive(Debug, Clone)]
pub struct RelativePriceStrength {
    short_period: usize,
    long_period: usize,
}

impl RelativePriceStrength {
    pub fn new() -> Self {
        Self { short_period: 63, long_period: 252 } // ~3 months, ~1 year
    }

    pub fn with_periods(short: usize, long: usize) -> Self {
        Self { short_period: short, long_period: long }
    }

    /// Calculate relative price strength.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_required = self.long_period.max(self.short_period + 1);
        if n < min_required {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; n];

        for i in (min_required - 1)..n {
            let current = close[i];

            // Short-term return
            let short_idx = i.saturating_sub(self.short_period);
            let short_ret = if close[short_idx].abs() > 1e-10 {
                (current / close[short_idx] - 1.0) * 100.0
            } else {
                f64::NAN
            };

            // Long-term return
            let long_idx = i.saturating_sub(self.long_period - 1);
            let long_ret = if close[long_idx].abs() > 1e-10 {
                (current / close[long_idx] - 1.0) * 100.0
            } else {
                f64::NAN
            };

            if short_ret.is_nan() || long_ret.is_nan() {
                continue;
            }

            // Weighted combination (IBD gives more weight to recent performance)
            result[i] = short_ret * 0.4 + long_ret * 0.2 + (short_ret + long_ret) / 2.0 * 0.4;
        }

        result
    }
}

impl Default for RelativePriceStrength {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for RelativePriceStrength {
    fn name(&self) -> &str {
        "Relative Price Strength"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.long_period.max(self.short_period + 1);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.long_period.max(self.short_period + 1)
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Volume Confirmation - Checks if price moves are confirmed by volume.
///
/// Breakouts should occur on above-average volume.
#[derive(Debug, Clone)]
pub struct VolumeConfirmation {
    period: usize,
    vol_threshold: f64,
}

impl VolumeConfirmation {
    pub fn new(period: usize) -> Self {
        Self { period, vol_threshold: 1.5 }
    }

    pub fn with_threshold(period: usize, threshold: f64) -> Self {
        Self { period, vol_threshold: threshold }
    }

    /// Calculate volume confirmation.
    /// Returns 1.0 for confirmed moves, 0.0 for unconfirmed.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.period {
            return vec![0.0; n];
        }

        let mut result = vec![0.0; self.period - 1];

        for i in (self.period - 1)..n {
            let start = i + 1 - self.period;
            let vol_window = &volume[start..i];

            let avg_vol: f64 = vol_window.iter().sum::<f64>() / vol_window.len() as f64;

            // Check if current volume is above threshold
            let vol_strong = volume[i] > avg_vol * self.vol_threshold;

            // Check if there's a price move
            let price_change = if i > 0 {
                (close[i] - close[i - 1]).abs() / close[i - 1]
            } else {
                0.0
            };

            let has_move = price_change > 0.01; // > 1% move

            if vol_strong && has_move {
                result.push(1.0);
            } else {
                result.push(0.0);
            }
        }

        result
    }
}

impl TechnicalIndicator for VolumeConfirmation {
    fn name(&self) -> &str {
        "Volume Confirmation"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close, &data.volume);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        1
    }
}

/// Support/Resistance Levels - Automatic S/R detection.
///
/// Identifies significant price levels based on swing points.
#[derive(Debug, Clone)]
pub struct SupportResistanceLevels {
    lookback: usize,
    tolerance: f64,
}

/// Output for Support/Resistance detection.
#[derive(Debug, Clone)]
pub struct SRLevelsOutput {
    /// Support levels
    pub support: Vec<f64>,
    /// Resistance levels
    pub resistance: Vec<f64>,
}

impl SupportResistanceLevels {
    pub fn new(lookback: usize) -> Self {
        Self { lookback, tolerance: 0.02 }
    }

    pub fn with_tolerance(lookback: usize, tolerance: f64) -> Self {
        Self { lookback, tolerance }
    }

    /// Calculate dynamic S/R levels.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        if n < self.lookback {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        let mut support = vec![f64::NAN; n];
        let mut resistance = vec![f64::NAN; n];

        for i in (self.lookback - 1)..n {
            let start = i + 1 - self.lookback;

            // Find swing low (support candidate)
            let min_idx = (start..=i).min_by(|&a, &b|
                low[a].partial_cmp(&low[b]).unwrap_or(std::cmp::Ordering::Equal)
            ).unwrap();

            // Find swing high (resistance candidate)
            let max_idx = (start..=i).max_by(|&a, &b|
                high[a].partial_cmp(&high[b]).unwrap_or(std::cmp::Ordering::Equal)
            ).unwrap();

            support[i] = low[min_idx];
            resistance[i] = high[max_idx];
        }

        (support, resistance)
    }
}

impl TechnicalIndicator for SupportResistanceLevels {
    fn name(&self) -> &str {
        "Support/Resistance Levels"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.lookback {
            return Err(IndicatorError::InsufficientData {
                required: self.lookback,
                got: data.close.len(),
            });
        }

        let (support, resistance) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(support, resistance))
    }

    fn min_periods(&self) -> usize {
        self.lookback
    }

    fn output_features(&self) -> usize {
        2
    }
}

/// Breakout Validation - Checks if a breakout is valid.
///
/// Valid breakouts have: price above resistance, volume confirmation, MA alignment.
#[derive(Debug, Clone)]
pub struct BreakoutValidation {
    lookback: usize,
    ma_period: usize,
}

impl BreakoutValidation {
    pub fn new(lookback: usize, ma_period: usize) -> Self {
        Self { lookback, ma_period }
    }

    /// Calculate breakout validation score (0-100).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len();
        let min_required = self.lookback.max(self.ma_period);
        if n < min_required {
            return vec![f64::NAN; n];
        }

        let sma = SMA::new(self.ma_period);
        let ma = sma.calculate(close);

        let mut result = vec![f64::NAN; min_required - 1];

        for i in (min_required - 1)..n {
            let start = i + 1 - self.lookback;

            // Find recent resistance
            let recent_high = high[start..i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // Check breakout conditions
            let mut score = 0.0;

            // 1. Price above recent resistance
            if close[i] > recent_high {
                score += 30.0;
            }

            // 2. Price above MA
            if !ma[i].is_nan() && close[i] > ma[i] {
                score += 25.0;
            }

            // 3. MA rising
            if i >= 5 && !ma[i].is_nan() && !ma[i - 5].is_nan() && ma[i] > ma[i - 5] {
                score += 20.0;
            }

            // 4. Volume confirmation
            let avg_vol: f64 = volume[start..i].iter().sum::<f64>() / (i - start) as f64;
            if avg_vol > 0.0 && volume[i] > avg_vol * 1.5 {
                score += 25.0;
            }

            result.push(score);
        }

        result
    }
}

impl TechnicalIndicator for BreakoutValidation {
    fn name(&self) -> &str {
        "Breakout Validation"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.lookback.max(self.ma_period);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.high, &data.low, &data.close, &data.volume);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.lookback.max(self.ma_period)
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for BreakoutValidation {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let scores = self.calculate(&data.high, &data.low, &data.close, &data.volume);

        if scores.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let score = *scores.last().unwrap();
        if score.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if score >= 75.0 {
            Ok(IndicatorSignal::Bullish)
        } else if score <= 25.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let scores = self.calculate(&data.high, &data.low, &data.close, &data.volume);

        Ok(scores.iter().map(|&s| {
            if s.is_nan() {
                IndicatorSignal::Neutral
            } else if s >= 75.0 {
                IndicatorSignal::Bullish
            } else if s <= 25.0 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

/// Trend Score - Composite trend strength indicator.
///
/// Combines multiple trend factors into a single score (0-100).
#[derive(Debug, Clone)]
pub struct TrendScore {
    short_ma: usize,
    long_ma: usize,
}

impl TrendScore {
    pub fn new() -> Self {
        Self { short_ma: 20, long_ma: 50 }
    }

    pub fn with_periods(short: usize, long: usize) -> Self {
        Self { short_ma: short, long_ma: long }
    }

    /// Calculate trend score.
    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        if n < self.long_ma {
            return vec![f64::NAN; n];
        }

        let short_sma = SMA::new(self.short_ma);
        let long_sma = SMA::new(self.long_ma);

        let ma_short = short_sma.calculate(close);
        let ma_long = long_sma.calculate(close);

        let mut result = vec![f64::NAN; n];

        for i in (self.long_ma - 1)..n {
            if ma_short[i].is_nan() || ma_long[i].is_nan() {
                continue;
            }

            let mut score: f64 = 50.0; // Base score

            // Factor 1: Price vs short MA
            if close[i] > ma_short[i] {
                score += 15.0;
            } else {
                score -= 15.0;
            }

            // Factor 2: Price vs long MA
            if close[i] > ma_long[i] {
                score += 15.0;
            } else {
                score -= 15.0;
            }

            // Factor 3: Short MA vs Long MA
            if ma_short[i] > ma_long[i] {
                score += 10.0;
            } else {
                score -= 10.0;
            }

            // Factor 4: Short MA slope
            if i >= 5 && !ma_short[i - 5].is_nan() {
                if ma_short[i] > ma_short[i - 5] {
                    score += 10.0;
                } else {
                    score -= 10.0;
                }
            }

            result[i] = score.clamp(0.0, 100.0);
        }

        result
    }
}

impl Default for TrendScore {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for TrendScore {
    fn name(&self) -> &str {
        "Trend Score"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.long_ma {
            return Err(IndicatorError::InsufficientData {
                required: self.long_ma,
                got: data.close.len(),
            });
        }

        let result = self.calculate(&data.close);
        Ok(IndicatorOutput::single(result))
    }

    fn min_periods(&self) -> usize {
        self.long_ma
    }

    fn output_features(&self) -> usize {
        1
    }
}

impl SignalIndicator for TrendScore {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let scores = self.calculate(&data.close);

        if scores.is_empty() {
            return Ok(IndicatorSignal::Neutral);
        }

        let score = *scores.last().unwrap();
        if score.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if score >= 70.0 {
            Ok(IndicatorSignal::Bullish)
        } else if score <= 30.0 {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let scores = self.calculate(&data.close);

        Ok(scores.iter().map(|&s| {
            if s.is_nan() {
                IndicatorSignal::Neutral
            } else if s >= 70.0 {
                IndicatorSignal::Bullish
            } else if s <= 30.0 {
                IndicatorSignal::Bearish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create trending data
        let high: Vec<f64> = (0..200).map(|i| 105.0 + i as f64 * 0.1 + (i as f64 * 0.1).sin() * 2.0).collect();
        let low: Vec<f64> = (0..200).map(|i| 95.0 + i as f64 * 0.1 + (i as f64 * 0.1).sin() * 2.0).collect();
        let close: Vec<f64> = (0..200).map(|i| 100.0 + i as f64 * 0.1 + (i as f64 * 0.1).sin() * 2.0).collect();
        let volume: Vec<f64> = (0..200).map(|i| 1000000.0 + (i as f64 * 0.2).sin() * 200000.0).collect();
        (high, low, close, volume)
    }

    #[test]
    fn test_stage_analysis() {
        let (_, _, close, _) = make_test_data();
        let stage = StageAnalysis::new();
        let output = stage.calculate(&close);

        assert_eq!(output.ma.len(), 200);
        assert_eq!(output.stage.len(), 200);
    }

    #[test]
    fn test_weinstein_ma() {
        let (_, _, close, _) = make_test_data();
        let ma = WeinsteinMA::new();
        let result = ma.calculate(&close);

        assert_eq!(result.len(), 200);
    }

    #[test]
    fn test_mansfield_rs() {
        let stock: Vec<f64> = (0..100).map(|i| 100.0 + i as f64 * 0.5).collect();
        let market: Vec<f64> = (0..100).map(|i| 1000.0 + i as f64 * 0.3).collect();

        let rs = MansfieldRS::new(20);
        let result = rs.calculate(&stock, &market);

        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_relative_price_strength() {
        let close: Vec<f64> = (0..300).map(|i| 100.0 + i as f64 * 0.1).collect();
        let rps = RelativePriceStrength::new();
        let result = rps.calculate(&close);

        assert_eq!(result.len(), 300);
    }

    #[test]
    fn test_volume_confirmation() {
        let (_, _, close, volume) = make_test_data();
        let vc = VolumeConfirmation::new(20);
        let result = vc.calculate(&close, &volume);

        assert_eq!(result.len(), 200);
    }

    #[test]
    fn test_support_resistance() {
        let (high, low, close, _) = make_test_data();
        let sr = SupportResistanceLevels::new(20);
        let (support, resistance) = sr.calculate(&high, &low, &close);

        assert_eq!(support.len(), 200);
        assert_eq!(resistance.len(), 200);

        // Support should be less than resistance
        for i in 19..200 {
            if !support[i].is_nan() && !resistance[i].is_nan() {
                assert!(support[i] <= resistance[i]);
            }
        }
    }

    #[test]
    fn test_breakout_validation() {
        let (high, low, close, volume) = make_test_data();
        let bv = BreakoutValidation::new(20, 50);
        let result = bv.calculate(&high, &low, &close, &volume);

        assert_eq!(result.len(), 200);

        // Scores should be 0-100
        for i in 49..200 {
            if !result[i].is_nan() {
                assert!(result[i] >= 0.0 && result[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_trend_score() {
        let (_, _, close, _) = make_test_data();
        let ts = TrendScore::new();
        let result = ts.calculate(&close);

        assert_eq!(result.len(), 200);

        // Scores should be 0-100
        for i in 49..200 {
            if !result[i].is_nan() {
                assert!(result[i] >= 0.0 && result[i] <= 100.0);
            }
        }
    }
}
