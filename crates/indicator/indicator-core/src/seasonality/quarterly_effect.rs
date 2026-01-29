//! Quarterly Effect Indicator (IND-240)
//!
//! Analyzes quarter-end flows and their impact on market behavior.
//! Quarter-end rebalancing by institutions often causes predictable patterns.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Phase within a quarter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuarterPhase {
    /// Early quarter (first month)
    Early,
    /// Mid quarter (second month)
    Mid,
    /// Late quarter (third month)
    Late,
    /// Quarter-end (last 5 trading days)
    QuarterEnd,
    /// Month-end within quarter
    MonthEnd,
}

/// Output structure for quarterly effect analysis.
#[derive(Debug, Clone)]
pub struct QuarterlyEffectOutput {
    /// Quarter-end flow strength (-1 to 1)
    pub flow_strength: Vec<f64>,
    /// Days until quarter end
    pub days_to_quarter_end: Vec<f64>,
    /// Historical quarter-end performance
    pub quarter_end_performance: Vec<f64>,
    /// Current quarter phase
    pub phase: Vec<QuarterPhase>,
}

/// Configuration for the Quarterly Effect indicator.
#[derive(Debug, Clone)]
pub struct QuarterlyEffectConfig {
    /// Lookback period for historical comparison (in quarters)
    pub lookback_quarters: usize,
    /// Number of days before quarter-end to consider as "quarter-end period"
    pub quarter_end_days: usize,
    /// Number of days before month-end to consider as "month-end period"
    pub month_end_days: usize,
}

impl Default for QuarterlyEffectConfig {
    fn default() -> Self {
        Self {
            lookback_quarters: 4,
            quarter_end_days: 5,
            month_end_days: 3,
        }
    }
}

/// Quarterly Effect indicator for analyzing quarter-end flows.
///
/// This indicator helps identify:
/// - Institutional rebalancing patterns at quarter-end
/// - Window dressing effects
/// - Month-end vs quarter-end flow differences
/// - Seasonal quarter patterns (Q1, Q2, Q3, Q4 effects)
#[derive(Debug, Clone)]
pub struct QuarterlyEffect {
    config: QuarterlyEffectConfig,
}

impl QuarterlyEffect {
    /// Create a new Quarterly Effect indicator with default parameters.
    pub fn new() -> Self {
        Self {
            config: QuarterlyEffectConfig::default(),
        }
    }

    /// Create a Quarterly Effect indicator with custom configuration.
    pub fn with_config(config: QuarterlyEffectConfig) -> Self {
        Self { config }
    }

    /// Estimate days until quarter end based on trading day index.
    /// In practice, this would use actual calendar data.
    fn estimate_quarter_position(&self, day_index: usize, total_days: usize) -> (f64, QuarterPhase) {
        // Approximate a quarter as ~63 trading days (252 / 4)
        let quarter_days = 63;
        let day_in_quarter = day_index % quarter_days;
        let days_to_end = quarter_days.saturating_sub(day_in_quarter);

        let phase = if days_to_end <= self.config.quarter_end_days {
            QuarterPhase::QuarterEnd
        } else if day_in_quarter % 21 >= (21 - self.config.month_end_days) {
            // Approximately 21 trading days per month
            QuarterPhase::MonthEnd
        } else if day_in_quarter < 21 {
            QuarterPhase::Early
        } else if day_in_quarter < 42 {
            QuarterPhase::Mid
        } else {
            QuarterPhase::Late
        };

        (days_to_end as f64, phase)
    }

    /// Calculate quarter-end flow strength based on price action and volume.
    fn calculate_flow_strength(&self, close: &[f64], volume: &[f64], idx: usize) -> f64 {
        if idx < 5 {
            return 0.0;
        }

        // Calculate recent price momentum
        let price_change = (close[idx] - close[idx - 5]) / close[idx - 5];

        // Calculate volume relative to average
        let avg_volume: f64 = volume[idx.saturating_sub(20)..idx].iter().sum::<f64>()
            / (idx - idx.saturating_sub(20)).max(1) as f64;
        let volume_ratio = if avg_volume > 0.0 {
            volume[idx] / avg_volume
        } else {
            1.0
        };

        // Flow strength combines price direction with volume intensity
        // Positive values suggest buying pressure, negative suggests selling
        let raw_flow = price_change * volume_ratio;

        // Normalize to -1 to 1 range using tanh
        raw_flow.tanh()
    }

    /// Calculate historical quarter-end performance.
    fn calculate_quarter_end_performance(&self, close: &[f64], idx: usize) -> f64 {
        // Look back at historical quarter-ends (every ~63 trading days)
        let quarter_days = 63;
        let mut total_return = 0.0;
        let mut count = 0;

        let mut check_idx = idx.saturating_sub(quarter_days);
        while check_idx > self.config.quarter_end_days && count < self.config.lookback_quarters {
            // Calculate return during quarter-end period
            let end_idx = check_idx;
            let start_idx = end_idx.saturating_sub(self.config.quarter_end_days);

            if start_idx < close.len() && end_idx < close.len() && close[start_idx] > 0.0 {
                let period_return = (close[end_idx] - close[start_idx]) / close[start_idx];
                total_return += period_return;
                count += 1;
            }

            check_idx = check_idx.saturating_sub(quarter_days);
        }

        if count > 0 {
            total_return / count as f64
        } else {
            0.0
        }
    }

    /// Calculate the quarterly effect indicators.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> QuarterlyEffectOutput {
        let n = close.len();
        let mut flow_strength = vec![f64::NAN; n];
        let mut days_to_quarter_end = vec![f64::NAN; n];
        let mut quarter_end_performance = vec![f64::NAN; n];
        let mut phase = vec![QuarterPhase::Early; n];

        for i in 0..n {
            let (days_to_end, current_phase) = self.estimate_quarter_position(i, n);
            days_to_quarter_end[i] = days_to_end;
            phase[i] = current_phase;

            if i >= 20 {
                flow_strength[i] = self.calculate_flow_strength(close, volume, i);
                quarter_end_performance[i] = self.calculate_quarter_end_performance(close, i);
            }
        }

        QuarterlyEffectOutput {
            flow_strength,
            days_to_quarter_end,
            quarter_end_performance,
            phase,
        }
    }

    /// Get the primary quarterly effect signal (flow strength).
    pub fn calculate_signal(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        self.calculate(close, volume).flow_strength
    }
}

impl Default for QuarterlyEffect {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for QuarterlyEffect {
    fn name(&self) -> &str {
        "QuarterlyEffect"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 21 {
            return Err(IndicatorError::InsufficientData {
                required: 21,
                got: data.close.len(),
            });
        }

        let output = self.calculate(&data.close, &data.volume);

        Ok(IndicatorOutput::triple(
            output.flow_strength,
            output.days_to_quarter_end,
            output.quarter_end_performance,
        ))
    }

    fn min_periods(&self) -> usize {
        21
    }
}

impl SignalIndicator for QuarterlyEffect {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        if data.close.len() < 21 {
            return Ok(IndicatorSignal::Neutral);
        }

        let output = self.calculate(&data.close, &data.volume);
        let n = output.flow_strength.len();

        if n == 0 || output.flow_strength[n - 1].is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        let flow = output.flow_strength[n - 1];
        let is_quarter_end = matches!(output.phase[n - 1], QuarterPhase::QuarterEnd);

        // Strong signal at quarter-end with positive historical performance
        if is_quarter_end {
            let hist_perf = output.quarter_end_performance[n - 1];
            if flow > 0.3 && hist_perf > 0.0 {
                return Ok(IndicatorSignal::Bullish);
            } else if flow < -0.3 && hist_perf < 0.0 {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let output = self.calculate(&data.close, &data.volume);
        let signals = output
            .flow_strength
            .iter()
            .enumerate()
            .map(|(i, &flow)| {
                if flow.is_nan() {
                    IndicatorSignal::Neutral
                } else {
                    let is_quarter_end = matches!(output.phase[i], QuarterPhase::QuarterEnd);
                    if is_quarter_end && flow > 0.3 {
                        IndicatorSignal::Bullish
                    } else if is_quarter_end && flow < -0.3 {
                        IndicatorSignal::Bearish
                    } else {
                        IndicatorSignal::Neutral
                    }
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(n: usize) -> (Vec<f64>, Vec<f64>) {
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1)).collect();
        let volume: Vec<f64> = (0..n).map(|i| 1000000.0 + (i as f64 * 1000.0)).collect();
        (close, volume)
    }

    #[test]
    fn test_quarterly_effect_basic() {
        let indicator = QuarterlyEffect::new();
        let (close, volume) = create_test_data(100);

        let output = indicator.calculate(&close, &volume);

        assert_eq!(output.flow_strength.len(), 100);
        assert_eq!(output.days_to_quarter_end.len(), 100);
        assert_eq!(output.phase.len(), 100);
    }

    #[test]
    fn test_quarterly_effect_phases() {
        let indicator = QuarterlyEffect::new();
        let (close, volume) = create_test_data(70);

        let output = indicator.calculate(&close, &volume);

        // Should have various phases across a quarter
        let has_early = output.phase.iter().any(|p| matches!(p, QuarterPhase::Early));
        let has_mid = output.phase.iter().any(|p| matches!(p, QuarterPhase::Mid));
        let has_late = output.phase.iter().any(|p| matches!(p, QuarterPhase::Late));

        assert!(has_early || has_mid || has_late);
    }

    #[test]
    fn test_quarterly_effect_flow_strength_range() {
        let indicator = QuarterlyEffect::new();
        let (close, volume) = create_test_data(100);

        let output = indicator.calculate(&close, &volume);

        // Flow strength should be in -1 to 1 range (due to tanh)
        for &flow in output.flow_strength.iter().filter(|v| !v.is_nan()) {
            assert!(flow >= -1.0 && flow <= 1.0);
        }
    }

    #[test]
    fn test_quarterly_effect_insufficient_data() {
        let indicator = QuarterlyEffect::new();
        let data = OHLCVSeries {
            open: vec![100.0; 10],
            high: vec![101.0; 10],
            low: vec![99.0; 10],
            close: vec![100.0; 10],
            volume: vec![1000.0; 10],
        };

        let result = indicator.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_quarterly_effect_technical_indicator() {
        let indicator = QuarterlyEffect::new();
        let data = OHLCVSeries {
            open: (0..50).map(|i| 100.0 + i as f64).collect(),
            high: (0..50).map(|i| 101.0 + i as f64).collect(),
            low: (0..50).map(|i| 99.0 + i as f64).collect(),
            close: (0..50).map(|i| 100.0 + i as f64 * 0.5).collect(),
            volume: vec![1000000.0; 50],
        };

        let result = indicator.compute(&data);
        assert!(result.is_ok());
    }
}
