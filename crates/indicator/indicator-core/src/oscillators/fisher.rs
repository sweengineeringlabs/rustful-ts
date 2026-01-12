//! Fisher Transform indicator.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Fisher Transform - IND-042
///
/// Transforms prices into a Gaussian normal distribution.
/// Fisher = 0.5 * ln((1 + Value) / (1 - Value))
#[derive(Debug, Clone)]
pub struct FisherTransform {
    period: usize,
}

impl FisherTransform {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn calculate(&self, high: &[f64], low: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        if n < self.period {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate median price
        let median: Vec<f64> = high.iter()
            .zip(low.iter())
            .map(|(h, l)| (h + l) / 2.0)
            .collect();

        let mut fisher = vec![f64::NAN; self.period - 1];
        let mut trigger = vec![f64::NAN; self.period];

        let mut prev_value = 0.0;
        let mut prev_fisher = 0.0;

        for i in (self.period - 1)..n {
            // Find highest and lowest in period
            let start_idx = i + 1 - self.period;
            let window = &median[start_idx..=i];
            let highest = window.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let lowest = window.iter().fold(f64::INFINITY, |a, &b| a.min(b));

            // Normalize to -1 to +1 range
            let value = if (highest - lowest) > 0.0 {
                let raw = 2.0 * ((median[i] - lowest) / (highest - lowest) - 0.5);
                // Smoothing
                0.66 * raw + 0.34 * prev_value
            } else {
                prev_value
            };

            // Clamp value to avoid log of zero
            let clamped = value.max(-0.999).min(0.999);

            // Fisher Transform
            let fisher_val = 0.5 * ((1.0 + clamped) / (1.0 - clamped)).ln();
            let smoothed_fisher = fisher_val + prev_fisher * 0.5;

            fisher.push(smoothed_fisher);

            prev_value = value;
            prev_fisher = smoothed_fisher;
        }

        // Trigger (previous Fisher)
        for i in self.period..n {
            trigger.push(fisher[i - 1]);
        }

        (fisher, trigger)
    }
}

impl Default for FisherTransform {
    fn default() -> Self {
        Self::new(10)
    }
}

impl TechnicalIndicator for FisherTransform {
    fn name(&self) -> &str {
        "FisherTransform"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period {
            return Err(IndicatorError::InsufficientData {
                required: self.period,
                got: data.high.len(),
            });
        }

        let (fisher, trigger) = self.calculate(&data.high, &data.low);
        Ok(IndicatorOutput::dual(fisher, trigger))
    }

    fn min_periods(&self) -> usize {
        self.period
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for FisherTransform {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (fisher, trigger) = self.calculate(&data.high, &data.low);

        if fisher.len() < 2 || trigger.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let fisher_last = fisher[fisher.len() - 1];
        let trigger_last = trigger[trigger.len() - 1];
        let fisher_prev = fisher[fisher.len() - 2];
        let trigger_prev = trigger[trigger.len() - 2];

        if fisher_last.is_nan() || trigger_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Crossover signals
        if fisher_last > trigger_last && fisher_prev <= trigger_prev {
            Ok(IndicatorSignal::Bullish)
        } else if fisher_last < trigger_last && fisher_prev >= trigger_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (fisher, trigger) = self.calculate(&data.high, &data.low);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..fisher.len().min(trigger.len()) {
            if fisher[i].is_nan() || trigger[i].is_nan() || fisher[i-1].is_nan() || trigger[i-1].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if fisher[i] > trigger[i] && fisher[i-1] <= trigger[i-1] {
                signals.push(IndicatorSignal::Bullish);
            } else if fisher[i] < trigger[i] && fisher[i-1] >= trigger[i-1] {
                signals.push(IndicatorSignal::Bearish);
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

    #[test]
    fn test_fisher_basic() {
        let fisher = FisherTransform::new(10);
        let n = 50;
        let high: Vec<f64> = (0..n).map(|i| 110.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 90.0 + (i as f64 * 0.1).sin() * 5.0).collect();

        let (fisher_line, trigger_line) = fisher.calculate(&high, &low);

        assert_eq!(fisher_line.len(), n);
        assert_eq!(trigger_line.len(), n);
    }
}
