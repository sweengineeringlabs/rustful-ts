//! Vortex Indicator.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Vortex Indicator - IND-012
///
/// VI+ and VI- measure trend direction.
/// Based on true range and directional movement.
#[derive(Debug, Clone)]
pub struct VortexIndicator {
    period: usize,
}

impl VortexIndicator {
    pub fn new(period: usize) -> Self {
        Self { period }
    }

    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = high.len();
        if n < self.period + 1 {
            return (vec![f64::NAN; n], vec![f64::NAN; n]);
        }

        // Calculate True Range and VM+ / VM-
        let mut tr = vec![0.0; n];
        let mut vm_plus = vec![0.0; n];
        let mut vm_minus = vec![0.0; n];

        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr[i] = hl.max(hc).max(lc);

            vm_plus[i] = (high[i] - low[i - 1]).abs();
            vm_minus[i] = (low[i] - high[i - 1]).abs();
        }

        let mut vi_plus = vec![f64::NAN; self.period];
        let mut vi_minus = vec![f64::NAN; self.period];

        for i in self.period..n {
            let tr_sum: f64 = tr[(i - self.period + 1)..=i].iter().sum();
            let vm_plus_sum: f64 = vm_plus[(i - self.period + 1)..=i].iter().sum();
            let vm_minus_sum: f64 = vm_minus[(i - self.period + 1)..=i].iter().sum();

            if tr_sum != 0.0 {
                vi_plus.push(vm_plus_sum / tr_sum);
                vi_minus.push(vm_minus_sum / tr_sum);
            } else {
                vi_plus.push(f64::NAN);
                vi_minus.push(f64::NAN);
            }
        }

        (vi_plus, vi_minus)
    }
}

impl Default for VortexIndicator {
    fn default() -> Self {
        Self::new(14)
    }
}

impl TechnicalIndicator for VortexIndicator {
    fn name(&self) -> &str {
        "Vortex"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.high.len() < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: data.high.len(),
            });
        }

        let (vi_plus, vi_minus) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(vi_plus, vi_minus))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn output_features(&self) -> usize {
        2
    }
}

impl SignalIndicator for VortexIndicator {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (vi_plus, vi_minus) = self.calculate(&data.high, &data.low, &data.close);

        if vi_plus.len() < 2 || vi_minus.len() < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let vip_last = vi_plus[vi_plus.len() - 1];
        let vim_last = vi_minus[vi_minus.len() - 1];
        let vip_prev = vi_plus[vi_plus.len() - 2];
        let vim_prev = vi_minus[vi_minus.len() - 2];

        if vip_last.is_nan() || vim_last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Bullish crossover: VI+ crosses above VI-
        if vip_last > vim_last && vip_prev <= vim_prev {
            Ok(IndicatorSignal::Bullish)
        } else if vip_last < vim_last && vip_prev >= vim_prev {
            Ok(IndicatorSignal::Bearish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (vi_plus, vi_minus) = self.calculate(&data.high, &data.low, &data.close);
        let mut signals = vec![IndicatorSignal::Neutral];

        for i in 1..vi_plus.len().min(vi_minus.len()) {
            if vi_plus[i].is_nan() || vi_minus[i].is_nan() {
                signals.push(IndicatorSignal::Neutral);
            } else if vi_plus[i] > vi_minus[i] && vi_plus[i-1] <= vi_minus[i-1] {
                signals.push(IndicatorSignal::Bullish);
            } else if vi_plus[i] < vi_minus[i] && vi_plus[i-1] >= vi_minus[i-1] {
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
    fn test_vortex_basic() {
        let vortex = VortexIndicator::new(14);
        let n = 30;
        let high: Vec<f64> = (0..n).map(|i| 110.0 + i as f64).collect();
        let low: Vec<f64> = (0..n).map(|i| 90.0 + i as f64).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let (vi_plus, vi_minus) = vortex.calculate(&high, &low, &close);

        assert_eq!(vi_plus.len(), n);
        assert_eq!(vi_minus.len(), n);
    }
}
