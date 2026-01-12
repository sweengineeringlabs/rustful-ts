//! Connors RSI indicator.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Connors RSI - IND-038
///
/// Composite oscillator: (RSI + Streak RSI + Percent Rank ROC) / 3
#[derive(Debug, Clone)]
pub struct ConnorsRSI {
    rsi_period: usize,
    streak_period: usize,
    rank_period: usize,
    overbought: f64,
    oversold: f64,
}

impl ConnorsRSI {
    pub fn new(rsi_period: usize, streak_period: usize, rank_period: usize) -> Self {
        Self {
            rsi_period,
            streak_period,
            rank_period,
            overbought: 70.0,
            oversold: 30.0,
        }
    }

    fn rsi(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period + 1 {
            return vec![f64::NAN; n];
        }

        let mut gains = Vec::with_capacity(n - 1);
        let mut losses = Vec::with_capacity(n - 1);

        for i in 1..n {
            let change = data[i] - data[i - 1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }

        let mut result = vec![f64::NAN; period];

        let mut avg_gain: f64 = gains[0..period].iter().sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses[0..period].iter().sum::<f64>() / period as f64;

        let rsi_val = if avg_loss == 0.0 { 100.0 } else { 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) };
        result.push(rsi_val);

        for i in period..(n - 1) {
            avg_gain = (avg_gain * (period - 1) as f64 + gains[i]) / period as f64;
            avg_loss = (avg_loss * (period - 1) as f64 + losses[i]) / period as f64;
            let rsi_val = if avg_loss == 0.0 { 100.0 } else { 100.0 - 100.0 / (1.0 + avg_gain / avg_loss) };
            result.push(rsi_val);
        }

        result
    }

    fn streak(data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < 2 {
            return vec![0.0; n];
        }

        let mut streaks = vec![0.0; n];

        for i in 1..n {
            if data[i] > data[i - 1] {
                if streaks[i - 1] > 0.0 {
                    streaks[i] = streaks[i - 1] + 1.0;
                } else {
                    streaks[i] = 1.0;
                }
            } else if data[i] < data[i - 1] {
                if streaks[i - 1] < 0.0 {
                    streaks[i] = streaks[i - 1] - 1.0;
                } else {
                    streaks[i] = -1.0;
                }
            } else {
                streaks[i] = 0.0;
            }
        }

        streaks
    }

    fn percent_rank(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period + 1 {
            return vec![f64::NAN; n];
        }

        let mut roc = vec![f64::NAN; 1];
        for i in 1..n {
            if data[i - 1] != 0.0 {
                roc.push(((data[i] - data[i - 1]) / data[i - 1]) * 100.0);
            } else {
                roc.push(0.0);
            }
        }

        let mut result = vec![f64::NAN; period];

        for i in period..n {
            let current_roc = roc[i];
            if current_roc.is_nan() {
                result.push(f64::NAN);
                continue;
            }

            let window = &roc[(i - period)..i];
            let count_below = window.iter()
                .filter(|&&x| !x.is_nan() && x < current_roc)
                .count();
            let total_valid = window.iter().filter(|x| !x.is_nan()).count();

            if total_valid > 0 {
                result.push((count_below as f64 / total_valid as f64) * 100.0);
            } else {
                result.push(50.0);
            }
        }

        result
    }

    pub fn calculate(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let min_req = self.rsi_period.max(self.streak_period).max(self.rank_period) + 1;

        if n < min_req {
            return vec![f64::NAN; n];
        }

        // Calculate components
        let rsi = Self::rsi(data, self.rsi_period);
        let streak = Self::streak(data);
        let streak_rsi = Self::rsi(&streak, self.streak_period);
        let pct_rank = Self::percent_rank(data, self.rank_period);

        // Combine: (RSI + Streak RSI + Percent Rank) / 3
        (0..n)
            .map(|i| {
                let r = rsi[i];
                let sr = if i < streak_rsi.len() { streak_rsi[i] } else { f64::NAN };
                let pr = pct_rank[i];

                if r.is_nan() || sr.is_nan() || pr.is_nan() {
                    f64::NAN
                } else {
                    (r + sr + pr) / 3.0
                }
            })
            .collect()
    }
}

impl Default for ConnorsRSI {
    fn default() -> Self {
        Self::new(3, 2, 100)
    }
}

impl TechnicalIndicator for ConnorsRSI {
    fn name(&self) -> &str {
        "ConnorsRSI"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_req = self.rsi_period.max(self.streak_period).max(self.rank_period) + 1;

        if data.close.len() < min_req {
            return Err(IndicatorError::InsufficientData {
                required: min_req,
                got: data.close.len(),
            });
        }

        let values = self.calculate(&data.close);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        self.rsi_period.max(self.streak_period).max(self.rank_period) + 1
    }
}

impl SignalIndicator for ConnorsRSI {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let values = self.calculate(&data.close);
        let last = values.last().copied().unwrap_or(f64::NAN);

        if last.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        if last >= self.overbought {
            Ok(IndicatorSignal::Bearish)
        } else if last <= self.oversold {
            Ok(IndicatorSignal::Bullish)
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.close);
        let signals = values.iter().map(|&val| {
            if val.is_nan() {
                IndicatorSignal::Neutral
            } else if val >= self.overbought {
                IndicatorSignal::Bearish
            } else if val <= self.oversold {
                IndicatorSignal::Bullish
            } else {
                IndicatorSignal::Neutral
            }
        }).collect();
        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connors_rsi_range() {
        let crsi = ConnorsRSI::default();
        let data: Vec<f64> = (0..150).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
        let result = crsi.calculate(&data);

        // ConnorsRSI should be in range [0, 100]
        for val in result.iter() {
            if !val.is_nan() {
                assert!(*val >= 0.0 && *val <= 100.0, "Value {} out of range", val);
            }
        }
    }
}
