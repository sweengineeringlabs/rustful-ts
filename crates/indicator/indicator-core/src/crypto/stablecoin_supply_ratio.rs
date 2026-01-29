//! Stablecoin Supply Ratio (SSR) - IND-276
//!
//! Measures Bitcoin's market cap relative to stablecoin market cap.
//! Indicates buying power available in stablecoins to purchase Bitcoin.
//!
//! SSR = BTC Market Cap / Stablecoin Market Cap
//!
//! Interpretation:
//! - Low SSR: High stablecoin supply relative to BTC = Potential buying pressure
//! - High SSR: Low stablecoin supply relative to BTC = Limited buying power
//! - Declining SSR: Growing stablecoin supply = Bullish (dry powder accumulating)

use indicator_spi::IndicatorSignal;

/// Stablecoin Supply Ratio output.
#[derive(Debug, Clone)]
pub struct StablecoinSupplyRatioOutput {
    /// Raw SSR values.
    pub ssr: Vec<f64>,
    /// SSR oscillator (deviation from moving average).
    pub ssr_oscillator: Vec<f64>,
    /// SSR rate of change.
    pub ssr_roc: Vec<f64>,
}

/// SSR signal interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SSRSignal {
    /// Very low SSR - strong buying power available.
    StrongBuyingPower,
    /// Low SSR - moderate buying power.
    ModerateBuyingPower,
    /// Neutral SSR.
    Neutral,
    /// High SSR - limited buying power.
    LimitedBuyingPower,
    /// Very high SSR - very limited buying power.
    VeryLimitedBuyingPower,
}

/// Stablecoin Supply Ratio (SSR) - IND-276
///
/// Compares Bitcoin market cap to total stablecoin market cap.
///
/// # Formula
/// ```text
/// SSR = BTC Market Cap / Stablecoin Market Cap
/// SSR Oscillator = SSR / SMA(SSR) - 1
/// ```
///
/// # Example
/// ```
/// use indicator_core::crypto::StablecoinSupplyRatio;
///
/// let ssr = StablecoinSupplyRatio::new(30);
/// let btc_caps = vec![800e9, 850e9, 900e9]; // BTC market caps
/// let stable_caps = vec![150e9, 155e9, 160e9]; // Stablecoin caps
/// let output = ssr.calculate(&btc_caps, &stable_caps);
/// ```
#[derive(Debug, Clone)]
pub struct StablecoinSupplyRatio {
    /// Period for moving average calculation.
    ma_period: usize,
    /// Period for rate of change.
    roc_period: usize,
    /// Low SSR threshold (bullish).
    low_threshold: f64,
    /// High SSR threshold (bearish).
    high_threshold: f64,
}

impl StablecoinSupplyRatio {
    /// Create a new SSR indicator.
    pub fn new(ma_period: usize) -> Self {
        Self {
            ma_period,
            roc_period: 7,
            low_threshold: 4.0,
            high_threshold: 8.0,
        }
    }

    /// Create with custom parameters.
    pub fn with_params(
        ma_period: usize,
        roc_period: usize,
        low_threshold: f64,
        high_threshold: f64,
    ) -> Self {
        Self {
            ma_period,
            roc_period,
            low_threshold,
            high_threshold,
        }
    }

    /// Calculate SSR from market cap series.
    pub fn calculate(&self, btc_caps: &[f64], stablecoin_caps: &[f64]) -> StablecoinSupplyRatioOutput {
        let n = btc_caps.len().min(stablecoin_caps.len());

        if n == 0 {
            return StablecoinSupplyRatioOutput {
                ssr: vec![],
                ssr_oscillator: vec![],
                ssr_roc: vec![],
            };
        }

        // Calculate raw SSR
        let ssr: Vec<f64> = (0..n)
            .map(|i| {
                if stablecoin_caps[i] > 0.0 {
                    btc_caps[i] / stablecoin_caps[i]
                } else {
                    f64::NAN
                }
            })
            .collect();

        // Calculate SSR oscillator (deviation from MA)
        let ssr_oscillator = self.calculate_oscillator(&ssr);

        // Calculate rate of change
        let ssr_roc = self.calculate_roc(&ssr);

        StablecoinSupplyRatioOutput {
            ssr,
            ssr_oscillator,
            ssr_roc,
        }
    }

    /// Calculate oscillator (deviation from moving average).
    fn calculate_oscillator(&self, ssr: &[f64]) -> Vec<f64> {
        let n = ssr.len();
        let mut result = vec![f64::NAN; n];

        if n < self.ma_period {
            return result;
        }

        for i in (self.ma_period - 1)..n {
            let start = i + 1 - self.ma_period;
            let mut sum = 0.0;
            let mut count = 0;

            for j in start..=i {
                if !ssr[j].is_nan() {
                    sum += ssr[j];
                    count += 1;
                }
            }

            if count > 0 && !ssr[i].is_nan() {
                let ma = sum / count as f64;
                if ma > 1e-10 {
                    result[i] = (ssr[i] / ma - 1.0) * 100.0;
                }
            }
        }

        result
    }

    /// Calculate rate of change.
    fn calculate_roc(&self, ssr: &[f64]) -> Vec<f64> {
        let n = ssr.len();
        let mut result = vec![f64::NAN; n];

        if n <= self.roc_period {
            return result;
        }

        for i in self.roc_period..n {
            let prev = ssr[i - self.roc_period];
            let curr = ssr[i];
            if !prev.is_nan() && !curr.is_nan() && prev > 1e-10 {
                result[i] = (curr / prev - 1.0) * 100.0;
            }
        }

        result
    }

    /// Get signal interpretation for an SSR value.
    pub fn interpret(&self, ssr_value: f64) -> SSRSignal {
        if ssr_value.is_nan() {
            SSRSignal::Neutral
        } else if ssr_value < self.low_threshold * 0.7 {
            SSRSignal::StrongBuyingPower
        } else if ssr_value < self.low_threshold {
            SSRSignal::ModerateBuyingPower
        } else if ssr_value > self.high_threshold * 1.3 {
            SSRSignal::VeryLimitedBuyingPower
        } else if ssr_value > self.high_threshold {
            SSRSignal::LimitedBuyingPower
        } else {
            SSRSignal::Neutral
        }
    }

    /// Interpret SSR with rate of change for trend confirmation.
    pub fn interpret_with_trend(&self, ssr_value: f64, roc: f64) -> SSRSignal {
        let base_signal = self.interpret(ssr_value);

        // Adjust based on trend
        if roc.is_nan() {
            return base_signal;
        }

        match base_signal {
            SSRSignal::Neutral => {
                if roc < -5.0 {
                    // SSR declining = more buying power
                    SSRSignal::ModerateBuyingPower
                } else if roc > 5.0 {
                    // SSR rising = less buying power
                    SSRSignal::LimitedBuyingPower
                } else {
                    SSRSignal::Neutral
                }
            }
            _ => base_signal,
        }
    }

    /// Convert SSR signal to trading signal.
    pub fn to_indicator_signal(&self, ssr_signal: SSRSignal) -> IndicatorSignal {
        match ssr_signal {
            SSRSignal::StrongBuyingPower => IndicatorSignal::Bullish,
            SSRSignal::ModerateBuyingPower => IndicatorSignal::Bullish,
            SSRSignal::Neutral => IndicatorSignal::Neutral,
            SSRSignal::LimitedBuyingPower => IndicatorSignal::Bearish,
            SSRSignal::VeryLimitedBuyingPower => IndicatorSignal::Bearish,
        }
    }

    /// Calculate buying power index (inverse of SSR, normalized).
    pub fn buying_power_index(&self, ssr: &[f64]) -> Vec<f64> {
        ssr.iter()
            .map(|&s| {
                if s.is_nan() || s <= 0.0 {
                    f64::NAN
                } else {
                    // Inverse SSR, normalized to 0-100 scale
                    // Typical SSR range: 2-15
                    let inverse = 1.0 / s;
                    (inverse * 50.0).min(100.0)
                }
            })
            .collect()
    }
}

impl Default for StablecoinSupplyRatio {
    fn default() -> Self {
        Self::new(30)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssr_basic() {
        let ssr = StablecoinSupplyRatio::new(10);
        let btc_caps = vec![800e9; 20]; // $800B
        let stable_caps = vec![100e9; 20]; // $100B
        // SSR = 8.0

        let output = ssr.calculate(&btc_caps, &stable_caps);

        assert_eq!(output.ssr.len(), 20);
        assert!((output.ssr[0] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_ssr_varying() {
        let ssr = StablecoinSupplyRatio::new(5);
        let btc_caps: Vec<f64> = (0..20).map(|i| 800e9 + (i as f64 * 10e9)).collect();
        let stable_caps = vec![100e9; 20];

        let output = ssr.calculate(&btc_caps, &stable_caps);

        // SSR should increase as BTC cap increases
        assert!(output.ssr[19] > output.ssr[0]);
    }

    #[test]
    fn test_ssr_interpretation() {
        let ssr = StablecoinSupplyRatio::default();

        assert_eq!(ssr.interpret(2.0), SSRSignal::StrongBuyingPower);
        assert_eq!(ssr.interpret(3.5), SSRSignal::ModerateBuyingPower);
        assert_eq!(ssr.interpret(6.0), SSRSignal::Neutral);
        assert_eq!(ssr.interpret(9.0), SSRSignal::LimitedBuyingPower);
        assert_eq!(ssr.interpret(12.0), SSRSignal::VeryLimitedBuyingPower);
    }

    #[test]
    fn test_ssr_oscillator() {
        let ssr_ind = StablecoinSupplyRatio::new(5);
        let btc_caps: Vec<f64> = (0..20).map(|i| 800e9 + (i as f64 * 20e9)).collect();
        let stable_caps = vec![100e9; 20];

        let output = ssr_ind.calculate(&btc_caps, &stable_caps);

        // Oscillator should be valid after warmup
        assert!(!output.ssr_oscillator[10].is_nan());
        // Rising SSR = positive oscillator
        assert!(output.ssr_oscillator[15] > 0.0);
    }

    #[test]
    fn test_ssr_roc() {
        let ssr_ind = StablecoinSupplyRatio::new(5);
        let btc_caps: Vec<f64> = (0..20).map(|i| 800e9 + (i as f64 * 20e9)).collect();
        let stable_caps = vec![100e9; 20];

        let output = ssr_ind.calculate(&btc_caps, &stable_caps);

        // ROC should be positive for rising SSR
        assert!(!output.ssr_roc[10].is_nan());
        assert!(output.ssr_roc[10] > 0.0);
    }

    #[test]
    fn test_ssr_zero_stablecoin_cap() {
        let ssr = StablecoinSupplyRatio::default();
        let btc_caps = vec![800e9, 800e9, 800e9];
        let stable_caps = vec![100e9, 0.0, 100e9];

        let output = ssr.calculate(&btc_caps, &stable_caps);

        assert!(!output.ssr[0].is_nan());
        assert!(output.ssr[1].is_nan());
        assert!(!output.ssr[2].is_nan());
    }

    #[test]
    fn test_ssr_empty_input() {
        let ssr = StablecoinSupplyRatio::default();
        let output = ssr.calculate(&[], &[]);

        assert!(output.ssr.is_empty());
        assert!(output.ssr_oscillator.is_empty());
        assert!(output.ssr_roc.is_empty());
    }

    #[test]
    fn test_ssr_signal_conversion() {
        let ssr = StablecoinSupplyRatio::default();

        assert_eq!(
            ssr.to_indicator_signal(SSRSignal::StrongBuyingPower),
            IndicatorSignal::Bullish
        );
        assert_eq!(
            ssr.to_indicator_signal(SSRSignal::VeryLimitedBuyingPower),
            IndicatorSignal::Bearish
        );
        assert_eq!(
            ssr.to_indicator_signal(SSRSignal::Neutral),
            IndicatorSignal::Neutral
        );
    }

    #[test]
    fn test_buying_power_index() {
        let ssr_ind = StablecoinSupplyRatio::default();
        let ssr_values = vec![4.0, 8.0, 10.0];

        let bpi = ssr_ind.buying_power_index(&ssr_values);

        // Lower SSR = higher buying power index
        assert!(bpi[0] > bpi[1]);
        assert!(bpi[1] > bpi[2]);
    }

    #[test]
    fn test_interpret_with_trend() {
        let ssr = StablecoinSupplyRatio::default();

        // Neutral SSR but declining = moderate buying power
        assert_eq!(ssr.interpret_with_trend(6.0, -10.0), SSRSignal::ModerateBuyingPower);
        // Neutral SSR but rising = limited buying power
        assert_eq!(ssr.interpret_with_trend(6.0, 10.0), SSRSignal::LimitedBuyingPower);
    }
}
