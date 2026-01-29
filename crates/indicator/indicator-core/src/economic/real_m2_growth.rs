//! Real M2 Growth Indicator (IND-321)
//!
//! Tracks real (inflation-adjusted) M2 money supply growth as a
//! leading indicator of economic liquidity and future growth.

use indicator_spi::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result,
    SignalIndicator, TechnicalIndicator,
};

/// Configuration for RealM2Growth indicator.
#[derive(Debug, Clone)]
pub struct RealM2GrowthConfig {
    /// Period for YoY growth calculation (default: 12 months)
    pub yoy_period: usize,
    /// Smoothing period
    pub smoothing_period: usize,
    /// Bullish growth threshold (percentage)
    pub bullish_threshold: f64,
    /// Bearish growth threshold (percentage)
    pub bearish_threshold: f64,
    /// Lookback for trend analysis
    pub trend_lookback: usize,
}

impl Default for RealM2GrowthConfig {
    fn default() -> Self {
        Self {
            yoy_period: 12,
            smoothing_period: 3,
            bullish_threshold: 3.0,
            bearish_threshold: -1.0,
            trend_lookback: 6,
        }
    }
}

/// Real M2 Growth Indicator (IND-321)
///
/// Measures the year-over-year growth in real (inflation-adjusted) M2
/// money supply, which is a key indicator of monetary conditions.
///
/// # Interpretation
/// - Positive real M2 growth: Expansionary monetary conditions
/// - Negative real M2 growth: Contractionary monetary conditions
/// - Sharp declines often precede economic slowdowns by 12-18 months
/// - Real M2 is nominal M2 minus inflation rate
///
/// # Example
/// ```ignore
/// let indicator = RealM2Growth::new(RealM2GrowthConfig::default())?;
/// let growth = indicator.calculate_real_growth(&m2_data, &inflation_data);
/// ```
#[derive(Debug, Clone)]
pub struct RealM2Growth {
    config: RealM2GrowthConfig,
}

impl RealM2Growth {
    /// Create a new RealM2Growth indicator.
    pub fn new(config: RealM2GrowthConfig) -> Result<Self> {
        if config.yoy_period < 1 {
            return Err(IndicatorError::InvalidParameter {
                name: "yoy_period".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }
        Ok(Self { config })
    }

    /// Create with default configuration.
    pub fn default_monthly() -> Result<Self> {
        Self::new(RealM2GrowthConfig::default())
    }

    /// Calculate nominal M2 YoY growth.
    pub fn calculate_nominal_growth(&self, m2: &[f64]) -> Vec<f64> {
        let n = m2.len();
        let period = self.config.yoy_period;
        let mut result = vec![f64::NAN; n];

        if n <= period {
            return result;
        }

        for i in period..n {
            if m2[i - period].abs() > 1e-10 {
                result[i] = (m2[i] / m2[i - period] - 1.0) * 100.0;
            }
        }

        result
    }

    /// Calculate real M2 growth (nominal - inflation).
    pub fn calculate_real_growth(&self, m2: &[f64], inflation: &[f64]) -> Vec<f64> {
        let nominal = self.calculate_nominal_growth(m2);
        let n = m2.len().min(inflation.len());
        let mut result = vec![f64::NAN; n];

        for i in 0..n {
            if !nominal[i].is_nan() {
                // Real growth = nominal growth - inflation
                result[i] = nominal[i] - inflation[i];
            }
        }

        result
    }

    /// Calculate real growth using single series (M2 already inflation-adjusted).
    pub fn calculate_growth(&self, real_m2: &[f64]) -> Vec<f64> {
        self.calculate_nominal_growth(real_m2)
    }

    /// Calculate smoothed growth rate.
    pub fn calculate_smoothed_growth(&self, m2: &[f64]) -> Vec<f64> {
        let growth = self.calculate_growth(m2);
        let n = growth.len();
        let period = self.config.smoothing_period;
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        for i in period - 1..n {
            let slice = &growth[i + 1 - period..=i];
            let valid: Vec<f64> = slice.iter().filter(|x| !x.is_nan()).copied().collect();

            if !valid.is_empty() {
                result[i] = valid.iter().sum::<f64>() / valid.len() as f64;
            }
        }

        result
    }

    /// Calculate month-over-month change in growth rate.
    pub fn calculate_growth_momentum(&self, m2: &[f64]) -> Vec<f64> {
        let growth = self.calculate_growth(m2);
        let n = growth.len();
        let mut result = vec![f64::NAN; n];

        for i in 1..n {
            if !growth[i].is_nan() && !growth[i - 1].is_nan() {
                result[i] = growth[i] - growth[i - 1];
            }
        }

        result
    }

    /// Calculate money velocity proxy (growth rate of money turnover).
    pub fn calculate_velocity_proxy(&self, m2: &[f64], gdp: &[f64]) -> Vec<f64> {
        let n = m2.len().min(gdp.len());
        let mut result = vec![f64::NAN; n];

        for i in 0..n {
            if m2[i].abs() > 1e-10 {
                // Velocity = GDP / M2
                result[i] = gdp[i] / m2[i];
            }
        }

        result
    }

    /// Calculate excess money growth (growth vs trend).
    pub fn calculate_excess_growth(&self, m2: &[f64], trend_growth: f64) -> Vec<f64> {
        let growth = self.calculate_growth(m2);
        let n = growth.len();
        let mut result = vec![f64::NAN; n];

        for i in 0..n {
            if !growth[i].is_nan() {
                result[i] = growth[i] - trend_growth;
            }
        }

        result
    }

    /// Calculate z-score of growth rate.
    pub fn calculate_zscore(&self, m2: &[f64], lookback: usize) -> Vec<f64> {
        let growth = self.calculate_growth(m2);
        let n = growth.len();
        let mut result = vec![f64::NAN; n];

        if n < lookback {
            return result;
        }

        for i in lookback - 1..n {
            let slice = &growth[i + 1 - lookback..=i];
            let valid: Vec<f64> = slice.iter().filter(|x| !x.is_nan()).copied().collect();

            if valid.len() < 2 {
                continue;
            }

            let mean = valid.iter().sum::<f64>() / valid.len() as f64;
            let variance = valid.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / valid.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > 1e-10 && !growth[i].is_nan() {
                result[i] = (growth[i] - mean) / std_dev;
            }
        }

        result
    }

    /// Detect liquidity regime.
    pub fn detect_regime(&self, m2: &[f64]) -> Vec<i32> {
        let smoothed = self.calculate_smoothed_growth(m2);
        let momentum = self.calculate_growth_momentum(m2);
        let n = m2.len();
        let mut result = vec![0; n];

        for i in 0..n {
            let growth = if !smoothed[i].is_nan() { smoothed[i] } else { continue };
            let mom = if !momentum[i].is_nan() { momentum[i] } else { 0.0 };

            // Abundant liquidity: high growth + accelerating
            if growth > self.config.bullish_threshold && mom > 0.0 {
                result[i] = 2;
            }
            // Adequate liquidity: positive growth
            else if growth > 0.0 {
                result[i] = 1;
            }
            // Tight liquidity: negative growth + decelerating
            else if growth < self.config.bearish_threshold && mom < 0.0 {
                result[i] = -2;
            }
            // Moderately tight: negative growth
            else if growth < 0.0 {
                result[i] = -1;
            }
        }

        result
    }

    /// Generate monetary policy signal.
    pub fn policy_signal(&self, m2: &[f64]) -> Vec<i32> {
        let smoothed = self.calculate_smoothed_growth(m2);
        let n = m2.len();
        let lookback = self.config.trend_lookback;
        let mut result = vec![0; n];

        if n < lookback {
            return result;
        }

        for i in lookback - 1..n {
            let current = if !smoothed[i].is_nan() { smoothed[i] } else { continue };

            // Compare to recent average
            let slice = &smoothed[i + 1 - lookback..=i];
            let valid: Vec<f64> = slice.iter().filter(|x| !x.is_nan()).copied().collect();

            if valid.is_empty() {
                continue;
            }

            let avg = valid.iter().sum::<f64>() / valid.len() as f64;

            // Policy appears accommodative if growth above threshold and rising
            if current > self.config.bullish_threshold && current > avg {
                result[i] = 1; // Accommodative
            }
            // Policy appears restrictive if growth below threshold and falling
            else if current < self.config.bearish_threshold && current < avg {
                result[i] = -1; // Restrictive
            }
        }

        result
    }
}

impl TechnicalIndicator for RealM2Growth {
    fn name(&self) -> &str {
        "RealM2Growth"
    }

    fn min_periods(&self) -> usize {
        self.config.yoy_period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < self.min_periods() {
            return Err(IndicatorError::InsufficientData {
                required: self.min_periods(),
                got: data.close.len(),
            });
        }
        Ok(IndicatorOutput::single(self.calculate_growth(&data.close)))
    }
}

impl SignalIndicator for RealM2Growth {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let regime = self.detect_regime(&data.close);

        if let Some(&last) = regime.last() {
            match last {
                1 | 2 => Ok(IndicatorSignal::Bullish),
                -1 | -2 => Ok(IndicatorSignal::Bearish),
                _ => Ok(IndicatorSignal::Neutral),
            }
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let regime = self.detect_regime(&data.close);
        Ok(regime
            .iter()
            .map(|&r| match r {
                1 | 2 => IndicatorSignal::Bullish,
                -1 | -2 => IndicatorSignal::Bearish,
                _ => IndicatorSignal::Neutral,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_m2_data() -> Vec<f64> {
        // Simulated M2 money supply (trillions)
        let mut data = vec![20.0];
        for i in 1..36 {
            // Growing at roughly 5% annually with some noise
            let growth = 1.0 + (0.05 / 12.0) + (i as f64 * 0.0001).sin() * 0.002;
            data.push(data[i - 1] * growth);
        }
        data
    }

    fn create_test_inflation_data() -> Vec<f64> {
        // Simulated YoY inflation rate (percentage)
        vec![
            2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9,
            1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
            3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.4, 3.3, 3.2, 3.1, 3.0, 2.9,
        ]
    }

    #[test]
    fn test_nominal_growth() {
        let m2 = create_test_m2_data();
        let indicator = RealM2Growth::default_monthly().unwrap();
        let growth = indicator.calculate_nominal_growth(&m2);

        assert_eq!(growth.len(), m2.len());

        // First 12 values should be NaN
        for i in 0..12 {
            assert!(growth[i].is_nan());
        }

        // Growth should be positive (we set ~5% annual)
        assert!(growth[12] > 0.0);
    }

    #[test]
    fn test_real_growth() {
        let m2 = create_test_m2_data();
        let inflation = create_test_inflation_data();
        let indicator = RealM2Growth::default_monthly().unwrap();
        let real = indicator.calculate_real_growth(&m2, &inflation);

        assert_eq!(real.len(), m2.len());

        // Real growth = nominal - inflation
        // Should still have some positive values
        let valid: Vec<f64> = real.iter().filter(|x| !x.is_nan()).copied().collect();
        assert!(!valid.is_empty());
    }

    #[test]
    fn test_smoothed_growth() {
        let m2 = create_test_m2_data();
        let indicator = RealM2Growth::default_monthly().unwrap();
        let smoothed = indicator.calculate_smoothed_growth(&m2);

        assert_eq!(smoothed.len(), m2.len());

        // Should have values after YoY + smoothing warmup
        assert!(!smoothed[14].is_nan());
    }

    #[test]
    fn test_growth_momentum() {
        let m2 = create_test_m2_data();
        let indicator = RealM2Growth::default_monthly().unwrap();
        let momentum = indicator.calculate_growth_momentum(&m2);

        assert_eq!(momentum.len(), m2.len());

        // Momentum should exist after YoY warmup + 1
        assert!(!momentum[13].is_nan());
    }

    #[test]
    fn test_excess_growth() {
        let m2 = create_test_m2_data();
        let indicator = RealM2Growth::default_monthly().unwrap();
        let excess = indicator.calculate_excess_growth(&m2, 5.0); // 5% trend

        assert_eq!(excess.len(), m2.len());

        // Excess growth = actual - trend
        for e in excess.iter().skip(12) {
            if !e.is_nan() {
                // Values should be reasonable
                assert!(e.abs() < 20.0);
            }
        }
    }

    #[test]
    fn test_zscore() {
        let m2 = create_test_m2_data();
        let indicator = RealM2Growth::default_monthly().unwrap();
        let zscore = indicator.calculate_zscore(&m2, 12);

        assert_eq!(zscore.len(), m2.len());

        // Z-score should be reasonable
        for z in zscore.iter().skip(23) {
            if !z.is_nan() {
                assert!(z.abs() < 5.0);
            }
        }
    }

    #[test]
    fn test_detect_regime() {
        let m2 = create_test_m2_data();
        let indicator = RealM2Growth::default_monthly().unwrap();
        let regime = indicator.detect_regime(&m2);

        assert_eq!(regime.len(), m2.len());

        // Regime values should be -2, -1, 0, 1, or 2
        for r in regime.iter() {
            assert!(*r >= -2 && *r <= 2);
        }
    }

    #[test]
    fn test_policy_signal() {
        let m2 = create_test_m2_data();
        let indicator = RealM2Growth::default_monthly().unwrap();
        let policy = indicator.policy_signal(&m2);

        assert_eq!(policy.len(), m2.len());

        // Policy values should be -1, 0, or 1
        for p in policy.iter() {
            assert!(*p >= -1 && *p <= 1);
        }
    }

    #[test]
    fn test_technical_indicator_impl() {
        let indicator = RealM2Growth::default_monthly().unwrap();
        let data = OHLCVSeries::from_close(create_test_m2_data());
        let result = indicator.compute(&data);

        assert!(result.is_ok());
    }

    #[test]
    fn test_signal_indicator_impl() {
        let indicator = RealM2Growth::default_monthly().unwrap();
        let data = OHLCVSeries::from_close(create_test_m2_data());
        let signals = indicator.signals(&data);

        assert!(signals.is_ok());
        assert_eq!(signals.unwrap().len(), data.close.len());
    }

    #[test]
    fn test_invalid_period() {
        let config = RealM2GrowthConfig {
            yoy_period: 0,
            ..Default::default()
        };
        let result = RealM2Growth::new(config);
        assert!(result.is_err());
    }
}
