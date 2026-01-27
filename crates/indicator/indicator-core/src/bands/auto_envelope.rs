//! Auto Envelope (Elder's Adaptive Envelope) implementation.
//!
//! ATR-based adaptive envelope that adjusts to volatility.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Auto Envelope (Elder's Adaptive Envelope).
///
/// An adaptive envelope indicator developed by Dr. Alexander Elder that uses
/// ATR (Average True Range) to automatically adjust band width based on
/// market volatility.
///
/// - Middle Band: EMA of close
/// - Upper Band: EMA + (ATR * multiplier)
/// - Lower Band: EMA - (ATR * multiplier)
///
/// The adaptive nature makes it more responsive than fixed percentage envelopes,
/// expanding during volatile periods and contracting during quiet markets.
#[derive(Debug, Clone)]
pub struct AutoEnvelope {
    /// Period for the EMA calculation.
    ema_period: usize,
    /// Period for the ATR calculation.
    atr_period: usize,
    /// Multiplier for ATR to determine band width.
    multiplier: f64,
}

impl AutoEnvelope {
    /// Create a new AutoEnvelope indicator.
    ///
    /// # Arguments
    /// * `ema_period` - Period for the EMA (typically 13 or 21)
    /// * `atr_period` - Period for the ATR (typically 14)
    /// * `multiplier` - ATR multiplier (typically 2.0 to 3.0)
    pub fn new(ema_period: usize, atr_period: usize, multiplier: f64) -> Self {
        Self {
            ema_period,
            atr_period,
            multiplier,
        }
    }

    /// Create with default Elder parameters (13-period EMA, 14-period ATR, 2.5 multiplier).
    pub fn elder_default() -> Self {
        Self::new(13, 14, 2.5)
    }

    /// Calculate True Range for each bar.
    fn true_range(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let n = high.len();
        if n == 0 {
            return vec![];
        }

        let mut tr = Vec::with_capacity(n);
        tr.push(high[0] - low[0]);

        for i in 1..n {
            let hl = high[i] - low[i];
            let hc = (high[i] - close[i - 1]).abs();
            let lc = (low[i] - close[i - 1]).abs();
            tr.push(hl.max(hc).max(lc));
        }

        tr
    }

    /// Calculate ATR using Wilder's smoothing.
    fn calculate_atr(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let tr = Self::true_range(high, low, close);
        let n = tr.len();

        if n < self.atr_period || self.atr_period == 0 {
            return vec![f64::NAN; n];
        }

        let mut atr = vec![f64::NAN; self.atr_period - 1];

        // Initial ATR is SMA of first period TRs
        let initial_atr: f64 = tr[0..self.atr_period].iter().sum::<f64>() / self.atr_period as f64;
        atr.push(initial_atr);

        // Smoothed ATR (Wilder's smoothing)
        let mut prev_atr = initial_atr;
        for i in self.atr_period..n {
            let curr_atr = (prev_atr * (self.atr_period - 1) as f64 + tr[i]) / self.atr_period as f64;
            atr.push(curr_atr);
            prev_atr = curr_atr;
        }

        atr
    }

    /// Calculate EMA values.
    fn calculate_ema(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        if n < self.ema_period || self.ema_period == 0 {
            return vec![f64::NAN; n];
        }

        let alpha = 2.0 / (self.ema_period as f64 + 1.0);
        let mut result = vec![f64::NAN; self.ema_period - 1];

        // Initial SMA as seed
        let initial_sma: f64 = data[0..self.ema_period].iter().sum::<f64>() / self.ema_period as f64;
        result.push(initial_sma);

        // EMA calculation
        let mut ema = initial_sma;
        for i in self.ema_period..n {
            ema = alpha * data[i] + (1.0 - alpha) * ema;
            result.push(ema);
        }

        result
    }

    /// Calculate Auto Envelope bands (middle, upper, lower).
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();

        // Calculate EMA of close (middle band)
        let middle = self.calculate_ema(close);

        // Calculate ATR
        let atr_values = self.calculate_atr(high, low, close);

        // Calculate upper and lower bands
        let mut upper = Vec::with_capacity(n);
        let mut lower = Vec::with_capacity(n);

        for i in 0..n {
            if middle[i].is_nan() || atr_values[i].is_nan() {
                upper.push(f64::NAN);
                lower.push(f64::NAN);
            } else {
                let band_offset = self.multiplier * atr_values[i];
                upper.push(middle[i] + band_offset);
                lower.push(middle[i] - band_offset);
            }
        }

        (middle, upper, lower)
    }

    /// Calculate the band width as a percentage of price.
    pub fn band_width_percent(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let (middle, upper, lower) = self.calculate(high, low, close);
        middle.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&m, (&u, &l))| {
                if m.is_nan() || u.is_nan() || l.is_nan() || m.abs() < 1e-10 {
                    f64::NAN
                } else {
                    (u - l) / m
                }
            })
            .collect()
    }

    /// Calculate %B (position within bands).
    pub fn percent_b(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let (_, upper, lower) = self.calculate(high, low, close);
        close.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if u.is_nan() || l.is_nan() || (u - l).abs() < 1e-10 {
                    f64::NAN
                } else {
                    (price - l) / (u - l)
                }
            })
            .collect()
    }
}

impl Default for AutoEnvelope {
    fn default() -> Self {
        Self::elder_default()
    }
}

impl TechnicalIndicator for AutoEnvelope {
    fn name(&self) -> &str {
        "AutoEnvelope"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.ema_period.max(self.atr_period);
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::triple(middle, upper, lower))
    }

    fn min_periods(&self) -> usize {
        self.ema_period.max(self.atr_period)
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for AutoEnvelope {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let (middle, upper, lower) = self.calculate(&data.high, &data.low, &data.close);

        let n = data.close.len();
        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let price = data.close[n - 1];
        let m = middle[n - 1];
        let u = upper[n - 1];
        let l = lower[n - 1];

        if u.is_nan() || l.is_nan() || m.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Price at/below lower band = potential buy (oversold)
        if price <= l {
            Ok(IndicatorSignal::Bullish)
        }
        // Price at/above upper band = potential sell (overbought)
        else if price >= u {
            Ok(IndicatorSignal::Bearish)
        }
        else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let (_, upper, lower) = self.calculate(&data.high, &data.low, &data.close);

        let signals: Vec<_> = data.close.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if u.is_nan() || l.is_nan() {
                    IndicatorSignal::Neutral
                } else if price <= l {
                    IndicatorSignal::Bullish
                } else if price >= u {
                    IndicatorSignal::Bearish
                } else {
                    IndicatorSignal::Neutral
                }
            })
            .collect();

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0).collect();
        (high, low, close)
    }

    #[test]
    fn test_auto_envelope_basic() {
        let ae = AutoEnvelope::new(10, 10, 2.0);
        let (high, low, close) = create_test_data(30);

        let (middle, upper, lower) = ae.calculate(&high, &low, &close);

        assert_eq!(middle.len(), 30);
        assert_eq!(upper.len(), 30);
        assert_eq!(lower.len(), 30);

        // Check warmup period
        for i in 0..9 {
            assert!(middle[i].is_nan());
        }

        // Check bands after warmup
        for i in 9..30 {
            if !middle[i].is_nan() && !upper[i].is_nan() {
                assert!(upper[i] > middle[i], "Upper should be above middle at index {}", i);
                assert!(lower[i] < middle[i], "Lower should be below middle at index {}", i);
            }
        }
    }

    #[test]
    fn test_auto_envelope_default() {
        let ae = AutoEnvelope::default();
        assert_eq!(ae.ema_period, 13);
        assert_eq!(ae.atr_period, 14);
        assert!((ae.multiplier - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_auto_envelope_elder_default() {
        let ae = AutoEnvelope::elder_default();
        assert_eq!(ae.ema_period, 13);
        assert_eq!(ae.atr_period, 14);
        assert!((ae.multiplier - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_auto_envelope_adapts_to_volatility() {
        let ae = AutoEnvelope::new(5, 5, 2.0);

        // Low volatility data
        let high_low_vol: Vec<f64> = (0..20).map(|_| 101.0).collect();
        let low_low_vol: Vec<f64> = (0..20).map(|_| 99.0).collect();
        let close_low_vol: Vec<f64> = (0..20).map(|_| 100.0).collect();

        // High volatility data
        let high_high_vol: Vec<f64> = (0..20).map(|_| 110.0).collect();
        let low_high_vol: Vec<f64> = (0..20).map(|_| 90.0).collect();
        let close_high_vol: Vec<f64> = (0..20).map(|_| 100.0).collect();

        let (_, upper_low, lower_low) = ae.calculate(&high_low_vol, &low_low_vol, &close_low_vol);
        let (_, upper_high, lower_high) = ae.calculate(&high_high_vol, &low_high_vol, &close_high_vol);

        // After warmup, high volatility should have wider bands
        let idx = 15;
        let width_low = upper_low[idx] - lower_low[idx];
        let width_high = upper_high[idx] - lower_high[idx];

        assert!(width_high > width_low, "High volatility should have wider bands");
    }

    #[test]
    fn test_auto_envelope_band_width_percent() {
        let ae = AutoEnvelope::new(10, 10, 2.0);
        let (high, low, close) = create_test_data(30);

        let bw_percent = ae.band_width_percent(&high, &low, &close);

        assert_eq!(bw_percent.len(), 30);

        // After warmup, bandwidth percent should be positive
        for i in 9..30 {
            if !bw_percent[i].is_nan() {
                assert!(bw_percent[i] > 0.0, "Band width percent should be positive");
            }
        }
    }

    #[test]
    fn test_auto_envelope_percent_b() {
        let ae = AutoEnvelope::new(10, 10, 2.0);
        let (high, low, close) = create_test_data(30);

        let percent_b = ae.percent_b(&high, &low, &close);

        assert_eq!(percent_b.len(), 30);

        // After warmup, %B should be finite
        for i in 9..30 {
            if !percent_b[i].is_nan() {
                assert!(percent_b[i].is_finite());
            }
        }
    }

    #[test]
    fn test_auto_envelope_signal_oversold() {
        let ae = AutoEnvelope::new(5, 5, 2.0);

        // Create downtrending data
        let high: Vec<f64> = (0..20).map(|i| 110.0 - i as f64 * 2.0).collect();
        let low: Vec<f64> = (0..20).map(|i| 100.0 - i as f64 * 2.0).collect();
        let close: Vec<f64> = (0..20).map(|i| 105.0 - i as f64 * 2.0).collect();

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 20],
        };

        let signals = ae.signals(&data).unwrap();
        assert_eq!(signals.len(), 20);
    }

    #[test]
    fn test_auto_envelope_signal_overbought() {
        let ae = AutoEnvelope::new(5, 5, 2.0);

        // Create uptrending data
        let high: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 2.0).collect();
        let low: Vec<f64> = (0..20).map(|i| 90.0 + i as f64 * 2.0).collect();
        let close: Vec<f64> = (0..20).map(|i| 95.0 + i as f64 * 2.0).collect();

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 20],
        };

        let signals = ae.signals(&data).unwrap();
        assert_eq!(signals.len(), 20);
    }

    #[test]
    fn test_auto_envelope_compute() {
        let ae = AutoEnvelope::new(10, 10, 2.0);
        let (high, low, close) = create_test_data(30);

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume: vec![1000.0; 30],
        };

        let output = ae.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_auto_envelope_insufficient_data() {
        let ae = AutoEnvelope::new(20, 20, 2.0);

        let data = OHLCVSeries {
            open: vec![100.0; 10],
            high: vec![101.0; 10],
            low: vec![99.0; 10],
            close: vec![100.0; 10],
            volume: vec![1000.0; 10],
        };

        let result = ae.compute(&data);
        assert!(result.is_err());

        match result {
            Err(IndicatorError::InsufficientData { required, got }) => {
                assert_eq!(required, 20);
                assert_eq!(got, 10);
            }
            _ => panic!("Expected InsufficientData error"),
        }
    }

    #[test]
    fn test_auto_envelope_min_periods() {
        let ae = AutoEnvelope::new(13, 14, 2.5);
        assert_eq!(ae.min_periods(), 14);

        let ae2 = AutoEnvelope::new(20, 10, 2.0);
        assert_eq!(ae2.min_periods(), 20);
    }

    #[test]
    fn test_auto_envelope_true_range() {
        let high = vec![105.0, 107.0, 104.0];
        let low = vec![95.0, 97.0, 94.0];
        let close = vec![100.0, 103.0, 98.0];

        let tr = AutoEnvelope::true_range(&high, &low, &close);

        assert_eq!(tr.len(), 3);
        // First TR = high - low = 10
        assert!((tr[0] - 10.0).abs() < 1e-10);
        // Second TR = max(H-L, |H-prevC|, |L-prevC|) = max(10, 7, 3) = 10
        assert!((tr[1] - 10.0).abs() < 1e-10);
        // Third TR = max(H-L, |H-prevC|, |L-prevC|) = max(10, 1, 9) = 10
        assert!((tr[2] - 10.0).abs() < 1e-10);
    }
}
