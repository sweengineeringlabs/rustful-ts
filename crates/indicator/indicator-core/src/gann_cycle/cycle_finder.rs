//! Cycle Finder Indicator (IND-328)
//!
//! Detects dominant cycles in price data using spectral analysis techniques.
//! Identifies the primary cyclical periods that drive price movements.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Dominant cycle information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominantCycle {
    /// Cycle period in bars
    pub period: f64,
    /// Cycle amplitude (strength)
    pub amplitude: f64,
    /// Current phase (0-360 degrees)
    pub phase: f64,
    /// Confidence level (0-1)
    pub confidence: f64,
}

/// Cycle Finder output structure
#[derive(Debug, Clone)]
pub struct CycleFinderOutput {
    /// Primary dominant cycle period
    pub dominant_period: Vec<f64>,
    /// Secondary cycle period
    pub secondary_period: Vec<f64>,
    /// Dominant cycle amplitude
    pub amplitude: Vec<f64>,
    /// Current phase of dominant cycle
    pub phase: Vec<f64>,
    /// Cycle mode indicator (-1 to 1, from trough to peak)
    pub cycle_mode: Vec<f64>,
    /// Trend/cycle decomposition - trend component
    pub trend_component: Vec<f64>,
    /// Trend/cycle decomposition - cycle component
    pub cycle_component: Vec<f64>,
    /// Cycle confidence score
    pub confidence: Vec<f64>,
}

/// Cycle Finder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleFinderConfig {
    /// Minimum cycle period to detect
    pub min_period: usize,
    /// Maximum cycle period to detect
    pub max_period: usize,
    /// Number of top cycles to track
    pub num_cycles: usize,
    /// Smoothing period for output
    pub smooth_period: usize,
    /// Band-pass filter bandwidth
    pub bandwidth: f64,
}

impl Default for CycleFinderConfig {
    fn default() -> Self {
        Self {
            min_period: 8,
            max_period: 50,
            num_cycles: 3,
            smooth_period: 3,
            bandwidth: 0.3,
        }
    }
}

/// Cycle Finder Indicator
///
/// Detects and tracks dominant market cycles using autocorrelation
/// and spectral analysis techniques.
///
/// # Method
/// 1. Detrend the price data
/// 2. Calculate autocorrelation at multiple lags
/// 3. Identify peaks in autocorrelation (cycle periods)
/// 4. Extract amplitude and phase for each cycle
///
/// # Output
/// - Dominant period: Primary cycle length in bars
/// - Amplitude: Strength of the cycle
/// - Phase: Current position in cycle (0-360 degrees)
/// - Cycle mode: Position from trough (-1) to peak (+1)
///
/// # Trading Rules
/// - Buy near cycle troughs (cycle_mode near -1)
/// - Sell near cycle peaks (cycle_mode near +1)
/// - Higher confidence = more reliable cycles
#[derive(Debug, Clone)]
pub struct CycleFinder {
    config: CycleFinderConfig,
}

impl CycleFinder {
    pub fn new() -> Self {
        Self {
            config: CycleFinderConfig::default(),
        }
    }

    pub fn with_config(config: CycleFinderConfig) -> Self {
        Self { config }
    }

    pub fn with_period_range(mut self, min: usize, max: usize) -> Self {
        self.config.min_period = min;
        self.config.max_period = max;
        self
    }

    /// Calculate autocorrelation at a specific lag
    fn autocorrelation(&self, data: &[f64], lag: usize) -> f64 {
        let n = data.len();
        if lag >= n {
            return 0.0;
        }

        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        if variance < 1e-10 {
            return 0.0;
        }

        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += (data[i] - mean) * (data[i + lag] - mean);
        }

        sum / ((n - lag) as f64 * variance)
    }

    /// Find dominant cycle using autocorrelation
    fn find_dominant_cycle(&self, data: &[f64]) -> DominantCycle {
        let mut best_period = self.config.min_period as f64;
        let mut best_autocorr = -1.0;

        // Scan for peak autocorrelation
        for period in self.config.min_period..=self.config.max_period {
            let ac = self.autocorrelation(data, period);
            if ac > best_autocorr {
                best_autocorr = ac;
                best_period = period as f64;
            }
        }

        // Estimate amplitude using the autocorrelation value
        let amplitude = best_autocorr.max(0.0);

        // Estimate phase by finding position in cycle
        let phase = self.estimate_phase(data, best_period as usize);

        // Confidence based on autocorrelation strength
        let confidence = (best_autocorr + 1.0) / 2.0;

        DominantCycle {
            period: best_period,
            amplitude,
            phase,
            confidence,
        }
    }

    /// Estimate current phase in cycle
    fn estimate_phase(&self, data: &[f64], period: usize) -> f64 {
        let n = data.len();
        if n < period {
            return 0.0;
        }

        // Use simplified phase estimation
        // Compare current value to recent min/max
        let recent = &data[n.saturating_sub(period)..];
        let min = recent.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = recent.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        if range < 1e-10 {
            return 0.0;
        }

        let current = data[n - 1];
        let normalized = (current - min) / range;

        // Find if we're rising or falling
        let is_rising = if n > 1 { data[n - 1] > data[n - 2] } else { true };

        // Convert to phase (0-360)
        if is_rising {
            normalized * 180.0
        } else {
            180.0 + (1.0 - normalized) * 180.0
        }
    }

    /// Detrend data using simple moving average
    fn detrend(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut detrended = vec![0.0; n];

        for i in 0..n {
            let start = i.saturating_sub(period / 2);
            let end = (i + period / 2 + 1).min(n);
            let trend: f64 = data[start..end].iter().sum::<f64>() / (end - start) as f64;
            detrended[i] = data[i] - trend;
        }

        detrended
    }

    /// Calculate band-pass filtered cycle component
    fn bandpass_filter(&self, data: &[f64], period: f64) -> Vec<f64> {
        let n = data.len();
        let mut filtered = vec![0.0; n];

        let bandwidth = self.config.bandwidth;
        let delta = 2.0 * PI / period;
        let beta = (1.0 - bandwidth * delta.sin()) / (1.0 + bandwidth * delta.sin()).cos();
        let gamma = (1.0 + beta) * delta.cos();
        let alpha = (1.0 - beta) / 2.0;

        for i in 2..n {
            filtered[i] = alpha * (data[i] - data[i - 2])
                + gamma * filtered[i - 1]
                - beta * filtered[i - 2];
        }

        filtered
    }

    /// Calculate Cycle Finder from OHLCV data
    pub fn calculate(&self, data: &OHLCVSeries) -> CycleFinderOutput {
        let n = data.close.len();
        let prices = &data.close;

        let mut dominant_period = vec![f64::NAN; n];
        let mut secondary_period = vec![f64::NAN; n];
        let mut amplitude = vec![f64::NAN; n];
        let mut phase = vec![f64::NAN; n];
        let mut cycle_mode = vec![f64::NAN; n];
        let mut trend_component = vec![f64::NAN; n];
        let mut cycle_component = vec![f64::NAN; n];
        let mut confidence = vec![f64::NAN; n];

        // Need minimum data for cycle detection
        let min_data = self.config.max_period * 2;

        for i in min_data..n {
            // Use rolling window for cycle detection
            let window = &prices[i.saturating_sub(self.config.max_period * 2)..=i];

            // Detrend the data
            let detrended = self.detrend(window, self.config.max_period);

            // Find dominant cycle
            let cycle = self.find_dominant_cycle(&detrended);

            dominant_period[i] = cycle.period;
            amplitude[i] = cycle.amplitude;
            phase[i] = cycle.phase;
            confidence[i] = cycle.confidence;

            // Calculate cycle mode (-1 to 1)
            // 0 phase = trough start, 90 = rising, 180 = peak, 270 = falling
            let phase_rad = cycle.phase.to_radians();
            cycle_mode[i] = phase_rad.sin();

            // Trend/cycle decomposition
            let trend_period = self.config.max_period;
            let start = i.saturating_sub(trend_period / 2);
            let end = (i + 1).min(n);
            trend_component[i] = prices[start..end].iter().sum::<f64>() / (end - start) as f64;
            cycle_component[i] = prices[i] - trend_component[i];

            // Find secondary cycle
            // Simple approach: look for next best autocorrelation peak
            let primary = cycle.period as usize;
            let mut best_secondary = 0.0;
            let mut secondary_ac = -1.0;

            for period in self.config.min_period..=self.config.max_period {
                // Skip periods close to primary
                if (period as i32 - primary as i32).abs() < 3 {
                    continue;
                }

                let ac = self.autocorrelation(&detrended, period);
                if ac > secondary_ac {
                    secondary_ac = ac;
                    best_secondary = period as f64;
                }
            }

            if best_secondary > 0.0 {
                secondary_period[i] = best_secondary;
            }
        }

        // Smooth outputs
        let smooth = self.config.smooth_period;
        if smooth > 1 {
            dominant_period = self.smooth_series(&dominant_period, smooth);
            amplitude = self.smooth_series(&amplitude, smooth);
            confidence = self.smooth_series(&confidence, smooth);
        }

        CycleFinderOutput {
            dominant_period,
            secondary_period,
            amplitude,
            phase,
            cycle_mode,
            trend_component,
            cycle_component,
            confidence,
        }
    }

    /// Simple moving average smoothing
    fn smooth_series(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut smoothed = vec![f64::NAN; n];

        for i in (period - 1)..n {
            let mut sum = 0.0;
            let mut count = 0;

            for j in 0..period {
                let val = data[i - j];
                if !val.is_nan() {
                    sum += val;
                    count += 1;
                }
            }

            if count > 0 {
                smoothed[i] = sum / count as f64;
            }
        }

        smoothed
    }
}

impl Default for CycleFinder {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for CycleFinder {
    fn name(&self) -> &str {
        "Cycle Finder"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.config.max_period * 2 + 1;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let result = self.calculate(data);

        // Primary: dominant_period, Secondary: cycle_mode, Tertiary: confidence
        Ok(IndicatorOutput::triple(
            result.dominant_period,
            result.cycle_mode,
            result.confidence,
        ))
    }

    fn min_periods(&self) -> usize {
        self.config.max_period * 2 + 1
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for CycleFinder {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let result = self.calculate(data);
        let n = result.cycle_mode.len();

        if n < 2 {
            return Ok(IndicatorSignal::Neutral);
        }

        let mode = result.cycle_mode[n - 1];
        let prev_mode = result.cycle_mode[n - 2];
        let conf = result.confidence[n - 1];

        if mode.is_nan() || prev_mode.is_nan() || conf.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Signal on cycle mode crossovers with sufficient confidence
        if conf > 0.5 {
            // Crossing up from trough
            if mode > -0.5 && prev_mode <= -0.5 {
                Ok(IndicatorSignal::Bullish)
            }
            // Crossing down from peak
            else if mode < 0.5 && prev_mode >= 0.5 {
                Ok(IndicatorSignal::Bearish)
            } else {
                Ok(IndicatorSignal::Neutral)
            }
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let result = self.calculate(data);
        let n = result.cycle_mode.len();

        let mut signals = vec![IndicatorSignal::Neutral; n];

        for i in 1..n {
            let mode = result.cycle_mode[i];
            let prev_mode = result.cycle_mode[i - 1];
            let conf = result.confidence[i];

            if mode.is_nan() || prev_mode.is_nan() || conf.is_nan() {
                continue;
            }

            if conf > 0.5 {
                if mode > -0.5 && prev_mode <= -0.5 {
                    signals[i] = IndicatorSignal::Bullish;
                } else if mode < 0.5 && prev_mode >= 0.5 {
                    signals[i] = IndicatorSignal::Bearish;
                }
            }
        }

        Ok(signals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_cyclic_data(bars: usize, period: usize) -> OHLCVSeries {
        let mut closes = Vec::with_capacity(bars);
        for i in 0..bars {
            // Create clear cycle with some noise
            let cycle = (2.0 * PI * i as f64 / period as f64).sin() * 10.0;
            closes.push(100.0 + cycle + (i as f64 * 0.1)); // Add slight trend
        }

        OHLCVSeries {
            open: closes.iter().map(|c| c - 0.5).collect(),
            high: closes.iter().map(|c| c + 2.0).collect(),
            low: closes.iter().map(|c| c - 2.0).collect(),
            close: closes,
            volume: vec![1000.0; bars],
        }
    }

    #[test]
    fn test_cycle_finder_initialization() {
        let cf = CycleFinder::new();
        assert_eq!(cf.name(), "Cycle Finder");
        assert_eq!(cf.output_features(), 3);
    }

    #[test]
    fn test_cycle_detection() {
        // Create data with a 20-bar cycle
        let data = create_cyclic_data(200, 20);
        let cf = CycleFinder::new().with_period_range(10, 40);
        let result = cf.calculate(&data);

        // Should detect a period near 20
        let detected_period = result.dominant_period[199];
        if !detected_period.is_nan() {
            assert!(detected_period >= 15.0 && detected_period <= 25.0,
                "Detected period {} not near expected 20", detected_period);
        }
    }

    #[test]
    fn test_autocorrelation() {
        let cf = CycleFinder::new();

        // Perfect periodic data should have high autocorrelation at period
        let period = 20;
        let data: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * i as f64 / period as f64).sin())
            .collect();

        let ac_at_period = cf.autocorrelation(&data, period);
        let ac_at_half = cf.autocorrelation(&data, period / 2);

        // Autocorrelation at period should be higher than at half period
        assert!(ac_at_period > ac_at_half);
    }

    #[test]
    fn test_cycle_mode_bounds() {
        let data = create_cyclic_data(200, 20);
        let cf = CycleFinder::new();
        let result = cf.calculate(&data);

        // Cycle mode should be bounded between -1 and 1
        for i in 0..200 {
            if !result.cycle_mode[i].is_nan() {
                assert!(result.cycle_mode[i] >= -1.0 && result.cycle_mode[i] <= 1.0);
            }
        }
    }

    #[test]
    fn test_cycle_finder_compute() {
        let data = create_cyclic_data(150, 20);
        let cf = CycleFinder::new();
        let output = cf.compute(&data).unwrap();

        assert_eq!(output.primary.len(), 150);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_cycle_finder_signals() {
        let data = create_cyclic_data(150, 20);
        let cf = CycleFinder::new();
        let signals = cf.signals(&data).unwrap();

        assert_eq!(signals.len(), 150);
    }

    #[test]
    fn test_insufficient_data() {
        let data = OHLCVSeries {
            open: vec![100.0; 50],
            high: vec![102.0; 50],
            low: vec![98.0; 50],
            close: vec![100.0; 50],
            volume: vec![1000.0; 50],
        };

        let cf = CycleFinder::new();
        let result = cf.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_trend_cycle_decomposition() {
        let data = create_cyclic_data(200, 20);
        let cf = CycleFinder::new();
        let result = cf.calculate(&data);

        // Trend + cycle should approximately equal original price
        for i in 100..200 {
            if !result.trend_component[i].is_nan() && !result.cycle_component[i].is_nan() {
                let reconstructed = result.trend_component[i] + result.cycle_component[i];
                let diff = (reconstructed - data.close[i]).abs();
                assert!(diff < 1.0, "Decomposition error at {}: {} vs {}", i, reconstructed, data.close[i]);
            }
        }
    }

    #[test]
    fn test_confidence_bounds() {
        let data = create_cyclic_data(200, 20);
        let cf = CycleFinder::new();
        let result = cf.calculate(&data);

        for i in 0..200 {
            if !result.confidence[i].is_nan() {
                assert!(result.confidence[i] >= 0.0 && result.confidence[i] <= 1.0);
            }
        }
    }
}
