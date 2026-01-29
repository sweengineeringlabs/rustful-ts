//! Flag and Pennant Pattern Indicator (IND-338)
//!
//! Continuation patterns following strong price moves.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Pattern type for flag and pennant patterns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlagPennantType {
    /// Bull flag (rectangular consolidation after uptrend).
    BullFlag,
    /// Bear flag (rectangular consolidation after downtrend).
    BearFlag,
    /// Bull pennant (triangular consolidation after uptrend).
    BullPennant,
    /// Bear pennant (triangular consolidation after downtrend).
    BearPennant,
    /// No pattern detected.
    None,
}

/// Configuration for Flag and Pennant detection.
#[derive(Debug, Clone)]
pub struct FlagPennantConfig {
    /// Minimum pole length (strong move before consolidation).
    pub min_pole_length: usize,
    /// Maximum pole length.
    pub max_pole_length: usize,
    /// Minimum flag/pennant length.
    pub min_flag_length: usize,
    /// Maximum flag/pennant length.
    pub max_flag_length: usize,
    /// Minimum pole move as percentage.
    pub min_pole_move: f64,
    /// Maximum flag retracement as percentage of pole.
    pub max_retracement: f64,
}

impl Default for FlagPennantConfig {
    fn default() -> Self {
        Self {
            min_pole_length: 5,
            max_pole_length: 20,
            min_flag_length: 5,
            max_flag_length: 15,
            min_pole_move: 0.05,
            max_retracement: 0.50,
        }
    }
}

/// Flag and Pennant indicator for continuation pattern detection.
///
/// Flags and pennants are short-term continuation patterns:
/// - Flag: Rectangular consolidation against the prior trend
/// - Pennant: Triangular consolidation (converging lines)
///
/// Both are preceded by a sharp move (the pole) and signal continuation.
#[derive(Debug, Clone)]
pub struct FlagPennant {
    config: FlagPennantConfig,
}

impl FlagPennant {
    /// Create a new Flag and Pennant indicator with default settings.
    pub fn new() -> Self {
        Self {
            config: FlagPennantConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: FlagPennantConfig) -> Self {
        Self { config }
    }

    /// Create with custom parameters.
    pub fn with_params(
        min_pole_length: usize,
        min_flag_length: usize,
        min_pole_move: f64,
    ) -> Self {
        Self {
            config: FlagPennantConfig {
                min_pole_length,
                min_flag_length,
                min_pole_move,
                ..Default::default()
            },
        }
    }

    /// Detect strong upward move (bull pole).
    fn detect_bull_pole(&self, close: &[f64], end_idx: usize) -> Option<(usize, f64)> {
        for pole_len in self.config.min_pole_length..=self.config.max_pole_length {
            if end_idx < pole_len {
                continue;
            }

            let start_idx = end_idx - pole_len;
            let start_price = close[start_idx];
            let end_price = close[end_idx];

            let move_pct = (end_price - start_price) / start_price;

            if move_pct >= self.config.min_pole_move {
                // Check if it's a mostly upward move
                let mut up_bars = 0;
                for i in start_idx..end_idx {
                    if close[i + 1] > close[i] {
                        up_bars += 1;
                    }
                }

                if up_bars as f64 / pole_len as f64 >= 0.6 {
                    return Some((start_idx, move_pct));
                }
            }
        }
        None
    }

    /// Detect strong downward move (bear pole).
    fn detect_bear_pole(&self, close: &[f64], end_idx: usize) -> Option<(usize, f64)> {
        for pole_len in self.config.min_pole_length..=self.config.max_pole_length {
            if end_idx < pole_len {
                continue;
            }

            let start_idx = end_idx - pole_len;
            let start_price = close[start_idx];
            let end_price = close[end_idx];

            let move_pct = (start_price - end_price) / start_price;

            if move_pct >= self.config.min_pole_move {
                // Check if it's a mostly downward move
                let mut down_bars = 0;
                for i in start_idx..end_idx {
                    if close[i + 1] < close[i] {
                        down_bars += 1;
                    }
                }

                if down_bars as f64 / pole_len as f64 >= 0.6 {
                    return Some((start_idx, move_pct));
                }
            }
        }
        None
    }

    /// Check if consolidation is rectangular (flag) or triangular (pennant).
    fn is_rectangular(&self, high: &[f64], low: &[f64], start: usize, end: usize) -> bool {
        if end <= start + 2 || end > high.len() {
            return false;
        }

        let initial_range = high[start] - low[start];
        let final_range = high[end - 1] - low[end - 1];

        // Rectangular if range doesn't change much
        if initial_range > 0.0 {
            let range_change = (final_range - initial_range).abs() / initial_range;
            range_change < 0.3
        } else {
            false
        }
    }

    /// Check if consolidation is triangular (converging).
    fn is_triangular(&self, high: &[f64], low: &[f64], start: usize, end: usize) -> bool {
        if end <= start + 2 || end > high.len() {
            return false;
        }

        let initial_range = high[start] - low[start];
        let final_range = high[end - 1] - low[end - 1];

        // Triangular if range contracts
        if initial_range > 0.0 && final_range > 0.0 {
            let convergence = 1.0 - (final_range / initial_range);
            convergence >= 0.2 // At least 20% convergence
        } else {
            false
        }
    }

    /// Detect pattern type at each point.
    pub fn detect_pattern_type(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<FlagPennantType> {
        let n = close.len();
        let min_total = self.config.min_pole_length + self.config.min_flag_length;
        let mut patterns = vec![FlagPennantType::None; n];

        if n < min_total {
            return patterns;
        }

        for i in min_total..n {
            // Check for bull patterns
            let pole_end = i.saturating_sub(self.config.min_flag_length);
            if let Some((_pole_start, pole_move)) = self.detect_bull_pole(close, pole_end) {
                let flag_start = pole_end;
                let flag_end = i;

                // Check retracement
                let pole_high = close[pole_end];
                let flag_low = low[flag_start..flag_end].iter().cloned().fold(f64::INFINITY, f64::min);
                let retracement = (pole_high - flag_low) / (pole_high * pole_move);

                if retracement <= self.config.max_retracement {
                    if self.is_rectangular(high, low, flag_start, flag_end) {
                        patterns[i] = FlagPennantType::BullFlag;
                    } else if self.is_triangular(high, low, flag_start, flag_end) {
                        patterns[i] = FlagPennantType::BullPennant;
                    }
                }
            }

            // Check for bear patterns
            if let Some((_pole_start, pole_move)) = self.detect_bear_pole(close, pole_end) {
                let flag_start = pole_end;
                let flag_end = i;

                // Check retracement
                let pole_low = close[pole_end];
                let flag_high = high[flag_start..flag_end].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let retracement = (flag_high - pole_low) / (pole_low * pole_move);

                if retracement <= self.config.max_retracement {
                    if self.is_rectangular(high, low, flag_start, flag_end) {
                        patterns[i] = FlagPennantType::BearFlag;
                    } else if self.is_triangular(high, low, flag_start, flag_end) {
                        patterns[i] = FlagPennantType::BearPennant;
                    }
                }
            }
        }

        patterns
    }

    /// Calculate pattern signals.
    ///
    /// Returns a vector where:
    /// - Positive values (1.0 or 2.0) indicate bullish patterns (flag/pennant)
    /// - Negative values (-1.0 or -2.0) indicate bearish patterns
    /// - 0.0 indicates no pattern
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let pattern_types = self.detect_pattern_type(high, low, close);

        pattern_types.iter().map(|pt| match pt {
            FlagPennantType::BullFlag => 1.0,
            FlagPennantType::BullPennant => 2.0,
            FlagPennantType::BearFlag => -1.0,
            FlagPennantType::BearPennant => -2.0,
            FlagPennantType::None => 0.0,
        }).collect()
    }

    /// Detect bull flag patterns.
    pub fn detect_bull_flag(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<bool> {
        self.detect_pattern_type(high, low, close)
            .iter()
            .map(|pt| *pt == FlagPennantType::BullFlag)
            .collect()
    }

    /// Detect bear flag patterns.
    pub fn detect_bear_flag(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<bool> {
        self.detect_pattern_type(high, low, close)
            .iter()
            .map(|pt| *pt == FlagPennantType::BearFlag)
            .collect()
    }

    /// Detect bull pennant patterns.
    pub fn detect_bull_pennant(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<bool> {
        self.detect_pattern_type(high, low, close)
            .iter()
            .map(|pt| *pt == FlagPennantType::BullPennant)
            .collect()
    }

    /// Detect bear pennant patterns.
    pub fn detect_bear_pennant(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<bool> {
        self.detect_pattern_type(high, low, close)
            .iter()
            .map(|pt| *pt == FlagPennantType::BearPennant)
            .collect()
    }

    /// Calculate pole strength (magnitude of the initial move).
    pub fn pole_strength(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut strength = vec![0.0; n];

        for i in self.config.min_pole_length..n {
            // Check for bull pole
            if let Some((_, move_pct)) = self.detect_bull_pole(close, i) {
                strength[i] = move_pct;
            }
            // Check for bear pole
            else if let Some((_, move_pct)) = self.detect_bear_pole(close, i) {
                strength[i] = -move_pct;
            }
        }

        strength
    }
}

impl Default for FlagPennant {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicator for FlagPennant {
    fn name(&self) -> &str {
        "FlagPennant"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let min_required = self.config.min_pole_length + self.config.min_flag_length;
        if data.close.len() < min_required {
            return Err(IndicatorError::InsufficientData {
                required: min_required,
                got: data.close.len(),
            });
        }

        let signals = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::single(signals))
    }

    fn min_periods(&self) -> usize {
        self.config.min_pole_length + self.config.min_flag_length
    }
}

impl SignalIndicator for FlagPennant {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let signals = self.calculate(&data.high, &data.low, &data.close);

        // Find the most recent signal
        for &s in signals.iter().rev() {
            if s > 0.0 {
                return Ok(IndicatorSignal::Bullish);
            } else if s < 0.0 {
                return Ok(IndicatorSignal::Bearish);
            }
        }

        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let values = self.calculate(&data.high, &data.low, &data.close);
        let signals = values.iter().map(|&s| {
            if s > 0.0 {
                IndicatorSignal::Bullish
            } else if s < 0.0 {
                IndicatorSignal::Bearish
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

    fn create_bull_flag_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();

        // Pole: strong upward move
        for i in 0..10 {
            let base = 100.0 + i as f64 * 2.0;
            high.push(base + 2.0);
            low.push(base - 1.0);
            close.push(base + 1.0);
        }

        // Flag: sideways to slightly down consolidation
        for i in 0..8 {
            let base = 118.0 - i as f64 * 0.3;
            high.push(base + 1.5);
            low.push(base - 1.5);
            close.push(base);
        }

        (high, low, close)
    }

    fn create_bear_flag_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();

        // Pole: strong downward move
        for i in 0..10 {
            let base = 120.0 - i as f64 * 2.0;
            high.push(base + 1.0);
            low.push(base - 2.0);
            close.push(base - 1.0);
        }

        // Flag: sideways to slightly up consolidation
        for i in 0..8 {
            let base = 102.0 + i as f64 * 0.3;
            high.push(base + 1.5);
            low.push(base - 1.5);
            close.push(base);
        }

        (high, low, close)
    }

    #[test]
    fn test_flag_pennant_creation() {
        let indicator = FlagPennant::new();
        assert_eq!(indicator.config.min_pole_length, 5);
        assert_eq!(indicator.config.min_flag_length, 5);
    }

    #[test]
    fn test_flag_pennant_with_params() {
        let indicator = FlagPennant::with_params(6, 7, 0.08);
        assert_eq!(indicator.config.min_pole_length, 6);
        assert_eq!(indicator.config.min_flag_length, 7);
        assert_eq!(indicator.config.min_pole_move, 0.08);
    }

    #[test]
    fn test_detect_bull_pole() {
        let (_, _, close) = create_bull_flag_data();
        let indicator = FlagPennant::with_params(5, 5, 0.05);

        let result = indicator.detect_bull_pole(&close, 9);
        assert!(result.is_some());
    }

    #[test]
    fn test_detect_bear_pole() {
        let (_, _, close) = create_bear_flag_data();
        let indicator = FlagPennant::with_params(5, 5, 0.05);

        let result = indicator.detect_bear_pole(&close, 9);
        assert!(result.is_some());
    }

    #[test]
    fn test_is_rectangular() {
        let indicator = FlagPennant::new();
        // Rectangular consolidation
        let high = vec![110.0, 110.5, 109.5, 110.0, 110.2];
        let low = vec![108.0, 107.5, 108.5, 108.0, 107.8];

        let is_rect = indicator.is_rectangular(&high, &low, 0, 5);
        assert!(is_rect);
    }

    #[test]
    fn test_is_triangular() {
        let indicator = FlagPennant::new();
        // Triangular consolidation (converging)
        let high = vec![112.0, 111.0, 110.0, 109.0, 108.0];
        let low = vec![100.0, 101.0, 102.0, 103.0, 104.0];

        let is_tri = indicator.is_triangular(&high, &low, 0, 5);
        assert!(is_tri);
    }

    #[test]
    fn test_detect_bull_flag() {
        let (high, low, close) = create_bull_flag_data();
        let indicator = FlagPennant::with_params(5, 5, 0.05);

        let flags = indicator.detect_bull_flag(&high, &low, &close);
        assert_eq!(flags.len(), high.len());
    }

    #[test]
    fn test_detect_bear_flag() {
        let (high, low, close) = create_bear_flag_data();
        let indicator = FlagPennant::with_params(5, 5, 0.05);

        let flags = indicator.detect_bear_flag(&high, &low, &close);
        assert_eq!(flags.len(), high.len());
    }

    #[test]
    fn test_pole_strength() {
        let (_, _, close) = create_bull_flag_data();
        let indicator = FlagPennant::with_params(5, 5, 0.05);

        let strength = indicator.pole_strength(&close);
        assert_eq!(strength.len(), close.len());
    }

    #[test]
    fn test_calculate() {
        let (high, low, close) = create_bull_flag_data();
        let indicator = FlagPennant::with_params(5, 5, 0.05);

        let signals = indicator.calculate(&high, &low, &close);
        assert_eq!(signals.len(), high.len());
    }

    #[test]
    fn test_min_periods() {
        let indicator = FlagPennant::with_params(6, 8, 0.05);
        assert_eq!(indicator.min_periods(), 14);
    }

    #[test]
    fn test_insufficient_data() {
        let indicator = FlagPennant::new();
        let data = OHLCVSeries {
            open: vec![100.0; 5],
            high: vec![105.0; 5],
            low: vec![95.0; 5],
            close: vec![102.0; 5],
            volume: vec![1000.0; 5],
        };

        let result = indicator.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_pattern_type_enum() {
        assert_eq!(FlagPennantType::BullFlag, FlagPennantType::BullFlag);
        assert_ne!(FlagPennantType::BullFlag, FlagPennantType::BearFlag);
        assert_ne!(FlagPennantType::BullPennant, FlagPennantType::BearPennant);
    }
}
