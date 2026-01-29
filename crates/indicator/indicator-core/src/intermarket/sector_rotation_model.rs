//! Sector Rotation Model (IND-401)
//!
//! Business cycle position indicator that models sector rotation
//! based on economic cycle phases and relative performance metrics.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

use super::MultiSeries;

/// Business cycle phase enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusinessCyclePhase {
    /// Early expansion - recovery from recession
    EarlyExpansion,
    /// Mid-cycle expansion - sustained growth
    MidExpansion,
    /// Late expansion - peak growth, inflation concerns
    LateExpansion,
    /// Early contraction - slowdown begins
    EarlyContraction,
    /// Late contraction - recession deepens
    LateContraction,
    /// Unknown phase
    Unknown,
}

impl BusinessCyclePhase {
    /// Get the numeric representation for calculations.
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::EarlyExpansion => 1.0,
            Self::MidExpansion => 2.0,
            Self::LateExpansion => 3.0,
            Self::EarlyContraction => 4.0,
            Self::LateContraction => 5.0,
            Self::Unknown => 0.0,
        }
    }

    /// Get recommended sectors for this phase.
    pub fn recommended_sectors(&self) -> Vec<&'static str> {
        match self {
            Self::EarlyExpansion => vec!["Financials", "Consumer Discretionary", "Real Estate"],
            Self::MidExpansion => vec!["Technology", "Industrials", "Materials"],
            Self::LateExpansion => vec!["Energy", "Materials", "Industrials"],
            Self::EarlyContraction => vec!["Healthcare", "Consumer Staples", "Utilities"],
            Self::LateContraction => vec!["Consumer Staples", "Utilities", "Healthcare"],
            Self::Unknown => vec![],
        }
    }
}

/// Sector rotation model output.
#[derive(Debug, Clone)]
pub struct SectorRotationModelOutput {
    /// Current business cycle phase.
    pub phase: BusinessCyclePhase,
    /// Phase score (0-5 scale).
    pub phase_score: f64,
    /// Momentum score for phase transition.
    pub transition_momentum: f64,
    /// Confidence in phase determination (0-1).
    pub confidence: f64,
    /// Sector scores for current phase.
    pub sector_scores: Vec<(String, f64)>,
}

/// Sector Rotation Model (IND-401)
///
/// Models business cycle position using sector relative performance,
/// yield curve signals, and economic indicators to determine optimal
/// sector allocation across different market phases.
///
/// # Methodology
/// 1. Calculate sector momentum and relative strength
/// 2. Analyze cross-sector correlations for phase signals
/// 3. Use yield curve shape as economic indicator proxy
/// 4. Combine signals to determine business cycle phase
/// 5. Score sectors based on historical phase performance
///
/// # Interpretation
/// - Phase 1-2: Early to mid expansion (favor cyclicals)
/// - Phase 3: Late expansion (favor commodities, inflation hedges)
/// - Phase 4-5: Contraction (favor defensives)
#[derive(Debug, Clone)]
pub struct SectorRotationModel {
    /// Period for momentum calculation.
    momentum_period: usize,
    /// Period for relative strength calculation.
    rs_period: usize,
    /// Smoothing period for phase transitions.
    smooth_period: usize,
    /// Sector names.
    sector_names: Vec<String>,
    /// Sector price series.
    sector_prices: Vec<Vec<f64>>,
    /// Economic indicator (e.g., yield curve spread).
    economic_indicator: Vec<f64>,
}

impl SectorRotationModel {
    /// Create a new Sector Rotation Model.
    ///
    /// # Arguments
    /// * `momentum_period` - Period for momentum calculation (e.g., 20)
    /// * `rs_period` - Period for relative strength calculation (e.g., 60)
    /// * `smooth_period` - Smoothing period for transitions (e.g., 5)
    pub fn new(momentum_period: usize, rs_period: usize, smooth_period: usize) -> Result<Self> {
        if momentum_period < 5 {
            return Err(IndicatorError::InvalidParameter {
                name: "momentum_period".to_string(),
                reason: "must be at least 5".to_string(),
            });
        }
        if rs_period < momentum_period {
            return Err(IndicatorError::InvalidParameter {
                name: "rs_period".to_string(),
                reason: "must be greater than or equal to momentum_period".to_string(),
            });
        }
        Ok(Self {
            momentum_period,
            rs_period,
            smooth_period,
            sector_names: Vec::new(),
            sector_prices: Vec::new(),
            economic_indicator: Vec::new(),
        })
    }

    /// Create with default parameters (20, 60, 5).
    pub fn default_params() -> Result<Self> {
        Self::new(20, 60, 5)
    }

    /// Set sectors from MultiSeries.
    pub fn with_sectors(mut self, sectors: MultiSeries) -> Self {
        self.sector_names = sectors.series.iter().map(|(n, _)| n.clone()).collect();
        self.sector_prices = sectors.series.into_iter().map(|(_, p)| p).collect();
        self
    }

    /// Add a single sector.
    pub fn add_sector(&mut self, name: &str, prices: Vec<f64>) {
        self.sector_names.push(name.to_string());
        self.sector_prices.push(prices);
    }

    /// Set economic indicator (e.g., yield curve spread).
    pub fn with_economic_indicator(mut self, indicator: Vec<f64>) -> Self {
        self.economic_indicator = indicator;
        self
    }

    /// Calculate rate of change.
    fn calculate_roc(prices: &[f64], period: usize) -> Vec<f64> {
        let n = prices.len();
        if n < period + 1 {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period];
        for i in period..n {
            let prev = prices[i - period];
            if prev.abs() > 1e-10 {
                result.push((prices[i] - prev) / prev * 100.0);
            } else {
                result.push(f64::NAN);
            }
        }
        result
    }

    /// Calculate simple moving average.
    fn sma(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < period {
            return vec![f64::NAN; n];
        }

        let mut result = vec![f64::NAN; period - 1];
        let mut sum: f64 = data[..period].iter().filter(|x| !x.is_nan()).sum();

        for i in (period - 1)..n {
            if i > period - 1 {
                if !data[i].is_nan() {
                    sum += data[i];
                }
                if !data[i - period].is_nan() {
                    sum -= data[i - period];
                }
            }
            let count = data[(i + 1 - period)..=i]
                .iter()
                .filter(|x| !x.is_nan())
                .count();
            if count > 0 {
                result.push(sum / count as f64);
            } else {
                result.push(f64::NAN);
            }
        }
        result
    }

    /// Determine business cycle phase from indicators.
    fn determine_phase(
        cyclical_momentum: f64,
        defensive_momentum: f64,
        economic_signal: f64,
    ) -> (BusinessCyclePhase, f64) {
        // Cyclical vs defensive ratio
        let ratio = if defensive_momentum.abs() > 1e-10 {
            cyclical_momentum / defensive_momentum.abs().max(1e-10)
        } else {
            cyclical_momentum.signum()
        };

        // Combine with economic indicator
        let combined = if economic_signal.is_nan() {
            ratio
        } else {
            0.7 * ratio + 0.3 * economic_signal
        };

        // Determine phase based on combined signal
        let (phase, score) = if combined > 1.5 {
            (BusinessCyclePhase::EarlyExpansion, 1.0)
        } else if combined > 0.5 {
            (BusinessCyclePhase::MidExpansion, 2.0)
        } else if combined > -0.5 {
            (BusinessCyclePhase::LateExpansion, 3.0)
        } else if combined > -1.5 {
            (BusinessCyclePhase::EarlyContraction, 4.0)
        } else {
            (BusinessCyclePhase::LateContraction, 5.0)
        };

        (phase, score)
    }

    /// Calculate confidence based on signal clarity.
    fn calculate_confidence(signals: &[f64]) -> f64 {
        if signals.is_empty() {
            return 0.0;
        }

        let mean: f64 = signals.iter().sum::<f64>() / signals.len() as f64;
        let variance: f64 = signals.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / signals.len() as f64;
        let std_dev = variance.sqrt();

        // Higher absolute mean and lower variance = higher confidence
        let signal_strength = mean.abs();
        let consistency = 1.0 / (1.0 + std_dev);

        (signal_strength * consistency).min(1.0)
    }

    /// Calculate sector rotation model outputs.
    pub fn calculate(&self) -> Vec<SectorRotationModelOutput> {
        if self.sector_prices.is_empty() {
            return Vec::new();
        }

        let n = self.sector_prices.iter().map(|p| p.len()).min().unwrap_or(0);
        if n < self.rs_period + 1 {
            return Vec::new();
        }

        // Calculate momentum for all sectors
        let momentums: Vec<Vec<f64>> = self
            .sector_prices
            .iter()
            .map(|p| Self::calculate_roc(p, self.momentum_period))
            .collect();

        // Calculate relative strength (vs equal-weighted average)
        let mut avg_prices: Vec<f64> = vec![0.0; n];
        for i in 0..n {
            let sum: f64 = self.sector_prices.iter().map(|p| p[i]).sum();
            avg_prices[i] = sum / self.sector_prices.len() as f64;
        }
        let avg_roc = Self::calculate_roc(&avg_prices, self.rs_period);

        // Sector relative strength
        let sector_rs: Vec<Vec<f64>> = self
            .sector_prices
            .iter()
            .map(|p| {
                let roc = Self::calculate_roc(p, self.rs_period);
                roc.iter()
                    .zip(avg_roc.iter())
                    .map(|(s, a)| {
                        if s.is_nan() || a.is_nan() {
                            f64::NAN
                        } else {
                            s - a
                        }
                    })
                    .collect()
            })
            .collect();

        // Identify cyclical and defensive sectors by name pattern
        let cyclical_indices: Vec<usize> = self
            .sector_names
            .iter()
            .enumerate()
            .filter(|(_, name)| {
                let n = name.to_lowercase();
                n.contains("tech") || n.contains("discretionary") || n.contains("industrial")
                    || n.contains("financial") || n.contains("material")
            })
            .map(|(i, _)| i)
            .collect();

        let defensive_indices: Vec<usize> = self
            .sector_names
            .iter()
            .enumerate()
            .filter(|(_, name)| {
                let n = name.to_lowercase();
                n.contains("staple") || n.contains("utility") || n.contains("health")
                    || n.contains("telecom")
            })
            .map(|(i, _)| i)
            .collect();

        let mut results = Vec::with_capacity(n);

        for i in 0..n {
            // Calculate cyclical and defensive momentum averages
            let cyclical_mom: f64 = if cyclical_indices.is_empty() {
                0.0
            } else {
                cyclical_indices
                    .iter()
                    .filter_map(|&idx| {
                        let v = momentums[idx][i];
                        if v.is_nan() { None } else { Some(v) }
                    })
                    .sum::<f64>()
                    / cyclical_indices.len().max(1) as f64
            };

            let defensive_mom: f64 = if defensive_indices.is_empty() {
                1.0 // Avoid division by zero
            } else {
                defensive_indices
                    .iter()
                    .filter_map(|&idx| {
                        let v = momentums[idx][i];
                        if v.is_nan() { None } else { Some(v) }
                    })
                    .sum::<f64>()
                    / defensive_indices.len().max(1) as f64
            };

            let economic_signal = self
                .economic_indicator
                .get(i)
                .copied()
                .unwrap_or(f64::NAN);

            let (phase, phase_score) =
                Self::determine_phase(cyclical_mom, defensive_mom, economic_signal);

            // Calculate transition momentum
            let transition_momentum = if i >= self.smooth_period {
                let recent_scores: Vec<f64> = (0..self.smooth_period)
                    .filter_map(|j| {
                        let idx = i - j;
                        Some(Self::determine_phase(
                            if cyclical_indices.is_empty() {
                                0.0
                            } else {
                                cyclical_indices
                                    .iter()
                                    .filter_map(|&k| {
                                        let v = momentums[k][idx];
                                        if v.is_nan() { None } else { Some(v) }
                                    })
                                    .sum::<f64>()
                                    / cyclical_indices.len().max(1) as f64
                            },
                            if defensive_indices.is_empty() {
                                1.0
                            } else {
                                defensive_indices
                                    .iter()
                                    .filter_map(|&k| {
                                        let v = momentums[k][idx];
                                        if v.is_nan() { None } else { Some(v) }
                                    })
                                    .sum::<f64>()
                                    / defensive_indices.len().max(1) as f64
                            },
                            self.economic_indicator.get(idx).copied().unwrap_or(f64::NAN),
                        ).1)
                    })
                    .collect();
                if recent_scores.len() >= 2 {
                    recent_scores[0] - recent_scores[recent_scores.len() - 1]
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Calculate confidence
            let signals: Vec<f64> = momentums
                .iter()
                .map(|m| m[i])
                .filter(|v| !v.is_nan())
                .collect();
            let confidence = Self::calculate_confidence(&signals);

            // Score sectors for current phase
            let sector_scores: Vec<(String, f64)> = self
                .sector_names
                .iter()
                .enumerate()
                .map(|(idx, name)| {
                    let mom = momentums[idx][i];
                    let rs = sector_rs[idx][i];
                    let score = if mom.is_nan() || rs.is_nan() {
                        f64::NAN
                    } else {
                        0.6 * mom + 0.4 * rs
                    };
                    (name.clone(), score)
                })
                .collect();

            results.push(SectorRotationModelOutput {
                phase,
                phase_score,
                transition_momentum,
                confidence,
                sector_scores,
            });
        }

        results
    }

    /// Get phase score series.
    pub fn phase_scores(&self) -> Vec<f64> {
        self.calculate().iter().map(|o| o.phase_score).collect()
    }

    /// Get transition momentum series.
    pub fn transition_momentums(&self) -> Vec<f64> {
        self.calculate()
            .iter()
            .map(|o| o.transition_momentum)
            .collect()
    }

    /// Get confidence series.
    pub fn confidences(&self) -> Vec<f64> {
        self.calculate().iter().map(|o| o.confidence).collect()
    }
}

impl TechnicalIndicator for SectorRotationModel {
    fn name(&self) -> &str {
        "Sector Rotation Model"
    }

    fn min_periods(&self) -> usize {
        self.rs_period + 1
    }

    fn compute(&self, _data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.sector_prices.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "sector_prices".to_string(),
                reason: "Sector prices must be set before computing".to_string(),
            });
        }

        let min_len = self.sector_prices.iter().map(|p| p.len()).min().unwrap_or(0);
        if min_len < self.rs_period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.rs_period + 1,
                got: min_len,
            });
        }

        let phase_scores = self.phase_scores();
        Ok(IndicatorOutput::single(phase_scores))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sectors(n: usize) -> SectorRotationModel {
        // Create synthetic sector data with different performance patterns
        let tech: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.5 + ((i as f64) * 0.1).sin() * 3.0)
            .collect();

        let healthcare: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.3 + ((i as f64) * 0.15).sin() * 2.0)
            .collect();

        let consumer_staples: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.2 + ((i as f64) * 0.2).sin() * 1.5)
            .collect();

        let financials: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.4 + ((i as f64) * 0.12).sin() * 2.5)
            .collect();

        let utilities: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.1 + ((i as f64) * 0.08).sin() * 1.0)
            .collect();

        let mut model = SectorRotationModel::new(20, 60, 5).unwrap();
        model.add_sector("Technology", tech);
        model.add_sector("Healthcare", healthcare);
        model.add_sector("Consumer Staples", consumer_staples);
        model.add_sector("Financials", financials);
        model.add_sector("Utilities", utilities);

        model
    }

    #[test]
    fn test_sector_rotation_model_basic() {
        let model = create_test_sectors(100);
        let outputs = model.calculate();

        assert_eq!(outputs.len(), 100);

        // Check last output has valid phase
        let last = outputs.last().unwrap();
        assert_ne!(last.phase, BusinessCyclePhase::Unknown);
        assert!(last.phase_score >= 0.0 && last.phase_score <= 5.0);
    }

    #[test]
    fn test_phase_determination() {
        let model = create_test_sectors(100);
        let outputs = model.calculate();

        // Should have a phase for later periods
        for output in outputs.iter().skip(70) {
            assert!(output.phase_score >= 1.0 && output.phase_score <= 5.0);
        }
    }

    #[test]
    fn test_sector_scores() {
        let model = create_test_sectors(100);
        let outputs = model.calculate();

        let last = outputs.last().unwrap();
        assert_eq!(last.sector_scores.len(), 5);

        // Check all sectors are represented
        let names: Vec<&str> = last.sector_scores.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"Technology"));
        assert!(names.contains(&"Healthcare"));
    }

    #[test]
    fn test_confidence_range() {
        let model = create_test_sectors(100);
        let confidences = model.confidences();

        for conf in &confidences {
            assert!(*conf >= 0.0 && *conf <= 1.0);
        }
    }

    #[test]
    fn test_transition_momentum() {
        let model = create_test_sectors(100);
        let transitions = model.transition_momentums();

        assert_eq!(transitions.len(), 100);
        // First few should be 0 due to smoothing warmup
        assert_eq!(transitions[0], 0.0);
    }

    #[test]
    fn test_with_economic_indicator() {
        let mut model = create_test_sectors(100);
        let yield_curve: Vec<f64> = (0..100)
            .map(|i| 0.5 * ((i as f64) * 0.1).sin())
            .collect();

        model = model.with_economic_indicator(yield_curve);
        let outputs = model.calculate();

        assert_eq!(outputs.len(), 100);
    }

    #[test]
    fn test_phase_recommended_sectors() {
        let phase = BusinessCyclePhase::EarlyExpansion;
        let recommended = phase.recommended_sectors();

        assert!(!recommended.is_empty());
        assert!(recommended.contains(&"Financials"));
    }

    #[test]
    fn test_technical_indicator_impl() {
        let model = create_test_sectors(100);
        let data = OHLCVSeries::from_close(vec![100.0; 100]);
        let result = model.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
    }

    #[test]
    fn test_empty_sectors_error() {
        let model = SectorRotationModel::new(20, 60, 5).unwrap();
        let data = OHLCVSeries::from_close(vec![100.0; 100]);
        let result = model.compute(&data);

        assert!(result.is_err());
    }

    #[test]
    fn test_insufficient_data() {
        let mut model = SectorRotationModel::new(20, 60, 5).unwrap();
        model.add_sector("Tech", vec![100.0; 30]);
        model.add_sector("Health", vec![100.0; 30]);

        let data = OHLCVSeries::from_close(vec![100.0; 30]);
        let result = model.compute(&data);

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_parameters() {
        let result = SectorRotationModel::new(2, 60, 5);
        assert!(result.is_err());

        let result = SectorRotationModel::new(20, 10, 5);
        assert!(result.is_err());
    }
}
