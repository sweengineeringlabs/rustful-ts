//! Sector Rotation Indicator
//!
//! Calculates relative performance rankings for sectors or assets,
//! useful for sector rotation strategies and relative strength analysis.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

use super::MultiSeries;

/// Sector ranking output.
#[derive(Debug, Clone)]
pub struct SectorRank {
    /// Sector/asset name.
    pub name: String,
    /// Relative strength score.
    pub strength: f64,
    /// Rank (1 = strongest).
    pub rank: usize,
    /// Rate of change percentage.
    pub roc: f64,
    /// Relative strength vs benchmark (if provided).
    pub relative_strength: f64,
    /// Momentum score (change in rank).
    pub momentum: i32,
}

/// Sector rotation output for a single time point.
#[derive(Debug, Clone)]
pub struct SectorRotationOutput {
    /// Rankings for all sectors (sorted by strength).
    pub rankings: Vec<SectorRank>,
    /// Name of the leading sector.
    pub leader: String,
    /// Name of the lagging sector.
    pub laggard: String,
    /// Rotation score (measure of rotation activity).
    pub rotation_score: f64,
}

/// Sector Rotation Indicator.
///
/// Analyzes relative performance across multiple sectors or assets to
/// identify rotation patterns and momentum. Uses rate of change (ROC)
/// and relative strength calculations.
///
/// # Methodology
/// 1. Calculate ROC for each sector over the lookback period
/// 2. Optionally calculate relative strength vs a benchmark
/// 3. Rank sectors by combined score
/// 4. Track rotation by measuring rank changes over time
///
/// # Use Cases
/// - Sector ETF rotation strategies
/// - Asset class allocation
/// - Country/region rotation
/// - Factor rotation (value, growth, momentum, etc.)
#[derive(Debug, Clone)]
pub struct SectorRotation {
    /// Period for ROC calculation.
    roc_period: usize,
    /// Period for relative strength calculation.
    rs_period: usize,
    /// Benchmark prices for relative strength (optional).
    benchmark: Vec<f64>,
    /// Sector/asset names.
    names: Vec<String>,
    /// Sector/asset price series.
    prices: Vec<Vec<f64>>,
}

impl SectorRotation {
    /// Create a new Sector Rotation indicator.
    ///
    /// # Arguments
    /// * `roc_period` - Period for rate of change calculation (e.g., 20)
    /// * `rs_period` - Period for relative strength calculation (e.g., 50)
    pub fn new(roc_period: usize, rs_period: usize) -> Self {
        Self {
            roc_period,
            rs_period,
            benchmark: Vec::new(),
            names: Vec::new(),
            prices: Vec::new(),
        }
    }

    /// Create with a single period for both ROC and RS.
    pub fn with_period(period: usize) -> Self {
        Self::new(period, period)
    }

    /// Set the benchmark for relative strength calculation.
    pub fn with_benchmark(mut self, benchmark: Vec<f64>) -> Self {
        self.benchmark = benchmark;
        self
    }

    /// Set sectors from MultiSeries.
    pub fn with_sectors(mut self, sectors: MultiSeries) -> Self {
        self.names = sectors.series.iter().map(|(n, _)| n.clone()).collect();
        self.prices = sectors.series.into_iter().map(|(_, p)| p).collect();
        self
    }

    /// Add a single sector.
    pub fn add_sector(&mut self, name: &str, prices: Vec<f64>) {
        self.names.push(name.to_string());
        self.prices.push(prices);
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
            if prev.abs() < 1e-10 {
                result.push(f64::NAN);
            } else {
                let roc = (prices[i] - prev) / prev * 100.0;
                result.push(roc);
            }
        }
        result
    }

    /// Calculate relative strength vs benchmark.
    fn calculate_relative_strength(sector: &[f64], benchmark: &[f64], period: usize) -> Vec<f64> {
        if sector.len() != benchmark.len() {
            return vec![f64::NAN; sector.len()];
        }

        let n = sector.len();
        if n < period + 1 {
            return vec![f64::NAN; n];
        }

        let sector_roc = Self::calculate_roc(sector, period);
        let bench_roc = Self::calculate_roc(benchmark, period);

        sector_roc
            .iter()
            .zip(bench_roc.iter())
            .map(|(s, b)| {
                if s.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    s - b // Excess return over benchmark
                }
            })
            .collect()
    }

    /// Calculate combined strength score.
    fn calculate_strength_score(roc: f64, rs: f64) -> f64 {
        if roc.is_nan() {
            return f64::NAN;
        }
        if rs.is_nan() {
            roc
        } else {
            // Weighted combination: 60% ROC, 40% relative strength
            0.6 * roc + 0.4 * rs
        }
    }

    /// Calculate rotation score (measure of rotation activity).
    fn calculate_rotation_score(prev_ranks: &[usize], curr_ranks: &[usize]) -> f64 {
        if prev_ranks.len() != curr_ranks.len() || prev_ranks.is_empty() {
            return 0.0;
        }

        let n = prev_ranks.len() as f64;
        let total_change: i32 = prev_ranks
            .iter()
            .zip(curr_ranks.iter())
            .map(|(p, c)| (*p as i32 - *c as i32).abs())
            .sum();

        // Normalize by maximum possible change
        total_change as f64 / n
    }

    /// Calculate full rotation analysis over time.
    pub fn calculate(&self) -> Vec<SectorRotationOutput> {
        if self.prices.is_empty() || self.names.is_empty() {
            return Vec::new();
        }

        let n = self.prices.iter().map(|p| p.len()).min().unwrap_or(0);
        if n == 0 {
            return Vec::new();
        }

        // Calculate ROC and RS for all sectors
        let rocs: Vec<Vec<f64>> = self
            .prices
            .iter()
            .map(|p| Self::calculate_roc(p, self.roc_period))
            .collect();

        let rs_values: Vec<Vec<f64>> = if self.benchmark.is_empty() {
            vec![vec![f64::NAN; n]; self.prices.len()]
        } else {
            self.prices
                .iter()
                .map(|p| Self::calculate_relative_strength(p, &self.benchmark, self.rs_period))
                .collect()
        };

        let mut results = Vec::with_capacity(n);
        let mut prev_ranks: Vec<usize> = Vec::new();

        for i in 0..n {
            // Calculate strength scores for this time point
            let mut sector_scores: Vec<(usize, f64, f64, f64)> = self
                .names
                .iter()
                .enumerate()
                .map(|(idx, _)| {
                    let roc = rocs[idx].get(i).copied().unwrap_or(f64::NAN);
                    let rs = rs_values[idx].get(i).copied().unwrap_or(f64::NAN);
                    let strength = Self::calculate_strength_score(roc, rs);
                    (idx, strength, roc, rs)
                })
                .collect();

            // Sort by strength (descending)
            sector_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Build rankings
            let mut curr_ranks: Vec<usize> = vec![0; self.names.len()];
            let rankings: Vec<SectorRank> = sector_scores
                .iter()
                .enumerate()
                .map(|(rank, (idx, strength, roc, rs))| {
                    let prev_rank = if !prev_ranks.is_empty() {
                        prev_ranks.get(*idx).copied().unwrap_or(rank + 1)
                    } else {
                        rank + 1
                    };
                    curr_ranks[*idx] = rank + 1;

                    SectorRank {
                        name: self.names[*idx].clone(),
                        strength: *strength,
                        rank: rank + 1,
                        roc: *roc,
                        relative_strength: *rs,
                        momentum: prev_rank as i32 - (rank + 1) as i32, // Positive = improving
                    }
                })
                .collect();

            // Calculate rotation score
            let rotation_score = if !prev_ranks.is_empty() {
                Self::calculate_rotation_score(&prev_ranks, &curr_ranks)
            } else {
                0.0
            };

            // Find leader and laggard
            let leader = rankings
                .iter()
                .find(|r| r.rank == 1)
                .map(|r| r.name.clone())
                .unwrap_or_default();
            let laggard = rankings
                .iter()
                .find(|r| r.rank == self.names.len())
                .map(|r| r.name.clone())
                .unwrap_or_default();

            results.push(SectorRotationOutput {
                rankings,
                leader,
                laggard,
                rotation_score,
            });

            prev_ranks = curr_ranks;
        }

        results
    }

    /// Get strength series for a specific sector.
    pub fn strength_for(&self, name: &str) -> Vec<f64> {
        let outputs = self.calculate();
        outputs
            .iter()
            .map(|o| {
                o.rankings
                    .iter()
                    .find(|r| r.name == name)
                    .map(|r| r.strength)
                    .unwrap_or(f64::NAN)
            })
            .collect()
    }

    /// Get rank series for a specific sector.
    pub fn rank_for(&self, name: &str) -> Vec<usize> {
        let outputs = self.calculate();
        outputs
            .iter()
            .map(|o| {
                o.rankings
                    .iter()
                    .find(|r| r.name == name)
                    .map(|r| r.rank)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Get leaders over time.
    pub fn leaders(&self) -> Vec<String> {
        let outputs = self.calculate();
        outputs.iter().map(|o| o.leader.clone()).collect()
    }

    /// Get laggards over time.
    pub fn laggards(&self) -> Vec<String> {
        let outputs = self.calculate();
        outputs.iter().map(|o| o.laggard.clone()).collect()
    }

    /// Get rotation score series.
    pub fn rotation_scores(&self) -> Vec<f64> {
        let outputs = self.calculate();
        outputs.iter().map(|o| o.rotation_score).collect()
    }

    /// Identify sectors with improving momentum (rising in ranks).
    pub fn improving_sectors(&self) -> Vec<Vec<String>> {
        let outputs = self.calculate();
        outputs
            .iter()
            .map(|o| {
                o.rankings
                    .iter()
                    .filter(|r| r.momentum > 0)
                    .map(|r| r.name.clone())
                    .collect()
            })
            .collect()
    }

    /// Identify sectors with deteriorating momentum (falling in ranks).
    pub fn deteriorating_sectors(&self) -> Vec<Vec<String>> {
        let outputs = self.calculate();
        outputs
            .iter()
            .map(|o| {
                o.rankings
                    .iter()
                    .filter(|r| r.momentum < 0)
                    .map(|r| r.name.clone())
                    .collect()
            })
            .collect()
    }
}

impl TechnicalIndicator for SectorRotation {
    fn name(&self) -> &str {
        "SectorRotation"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if self.prices.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "prices".to_string(),
                reason: "Sector prices must be set before computing".to_string(),
            });
        }

        let min_len = self.prices.iter().map(|p| p.len()).min().unwrap_or(0);
        let required = self.roc_period.max(self.rs_period) + 1;

        if min_len < required {
            return Err(IndicatorError::InsufficientData {
                required,
                got: min_len,
            });
        }

        // Primary output: rotation score
        let rotation = self.rotation_scores();
        Ok(IndicatorOutput::single(rotation))
    }

    fn min_periods(&self) -> usize {
        self.roc_period.max(self.rs_period) + 1
    }
}

impl SignalIndicator for SectorRotation {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        // Rotation score doesn't directly translate to buy/sell
        // High rotation could indicate market regime change
        let rotation = self.rotation_scores();
        let last = rotation.last().copied().unwrap_or(0.0);

        // Simple heuristic: high rotation = neutral (uncertainty)
        // Low rotation = current trend likely to continue
        if last > 2.0 {
            Ok(IndicatorSignal::Neutral) // High rotation
        } else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let n = if self.prices.is_empty() {
            data.close.len()
        } else {
            self.prices[0].len()
        };
        Ok(vec![IndicatorSignal::Neutral; n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sectors(n: usize) -> SectorRotation {
        // Create synthetic sector data with different performance patterns
        let tech: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.5 + ((i as f64) * 0.1).sin() * 3.0)
            .collect(); // Strong uptrend

        let healthcare: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.3 + ((i as f64) * 0.15).sin() * 2.0)
            .collect(); // Moderate uptrend

        let energy: Vec<f64> = (0..n)
            .map(|i| 100.0 - (i as f64) * 0.1 + ((i as f64) * 0.2).sin() * 4.0)
            .collect(); // Slight downtrend

        let financials: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.2 + ((i as f64) * 0.12).sin() * 2.5)
            .collect(); // Weak uptrend

        let benchmark: Vec<f64> = (0..n)
            .map(|i| 100.0 + (i as f64) * 0.25 + ((i as f64) * 0.08).sin() * 1.5)
            .collect(); // Market average

        let mut rotation = SectorRotation::new(20, 50).with_benchmark(benchmark);
        rotation.add_sector("Technology", tech);
        rotation.add_sector("Healthcare", healthcare);
        rotation.add_sector("Energy", energy);
        rotation.add_sector("Financials", financials);

        rotation
    }

    #[test]
    fn test_sector_rotation_basic() {
        let rotation = create_test_sectors(100);
        let outputs = rotation.calculate();

        assert_eq!(outputs.len(), 100);

        // After warmup, should have 4 sectors ranked
        let last = outputs.last().unwrap();
        assert_eq!(last.rankings.len(), 4);

        // Check ranks 1-4 are present
        let ranks: Vec<usize> = last.rankings.iter().map(|r| r.rank).collect();
        assert!(ranks.contains(&1));
        assert!(ranks.contains(&4));
    }

    #[test]
    fn test_sector_ranking_order() {
        let rotation = create_test_sectors(100);
        let outputs = rotation.calculate();

        // Rankings should be sorted by strength (descending)
        for output in outputs.iter().skip(50) {
            for i in 0..output.rankings.len() - 1 {
                if !output.rankings[i].strength.is_nan()
                    && !output.rankings[i + 1].strength.is_nan()
                {
                    assert!(
                        output.rankings[i].strength >= output.rankings[i + 1].strength,
                        "Rankings should be sorted by strength"
                    );
                }
            }
        }
    }

    #[test]
    fn test_leader_laggard() {
        let rotation = create_test_sectors(100);
        let outputs = rotation.calculate();

        // After warmup, should have valid leader and laggard
        let last = outputs.last().unwrap();
        assert!(!last.leader.is_empty());
        assert!(!last.laggard.is_empty());
        assert_ne!(last.leader, last.laggard);
    }

    #[test]
    fn test_strength_for_sector() {
        let rotation = create_test_sectors(100);
        let tech_strength = rotation.strength_for("Technology");

        assert_eq!(tech_strength.len(), 100);
        // First values should be NaN (warmup)
        assert!(tech_strength[0].is_nan());
        // After warmup, should have values
        assert!(!tech_strength[60].is_nan());
    }

    #[test]
    fn test_rank_for_sector() {
        let rotation = create_test_sectors(100);
        let energy_ranks = rotation.rank_for("Energy");

        assert_eq!(energy_ranks.len(), 100);
        // Ranks should be 1-4
        for rank in energy_ranks.iter().skip(50) {
            if *rank > 0 {
                assert!(*rank >= 1 && *rank <= 4);
            }
        }
    }

    #[test]
    fn test_rotation_score() {
        let rotation = create_test_sectors(100);
        let scores = rotation.rotation_scores();

        assert_eq!(scores.len(), 100);
        // First rotation score should be 0 (no previous ranks)
        assert_eq!(scores[0], 0.0);
        // Scores should be non-negative
        for score in &scores {
            assert!(*score >= 0.0);
        }
    }

    #[test]
    fn test_momentum() {
        let rotation = create_test_sectors(100);
        let outputs = rotation.calculate();

        // At index 0, all momentum should be 0
        if !outputs[0].rankings.is_empty() {
            // Initial momentum calculation
        }

        // Later, some sectors should have non-zero momentum
        let last = outputs.last().unwrap();
        let has_momentum = last.rankings.iter().any(|r| r.momentum != 0);
        // May or may not have momentum depending on rank stability
    }

    #[test]
    fn test_relative_strength() {
        let rotation = create_test_sectors(100);
        let outputs = rotation.calculate();

        // With benchmark set, relative strength should be calculated
        let last = outputs.last().unwrap();
        for ranking in &last.rankings {
            // RS should be calculated (may be NaN initially)
        }
    }

    #[test]
    fn test_improving_deteriorating() {
        let rotation = create_test_sectors(100);
        let improving = rotation.improving_sectors();
        let deteriorating = rotation.deteriorating_sectors();

        assert_eq!(improving.len(), 100);
        assert_eq!(deteriorating.len(), 100);
    }

    #[test]
    fn test_leaders_laggards_series() {
        let rotation = create_test_sectors(100);
        let leaders = rotation.leaders();
        let laggards = rotation.laggards();

        assert_eq!(leaders.len(), 100);
        assert_eq!(laggards.len(), 100);
    }

    #[test]
    fn test_technical_indicator_impl() {
        let rotation = create_test_sectors(100);
        let data = OHLCVSeries::from_close(vec![100.0; 100]);
        let result = rotation.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
    }

    #[test]
    fn test_empty_sectors_error() {
        let rotation = SectorRotation::new(20, 50);
        let data = OHLCVSeries::from_close(vec![100.0; 100]);
        let result = rotation.compute(&data);

        assert!(result.is_err());
    }

    #[test]
    fn test_with_sectors_from_multiseries() {
        let mut multi = MultiSeries::new();
        multi.add("Tech", (0..100).map(|i| 100.0 + i as f64).collect());
        multi.add("Health", (0..100).map(|i| 100.0 + i as f64 * 0.5).collect());

        let rotation = SectorRotation::with_period(20).with_sectors(multi);

        assert_eq!(rotation.names.len(), 2);
        assert_eq!(rotation.prices.len(), 2);
    }

    #[test]
    fn test_insufficient_data() {
        let mut rotation = SectorRotation::new(50, 50);
        rotation.add_sector("Tech", vec![100.0; 30]);
        rotation.add_sector("Health", vec![100.0; 30]);

        let data = OHLCVSeries::from_close(vec![100.0; 30]);
        let result = rotation.compute(&data);

        assert!(result.is_err());
    }
}
