//! Fibonacci Clusters - Multi-swing confluence zones.
//!
//! IND-391: Fibonacci Clusters identify price zones where multiple
//! Fibonacci retracement levels from different swing points converge,
//! creating strong support/resistance areas.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Fibonacci cluster data showing confluence strength at price levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterZone {
    /// Center price of the cluster zone
    pub price_level: f64,
    /// Number of Fibonacci levels converging here
    pub confluence_count: usize,
    /// Strength score (0-100)
    pub strength: f64,
    /// Is this a support zone (below current price)
    pub is_support: bool,
}

/// Fibonacci Clusters Indicator
///
/// Identifies price zones where multiple Fibonacci retracement levels
/// from different swing highs/lows converge. These confluence zones
/// represent stronger support/resistance levels.
///
/// # Interpretation
/// - High confluence = stronger support/resistance
/// - Multiple swings confirming same level increases significance
/// - Useful for identifying key reversal zones
///
/// # Calculation
/// 1. Identify all significant swing highs and lows
/// 2. Calculate Fibonacci retracements for each swing pair
/// 3. Count how many Fib levels fall within each price zone
/// 4. Higher count = stronger cluster
#[derive(Debug, Clone)]
pub struct FibonacciClusters {
    /// Lookback period to find swings
    lookback: usize,
    /// Swing detection strength
    swing_strength: usize,
    /// Price zone width as percentage
    zone_width_pct: f64,
    /// Number of price bins for clustering
    num_bins: usize,
    /// Minimum swings required
    min_swings: usize,
}

impl FibonacciClusters {
    /// Create a new Fibonacci Clusters indicator.
    ///
    /// # Arguments
    /// * `lookback` - Period to search for swing points (minimum 20)
    /// * `swing_strength` - Bars on each side to confirm swing (minimum 2)
    /// * `zone_width_pct` - Width of cluster zone as percent of price (minimum 0.1)
    /// * `num_bins` - Number of bins for price clustering (minimum 10)
    pub fn new(
        lookback: usize,
        swing_strength: usize,
        zone_width_pct: f64,
        num_bins: usize,
    ) -> Result<Self> {
        if lookback < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        if swing_strength < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "swing_strength".to_string(),
                reason: "must be at least 2".to_string(),
            });
        }
        if zone_width_pct < 0.1 {
            return Err(IndicatorError::InvalidParameter {
                name: "zone_width_pct".to_string(),
                reason: "must be at least 0.1".to_string(),
            });
        }
        if num_bins < 10 {
            return Err(IndicatorError::InvalidParameter {
                name: "num_bins".to_string(),
                reason: "must be at least 10".to_string(),
            });
        }
        Ok(Self {
            lookback,
            swing_strength,
            zone_width_pct,
            num_bins,
            min_swings: 2,
        })
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self {
            lookback: 50,
            swing_strength: 3,
            zone_width_pct: 0.5,
            num_bins: 50,
            min_swings: 2,
        }
    }

    /// Set minimum swings required.
    pub fn with_min_swings(mut self, min_swings: usize) -> Self {
        self.min_swings = min_swings.max(1);
        self
    }

    /// Find all swing highs in the data range.
    fn find_swing_highs(&self, high: &[f64], start: usize, end: usize) -> Vec<(usize, f64)> {
        let mut swings = Vec::new();

        for i in (start + self.swing_strength)..(end.saturating_sub(self.swing_strength)) {
            let mut is_swing = true;

            for j in (i.saturating_sub(self.swing_strength))..i {
                if high[j] >= high[i] {
                    is_swing = false;
                    break;
                }
            }

            if is_swing {
                for j in (i + 1)..=(i + self.swing_strength).min(end - 1) {
                    if high[j] >= high[i] {
                        is_swing = false;
                        break;
                    }
                }
            }

            if is_swing {
                swings.push((i, high[i]));
            }
        }

        swings
    }

    /// Find all swing lows in the data range.
    fn find_swing_lows(&self, low: &[f64], start: usize, end: usize) -> Vec<(usize, f64)> {
        let mut swings = Vec::new();

        for i in (start + self.swing_strength)..(end.saturating_sub(self.swing_strength)) {
            let mut is_swing = true;

            for j in (i.saturating_sub(self.swing_strength))..i {
                if low[j] <= low[i] {
                    is_swing = false;
                    break;
                }
            }

            if is_swing {
                for j in (i + 1)..=(i + self.swing_strength).min(end - 1) {
                    if low[j] <= low[i] {
                        is_swing = false;
                        break;
                    }
                }
            }

            if is_swing {
                swings.push((i, low[i]));
            }
        }

        swings
    }

    /// Calculate Fibonacci levels between two prices.
    fn fib_levels(&self, high: f64, low: f64) -> Vec<f64> {
        let range = high - low;
        vec![
            high,                      // 0%
            high - range * 0.236,      // 23.6%
            high - range * 0.382,      // 38.2%
            high - range * 0.500,      // 50%
            high - range * 0.618,      // 61.8%
            high - range * 0.786,      // 78.6%
            low,                       // 100%
        ]
    }

    /// Calculate cluster strength at each price level.
    pub fn calculate(&self, high: &[f64], low: &[f64], close: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut cluster_strength = vec![0.0; n];
        let mut nearest_cluster = vec![f64::NAN; n];

        if n < self.lookback {
            return (cluster_strength, nearest_cluster);
        }

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            // Find all swings
            let swing_highs = self.find_swing_highs(high, start, i);
            let swing_lows = self.find_swing_lows(low, start, i);

            if swing_highs.len() < self.min_swings || swing_lows.len() < self.min_swings {
                continue;
            }

            // Find price range for binning
            let mut min_price = f64::INFINITY;
            let mut max_price = f64::NEG_INFINITY;
            for j in start..=i {
                min_price = min_price.min(low[j]);
                max_price = max_price.max(high[j]);
            }

            let price_range = max_price - min_price;
            if price_range < 1e-10 {
                continue;
            }

            // Create bins and count Fib level occurrences
            let bin_size = price_range / self.num_bins as f64;
            let mut bin_counts = vec![0usize; self.num_bins];

            // Generate Fib levels for all swing pairs
            for &(_, sh_price) in &swing_highs {
                for &(_, sl_price) in &swing_lows {
                    if sh_price > sl_price {
                        let levels = self.fib_levels(sh_price, sl_price);
                        for level in levels {
                            if level >= min_price && level <= max_price {
                                let bin = ((level - min_price) / bin_size).floor() as usize;
                                let bin = bin.min(self.num_bins - 1);
                                bin_counts[bin] += 1;
                            }
                        }
                    }
                }
            }

            // Find maximum confluence
            let max_count = *bin_counts.iter().max().unwrap_or(&0);
            if max_count == 0 {
                continue;
            }

            // Calculate strength as percentage of maximum possible confluence
            let max_possible = swing_highs.len() * swing_lows.len() * 7; // 7 fib levels per pair
            cluster_strength[i] = (max_count as f64 / max_possible as f64) * 100.0;

            // Find nearest cluster to current price
            let current_bin = ((close[i] - min_price) / bin_size).floor() as usize;
            let current_bin = current_bin.min(self.num_bins - 1);

            let mut best_dist = usize::MAX;
            let mut best_bin = current_bin;

            for (bin, &count) in bin_counts.iter().enumerate() {
                if count > max_count / 2 {
                    // Significant cluster
                    let dist = if bin > current_bin {
                        bin - current_bin
                    } else {
                        current_bin - bin
                    };
                    if dist < best_dist {
                        best_dist = dist;
                        best_bin = bin;
                    }
                }
            }

            nearest_cluster[i] = min_price + (best_bin as f64 + 0.5) * bin_size;
        }

        (cluster_strength, nearest_cluster)
    }

    /// Get detailed cluster zones at a specific bar.
    pub fn get_cluster_zones(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        bar_index: usize,
    ) -> Vec<ClusterZone> {
        let mut zones = Vec::new();

        if bar_index >= close.len() || bar_index < self.lookback {
            return zones;
        }

        let start = bar_index.saturating_sub(self.lookback);

        // Find all swings
        let swing_highs = self.find_swing_highs(high, start, bar_index);
        let swing_lows = self.find_swing_lows(low, start, bar_index);

        if swing_highs.len() < self.min_swings || swing_lows.len() < self.min_swings {
            return zones;
        }

        // Find price range
        let mut min_price = f64::INFINITY;
        let mut max_price = f64::NEG_INFINITY;
        for j in start..=bar_index {
            min_price = min_price.min(low[j]);
            max_price = max_price.max(high[j]);
        }

        let price_range = max_price - min_price;
        if price_range < 1e-10 {
            return zones;
        }

        let bin_size = price_range / self.num_bins as f64;
        let mut bin_counts = vec![0usize; self.num_bins];

        // Count Fib levels in each bin
        for &(_, sh_price) in &swing_highs {
            for &(_, sl_price) in &swing_lows {
                if sh_price > sl_price {
                    let levels = self.fib_levels(sh_price, sl_price);
                    for level in levels {
                        if level >= min_price && level <= max_price {
                            let bin = ((level - min_price) / bin_size).floor() as usize;
                            let bin = bin.min(self.num_bins - 1);
                            bin_counts[bin] += 1;
                        }
                    }
                }
            }
        }

        // Find significant clusters
        let max_count = *bin_counts.iter().max().unwrap_or(&0);
        let threshold = (max_count as f64 * 0.3).ceil() as usize;

        for (bin, &count) in bin_counts.iter().enumerate() {
            if count >= threshold && count >= 2 {
                let price_level = min_price + (bin as f64 + 0.5) * bin_size;
                let max_possible = swing_highs.len() * swing_lows.len() * 7;
                let strength = (count as f64 / max_possible as f64) * 100.0;

                zones.push(ClusterZone {
                    price_level,
                    confluence_count: count,
                    strength,
                    is_support: price_level < close[bar_index],
                });
            }
        }

        // Sort by strength
        zones.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());

        zones
    }
}

impl Default for FibonacciClusters {
    fn default() -> Self {
        Self::default_params()
    }
}

impl TechnicalIndicator for FibonacciClusters {
    fn name(&self) -> &str {
        "Fibonacci Clusters"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        let (strength, nearest) = self.calculate(&data.high, &data.low, &data.close);
        Ok(IndicatorOutput::dual(strength, nearest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        // Create data with clear swings for cluster detection
        let mut high = Vec::new();
        let mut low = Vec::new();
        let mut close = Vec::new();

        for i in 0..80 {
            let base = 100.0;
            let trend = (i as f64) * 0.2;
            let swing = (i as f64 * 0.15).sin() * 10.0;
            close.push(base + trend + swing);
            high.push(base + trend + swing + 2.0);
            low.push(base + trend + swing - 2.0);
        }

        (high, low, close)
    }

    #[test]
    fn test_fib_clusters_creation() {
        let clusters = FibonacciClusters::new(30, 3, 0.5, 20);
        assert!(clusters.is_ok());

        let clusters = FibonacciClusters::new(10, 3, 0.5, 20);
        assert!(clusters.is_err());

        let clusters = FibonacciClusters::new(30, 1, 0.5, 20);
        assert!(clusters.is_err());

        let clusters = FibonacciClusters::new(30, 3, 0.05, 20);
        assert!(clusters.is_err());

        let clusters = FibonacciClusters::new(30, 3, 0.5, 5);
        assert!(clusters.is_err());
    }

    #[test]
    fn test_fib_clusters_calculation() {
        let (high, low, close) = make_test_data();
        let clusters = FibonacciClusters::new(30, 2, 0.5, 30).unwrap();
        let (strength, nearest) = clusters.calculate(&high, &low, &close);

        assert_eq!(strength.len(), close.len());
        assert_eq!(nearest.len(), close.len());

        // Should have valid values after lookback
        let valid_strength = strength.iter().filter(|v| **v > 0.0).count();
        assert!(valid_strength > 0);
    }

    #[test]
    fn test_fib_clusters_strength_range() {
        let (high, low, close) = make_test_data();
        let clusters = FibonacciClusters::default_params();
        let (strength, _) = clusters.calculate(&high, &low, &close);

        // Strength should be between 0 and 100
        for s in strength.iter() {
            assert!(*s >= 0.0 && *s <= 100.0);
        }
    }

    #[test]
    fn test_fib_clusters_zones() {
        let (high, low, close) = make_test_data();
        let clusters = FibonacciClusters::new(30, 2, 0.5, 30).unwrap();

        let zones = clusters.get_cluster_zones(&high, &low, &close, 60);

        // Should find some zones
        if !zones.is_empty() {
            // Zones should be sorted by strength (descending)
            for i in 1..zones.len() {
                assert!(zones[i - 1].strength >= zones[i].strength);
            }
        }
    }

    #[test]
    fn test_fib_clusters_zone_classification() {
        let (high, low, close) = make_test_data();
        let clusters = FibonacciClusters::default_params();

        let zones = clusters.get_cluster_zones(&high, &low, &close, 70);

        for zone in zones.iter() {
            // is_support should be true if zone is below current price
            if zone.is_support {
                assert!(zone.price_level < close[70]);
            } else {
                assert!(zone.price_level >= close[70]);
            }
        }
    }

    #[test]
    fn test_fib_clusters_with_min_swings() {
        let clusters = FibonacciClusters::default_params().with_min_swings(3);
        assert_eq!(clusters.min_swings, 3);
    }

    #[test]
    fn test_fib_clusters_technical_indicator() {
        let clusters = FibonacciClusters::default_params();
        assert_eq!(clusters.name(), "Fibonacci Clusters");
        assert_eq!(clusters.min_periods(), 51);
    }

    #[test]
    fn test_fib_clusters_compute() {
        let (high, low, close) = make_test_data();
        let volume = vec![1000.0; close.len()];

        let data = OHLCVSeries {
            open: close.clone(),
            high,
            low,
            close,
            volume,
        };

        let clusters = FibonacciClusters::default_params();
        let result = clusters.compute(&data);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.values.len(), 2); // strength and nearest
    }

    #[test]
    fn test_fib_clusters_swing_detection() {
        let (high, low, _) = make_test_data();
        let clusters = FibonacciClusters::new(30, 2, 0.5, 30).unwrap();

        let swing_highs = clusters.find_swing_highs(&high, 0, 50);
        let swing_lows = clusters.find_swing_lows(&low, 0, 50);

        // Should find some swings in the test data
        assert!(!swing_highs.is_empty());
        assert!(!swing_lows.is_empty());
    }
}
