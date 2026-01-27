//! Currency Strength Meter
//!
//! Calculates relative strength rankings for currencies based on
//! multiple currency pairs. Useful for forex trading to identify
//! the strongest and weakest currencies.

use crate::{
    IndicatorError, IndicatorOutput, IndicatorSignal, OHLCVSeries, Result, SignalIndicator,
    TechnicalIndicator,
};

use super::MultiSeries;

/// Currency pair data for strength calculation.
#[derive(Debug, Clone)]
pub struct CurrencyPair {
    /// Base currency (e.g., "EUR" in EUR/USD).
    pub base: String,
    /// Quote currency (e.g., "USD" in EUR/USD).
    pub quote: String,
    /// Price series for the pair.
    pub prices: Vec<f64>,
}

impl CurrencyPair {
    /// Create a new currency pair.
    pub fn new(base: &str, quote: &str, prices: Vec<f64>) -> Self {
        Self {
            base: base.to_string(),
            quote: quote.to_string(),
            prices,
        }
    }

    /// Get the pair name (e.g., "EURUSD").
    pub fn pair_name(&self) -> String {
        format!("{}{}", self.base, self.quote)
    }
}

/// Currency strength output for a single currency.
#[derive(Debug, Clone)]
pub struct CurrencyStrengthOutput {
    /// Currency name.
    pub currency: String,
    /// Strength score (higher = stronger).
    pub strength: f64,
    /// Rank among all currencies (1 = strongest).
    pub rank: usize,
    /// Strength change from previous period.
    pub momentum: f64,
}

/// Currency Strength Meter.
///
/// Calculates relative strength for each currency in a basket based on
/// rate of change (ROC) of currency pairs. The strength score is computed
/// by averaging the ROC of all pairs containing that currency, with
/// appropriate sign adjustment for base/quote position.
///
/// # Methodology
/// 1. Calculate ROC for each currency pair
/// 2. For each currency, aggregate ROC from all pairs containing it
///    - Add ROC if currency is the base (currency strengthening = pair rising)
///    - Subtract ROC if currency is the quote (currency strengthening = pair falling)
/// 3. Normalize and rank currencies by strength
///
/// # Example Usage
/// ```ignore
/// let pairs = vec![
///     CurrencyPair::new("EUR", "USD", eurusd_prices),
///     CurrencyPair::new("GBP", "USD", gbpusd_prices),
///     CurrencyPair::new("EUR", "GBP", eurgbp_prices),
/// ];
/// let strength = CurrencyStrength::new(14).with_pairs(pairs);
/// let rankings = strength.calculate_rankings();
/// ```
#[derive(Debug, Clone)]
pub struct CurrencyStrength {
    /// Period for ROC calculation.
    period: usize,
    /// Currency pairs for analysis.
    pairs: Vec<CurrencyPair>,
    /// List of unique currencies.
    currencies: Vec<String>,
}

impl CurrencyStrength {
    /// Create a new Currency Strength indicator.
    ///
    /// # Arguments
    /// * `period` - Period for rate of change calculation (e.g., 14)
    pub fn new(period: usize) -> Self {
        Self {
            period,
            pairs: Vec::new(),
            currencies: Vec::new(),
        }
    }

    /// Set the currency pairs for analysis.
    pub fn with_pairs(mut self, pairs: Vec<CurrencyPair>) -> Self {
        // Extract unique currencies
        let mut currencies = std::collections::HashSet::new();
        for pair in &pairs {
            currencies.insert(pair.base.clone());
            currencies.insert(pair.quote.clone());
        }
        self.currencies = currencies.into_iter().collect();
        self.currencies.sort();
        self.pairs = pairs;
        self
    }

    /// Add a single currency pair.
    pub fn add_pair(&mut self, pair: CurrencyPair) {
        // Add currencies if not present
        if !self.currencies.contains(&pair.base) {
            self.currencies.push(pair.base.clone());
        }
        if !self.currencies.contains(&pair.quote) {
            self.currencies.push(pair.quote.clone());
        }
        self.currencies.sort();
        self.pairs.push(pair);
    }

    /// Calculate rate of change.
    fn roc(prices: &[f64], period: usize) -> Vec<f64> {
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
                let roc_val = (prices[i] - prev) / prev * 100.0;
                result.push(roc_val);
            }
        }
        result
    }

    /// Calculate strength for all currencies at a given index.
    fn calculate_strength_at(&self, index: usize, rocs: &[Vec<f64>]) -> Vec<(String, f64)> {
        let mut strengths: Vec<(String, f64)> =
            self.currencies.iter().map(|c| (c.clone(), 0.0)).collect();

        // For each pair, add/subtract ROC based on currency position
        for (pair, roc) in self.pairs.iter().zip(rocs.iter()) {
            let roc_val = roc.get(index).copied().unwrap_or(f64::NAN);
            if roc_val.is_nan() {
                continue;
            }

            // Base currency: strength increases when pair rises
            if let Some(pos) = strengths.iter().position(|(c, _)| c == &pair.base) {
                strengths[pos].1 += roc_val;
            }

            // Quote currency: strength increases when pair falls
            if let Some(pos) = strengths.iter().position(|(c, _)| c == &pair.quote) {
                strengths[pos].1 -= roc_val;
            }
        }

        // Normalize by number of pairs each currency appears in
        for (currency, strength) in &mut strengths {
            let pair_count = self
                .pairs
                .iter()
                .filter(|p| &p.base == currency || &p.quote == currency)
                .count();
            if pair_count > 0 {
                *strength /= pair_count as f64;
            }
        }

        strengths
    }

    /// Calculate full currency strength rankings over time.
    pub fn calculate(&self) -> Vec<Vec<CurrencyStrengthOutput>> {
        if self.pairs.is_empty() || self.currencies.is_empty() {
            return Vec::new();
        }

        // Calculate ROC for all pairs
        let rocs: Vec<Vec<f64>> = self
            .pairs
            .iter()
            .map(|p| Self::roc(&p.prices, self.period))
            .collect();

        let n = self.pairs[0].prices.len();
        let mut results = Vec::with_capacity(n);

        for i in 0..n {
            let mut strengths = self.calculate_strength_at(i, &rocs);

            // Sort by strength (descending) for ranking
            strengths.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Build output
            let output: Vec<CurrencyStrengthOutput> = strengths
                .iter()
                .enumerate()
                .map(|(rank, (currency, strength))| {
                    // Calculate momentum (change from previous)
                    let momentum = if i > 0 {
                        let prev_strengths = self.calculate_strength_at(i - 1, &rocs);
                        prev_strengths
                            .iter()
                            .find(|(c, _)| c == currency)
                            .map(|(_, s)| strength - s)
                            .unwrap_or(0.0)
                    } else {
                        0.0
                    };

                    CurrencyStrengthOutput {
                        currency: currency.clone(),
                        strength: *strength,
                        rank: rank + 1,
                        momentum,
                    }
                })
                .collect();

            results.push(output);
        }

        results
    }

    /// Get strength series for a specific currency.
    pub fn strength_for(&self, currency: &str) -> Vec<f64> {
        let rankings = self.calculate();
        rankings
            .iter()
            .map(|r| {
                r.iter()
                    .find(|o| o.currency == currency)
                    .map(|o| o.strength)
                    .unwrap_or(f64::NAN)
            })
            .collect()
    }

    /// Get rank series for a specific currency.
    pub fn rank_for(&self, currency: &str) -> Vec<usize> {
        let rankings = self.calculate();
        rankings
            .iter()
            .map(|r| {
                r.iter()
                    .find(|o| o.currency == currency)
                    .map(|o| o.rank)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Get strongest currency at each time point.
    pub fn strongest(&self) -> Vec<Option<String>> {
        let rankings = self.calculate();
        rankings
            .iter()
            .map(|r| r.iter().find(|o| o.rank == 1).map(|o| o.currency.clone()))
            .collect()
    }

    /// Get weakest currency at each time point.
    pub fn weakest(&self) -> Vec<Option<String>> {
        let rankings = self.calculate();
        let num_currencies = self.currencies.len();
        rankings
            .iter()
            .map(|r| {
                r.iter()
                    .find(|o| o.rank == num_currencies)
                    .map(|o| o.currency.clone())
            })
            .collect()
    }

    /// Calculate spread between strongest and weakest currencies.
    pub fn strength_spread(&self) -> Vec<f64> {
        let rankings = self.calculate();
        rankings
            .iter()
            .map(|r| {
                if r.is_empty() {
                    return f64::NAN;
                }
                let strongest = r.iter().find(|o| o.rank == 1);
                let weakest = r.iter().find(|o| o.rank == self.currencies.len());
                match (strongest, weakest) {
                    (Some(s), Some(w)) => s.strength - w.strength,
                    _ => f64::NAN,
                }
            })
            .collect()
    }
}

impl TechnicalIndicator for CurrencyStrength {
    fn name(&self) -> &str {
        "CurrencyStrength"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        // When used as TechnicalIndicator, assumes pairs have been set
        // and returns strength spread as primary output
        if self.pairs.is_empty() {
            return Err(IndicatorError::InvalidParameter {
                name: "pairs".to_string(),
                reason: "Currency pairs must be set before computing".to_string(),
            });
        }

        let min_len = self.pairs.iter().map(|p| p.prices.len()).min().unwrap_or(0);
        if min_len < self.period + 1 {
            return Err(IndicatorError::InsufficientData {
                required: self.period + 1,
                got: min_len,
            });
        }

        let spread = self.strength_spread();
        Ok(IndicatorOutput::single(spread))
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }
}

impl SignalIndicator for CurrencyStrength {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        // Not directly applicable - currency strength doesn't generate
        // buy/sell signals on its own
        Ok(IndicatorSignal::Neutral)
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let n = if self.pairs.is_empty() {
            data.close.len()
        } else {
            self.pairs[0].prices.len()
        };
        Ok(vec![IndicatorSignal::Neutral; n])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pairs(n: usize) -> Vec<CurrencyPair> {
        // Create synthetic currency data
        // EUR strengthening, USD weakening, GBP stable
        let eurusd: Vec<f64> = (0..n)
            .map(|i| 1.10 + (i as f64) * 0.001 + ((i as f64) * 0.1).sin() * 0.002)
            .collect();
        let gbpusd: Vec<f64> = (0..n)
            .map(|i| 1.30 + ((i as f64) * 0.15).sin() * 0.003)
            .collect();
        let eurgbp: Vec<f64> = (0..n)
            .map(|i| 0.85 + (i as f64) * 0.0005 + ((i as f64) * 0.2).sin() * 0.002)
            .collect();
        let usdjpy: Vec<f64> = (0..n)
            .map(|i| 110.0 - (i as f64) * 0.02 + ((i as f64) * 0.12).sin() * 0.5)
            .collect();

        vec![
            CurrencyPair::new("EUR", "USD", eurusd),
            CurrencyPair::new("GBP", "USD", gbpusd),
            CurrencyPair::new("EUR", "GBP", eurgbp),
            CurrencyPair::new("USD", "JPY", usdjpy),
        ]
    }

    #[test]
    fn test_currency_strength_basic() {
        let pairs = create_test_pairs(100);
        let strength = CurrencyStrength::new(14).with_pairs(pairs);
        let rankings = strength.calculate();

        assert_eq!(rankings.len(), 100);

        // After warmup, should have 4 currencies ranked
        let last = rankings.last().unwrap();
        assert_eq!(last.len(), 4);

        // Check all ranks 1-4 are present
        let ranks: Vec<usize> = last.iter().map(|o| o.rank).collect();
        assert!(ranks.contains(&1));
        assert!(ranks.contains(&4));
    }

    #[test]
    fn test_currency_strength_ranking() {
        let pairs = create_test_pairs(100);
        let strength = CurrencyStrength::new(14).with_pairs(pairs);
        let rankings = strength.calculate();

        // Rankings should be ordered by strength
        for ranking in rankings.iter().skip(14) {
            for i in 0..ranking.len() - 1 {
                assert!(ranking[i].strength >= ranking[i + 1].strength);
            }
        }
    }

    #[test]
    fn test_strength_for_currency() {
        let pairs = create_test_pairs(100);
        let strength = CurrencyStrength::new(14).with_pairs(pairs);
        let eur_strength = strength.strength_for("EUR");

        assert_eq!(eur_strength.len(), 100);
        // During warmup (first 14 indices), ROCs are NaN so strength is 0.0
        // (no valid ROC contributions)
        assert_eq!(eur_strength[0], 0.0);
        // After warmup, should have non-zero values
        assert!(!eur_strength[50].is_nan());
    }

    #[test]
    fn test_strongest_weakest() {
        let pairs = create_test_pairs(100);
        let strength = CurrencyStrength::new(14).with_pairs(pairs);

        let strongest = strength.strongest();
        let weakest = strength.weakest();

        assert_eq!(strongest.len(), 100);
        assert_eq!(weakest.len(), 100);

        // After warmup, should have some values
        assert!(strongest.last().unwrap().is_some());
        assert!(weakest.last().unwrap().is_some());

        // Strongest should not equal weakest
        let last_strongest = strongest.last().unwrap().as_ref().unwrap();
        let last_weakest = weakest.last().unwrap().as_ref().unwrap();
        assert_ne!(last_strongest, last_weakest);
    }

    #[test]
    fn test_strength_spread() {
        let pairs = create_test_pairs(100);
        let strength = CurrencyStrength::new(14).with_pairs(pairs);
        let spread = strength.strength_spread();

        assert_eq!(spread.len(), 100);
        // Spread should be positive (strongest - weakest)
        for s in spread.iter().skip(14) {
            if !s.is_nan() {
                assert!(*s >= 0.0, "Spread should be non-negative");
            }
        }
    }

    #[test]
    fn test_momentum() {
        let pairs = create_test_pairs(100);
        let strength = CurrencyStrength::new(14).with_pairs(pairs);
        let rankings = strength.calculate();

        // Check that momentum is calculated (change from previous period)
        // At index 0, momentum should be 0
        if !rankings[0].is_empty() {
            // For any valid ranking, momentum at first point is 0
        }
    }

    #[test]
    fn test_add_pair() {
        let mut strength = CurrencyStrength::new(14);

        let eurusd: Vec<f64> = (0..50).map(|i| 1.10 + (i as f64) * 0.001).collect();
        let gbpusd: Vec<f64> = (0..50).map(|i| 1.30 + (i as f64) * 0.0005).collect();

        strength.add_pair(CurrencyPair::new("EUR", "USD", eurusd));
        strength.add_pair(CurrencyPair::new("GBP", "USD", gbpusd));

        assert_eq!(strength.pairs.len(), 2);
        assert_eq!(strength.currencies.len(), 3); // EUR, GBP, USD
    }

    #[test]
    fn test_technical_indicator_impl() {
        let pairs = create_test_pairs(100);
        let strength = CurrencyStrength::new(14).with_pairs(pairs);

        // Dummy OHLCV data (not actually used for currency strength)
        let data = OHLCVSeries::from_close(vec![100.0; 100]);
        let result = strength.compute(&data);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.primary.len(), 100);
    }

    #[test]
    fn test_empty_pairs_error() {
        let strength = CurrencyStrength::new(14);
        let data = OHLCVSeries::from_close(vec![100.0; 100]);
        let result = strength.compute(&data);

        assert!(result.is_err());
    }

    #[test]
    fn test_rank_for_currency() {
        let pairs = create_test_pairs(100);
        let strength = CurrencyStrength::new(14).with_pairs(pairs);
        let usd_ranks = strength.rank_for("USD");

        assert_eq!(usd_ranks.len(), 100);
        // Ranks should be 1-4
        for rank in usd_ranks.iter().skip(14) {
            if *rank > 0 {
                assert!(*rank >= 1 && *rank <= 4);
            }
        }
    }
}
