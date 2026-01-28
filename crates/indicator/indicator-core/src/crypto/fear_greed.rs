//! Fear & Greed Index Components - IND-089
//!
//! A composite sentiment indicator combining multiple market factors.
//!
//! Components (typical weights):
//! - Volatility (25%): Current vs 30/90-day average
//! - Market Momentum/Volume (25%): Price vs MA, volume trends
//! - Social Media (15%): Sentiment from social platforms
//! - Dominance (10%): Bitcoin dominance trends
//! - Trends (10%): Google Trends data
//! - Surveys (15%): Investor polls
//!
//! Scale: 0-100
//! - 0-25: Extreme Fear (buy opportunity)
//! - 25-50: Fear
//! - 50-75: Greed
//! - 75-100: Extreme Greed (sell opportunity)

use indicator_spi::IndicatorSignal;

/// Fear & Greed output.
#[derive(Debug, Clone)]
pub struct FearGreedOutput {
    /// Composite Fear & Greed index (0-100).
    pub index: Vec<f64>,
    /// Volatility component (0-100).
    pub volatility_score: Vec<f64>,
    /// Momentum component (0-100).
    pub momentum_score: Vec<f64>,
    /// Volume component (0-100).
    pub volume_score: Vec<f64>,
}

/// Fear & Greed sentiment level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FearGreedLevel {
    /// Index 0-25: Extreme Fear.
    ExtremeFear,
    /// Index 25-50: Fear.
    Fear,
    /// Index 50-75: Greed.
    Greed,
    /// Index 75-100: Extreme Greed.
    ExtremeGreed,
}

/// Component weights for Fear & Greed calculation.
#[derive(Debug, Clone)]
pub struct FearGreedWeights {
    pub volatility: f64,
    pub momentum: f64,
    pub volume: f64,
}

impl Default for FearGreedWeights {
    fn default() -> Self {
        Self {
            volatility: 0.40,
            momentum: 0.35,
            volume: 0.25,
        }
    }
}

/// Fear & Greed Index Components - IND-089
///
/// Calculates a composite fear/greed index from market data.
///
/// # Example
/// ```
/// use indicator_core::crypto::FearGreedIndex;
///
/// let fgi = FearGreedIndex::new(14, 30);
/// let prices = vec![100.0, 105.0, 102.0, 108.0];
/// let volumes = vec![1e6, 1.2e6, 0.9e6, 1.5e6];
/// let output = fgi.calculate(&prices, &volumes);
/// ```
#[derive(Debug, Clone)]
pub struct FearGreedIndex {
    /// Period for short-term calculations.
    short_period: usize,
    /// Period for long-term baseline.
    long_period: usize,
    /// Component weights.
    weights: FearGreedWeights,
}

impl FearGreedIndex {
    /// Create a new Fear & Greed Index indicator.
    pub fn new(short_period: usize, long_period: usize) -> Self {
        Self {
            short_period,
            long_period,
            weights: FearGreedWeights::default(),
        }
    }

    /// Create with custom weights.
    pub fn with_weights(short_period: usize, long_period: usize, weights: FearGreedWeights) -> Self {
        Self {
            short_period,
            long_period,
            weights,
        }
    }

    /// Calculate Fear & Greed Index from price and volume data.
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> FearGreedOutput {
        let n = close.len().min(volume.len());

        if n < self.long_period {
            return FearGreedOutput {
                index: vec![f64::NAN; n],
                volatility_score: vec![f64::NAN; n],
                momentum_score: vec![f64::NAN; n],
                volume_score: vec![f64::NAN; n],
            };
        }

        // Calculate component scores
        let volatility_score = self.calculate_volatility_score(close);
        let momentum_score = self.calculate_momentum_score(close);
        let volume_score = self.calculate_volume_score(volume);

        // Calculate composite index
        let index: Vec<f64> = (0..n)
            .map(|i| {
                if volatility_score[i].is_nan()
                    || momentum_score[i].is_nan()
                    || volume_score[i].is_nan()
                {
                    f64::NAN
                } else {
                    let composite = self.weights.volatility * volatility_score[i]
                        + self.weights.momentum * momentum_score[i]
                        + self.weights.volume * volume_score[i];
                    composite.max(0.0).min(100.0)
                }
            })
            .collect();

        FearGreedOutput {
            index,
            volatility_score,
            momentum_score,
            volume_score,
        }
    }

    /// Calculate volatility score (inverse - high volatility = fear).
    fn calculate_volatility_score(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n < self.long_period {
            return result;
        }

        // Calculate returns
        let returns: Vec<f64> = (1..n)
            .map(|i| (close[i] / close[i - 1] - 1.0))
            .collect();

        // Calculate rolling volatility
        for i in self.long_period..n {
            let window: Vec<f64> = returns[(i - self.long_period)..(i - 1)].to_vec();
            if window.is_empty() {
                continue;
            }

            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance: f64 =
                window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
            let volatility = variance.sqrt();

            // Convert to score (0-100, inverse - high vol = low score = fear)
            // Assuming typical daily vol ranges from 0.5% to 5%
            let vol_pct = volatility * 100.0;
            let score = 100.0 - (vol_pct - 0.5).max(0.0) / 4.5 * 100.0;
            result[i] = score.max(0.0).min(100.0);
        }

        result
    }

    /// Calculate momentum score (price vs MA).
    fn calculate_momentum_score(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![f64::NAN; n];

        if n < self.long_period {
            return result;
        }

        // Calculate long-term SMA
        let ma = self.calculate_sma(close, self.long_period);

        for i in (self.long_period - 1)..n {
            if ma[i].is_nan() || ma[i] < 1e-10 {
                continue;
            }

            // Price position relative to MA
            let deviation = (close[i] - ma[i]) / ma[i];

            // Convert to 0-100 scale
            // Assuming typical deviations range from -20% to +20%
            let score = 50.0 + deviation / 0.20 * 50.0;
            result[i] = score.max(0.0).min(100.0);
        }

        result
    }

    /// Calculate volume score.
    fn calculate_volume_score(&self, volume: &[f64]) -> Vec<f64> {
        let n = volume.len();
        let mut result = vec![f64::NAN; n];

        if n < self.long_period {
            return result;
        }

        // Calculate volume MA
        let vol_ma = self.calculate_sma(volume, self.long_period);

        for i in (self.long_period - 1)..n {
            if vol_ma[i].is_nan() || vol_ma[i] < 1e-10 {
                continue;
            }

            // Volume relative to average
            let ratio = volume[i] / vol_ma[i];

            // Convert to 0-100 scale
            // High volume = more greed (FOMO), low volume = fear
            let score = 50.0 + (ratio - 1.0) / 1.0 * 50.0;
            result[i] = score.max(0.0).min(100.0);
        }

        result
    }

    /// Calculate SMA.
    fn calculate_sma(&self, data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![f64::NAN; n];

        if n < period {
            return result;
        }

        let mut sum: f64 = data[..period].iter().sum();
        result[period - 1] = sum / period as f64;

        for i in period..n {
            sum = sum - data[i - period] + data[i];
            result[i] = sum / period as f64;
        }

        result
    }

    /// Get sentiment level from index value.
    pub fn get_level(&self, index_value: f64) -> FearGreedLevel {
        if index_value.is_nan() {
            FearGreedLevel::Fear // Default to cautious
        } else if index_value < 25.0 {
            FearGreedLevel::ExtremeFear
        } else if index_value < 50.0 {
            FearGreedLevel::Fear
        } else if index_value < 75.0 {
            FearGreedLevel::Greed
        } else {
            FearGreedLevel::ExtremeGreed
        }
    }

    /// Convert level to trading signal (contrarian).
    pub fn to_indicator_signal(&self, level: FearGreedLevel) -> IndicatorSignal {
        match level {
            FearGreedLevel::ExtremeFear => IndicatorSignal::Bullish,
            FearGreedLevel::Fear => IndicatorSignal::Neutral,
            FearGreedLevel::Greed => IndicatorSignal::Neutral,
            FearGreedLevel::ExtremeGreed => IndicatorSignal::Bearish,
        }
    }

    /// Calculate the current index value.
    pub fn current_value(&self, close: &[f64], volume: &[f64]) -> f64 {
        let output = self.calculate(close, volume);
        output.index.last().copied().unwrap_or(f64::NAN)
    }
}

impl Default for FearGreedIndex {
    fn default() -> Self {
        Self::new(14, 30)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data(n: usize, trend: f64) -> (Vec<f64>, Vec<f64>) {
        let prices: Vec<f64> = (0..n)
            .map(|i| 100.0 * (1.0 + trend * i as f64 / n as f64))
            .collect();
        let volumes: Vec<f64> = (0..n).map(|i| 1e6 * (1.0 + 0.1 * (i as f64).sin())).collect();
        (prices, volumes)
    }

    #[test]
    fn test_fear_greed_basic() {
        let fgi = FearGreedIndex::new(5, 10);
        let (prices, volumes) = create_test_data(30, 0.1);

        let output = fgi.calculate(&prices, &volumes);

        assert_eq!(output.index.len(), 30);
        assert_eq!(output.volatility_score.len(), 30);
        assert_eq!(output.momentum_score.len(), 30);
        assert_eq!(output.volume_score.len(), 30);
    }

    #[test]
    fn test_fear_greed_range() {
        let fgi = FearGreedIndex::new(5, 10);
        let (prices, volumes) = create_test_data(50, 0.2);

        let output = fgi.calculate(&prices, &volumes);

        // All values should be in 0-100 range
        for i in 10..50 {
            if !output.index[i].is_nan() {
                assert!(output.index[i] >= 0.0 && output.index[i] <= 100.0);
            }
        }
    }

    #[test]
    fn test_fear_greed_levels() {
        let fgi = FearGreedIndex::default();

        assert_eq!(fgi.get_level(10.0), FearGreedLevel::ExtremeFear);
        assert_eq!(fgi.get_level(35.0), FearGreedLevel::Fear);
        assert_eq!(fgi.get_level(60.0), FearGreedLevel::Greed);
        assert_eq!(fgi.get_level(85.0), FearGreedLevel::ExtremeGreed);
    }

    #[test]
    fn test_fear_greed_signal_conversion() {
        let fgi = FearGreedIndex::default();

        assert_eq!(
            fgi.to_indicator_signal(FearGreedLevel::ExtremeFear),
            IndicatorSignal::Bullish
        );
        assert_eq!(
            fgi.to_indicator_signal(FearGreedLevel::ExtremeGreed),
            IndicatorSignal::Bearish
        );
        assert_eq!(
            fgi.to_indicator_signal(FearGreedLevel::Fear),
            IndicatorSignal::Neutral
        );
    }

    #[test]
    fn test_fear_greed_insufficient_data() {
        let fgi = FearGreedIndex::new(10, 30);
        let (prices, volumes) = create_test_data(10, 0.1);

        let output = fgi.calculate(&prices, &volumes);

        // All values should be NaN with insufficient data
        assert!(output.index.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn test_fear_greed_custom_weights() {
        let weights = FearGreedWeights {
            volatility: 0.5,
            momentum: 0.3,
            volume: 0.2,
        };
        let fgi = FearGreedIndex::with_weights(5, 10, weights);
        let (prices, volumes) = create_test_data(30, 0.1);

        let output = fgi.calculate(&prices, &volumes);

        // Should still produce valid output
        assert!(!output.index.iter().all(|x| x.is_nan()));
    }

    #[test]
    fn test_fear_greed_current_value() {
        let fgi = FearGreedIndex::new(5, 10);
        let (prices, volumes) = create_test_data(30, 0.1);

        let current = fgi.current_value(&prices, &volumes);

        // Should return a valid value
        assert!(!current.is_nan());
        assert!(current >= 0.0 && current <= 100.0);
    }

    #[test]
    fn test_weights_default() {
        let weights = FearGreedWeights::default();
        assert!((weights.volatility + weights.momentum + weights.volume - 1.0).abs() < 1e-10);
    }
}
