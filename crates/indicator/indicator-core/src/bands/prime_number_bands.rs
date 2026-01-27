//! Prime Number Bands implementation.
//!
//! Bands based on prime numbers for support and resistance levels.

use crate::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCVSeries,
};

/// Prime Number Bands indicator.
///
/// Creates support and resistance bands by finding the nearest prime numbers
/// above and below the current price. The theory is that prices tend to
/// gravitate toward prime number levels, which can act as psychological
/// support/resistance.
///
/// - Upper Band: Nearest prime number above the price
/// - Lower Band: Nearest prime number below the price
/// - Middle Band: Current price (or midpoint between bands)
///
/// This indicator works best on instruments with prices in reasonable ranges
/// (e.g., 10-1000). For very large prices, consider scaling.
#[derive(Debug, Clone)]
pub struct PrimeNumberBands {
    /// Scaling factor for price (useful for high-priced instruments).
    scale: f64,
    /// Whether to use the midpoint as middle band (vs. actual price).
    use_midpoint: bool,
}

impl PrimeNumberBands {
    /// Create a new PrimeNumberBands indicator.
    ///
    /// # Arguments
    /// * `scale` - Scale factor for price (1.0 = no scaling, 0.1 = divide by 10)
    pub fn new(scale: f64) -> Self {
        Self {
            scale,
            use_midpoint: true,
        }
    }

    /// Create with no scaling (for instruments in reasonable price ranges).
    pub fn no_scale() -> Self {
        Self {
            scale: 1.0,
            use_midpoint: true,
        }
    }

    /// Create with price scaling and midpoint option.
    pub fn with_options(scale: f64, use_midpoint: bool) -> Self {
        Self { scale, use_midpoint }
    }

    /// Check if a number is prime.
    fn is_prime(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        if n == 2 {
            return true;
        }
        if n % 2 == 0 {
            return false;
        }

        let sqrt_n = (n as f64).sqrt() as u64;
        for i in (3..=sqrt_n).step_by(2) {
            if n % i == 0 {
                return false;
            }
        }
        true
    }

    /// Find the nearest prime number greater than or equal to n.
    fn next_prime(n: u64) -> u64 {
        if n <= 2 {
            return 2;
        }

        let mut candidate = if n % 2 == 0 { n + 1 } else { n };

        while !Self::is_prime(candidate) {
            candidate += 2;
        }

        candidate
    }

    /// Find the nearest prime number less than or equal to n.
    fn prev_prime(n: u64) -> u64 {
        if n <= 2 {
            return 2;
        }

        let mut candidate = if n % 2 == 0 { n - 1 } else { n };

        while !Self::is_prime(candidate) && candidate > 2 {
            candidate -= 2;
        }

        if candidate < 2 {
            2
        } else {
            candidate
        }
    }

    /// Find the nearest prime above the price.
    fn upper_prime(&self, price: f64) -> f64 {
        let scaled_price = (price * self.scale).ceil() as u64;
        let prime = Self::next_prime(scaled_price);
        prime as f64 / self.scale
    }

    /// Find the nearest prime below the price.
    fn lower_prime(&self, price: f64) -> f64 {
        let scaled_price = (price * self.scale).floor() as u64;
        if scaled_price == 0 {
            return 2.0 / self.scale;
        }
        let prime = Self::prev_prime(scaled_price);
        prime as f64 / self.scale
    }

    /// Calculate Prime Number Bands (upper, middle, lower).
    pub fn calculate(&self, close: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = close.len();
        let mut upper = Vec::with_capacity(n);
        let mut middle = Vec::with_capacity(n);
        let mut lower = Vec::with_capacity(n);

        for &price in close {
            if price <= 0.0 || price.is_nan() || price.is_infinite() {
                upper.push(f64::NAN);
                middle.push(f64::NAN);
                lower.push(f64::NAN);
            } else {
                let u = self.upper_prime(price);
                let l = self.lower_prime(price);

                upper.push(u);
                lower.push(l);

                if self.use_midpoint {
                    middle.push((u + l) / 2.0);
                } else {
                    middle.push(price);
                }
            }
        }

        (upper, middle, lower)
    }

    /// Calculate distance to nearest prime (positive = above, negative = below).
    pub fn prime_distance(&self, close: &[f64]) -> Vec<f64> {
        let (upper, _, lower) = self.calculate(close);
        close.iter()
            .zip(upper.iter().zip(lower.iter()))
            .map(|(&price, (&u, &l))| {
                if price.is_nan() || u.is_nan() || l.is_nan() {
                    f64::NAN
                } else {
                    let dist_up = u - price;
                    let dist_down = price - l;
                    if dist_up < dist_down {
                        dist_up
                    } else {
                        -dist_down
                    }
                }
            })
            .collect()
    }

    /// Calculate position within bands (0 = at lower, 1 = at upper).
    pub fn position(&self, close: &[f64]) -> Vec<f64> {
        let (upper, _, lower) = self.calculate(close);
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

    /// Check if price is at a prime number level.
    pub fn at_prime(&self, close: &[f64]) -> Vec<bool> {
        close.iter()
            .map(|&price| {
                if price <= 0.0 || price.is_nan() || price.is_infinite() {
                    false
                } else {
                    let scaled = (price * self.scale).round() as u64;
                    Self::is_prime(scaled)
                }
            })
            .collect()
    }
}

impl Default for PrimeNumberBands {
    fn default() -> Self {
        Self::no_scale()
    }
}

impl TechnicalIndicator for PrimeNumberBands {
    fn name(&self) -> &str {
        "PrimeNumberBands"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        let (upper, middle, lower) = self.calculate(&data.close);
        Ok(IndicatorOutput::triple(upper, middle, lower))
    }

    fn min_periods(&self) -> usize {
        1
    }

    fn output_features(&self) -> usize {
        3
    }
}

impl SignalIndicator for PrimeNumberBands {
    fn signal(&self, data: &OHLCVSeries) -> Result<IndicatorSignal> {
        let position = self.position(&data.close);

        let n = position.len();
        if n == 0 {
            return Ok(IndicatorSignal::Neutral);
        }

        let pos = position[n - 1];
        if pos.is_nan() {
            return Ok(IndicatorSignal::Neutral);
        }

        // Price near lower prime = potential support = bullish
        if pos < 0.25 {
            Ok(IndicatorSignal::Bullish)
        }
        // Price near upper prime = potential resistance = bearish
        else if pos > 0.75 {
            Ok(IndicatorSignal::Bearish)
        }
        else {
            Ok(IndicatorSignal::Neutral)
        }
    }

    fn signals(&self, data: &OHLCVSeries) -> Result<Vec<IndicatorSignal>> {
        let position = self.position(&data.close);

        let signals: Vec<_> = position.iter()
            .map(|&pos| {
                if pos.is_nan() {
                    IndicatorSignal::Neutral
                } else if pos < 0.25 {
                    IndicatorSignal::Bullish
                } else if pos > 0.75 {
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

    #[test]
    fn test_is_prime() {
        // Known primes
        assert!(PrimeNumberBands::is_prime(2));
        assert!(PrimeNumberBands::is_prime(3));
        assert!(PrimeNumberBands::is_prime(5));
        assert!(PrimeNumberBands::is_prime(7));
        assert!(PrimeNumberBands::is_prime(11));
        assert!(PrimeNumberBands::is_prime(13));
        assert!(PrimeNumberBands::is_prime(97));
        assert!(PrimeNumberBands::is_prime(101));

        // Known non-primes
        assert!(!PrimeNumberBands::is_prime(0));
        assert!(!PrimeNumberBands::is_prime(1));
        assert!(!PrimeNumberBands::is_prime(4));
        assert!(!PrimeNumberBands::is_prime(6));
        assert!(!PrimeNumberBands::is_prime(9));
        assert!(!PrimeNumberBands::is_prime(100));
    }

    #[test]
    fn test_next_prime() {
        assert_eq!(PrimeNumberBands::next_prime(0), 2);
        assert_eq!(PrimeNumberBands::next_prime(1), 2);
        assert_eq!(PrimeNumberBands::next_prime(2), 2);
        assert_eq!(PrimeNumberBands::next_prime(3), 3);
        assert_eq!(PrimeNumberBands::next_prime(4), 5);
        assert_eq!(PrimeNumberBands::next_prime(10), 11);
        assert_eq!(PrimeNumberBands::next_prime(100), 101);
    }

    #[test]
    fn test_prev_prime() {
        assert_eq!(PrimeNumberBands::prev_prime(2), 2);
        assert_eq!(PrimeNumberBands::prev_prime(3), 3);
        assert_eq!(PrimeNumberBands::prev_prime(4), 3);
        assert_eq!(PrimeNumberBands::prev_prime(10), 7);
        assert_eq!(PrimeNumberBands::prev_prime(100), 97);
    }

    #[test]
    fn test_prime_number_bands_basic() {
        let pnb = PrimeNumberBands::no_scale();
        let close = vec![10.0, 15.0, 20.0, 25.0, 30.0];

        let (upper, middle, lower) = pnb.calculate(&close);

        assert_eq!(upper.len(), 5);
        assert_eq!(middle.len(), 5);
        assert_eq!(lower.len(), 5);

        // Check specific values
        // 10: lower=7, upper=11
        assert!((lower[0] - 7.0).abs() < 1e-10);
        assert!((upper[0] - 11.0).abs() < 1e-10);

        // 15: lower=13, upper=17
        assert!((lower[1] - 13.0).abs() < 1e-10);
        assert!((upper[1] - 17.0).abs() < 1e-10);

        // 20: lower=19, upper=23
        assert!((lower[2] - 19.0).abs() < 1e-10);
        assert!((upper[2] - 23.0).abs() < 1e-10);
    }

    #[test]
    fn test_prime_number_bands_middle() {
        let pnb = PrimeNumberBands::with_options(1.0, true);
        let close = vec![10.0];

        let (upper, middle, lower) = pnb.calculate(&close);

        // Middle should be midpoint: (11 + 7) / 2 = 9
        assert!((middle[0] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_prime_number_bands_no_midpoint() {
        let pnb = PrimeNumberBands::with_options(1.0, false);
        let close = vec![10.0];

        let (_, middle, _) = pnb.calculate(&close);

        // Middle should be actual price
        assert!((middle[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_prime_number_bands_scaling() {
        let pnb = PrimeNumberBands::new(0.1); // Scale down by 10
        let close = vec![100.0]; // Becomes 10 after scaling

        let (upper, _, lower) = pnb.calculate(&close);

        // 10 scaled: lower=7/0.1=70, upper=11/0.1=110
        assert!((lower[0] - 70.0).abs() < 1e-10);
        assert!((upper[0] - 110.0).abs() < 1e-10);
    }

    #[test]
    fn test_prime_number_bands_at_prime() {
        let pnb = PrimeNumberBands::no_scale();
        let close = vec![7.0, 8.0, 11.0, 12.0, 13.0];

        let at_prime = pnb.at_prime(&close);

        assert!(at_prime[0]); // 7 is prime
        assert!(!at_prime[1]); // 8 is not prime
        assert!(at_prime[2]); // 11 is prime
        assert!(!at_prime[3]); // 12 is not prime
        assert!(at_prime[4]); // 13 is prime
    }

    #[test]
    fn test_prime_number_bands_prime_distance() {
        let pnb = PrimeNumberBands::no_scale();
        let close = vec![10.0, 8.0];

        let distance = pnb.prime_distance(&close);

        // 10 is closer to 11 (distance 1) than to 7 (distance 3)
        assert!((distance[0] - 1.0).abs() < 1e-10);

        // 8 is closer to 7 (distance 1) than to 11 (distance 3)
        assert!((distance[1] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_prime_number_bands_position() {
        let pnb = PrimeNumberBands::no_scale();
        // Use prices that are NOT primes themselves to get clear band positions
        let close = vec![8.0, 9.0, 10.0];

        let position = pnb.position(&close);

        // 8: lower=7, upper=11, pos = (8-7)/(11-7) = 0.25
        assert!((position[0] - 0.25).abs() < 1e-10);

        // 9: lower=7, upper=11, pos = (9-7)/(11-7) = 0.5
        assert!((position[1] - 0.5).abs() < 1e-10);

        // 10: lower=7, upper=11, pos = (10-7)/(11-7) = 0.75
        assert!((position[2] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_prime_number_bands_default() {
        let pnb = PrimeNumberBands::default();
        assert!((pnb.scale - 1.0).abs() < 1e-10);
        assert!(pnb.use_midpoint);
    }

    #[test]
    fn test_prime_number_bands_negative_price() {
        let pnb = PrimeNumberBands::no_scale();
        let close = vec![-10.0, 0.0, 10.0];

        let (upper, middle, lower) = pnb.calculate(&close);

        // Negative and zero prices should return NaN
        assert!(upper[0].is_nan());
        assert!(middle[0].is_nan());
        assert!(lower[0].is_nan());

        assert!(upper[1].is_nan());
        assert!(middle[1].is_nan());
        assert!(lower[1].is_nan());

        // Positive price should work
        assert!(!upper[2].is_nan());
        assert!(!middle[2].is_nan());
        assert!(!lower[2].is_nan());
    }

    #[test]
    fn test_prime_number_bands_signal_bullish() {
        let pnb = PrimeNumberBands::no_scale();

        // Price near lower prime (7)
        let close = vec![7.5, 7.2, 7.1];

        let data = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x + 0.5).collect(),
            low: close.iter().map(|x| x - 0.5).collect(),
            close,
            volume: vec![1000.0; 3],
        };

        let signal = pnb.signal(&data).unwrap();

        // Position at 7.1: (7.1-7)/(11-7) = 0.025 < 0.25 = bullish
        assert_eq!(signal, IndicatorSignal::Bullish);
    }

    #[test]
    fn test_prime_number_bands_signal_bearish() {
        let pnb = PrimeNumberBands::no_scale();

        // Price near upper prime (11)
        let close = vec![10.5, 10.8, 10.9];

        let data = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x + 0.5).collect(),
            low: close.iter().map(|x| x - 0.5).collect(),
            close,
            volume: vec![1000.0; 3],
        };

        let signal = pnb.signal(&data).unwrap();

        // Position at 10.9: (10.9-7)/(11-7) = 0.975 > 0.75 = bearish
        assert_eq!(signal, IndicatorSignal::Bearish);
    }

    #[test]
    fn test_prime_number_bands_compute() {
        let pnb = PrimeNumberBands::no_scale();
        let close: Vec<f64> = (10..40).map(|i| i as f64).collect();

        let data = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x + 1.0).collect(),
            low: close.iter().map(|x| x - 1.0).collect(),
            close,
            volume: vec![1000.0; 30],
        };

        let output = pnb.compute(&data).unwrap();
        assert_eq!(output.primary.len(), 30);
        assert!(output.secondary.is_some());
        assert!(output.tertiary.is_some());
    }

    #[test]
    fn test_prime_number_bands_insufficient_data() {
        let pnb = PrimeNumberBands::no_scale();

        let data = OHLCVSeries {
            open: vec![],
            high: vec![],
            low: vec![],
            close: vec![],
            volume: vec![],
        };

        let result = pnb.compute(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_prime_number_bands_signals() {
        let pnb = PrimeNumberBands::no_scale();
        let close: Vec<f64> = (5..15).map(|i| i as f64).collect();

        let data = OHLCVSeries {
            open: close.clone(),
            high: close.iter().map(|x| x + 1.0).collect(),
            low: close.iter().map(|x| x - 1.0).collect(),
            close,
            volume: vec![1000.0; 10],
        };

        let signals = pnb.signals(&data).unwrap();
        assert_eq!(signals.len(), 10);
    }

    #[test]
    fn test_prime_number_bands_large_primes() {
        let pnb = PrimeNumberBands::no_scale();
        let close = vec![997.0, 1000.0, 1009.0];

        let (upper, _, lower) = pnb.calculate(&close);

        // 997 is prime
        assert!((lower[0] - 997.0).abs() < 1e-10);

        // 1000: lower=997, upper=1009
        assert!((lower[1] - 997.0).abs() < 1e-10);
        assert!((upper[1] - 1009.0).abs() < 1e-10);

        // 1009 is prime
        assert!((upper[2] - 1009.0).abs() < 1e-10);
    }
}
