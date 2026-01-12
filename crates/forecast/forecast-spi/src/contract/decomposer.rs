//! Trait for time series decomposition

use crate::model::DecompositionResult;

/// Trait for time series decomposition
pub trait Decomposer: Send + Sync {
    /// Decompose a time series into trend, seasonal, and residual components
    fn decompose(&self, data: &[f64], period: usize) -> DecompositionResult;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock implementation: Additive decomposer using simple moving average
    struct SimpleAdditiveDecomposer;

    impl Decomposer for SimpleAdditiveDecomposer {
        fn decompose(&self, data: &[f64], period: usize) -> DecompositionResult {
            let n = data.len();
            if n == 0 || period == 0 {
                return DecompositionResult {
                    trend: vec![],
                    seasonal: vec![],
                    residual: vec![],
                };
            }

            // Simple moving average for trend
            let mut trend = vec![0.0; n];
            let half_period = period / 2;
            for i in 0..n {
                let start = i.saturating_sub(half_period);
                let end = (i + half_period + 1).min(n);
                let sum: f64 = data[start..end].iter().sum();
                trend[i] = sum / (end - start) as f64;
            }

            // Detrended data
            let detrended: Vec<f64> = data.iter().zip(&trend).map(|(d, t)| d - t).collect();

            // Simple seasonal component (average of detrended for each position in period)
            let mut seasonal = vec![0.0; n];
            for i in 0..n {
                let pos = i % period;
                let mut sum = 0.0;
                let mut count = 0;
                for j in (pos..n).step_by(period) {
                    sum += detrended[j];
                    count += 1;
                }
                seasonal[i] = sum / count as f64;
            }

            // Residual = data - trend - seasonal
            let residual: Vec<f64> = data
                .iter()
                .zip(&trend)
                .zip(&seasonal)
                .map(|((d, t), s)| d - t - s)
                .collect();

            DecompositionResult {
                trend,
                seasonal,
                residual,
            }
        }
    }

    /// Mock implementation: Multiplicative decomposer
    struct SimpleMultiplicativeDecomposer;

    impl Decomposer for SimpleMultiplicativeDecomposer {
        fn decompose(&self, data: &[f64], period: usize) -> DecompositionResult {
            let n = data.len();
            if n == 0 || period == 0 {
                return DecompositionResult {
                    trend: vec![],
                    seasonal: vec![],
                    residual: vec![],
                };
            }

            // Simple moving average for trend
            let mut trend = vec![0.0; n];
            let half_period = period / 2;
            for i in 0..n {
                let start = i.saturating_sub(half_period);
                let end = (i + half_period + 1).min(n);
                let sum: f64 = data[start..end].iter().sum();
                trend[i] = sum / (end - start) as f64;
            }

            // Detrended data (multiplicative)
            let detrended: Vec<f64> = data
                .iter()
                .zip(&trend)
                .map(|(d, t)| if *t != 0.0 { d / t } else { 1.0 })
                .collect();

            // Simple seasonal component
            let mut seasonal = vec![1.0; n];
            for i in 0..n {
                let pos = i % period;
                let mut sum = 0.0;
                let mut count = 0;
                for j in (pos..n).step_by(period) {
                    sum += detrended[j];
                    count += 1;
                }
                seasonal[i] = sum / count as f64;
            }

            // Residual = data / (trend * seasonal)
            let residual: Vec<f64> = data
                .iter()
                .zip(&trend)
                .zip(&seasonal)
                .map(|((d, t), s)| {
                    let divisor = t * s;
                    if divisor != 0.0 {
                        d / divisor
                    } else {
                        1.0
                    }
                })
                .collect();

            DecompositionResult {
                trend,
                seasonal,
                residual,
            }
        }
    }

    /// Mock implementation: Trivial decomposer (no decomposition)
    struct TrivialDecomposer;

    impl Decomposer for TrivialDecomposer {
        fn decompose(&self, data: &[f64], _period: usize) -> DecompositionResult {
            DecompositionResult {
                trend: data.to_vec(),
                seasonal: vec![0.0; data.len()],
                residual: vec![0.0; data.len()],
            }
        }
    }

    #[test]
    fn test_simple_additive_decomposer_basic() {
        let decomposer = SimpleAdditiveDecomposer;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = decomposer.decompose(&data, 2);

        assert_eq!(result.trend.len(), data.len());
        assert_eq!(result.seasonal.len(), data.len());
        assert_eq!(result.residual.len(), data.len());
    }

    #[test]
    fn test_simple_additive_decomposer_empty_data() {
        let decomposer = SimpleAdditiveDecomposer;
        let data: Vec<f64> = vec![];
        let result = decomposer.decompose(&data, 12);

        assert!(result.trend.is_empty());
        assert!(result.seasonal.is_empty());
        assert!(result.residual.is_empty());
    }

    #[test]
    fn test_simple_additive_decomposer_zero_period() {
        let decomposer = SimpleAdditiveDecomposer;
        let data = vec![1.0, 2.0, 3.0];
        let result = decomposer.decompose(&data, 0);

        assert!(result.trend.is_empty());
        assert!(result.seasonal.is_empty());
        assert!(result.residual.is_empty());
    }

    #[test]
    fn test_simple_additive_decomposer_period_one() {
        let decomposer = SimpleAdditiveDecomposer;
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = decomposer.decompose(&data, 1);

        assert_eq!(result.trend.len(), 4);
        assert_eq!(result.seasonal.len(), 4);
        assert_eq!(result.residual.len(), 4);
    }

    #[test]
    fn test_simple_additive_decomposer_seasonal_pattern() {
        let decomposer = SimpleAdditiveDecomposer;
        // Create data with clear seasonal pattern (period=4)
        let data = vec![
            10.0, 20.0, 15.0, 5.0, // First cycle
            11.0, 21.0, 16.0, 6.0, // Second cycle
            12.0, 22.0, 17.0, 7.0, // Third cycle
        ];
        let result = decomposer.decompose(&data, 4);

        assert_eq!(result.trend.len(), data.len());
        assert_eq!(result.seasonal.len(), data.len());
        assert_eq!(result.residual.len(), data.len());
    }

    #[test]
    fn test_simple_multiplicative_decomposer_basic() {
        let decomposer = SimpleMultiplicativeDecomposer;
        let data = vec![100.0, 200.0, 150.0, 250.0, 110.0, 210.0];
        let result = decomposer.decompose(&data, 2);

        assert_eq!(result.trend.len(), data.len());
        assert_eq!(result.seasonal.len(), data.len());
        assert_eq!(result.residual.len(), data.len());
    }

    #[test]
    fn test_trivial_decomposer() {
        let decomposer = TrivialDecomposer;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = decomposer.decompose(&data, 12);

        assert_eq!(result.trend, data);
        assert!(result.seasonal.iter().all(|&x| x == 0.0));
        assert!(result.residual.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_decomposer_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SimpleAdditiveDecomposer>();
        assert_send::<SimpleMultiplicativeDecomposer>();
        assert_send::<TrivialDecomposer>();
    }

    #[test]
    fn test_decomposer_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<SimpleAdditiveDecomposer>();
        assert_sync::<SimpleMultiplicativeDecomposer>();
        assert_sync::<TrivialDecomposer>();
    }

    #[test]
    fn test_decomposer_as_trait_object() {
        let decomposer: Box<dyn Decomposer> = Box::new(SimpleAdditiveDecomposer);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = decomposer.decompose(&data, 2);

        assert_eq!(result.trend.len(), 4);
    }

    #[test]
    fn test_decomposer_with_single_element() {
        let decomposer = SimpleAdditiveDecomposer;
        let data = vec![42.0];
        let result = decomposer.decompose(&data, 1);

        assert_eq!(result.trend.len(), 1);
        assert_eq!(result.seasonal.len(), 1);
        assert_eq!(result.residual.len(), 1);
    }

    #[test]
    fn test_decomposer_large_period() {
        let decomposer = SimpleAdditiveDecomposer;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // Period larger than data length
        let result = decomposer.decompose(&data, 100);

        assert_eq!(result.trend.len(), data.len());
        assert_eq!(result.seasonal.len(), data.len());
        assert_eq!(result.residual.len(), data.len());
    }

    #[test]
    fn test_decomposer_constant_data() {
        let decomposer = SimpleAdditiveDecomposer;
        let data = vec![5.0; 12];
        let result = decomposer.decompose(&data, 4);

        // For constant data, trend should be constant
        for &t in &result.trend {
            assert!((t - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_decomposer_negative_values() {
        let decomposer = SimpleAdditiveDecomposer;
        let data = vec![-10.0, -5.0, -15.0, -10.0, -5.0, -15.0];
        let result = decomposer.decompose(&data, 3);

        assert_eq!(result.trend.len(), data.len());
        assert_eq!(result.seasonal.len(), data.len());
        assert_eq!(result.residual.len(), data.len());
    }

    #[test]
    fn test_decomposer_reconstruction_additive() {
        let decomposer = TrivialDecomposer;
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let result = decomposer.decompose(&data, 2);

        // For trivial decomposer: data = trend + seasonal + residual
        for i in 0..data.len() {
            let reconstructed = result.trend[i] + result.seasonal[i] + result.residual[i];
            assert!((reconstructed - data[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_decomposer_with_trend() {
        let decomposer = SimpleAdditiveDecomposer;
        // Linear trend with no seasonality
        let data: Vec<f64> = (0..20).map(|i| 100.0 + 5.0 * i as f64).collect();
        let result = decomposer.decompose(&data, 4);

        // Trend should be roughly linear
        assert_eq!(result.trend.len(), 20);
        // First trend value should be less than last
        assert!(result.trend[0] < result.trend[19]);
    }

    #[test]
    fn test_decomposer_different_periods() {
        let decomposer = SimpleAdditiveDecomposer;
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();

        for period in [2, 3, 4, 6, 12] {
            let result = decomposer.decompose(&data, period);
            assert_eq!(result.trend.len(), 24);
            assert_eq!(result.seasonal.len(), 24);
            assert_eq!(result.residual.len(), 24);
        }
    }

    #[test]
    fn test_decomposer_special_values() {
        let decomposer = TrivialDecomposer;
        let data = vec![f64::INFINITY, 0.0, f64::NEG_INFINITY];
        let result = decomposer.decompose(&data, 1);

        assert!(result.trend[0].is_infinite());
        assert_eq!(result.trend[1], 0.0);
        assert!(result.trend[2].is_infinite());
    }
}
