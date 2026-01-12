//! Decomposition result model

/// Result of time series decomposition
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// Trend component
    pub trend: Vec<f64>,
    /// Seasonal component
    pub seasonal: Vec<f64>,
    /// Residual component
    pub residual: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomposition_result_creation() {
        let result = DecompositionResult {
            trend: vec![1.0, 2.0, 3.0],
            seasonal: vec![0.1, 0.2, 0.3],
            residual: vec![0.01, 0.02, 0.03],
        };

        assert_eq!(result.trend, vec![1.0, 2.0, 3.0]);
        assert_eq!(result.seasonal, vec![0.1, 0.2, 0.3]);
        assert_eq!(result.residual, vec![0.01, 0.02, 0.03]);
    }

    #[test]
    fn test_decomposition_result_empty_vectors() {
        let result = DecompositionResult {
            trend: vec![],
            seasonal: vec![],
            residual: vec![],
        };

        assert!(result.trend.is_empty());
        assert!(result.seasonal.is_empty());
        assert!(result.residual.is_empty());
    }

    #[test]
    fn test_decomposition_result_single_element() {
        let result = DecompositionResult {
            trend: vec![42.0],
            seasonal: vec![-1.5],
            residual: vec![0.5],
        };

        assert_eq!(result.trend.len(), 1);
        assert_eq!(result.seasonal.len(), 1);
        assert_eq!(result.residual.len(), 1);
        assert_eq!(result.trend[0], 42.0);
        assert_eq!(result.seasonal[0], -1.5);
        assert_eq!(result.residual[0], 0.5);
    }

    #[test]
    fn test_decomposition_result_large_data() {
        let size = 10000;
        let result = DecompositionResult {
            trend: (0..size).map(|i| i as f64).collect(),
            seasonal: (0..size).map(|i| (i as f64).sin()).collect(),
            residual: vec![0.0; size],
        };

        assert_eq!(result.trend.len(), size);
        assert_eq!(result.seasonal.len(), size);
        assert_eq!(result.residual.len(), size);
    }

    #[test]
    fn test_decomposition_result_special_float_values() {
        let result = DecompositionResult {
            trend: vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN],
            seasonal: vec![f64::MIN, f64::MAX, 0.0],
            residual: vec![f64::EPSILON, -f64::EPSILON, f64::MIN_POSITIVE],
        };

        assert!(result.trend[0].is_infinite());
        assert!(result.trend[1].is_infinite());
        assert!(result.trend[2].is_nan());
        assert_eq!(result.seasonal[0], f64::MIN);
        assert_eq!(result.seasonal[1], f64::MAX);
        assert_eq!(result.residual[0], f64::EPSILON);
    }

    #[test]
    fn test_decomposition_result_negative_values() {
        let result = DecompositionResult {
            trend: vec![-100.0, -50.0, -25.0],
            seasonal: vec![-0.5, -0.25, -0.125],
            residual: vec![-1e-10, -1e-20, -1e-30],
        };

        assert!(result.trend.iter().all(|&x| x < 0.0));
        assert!(result.seasonal.iter().all(|&x| x < 0.0));
        assert!(result.residual.iter().all(|&x| x < 0.0));
    }

    #[test]
    fn test_decomposition_result_clone() {
        let original = DecompositionResult {
            trend: vec![1.0, 2.0, 3.0],
            seasonal: vec![0.1, 0.2, 0.3],
            residual: vec![0.01, 0.02, 0.03],
        };

        let cloned = original.clone();

        assert_eq!(original.trend, cloned.trend);
        assert_eq!(original.seasonal, cloned.seasonal);
        assert_eq!(original.residual, cloned.residual);
    }

    #[test]
    fn test_decomposition_result_clone_independence() {
        let original = DecompositionResult {
            trend: vec![1.0, 2.0, 3.0],
            seasonal: vec![0.1, 0.2, 0.3],
            residual: vec![0.01, 0.02, 0.03],
        };

        let mut cloned = original.clone();
        cloned.trend[0] = 999.0;
        cloned.seasonal[0] = 999.0;
        cloned.residual[0] = 999.0;

        // Original should be unchanged
        assert_eq!(original.trend[0], 1.0);
        assert_eq!(original.seasonal[0], 0.1);
        assert_eq!(original.residual[0], 0.01);
    }

    #[test]
    fn test_decomposition_result_debug() {
        let result = DecompositionResult {
            trend: vec![1.0],
            seasonal: vec![0.5],
            residual: vec![0.1],
        };

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("DecompositionResult"));
        assert!(debug_str.contains("trend"));
        assert!(debug_str.contains("seasonal"));
        assert!(debug_str.contains("residual"));
    }

    #[test]
    fn test_decomposition_result_field_access_mutability() {
        let mut result = DecompositionResult {
            trend: vec![1.0, 2.0, 3.0],
            seasonal: vec![0.1, 0.2, 0.3],
            residual: vec![0.01, 0.02, 0.03],
        };

        result.trend.push(4.0);
        result.seasonal.push(0.4);
        result.residual.push(0.04);

        assert_eq!(result.trend.len(), 4);
        assert_eq!(result.seasonal.len(), 4);
        assert_eq!(result.residual.len(), 4);
    }

    #[test]
    fn test_decomposition_result_different_lengths() {
        // Note: The struct allows different lengths for flexibility,
        // but implementations should ensure they match
        let result = DecompositionResult {
            trend: vec![1.0, 2.0, 3.0],
            seasonal: vec![0.1, 0.2],
            residual: vec![0.01],
        };

        assert_eq!(result.trend.len(), 3);
        assert_eq!(result.seasonal.len(), 2);
        assert_eq!(result.residual.len(), 1);
    }

    #[test]
    fn test_decomposition_result_realistic_data() {
        // Simulate a realistic decomposition scenario
        let n = 24; // 24 months of data
        let trend: Vec<f64> = (0..n).map(|i| 100.0 + 2.0 * i as f64).collect();
        let seasonal: Vec<f64> = (0..n)
            .map(|i| 10.0 * ((2.0 * std::f64::consts::PI * i as f64 / 12.0).sin()))
            .collect();
        let residual: Vec<f64> = vec![0.5; n];

        let result = DecompositionResult {
            trend,
            seasonal,
            residual,
        };

        assert_eq!(result.trend.len(), n);
        assert_eq!(result.seasonal.len(), n);
        assert_eq!(result.residual.len(), n);

        // Check trend is increasing
        for i in 1..n {
            assert!(result.trend[i] > result.trend[i - 1]);
        }

        // Check seasonal pattern
        assert!(result.seasonal[3] > result.seasonal[9]); // Peak vs trough
    }

    #[test]
    fn test_decomposition_result_reconstruction() {
        let result = DecompositionResult {
            trend: vec![100.0, 101.0, 102.0],
            seasonal: vec![5.0, -5.0, 5.0],
            residual: vec![0.1, -0.1, 0.2],
        };

        // Verify we can reconstruct original series
        let reconstructed: Vec<f64> = (0..3)
            .map(|i| result.trend[i] + result.seasonal[i] + result.residual[i])
            .collect();

        assert!((reconstructed[0] - 105.1).abs() < 1e-10);
        assert!((reconstructed[1] - 95.9).abs() < 1e-10);
        assert!((reconstructed[2] - 107.2).abs() < 1e-10);
    }
}
