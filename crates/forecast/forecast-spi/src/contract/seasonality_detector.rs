//! Trait for seasonality detection

/// Trait for seasonality detection
pub trait SeasonalityDetector: Send + Sync {
    /// Detect the dominant seasonality period in the data
    fn detect(&self, data: &[f64], max_period: usize) -> Option<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock implementation: Autocorrelation-based seasonality detector
    struct AutocorrelationDetector {
        min_period: usize,
        threshold: f64,
    }

    impl AutocorrelationDetector {
        fn new(min_period: usize, threshold: f64) -> Self {
            Self {
                min_period,
                threshold,
            }
        }

        fn autocorrelation(data: &[f64], lag: usize) -> f64 {
            if lag >= data.len() {
                return 0.0;
            }
            let n = data.len() - lag;
            if n == 0 {
                return 0.0;
            }
            let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for i in 0..data.len() {
                denominator += (data[i] - mean).powi(2);
            }
            for i in 0..n {
                numerator += (data[i] - mean) * (data[i + lag] - mean);
            }
            if denominator == 0.0 {
                0.0
            } else {
                numerator / denominator
            }
        }
    }

    impl SeasonalityDetector for AutocorrelationDetector {
        fn detect(&self, data: &[f64], max_period: usize) -> Option<usize> {
            if data.len() < self.min_period * 2 {
                return None;
            }

            let effective_max = max_period.min(data.len() / 2);
            let mut best_period = None;
            let mut best_autocorr = self.threshold;

            for period in self.min_period..=effective_max {
                let autocorr = Self::autocorrelation(data, period);
                if autocorr > best_autocorr {
                    best_autocorr = autocorr;
                    best_period = Some(period);
                }
            }

            best_period
        }
    }

    /// Mock implementation: Fixed period detector (always returns a fixed period)
    struct FixedPeriodDetector {
        period: usize,
    }

    impl FixedPeriodDetector {
        fn new(period: usize) -> Self {
            Self { period }
        }
    }

    impl SeasonalityDetector for FixedPeriodDetector {
        fn detect(&self, data: &[f64], max_period: usize) -> Option<usize> {
            if data.len() < self.period * 2 || self.period > max_period {
                None
            } else {
                Some(self.period)
            }
        }
    }

    /// Mock implementation: No seasonality detector (always returns None)
    struct NoSeasonalityDetector;

    impl SeasonalityDetector for NoSeasonalityDetector {
        fn detect(&self, _data: &[f64], _max_period: usize) -> Option<usize> {
            None
        }
    }

    /// Mock implementation: First valid period detector
    struct FirstValidPeriodDetector;

    impl SeasonalityDetector for FirstValidPeriodDetector {
        fn detect(&self, data: &[f64], max_period: usize) -> Option<usize> {
            if data.len() >= 4 && max_period >= 2 {
                Some(2)
            } else {
                None
            }
        }
    }

    #[test]
    fn test_autocorrelation_detector_creation() {
        let detector = AutocorrelationDetector::new(2, 0.5);
        assert_eq!(detector.min_period, 2);
        assert_eq!(detector.threshold, 0.5);
    }

    #[test]
    fn test_autocorrelation_detector_with_clear_seasonality() {
        let detector = AutocorrelationDetector::new(2, 0.3);
        // Create data with clear period of 4
        let data: Vec<f64> = (0..40)
            .map(|i| {
                let seasonal = 10.0 * ((2.0 * std::f64::consts::PI * i as f64 / 4.0).sin());
                seasonal
            })
            .collect();

        let period = detector.detect(&data, 20);
        // Should detect period close to 4
        assert!(period.is_some());
        let detected = period.unwrap();
        assert!(detected >= 3 && detected <= 5, "Detected period: {}", detected);
    }

    #[test]
    fn test_autocorrelation_detector_no_seasonality() {
        let detector = AutocorrelationDetector::new(2, 0.8);
        // Random-like data with no clear pattern
        let data = vec![1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 1.0, 9.0];

        let period = detector.detect(&data, 5);
        // With high threshold, should not detect any period
        assert!(period.is_none());
    }

    #[test]
    fn test_autocorrelation_detector_insufficient_data() {
        let detector = AutocorrelationDetector::new(5, 0.3);
        let data = vec![1.0, 2.0, 3.0]; // Too short for min_period of 5

        let period = detector.detect(&data, 10);
        assert!(period.is_none());
    }

    #[test]
    fn test_autocorrelation_detector_empty_data() {
        let detector = AutocorrelationDetector::new(2, 0.3);
        let data: Vec<f64> = vec![];

        let period = detector.detect(&data, 10);
        assert!(period.is_none());
    }

    #[test]
    fn test_fixed_period_detector() {
        let detector = FixedPeriodDetector::new(12);
        let data = vec![1.0; 50]; // Enough data

        let period = detector.detect(&data, 20);
        assert_eq!(period, Some(12));
    }

    #[test]
    fn test_fixed_period_detector_insufficient_data() {
        let detector = FixedPeriodDetector::new(12);
        let data = vec![1.0; 20]; // Not enough for period * 2

        let period = detector.detect(&data, 20);
        assert!(period.is_none());
    }

    #[test]
    fn test_fixed_period_detector_max_period_exceeded() {
        let detector = FixedPeriodDetector::new(12);
        let data = vec![1.0; 50];

        let period = detector.detect(&data, 10); // max_period < fixed period
        assert!(period.is_none());
    }

    #[test]
    fn test_no_seasonality_detector() {
        let detector = NoSeasonalityDetector;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let period = detector.detect(&data, 100);
        assert!(period.is_none());
    }

    #[test]
    fn test_first_valid_period_detector() {
        let detector = FirstValidPeriodDetector;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let period = detector.detect(&data, 10);
        assert_eq!(period, Some(2));
    }

    #[test]
    fn test_first_valid_period_detector_short_data() {
        let detector = FirstValidPeriodDetector;
        let data = vec![1.0, 2.0, 3.0]; // len < 4

        let period = detector.detect(&data, 10);
        assert!(period.is_none());
    }

    #[test]
    fn test_seasonality_detector_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<AutocorrelationDetector>();
        assert_send::<FixedPeriodDetector>();
        assert_send::<NoSeasonalityDetector>();
        assert_send::<FirstValidPeriodDetector>();
    }

    #[test]
    fn test_seasonality_detector_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<AutocorrelationDetector>();
        assert_sync::<FixedPeriodDetector>();
        assert_sync::<NoSeasonalityDetector>();
        assert_sync::<FirstValidPeriodDetector>();
    }

    #[test]
    fn test_seasonality_detector_as_trait_object() {
        let detector: Box<dyn SeasonalityDetector> =
            Box::new(FixedPeriodDetector::new(7));
        let data = vec![1.0; 20];

        let period = detector.detect(&data, 10);
        assert_eq!(period, Some(7));
    }

    #[test]
    fn test_autocorrelation_constant_data() {
        let detector = AutocorrelationDetector::new(2, 0.1);
        let data = vec![5.0; 20]; // All same values

        let period = detector.detect(&data, 10);
        // Constant data has no meaningful seasonality
        assert!(period.is_none());
    }

    #[test]
    fn test_autocorrelation_linear_trend() {
        let detector = AutocorrelationDetector::new(2, 0.5);
        let data: Vec<f64> = (0..30).map(|i| i as f64).collect();

        // Linear trend should not show periodic seasonality with high threshold
        let period = detector.detect(&data, 15);
        // May or may not detect a period depending on autocorrelation behavior
        // Just verify it doesn't crash
        let _ = period;
    }

    #[test]
    fn test_autocorrelation_different_periods() {
        let detector = AutocorrelationDetector::new(2, 0.3);

        for expected_period in [3, 5, 7, 12] {
            let data: Vec<f64> = (0..expected_period * 10)
                .map(|i| {
                    10.0 * ((2.0 * std::f64::consts::PI * i as f64 / expected_period as f64).sin())
                })
                .collect();

            let period = detector.detect(&data, expected_period * 2);
            if let Some(detected) = period {
                // Allow some tolerance
                assert!(
                    (detected as i32 - expected_period as i32).abs() <= 1,
                    "Expected ~{}, got {}",
                    expected_period,
                    detected
                );
            }
        }
    }

    #[test]
    fn test_max_period_limiting() {
        let detector = AutocorrelationDetector::new(2, 0.1);
        let data: Vec<f64> = (0..100)
            .map(|i| 10.0 * ((2.0 * std::f64::consts::PI * i as f64 / 20.0).sin()))
            .collect();

        // With max_period = 10, should not detect period of 20
        let period = detector.detect(&data, 10);
        if let Some(p) = period {
            assert!(p <= 10);
        }
    }

    #[test]
    fn test_seasonality_detector_negative_values() {
        let detector = AutocorrelationDetector::new(2, 0.3);
        let data: Vec<f64> = (0..40)
            .map(|i| -10.0 + 5.0 * ((2.0 * std::f64::consts::PI * i as f64 / 4.0).sin()))
            .collect();

        // Should still work with negative values
        let period = detector.detect(&data, 20);
        // Verify it runs without error
        let _ = period;
    }

    #[test]
    fn test_autocorrelation_helper_function() {
        // Direct test of autocorrelation calculation
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let lag0 = AutocorrelationDetector::autocorrelation(&data, 0);
        // Autocorrelation at lag 0 should be 1.0 (perfect correlation with self)
        assert!((lag0 - 1.0).abs() < 1e-10);

        let lag_too_large = AutocorrelationDetector::autocorrelation(&data, 10);
        assert_eq!(lag_too_large, 0.0);
    }

    #[test]
    fn test_seasonality_detector_single_element() {
        let detector = AutocorrelationDetector::new(2, 0.3);
        let data = vec![42.0];

        let period = detector.detect(&data, 10);
        assert!(period.is_none());
    }

    #[test]
    fn test_seasonality_detector_two_elements() {
        let detector = AutocorrelationDetector::new(2, 0.3);
        let data = vec![1.0, 2.0];

        let period = detector.detect(&data, 10);
        assert!(period.is_none()); // Not enough for min_period * 2
    }

    #[test]
    fn test_seasonality_detector_special_values() {
        let detector = NoSeasonalityDetector;
        let data = vec![f64::INFINITY, f64::NEG_INFINITY, f64::NAN];

        // Should not crash
        let period = detector.detect(&data, 10);
        assert!(period.is_none());
    }
}
