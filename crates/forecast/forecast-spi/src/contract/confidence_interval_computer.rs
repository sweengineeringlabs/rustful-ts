//! Trait for confidence interval computation

use crate::model::ConfidenceInterval;

/// Trait for confidence interval computation
pub trait ConfidenceIntervalComputer: Send + Sync {
    /// Compute confidence intervals for forecasts
    fn compute(
        &self,
        forecast: &[f64],
        residuals: &[f64],
        confidence_level: f64,
    ) -> ConfidenceInterval;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock implementation: Normal distribution-based confidence intervals
    struct NormalConfidenceIntervalComputer;

    impl NormalConfidenceIntervalComputer {
        fn z_score(confidence_level: f64) -> f64 {
            // Approximate z-scores for common confidence levels
            if (confidence_level - 0.90).abs() < 0.001 {
                1.645
            } else if (confidence_level - 0.95).abs() < 0.001 {
                1.96
            } else if (confidence_level - 0.99).abs() < 0.001 {
                2.576
            } else {
                // Rough approximation for other levels
                1.96
            }
        }

        fn std_dev(data: &[f64]) -> f64 {
            if data.is_empty() {
                return 0.0;
            }
            let mean = data.iter().sum::<f64>() / data.len() as f64;
            let variance =
                data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
            variance.sqrt()
        }
    }

    impl ConfidenceIntervalComputer for NormalConfidenceIntervalComputer {
        fn compute(
            &self,
            forecast: &[f64],
            residuals: &[f64],
            confidence_level: f64,
        ) -> ConfidenceInterval {
            let z = Self::z_score(confidence_level);
            let std = Self::std_dev(residuals);
            let margin = z * std;

            ConfidenceInterval {
                forecast: forecast.to_vec(),
                lower: forecast.iter().map(|&f| f - margin).collect(),
                upper: forecast.iter().map(|&f| f + margin).collect(),
                confidence_level,
            }
        }
    }

    /// Mock implementation: Widening confidence intervals (uncertainty grows over horizon)
    struct WideningConfidenceIntervalComputer {
        growth_rate: f64,
    }

    impl WideningConfidenceIntervalComputer {
        fn new(growth_rate: f64) -> Self {
            Self { growth_rate }
        }
    }

    impl ConfidenceIntervalComputer for WideningConfidenceIntervalComputer {
        fn compute(
            &self,
            forecast: &[f64],
            residuals: &[f64],
            confidence_level: f64,
        ) -> ConfidenceInterval {
            let base_std = if residuals.is_empty() {
                1.0
            } else {
                let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
                let variance = residuals
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>()
                    / residuals.len() as f64;
                variance.sqrt().max(0.001)
            };

            let z = 1.96; // 95% approximation

            let lower: Vec<f64> = forecast
                .iter()
                .enumerate()
                .map(|(i, &f)| {
                    let margin = z * base_std * (1.0 + self.growth_rate * i as f64);
                    f - margin
                })
                .collect();

            let upper: Vec<f64> = forecast
                .iter()
                .enumerate()
                .map(|(i, &f)| {
                    let margin = z * base_std * (1.0 + self.growth_rate * i as f64);
                    f + margin
                })
                .collect();

            ConfidenceInterval {
                forecast: forecast.to_vec(),
                lower,
                upper,
                confidence_level,
            }
        }
    }

    /// Mock implementation: Fixed margin confidence intervals
    struct FixedMarginConfidenceIntervalComputer {
        margin: f64,
    }

    impl FixedMarginConfidenceIntervalComputer {
        fn new(margin: f64) -> Self {
            Self { margin }
        }
    }

    impl ConfidenceIntervalComputer for FixedMarginConfidenceIntervalComputer {
        fn compute(
            &self,
            forecast: &[f64],
            _residuals: &[f64],
            confidence_level: f64,
        ) -> ConfidenceInterval {
            ConfidenceInterval {
                forecast: forecast.to_vec(),
                lower: forecast.iter().map(|&f| f - self.margin).collect(),
                upper: forecast.iter().map(|&f| f + self.margin).collect(),
                confidence_level,
            }
        }
    }

    /// Mock implementation: Percentage-based confidence intervals
    struct PercentageConfidenceIntervalComputer {
        percentage: f64,
    }

    impl PercentageConfidenceIntervalComputer {
        fn new(percentage: f64) -> Self {
            Self { percentage }
        }
    }

    impl ConfidenceIntervalComputer for PercentageConfidenceIntervalComputer {
        fn compute(
            &self,
            forecast: &[f64],
            _residuals: &[f64],
            confidence_level: f64,
        ) -> ConfidenceInterval {
            ConfidenceInterval {
                forecast: forecast.to_vec(),
                lower: forecast
                    .iter()
                    .map(|&f| f * (1.0 - self.percentage))
                    .collect(),
                upper: forecast
                    .iter()
                    .map(|&f| f * (1.0 + self.percentage))
                    .collect(),
                confidence_level,
            }
        }
    }

    #[test]
    fn test_normal_ci_computer_basic() {
        let computer = NormalConfidenceIntervalComputer;
        let forecast = vec![100.0, 110.0, 120.0];
        let residuals = vec![1.0, -1.0, 2.0, -2.0, 0.5];

        let ci = computer.compute(&forecast, &residuals, 0.95);

        assert_eq!(ci.forecast, forecast);
        assert_eq!(ci.confidence_level, 0.95);
        assert_eq!(ci.lower.len(), forecast.len());
        assert_eq!(ci.upper.len(), forecast.len());

        // Verify bounds relationship
        for i in 0..forecast.len() {
            assert!(ci.lower[i] < ci.forecast[i]);
            assert!(ci.upper[i] > ci.forecast[i]);
        }
    }

    #[test]
    fn test_normal_ci_computer_empty_forecast() {
        let computer = NormalConfidenceIntervalComputer;
        let forecast: Vec<f64> = vec![];
        let residuals = vec![1.0, 2.0, 3.0];

        let ci = computer.compute(&forecast, &residuals, 0.95);

        assert!(ci.forecast.is_empty());
        assert!(ci.lower.is_empty());
        assert!(ci.upper.is_empty());
    }

    #[test]
    fn test_normal_ci_computer_empty_residuals() {
        let computer = NormalConfidenceIntervalComputer;
        let forecast = vec![100.0, 110.0];
        let residuals: Vec<f64> = vec![];

        let ci = computer.compute(&forecast, &residuals, 0.95);

        // With empty residuals, std dev is 0, so bounds equal forecast
        assert_eq!(ci.forecast, forecast);
        assert_eq!(ci.lower, forecast);
        assert_eq!(ci.upper, forecast);
    }

    #[test]
    fn test_normal_ci_computer_different_confidence_levels() {
        let computer = NormalConfidenceIntervalComputer;
        let forecast = vec![100.0];
        let residuals = vec![1.0, -1.0, 0.5, -0.5];

        let ci_90 = computer.compute(&forecast, &residuals, 0.90);
        let ci_95 = computer.compute(&forecast, &residuals, 0.95);
        let ci_99 = computer.compute(&forecast, &residuals, 0.99);

        // Higher confidence should give wider intervals
        let width_90 = ci_90.upper[0] - ci_90.lower[0];
        let width_95 = ci_95.upper[0] - ci_95.lower[0];
        let width_99 = ci_99.upper[0] - ci_99.lower[0];

        assert!(width_95 > width_90);
        assert!(width_99 > width_95);
    }

    #[test]
    fn test_widening_ci_computer() {
        let computer = WideningConfidenceIntervalComputer::new(0.1);
        let forecast = vec![100.0; 5];
        let residuals = vec![1.0, -1.0, 0.5];

        let ci = computer.compute(&forecast, &residuals, 0.95);

        // Verify intervals widen over horizon
        for i in 1..ci.forecast.len() {
            let prev_width = ci.upper[i - 1] - ci.lower[i - 1];
            let curr_width = ci.upper[i] - ci.lower[i];
            assert!(
                curr_width > prev_width,
                "Width should increase: {} vs {}",
                prev_width,
                curr_width
            );
        }
    }

    #[test]
    fn test_fixed_margin_ci_computer() {
        let margin = 10.0;
        let computer = FixedMarginConfidenceIntervalComputer::new(margin);
        let forecast = vec![100.0, 200.0, 300.0];
        let residuals = vec![1.0, 2.0, 3.0]; // Should be ignored

        let ci = computer.compute(&forecast, &residuals, 0.95);

        for i in 0..forecast.len() {
            assert_eq!(ci.lower[i], forecast[i] - margin);
            assert_eq!(ci.upper[i], forecast[i] + margin);
        }
    }

    #[test]
    fn test_percentage_ci_computer() {
        let percentage = 0.1; // 10%
        let computer = PercentageConfidenceIntervalComputer::new(percentage);
        let forecast = vec![100.0, 200.0, 50.0];
        let residuals: Vec<f64> = vec![];

        let ci = computer.compute(&forecast, &residuals, 0.95);

        assert!((ci.lower[0] - 90.0).abs() < 1e-10); // 100 * 0.9
        assert!((ci.upper[0] - 110.0).abs() < 1e-10); // 100 * 1.1
        assert!((ci.lower[1] - 180.0).abs() < 1e-10); // 200 * 0.9
        assert!((ci.upper[1] - 220.0).abs() < 1e-10); // 200 * 1.1
    }

    #[test]
    fn test_ci_computer_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<NormalConfidenceIntervalComputer>();
        assert_send::<WideningConfidenceIntervalComputer>();
        assert_send::<FixedMarginConfidenceIntervalComputer>();
        assert_send::<PercentageConfidenceIntervalComputer>();
    }

    #[test]
    fn test_ci_computer_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<NormalConfidenceIntervalComputer>();
        assert_sync::<WideningConfidenceIntervalComputer>();
        assert_sync::<FixedMarginConfidenceIntervalComputer>();
        assert_sync::<PercentageConfidenceIntervalComputer>();
    }

    #[test]
    fn test_ci_computer_as_trait_object() {
        let computer: Box<dyn ConfidenceIntervalComputer> =
            Box::new(FixedMarginConfidenceIntervalComputer::new(5.0));
        let forecast = vec![100.0, 200.0];
        let residuals = vec![1.0];

        let ci = computer.compute(&forecast, &residuals, 0.95);

        assert_eq!(ci.lower, vec![95.0, 195.0]);
        assert_eq!(ci.upper, vec![105.0, 205.0]);
    }

    #[test]
    fn test_ci_computer_single_element() {
        let computer = NormalConfidenceIntervalComputer;
        let forecast = vec![50.0];
        let residuals = vec![1.0, -1.0];

        let ci = computer.compute(&forecast, &residuals, 0.95);

        assert_eq!(ci.forecast.len(), 1);
        assert_eq!(ci.lower.len(), 1);
        assert_eq!(ci.upper.len(), 1);
    }

    #[test]
    fn test_ci_computer_negative_values() {
        let computer = NormalConfidenceIntervalComputer;
        let forecast = vec![-100.0, -50.0, 0.0];
        let residuals = vec![1.0, -1.0, 0.5];

        let ci = computer.compute(&forecast, &residuals, 0.95);

        // All bounds should be computed correctly
        for i in 0..forecast.len() {
            assert!(ci.lower[i] < ci.forecast[i]);
            assert!(ci.upper[i] > ci.forecast[i]);
        }
    }

    #[test]
    fn test_ci_computer_constant_residuals() {
        let computer = NormalConfidenceIntervalComputer;
        let forecast = vec![100.0, 200.0];
        let residuals = vec![0.0, 0.0, 0.0]; // Zero variance

        let ci = computer.compute(&forecast, &residuals, 0.95);

        // With zero variance, bounds should equal forecast
        assert_eq!(ci.lower, forecast);
        assert_eq!(ci.upper, forecast);
    }

    #[test]
    fn test_ci_computer_large_residuals() {
        let computer = NormalConfidenceIntervalComputer;
        let forecast = vec![100.0];
        let residuals = vec![100.0, -100.0, 50.0, -50.0];

        let ci = computer.compute(&forecast, &residuals, 0.95);

        // Large residuals should give wide intervals
        let width = ci.upper[0] - ci.lower[0];
        assert!(width > 100.0);
    }

    #[test]
    fn test_ci_computer_z_score_helper() {
        assert!((NormalConfidenceIntervalComputer::z_score(0.90) - 1.645).abs() < 0.01);
        assert!((NormalConfidenceIntervalComputer::z_score(0.95) - 1.96).abs() < 0.01);
        assert!((NormalConfidenceIntervalComputer::z_score(0.99) - 2.576).abs() < 0.01);
    }

    #[test]
    fn test_ci_computer_std_dev_helper() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = NormalConfidenceIntervalComputer::std_dev(&data);
        // Expected std dev is approximately 2.0
        assert!((std - 2.0).abs() < 0.1);

        let empty: Vec<f64> = vec![];
        assert_eq!(NormalConfidenceIntervalComputer::std_dev(&empty), 0.0);
    }

    #[test]
    fn test_ci_computer_preserves_confidence_level() {
        let computer = NormalConfidenceIntervalComputer;
        let forecast = vec![100.0];
        let residuals = vec![1.0];

        for level in [0.50, 0.75, 0.90, 0.95, 0.99] {
            let ci = computer.compute(&forecast, &residuals, level);
            assert_eq!(ci.confidence_level, level);
        }
    }

    #[test]
    fn test_ci_computer_symmetric_intervals() {
        let computer = NormalConfidenceIntervalComputer;
        let forecast = vec![100.0, 200.0, 300.0];
        let residuals = vec![1.0, -1.0, 2.0, -2.0];

        let ci = computer.compute(&forecast, &residuals, 0.95);

        for i in 0..forecast.len() {
            let lower_diff = forecast[i] - ci.lower[i];
            let upper_diff = ci.upper[i] - forecast[i];
            assert!(
                (lower_diff - upper_diff).abs() < 1e-10,
                "Intervals should be symmetric"
            );
        }
    }

    #[test]
    fn test_ci_computer_special_float_values() {
        let computer = FixedMarginConfidenceIntervalComputer::new(10.0);
        let forecast = vec![f64::MAX / 2.0];
        let residuals = vec![1.0];

        // Should handle large values without panic
        let ci = computer.compute(&forecast, &residuals, 0.95);
        assert!(ci.forecast[0].is_finite());
    }

    #[test]
    fn test_widening_ci_zero_growth() {
        let computer = WideningConfidenceIntervalComputer::new(0.0);
        let forecast = vec![100.0; 5];
        let residuals = vec![1.0, -1.0];

        let ci = computer.compute(&forecast, &residuals, 0.95);

        // With zero growth, all intervals should have same width
        let first_width = ci.upper[0] - ci.lower[0];
        for i in 1..ci.forecast.len() {
            let width = ci.upper[i] - ci.lower[i];
            assert!((width - first_width).abs() < 1e-10);
        }
    }

    #[test]
    fn test_percentage_ci_negative_forecast() {
        let computer = PercentageConfidenceIntervalComputer::new(0.1);
        let forecast = vec![-100.0];
        let residuals: Vec<f64> = vec![];

        let ci = computer.compute(&forecast, &residuals, 0.95);

        // For negative values: -100 * 0.9 = -90 (lower), -100 * 1.1 = -110 (upper)
        // Note: the "lower" bound is actually larger in absolute terms
        assert!((ci.lower[0] - (-90.0)).abs() < 1e-10); // -100 * (1 - 0.1) = -90
        assert!((ci.upper[0] - (-110.0)).abs() < 1e-10); // -100 * (1 + 0.1) = -110
    }
}
