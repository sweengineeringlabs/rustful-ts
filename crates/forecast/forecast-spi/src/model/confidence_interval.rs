//! Confidence interval model

/// Confidence interval result
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    /// Point forecast
    pub forecast: Vec<f64>,
    /// Lower bound of confidence interval
    pub lower: Vec<f64>,
    /// Upper bound of confidence interval
    pub upper: Vec<f64>,
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_interval_creation() {
        let ci = ConfidenceInterval {
            forecast: vec![100.0, 110.0, 120.0],
            lower: vec![90.0, 100.0, 110.0],
            upper: vec![110.0, 120.0, 130.0],
            confidence_level: 0.95,
        };

        assert_eq!(ci.forecast, vec![100.0, 110.0, 120.0]);
        assert_eq!(ci.lower, vec![90.0, 100.0, 110.0]);
        assert_eq!(ci.upper, vec![110.0, 120.0, 130.0]);
        assert_eq!(ci.confidence_level, 0.95);
    }

    #[test]
    fn test_confidence_interval_empty_vectors() {
        let ci = ConfidenceInterval {
            forecast: vec![],
            lower: vec![],
            upper: vec![],
            confidence_level: 0.90,
        };

        assert!(ci.forecast.is_empty());
        assert!(ci.lower.is_empty());
        assert!(ci.upper.is_empty());
        assert_eq!(ci.confidence_level, 0.90);
    }

    #[test]
    fn test_confidence_interval_single_element() {
        let ci = ConfidenceInterval {
            forecast: vec![50.0],
            lower: vec![45.0],
            upper: vec![55.0],
            confidence_level: 0.99,
        };

        assert_eq!(ci.forecast.len(), 1);
        assert_eq!(ci.lower.len(), 1);
        assert_eq!(ci.upper.len(), 1);
        assert_eq!(ci.forecast[0], 50.0);
        assert_eq!(ci.lower[0], 45.0);
        assert_eq!(ci.upper[0], 55.0);
    }

    #[test]
    fn test_confidence_interval_common_levels() {
        // Test common confidence levels: 90%, 95%, 99%
        let levels = vec![0.90, 0.95, 0.99];

        for level in levels {
            let ci = ConfidenceInterval {
                forecast: vec![100.0],
                lower: vec![90.0],
                upper: vec![110.0],
                confidence_level: level,
            };
            assert_eq!(ci.confidence_level, level);
        }
    }

    #[test]
    fn test_confidence_interval_extreme_levels() {
        // Edge case: very low confidence
        let ci_low = ConfidenceInterval {
            forecast: vec![100.0],
            lower: vec![99.0],
            upper: vec![101.0],
            confidence_level: 0.50,
        };
        assert_eq!(ci_low.confidence_level, 0.50);

        // Edge case: very high confidence
        let ci_high = ConfidenceInterval {
            forecast: vec![100.0],
            lower: vec![50.0],
            upper: vec![150.0],
            confidence_level: 0.999,
        };
        assert_eq!(ci_high.confidence_level, 0.999);
    }

    #[test]
    fn test_confidence_interval_bounds_relationship() {
        let ci = ConfidenceInterval {
            forecast: vec![100.0, 110.0, 120.0],
            lower: vec![90.0, 100.0, 110.0],
            upper: vec![110.0, 120.0, 130.0],
            confidence_level: 0.95,
        };

        // Verify lower <= forecast <= upper for all indices
        for i in 0..ci.forecast.len() {
            assert!(ci.lower[i] <= ci.forecast[i]);
            assert!(ci.forecast[i] <= ci.upper[i]);
        }
    }

    #[test]
    fn test_confidence_interval_symmetric_bounds() {
        let forecast = vec![100.0, 200.0, 300.0];
        let margin = 10.0;
        let ci = ConfidenceInterval {
            forecast: forecast.clone(),
            lower: forecast.iter().map(|&x| x - margin).collect(),
            upper: forecast.iter().map(|&x| x + margin).collect(),
            confidence_level: 0.95,
        };

        // Verify symmetric bounds
        for i in 0..ci.forecast.len() {
            let lower_diff = ci.forecast[i] - ci.lower[i];
            let upper_diff = ci.upper[i] - ci.forecast[i];
            assert!((lower_diff - upper_diff).abs() < 1e-10);
        }
    }

    #[test]
    fn test_confidence_interval_asymmetric_bounds() {
        // Some distributions have asymmetric confidence intervals
        let ci = ConfidenceInterval {
            forecast: vec![100.0],
            lower: vec![80.0], // 20 below
            upper: vec![150.0], // 50 above
            confidence_level: 0.95,
        };

        assert_eq!(ci.forecast[0] - ci.lower[0], 20.0);
        assert_eq!(ci.upper[0] - ci.forecast[0], 50.0);
    }

    #[test]
    fn test_confidence_interval_large_data() {
        let size = 1000;
        let ci = ConfidenceInterval {
            forecast: (0..size).map(|i| 100.0 + i as f64).collect(),
            lower: (0..size).map(|i| 90.0 + i as f64).collect(),
            upper: (0..size).map(|i| 110.0 + i as f64).collect(),
            confidence_level: 0.95,
        };

        assert_eq!(ci.forecast.len(), size);
        assert_eq!(ci.lower.len(), size);
        assert_eq!(ci.upper.len(), size);
    }

    #[test]
    fn test_confidence_interval_special_float_values() {
        let ci = ConfidenceInterval {
            forecast: vec![f64::INFINITY, 0.0, f64::NEG_INFINITY],
            lower: vec![f64::NEG_INFINITY, -1.0, f64::NEG_INFINITY],
            upper: vec![f64::INFINITY, 1.0, 0.0],
            confidence_level: 0.95,
        };

        assert!(ci.forecast[0].is_infinite());
        assert!(ci.lower[0].is_infinite());
        assert!(ci.upper[0].is_infinite());
    }

    #[test]
    fn test_confidence_interval_negative_values() {
        let ci = ConfidenceInterval {
            forecast: vec![-100.0, -50.0, -25.0],
            lower: vec![-120.0, -70.0, -45.0],
            upper: vec![-80.0, -30.0, -5.0],
            confidence_level: 0.95,
        };

        // Verify all values are negative
        assert!(ci.forecast.iter().all(|&x| x < 0.0));
        assert!(ci.lower.iter().all(|&x| x < 0.0));
        assert!(ci.upper.iter().all(|&x| x < 0.0));

        // Verify ordering still holds
        for i in 0..ci.forecast.len() {
            assert!(ci.lower[i] <= ci.forecast[i]);
            assert!(ci.forecast[i] <= ci.upper[i]);
        }
    }

    #[test]
    fn test_confidence_interval_clone() {
        let original = ConfidenceInterval {
            forecast: vec![100.0, 110.0],
            lower: vec![90.0, 100.0],
            upper: vec![110.0, 120.0],
            confidence_level: 0.95,
        };

        let cloned = original.clone();

        assert_eq!(original.forecast, cloned.forecast);
        assert_eq!(original.lower, cloned.lower);
        assert_eq!(original.upper, cloned.upper);
        assert_eq!(original.confidence_level, cloned.confidence_level);
    }

    #[test]
    fn test_confidence_interval_clone_independence() {
        let original = ConfidenceInterval {
            forecast: vec![100.0, 110.0],
            lower: vec![90.0, 100.0],
            upper: vec![110.0, 120.0],
            confidence_level: 0.95,
        };

        let mut cloned = original.clone();
        cloned.forecast[0] = 999.0;
        cloned.lower[0] = 999.0;
        cloned.upper[0] = 999.0;
        cloned.confidence_level = 0.50;

        // Original should be unchanged
        assert_eq!(original.forecast[0], 100.0);
        assert_eq!(original.lower[0], 90.0);
        assert_eq!(original.upper[0], 110.0);
        assert_eq!(original.confidence_level, 0.95);
    }

    #[test]
    fn test_confidence_interval_debug() {
        let ci = ConfidenceInterval {
            forecast: vec![100.0],
            lower: vec![90.0],
            upper: vec![110.0],
            confidence_level: 0.95,
        };

        let debug_str = format!("{:?}", ci);
        assert!(debug_str.contains("ConfidenceInterval"));
        assert!(debug_str.contains("forecast"));
        assert!(debug_str.contains("lower"));
        assert!(debug_str.contains("upper"));
        assert!(debug_str.contains("confidence_level"));
    }

    #[test]
    fn test_confidence_interval_field_mutability() {
        let mut ci = ConfidenceInterval {
            forecast: vec![100.0],
            lower: vec![90.0],
            upper: vec![110.0],
            confidence_level: 0.95,
        };

        ci.forecast.push(200.0);
        ci.lower.push(190.0);
        ci.upper.push(210.0);
        ci.confidence_level = 0.99;

        assert_eq!(ci.forecast.len(), 2);
        assert_eq!(ci.lower.len(), 2);
        assert_eq!(ci.upper.len(), 2);
        assert_eq!(ci.confidence_level, 0.99);
    }

    #[test]
    fn test_confidence_interval_widening_over_horizon() {
        // Typical pattern: CI widens for longer forecast horizons
        let ci = ConfidenceInterval {
            forecast: vec![100.0, 100.0, 100.0, 100.0],
            lower: vec![95.0, 90.0, 85.0, 80.0],
            upper: vec![105.0, 110.0, 115.0, 120.0],
            confidence_level: 0.95,
        };

        // Verify intervals widen
        for i in 1..ci.forecast.len() {
            let prev_width = ci.upper[i - 1] - ci.lower[i - 1];
            let curr_width = ci.upper[i] - ci.lower[i];
            assert!(curr_width > prev_width);
        }
    }

    #[test]
    fn test_confidence_interval_interval_width() {
        let ci = ConfidenceInterval {
            forecast: vec![100.0, 200.0, 300.0],
            lower: vec![90.0, 180.0, 270.0],
            upper: vec![110.0, 220.0, 330.0],
            confidence_level: 0.95,
        };

        let widths: Vec<f64> = (0..ci.forecast.len())
            .map(|i| ci.upper[i] - ci.lower[i])
            .collect();

        assert_eq!(widths, vec![20.0, 40.0, 60.0]);
    }

    #[test]
    fn test_confidence_interval_zero_width() {
        // Degenerate case: point forecast only (zero uncertainty)
        let ci = ConfidenceInterval {
            forecast: vec![100.0, 200.0],
            lower: vec![100.0, 200.0],
            upper: vec![100.0, 200.0],
            confidence_level: 1.0,
        };

        for i in 0..ci.forecast.len() {
            assert_eq!(ci.lower[i], ci.forecast[i]);
            assert_eq!(ci.upper[i], ci.forecast[i]);
        }
    }

    #[test]
    fn test_confidence_interval_boundary_level_values() {
        // Test boundary confidence levels
        let ci_zero = ConfidenceInterval {
            forecast: vec![100.0],
            lower: vec![100.0],
            upper: vec![100.0],
            confidence_level: 0.0,
        };
        assert_eq!(ci_zero.confidence_level, 0.0);

        let ci_one = ConfidenceInterval {
            forecast: vec![100.0],
            lower: vec![f64::NEG_INFINITY],
            upper: vec![f64::INFINITY],
            confidence_level: 1.0,
        };
        assert_eq!(ci_one.confidence_level, 1.0);
    }
}
