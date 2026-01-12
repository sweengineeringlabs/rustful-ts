//! Selected model types for AutoML.

/// A selected model with its optimized parameters.
#[derive(Debug, Clone)]
pub enum SelectedModel {
    Arima { p: usize, d: usize, q: usize },
    SES { alpha: f64 },
    Holt { alpha: f64, beta: f64 },
    HoltWinters { alpha: f64, beta: f64, gamma: f64, period: usize },
    LinearRegression,
    KNN { k: usize, window: usize },
}

/// Alias for backward compatibility.
pub type ModelType = SelectedModel;

impl std::fmt::Display for SelectedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectedModel::Arima { p, d, q } => write!(f, "ARIMA({},{},{})", p, d, q),
            SelectedModel::SES { alpha } => write!(f, "SES(alpha={:.2})", alpha),
            SelectedModel::Holt { alpha, beta } => {
                write!(f, "Holt(alpha={:.2}, beta={:.2})", alpha, beta)
            }
            SelectedModel::HoltWinters {
                alpha,
                beta,
                gamma,
                period,
            } => {
                write!(
                    f,
                    "HoltWinters(alpha={:.2}, beta={:.2}, gamma={:.2}, period={})",
                    alpha, beta, gamma, period
                )
            }
            SelectedModel::LinearRegression => write!(f, "LinearRegression"),
            SelectedModel::KNN { k, window } => write!(f, "KNN(k={}, window={})", k, window),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== ARIMA Tests ==========

    #[test]
    fn test_arima_construction() {
        let model = SelectedModel::Arima { p: 1, d: 1, q: 1 };
        if let SelectedModel::Arima { p, d, q } = model {
            assert_eq!(p, 1);
            assert_eq!(d, 1);
            assert_eq!(q, 1);
        } else {
            panic!("Expected Arima variant");
        }
    }

    #[test]
    fn test_arima_display() {
        let model = SelectedModel::Arima { p: 2, d: 1, q: 3 };
        assert_eq!(model.to_string(), "ARIMA(2,1,3)");
    }

    #[test]
    fn test_arima_display_zero_params() {
        let model = SelectedModel::Arima { p: 0, d: 0, q: 0 };
        assert_eq!(model.to_string(), "ARIMA(0,0,0)");
    }

    #[test]
    fn test_arima_display_large_params() {
        let model = SelectedModel::Arima {
            p: 100,
            d: 50,
            q: 75,
        };
        assert_eq!(model.to_string(), "ARIMA(100,50,75)");
    }

    // ========== SES Tests ==========

    #[test]
    fn test_ses_construction() {
        let model = SelectedModel::SES { alpha: 0.5 };
        if let SelectedModel::SES { alpha } = model {
            assert!((alpha - 0.5).abs() < f64::EPSILON);
        } else {
            panic!("Expected SES variant");
        }
    }

    #[test]
    fn test_ses_display() {
        let model = SelectedModel::SES { alpha: 0.3 };
        assert_eq!(model.to_string(), "SES(alpha=0.30)");
    }

    #[test]
    fn test_ses_display_edge_values() {
        let model_zero = SelectedModel::SES { alpha: 0.0 };
        assert_eq!(model_zero.to_string(), "SES(alpha=0.00)");

        let model_one = SelectedModel::SES { alpha: 1.0 };
        assert_eq!(model_one.to_string(), "SES(alpha=1.00)");
    }

    #[test]
    fn test_ses_display_rounding() {
        let model = SelectedModel::SES { alpha: 0.333333 };
        assert_eq!(model.to_string(), "SES(alpha=0.33)");

        let model_round_up = SelectedModel::SES { alpha: 0.999 };
        assert_eq!(model_round_up.to_string(), "SES(alpha=1.00)");
    }

    // ========== Holt Tests ==========

    #[test]
    fn test_holt_construction() {
        let model = SelectedModel::Holt {
            alpha: 0.5,
            beta: 0.3,
        };
        if let SelectedModel::Holt { alpha, beta } = model {
            assert!((alpha - 0.5).abs() < f64::EPSILON);
            assert!((beta - 0.3).abs() < f64::EPSILON);
        } else {
            panic!("Expected Holt variant");
        }
    }

    #[test]
    fn test_holt_display() {
        let model = SelectedModel::Holt {
            alpha: 0.5,
            beta: 0.3,
        };
        assert_eq!(model.to_string(), "Holt(alpha=0.50, beta=0.30)");
    }

    #[test]
    fn test_holt_display_edge_values() {
        let model = SelectedModel::Holt {
            alpha: 0.0,
            beta: 1.0,
        };
        assert_eq!(model.to_string(), "Holt(alpha=0.00, beta=1.00)");
    }

    // ========== HoltWinters Tests ==========

    #[test]
    fn test_holtwinters_construction() {
        let model = SelectedModel::HoltWinters {
            alpha: 0.5,
            beta: 0.3,
            gamma: 0.2,
            period: 12,
        };
        if let SelectedModel::HoltWinters {
            alpha,
            beta,
            gamma,
            period,
        } = model
        {
            assert!((alpha - 0.5).abs() < f64::EPSILON);
            assert!((beta - 0.3).abs() < f64::EPSILON);
            assert!((gamma - 0.2).abs() < f64::EPSILON);
            assert_eq!(period, 12);
        } else {
            panic!("Expected HoltWinters variant");
        }
    }

    #[test]
    fn test_holtwinters_display() {
        let model = SelectedModel::HoltWinters {
            alpha: 0.5,
            beta: 0.3,
            gamma: 0.2,
            period: 12,
        };
        assert_eq!(
            model.to_string(),
            "HoltWinters(alpha=0.50, beta=0.30, gamma=0.20, period=12)"
        );
    }

    #[test]
    fn test_holtwinters_display_various_periods() {
        let quarterly = SelectedModel::HoltWinters {
            alpha: 0.1,
            beta: 0.1,
            gamma: 0.1,
            period: 4,
        };
        assert!(quarterly.to_string().contains("period=4"));

        let weekly = SelectedModel::HoltWinters {
            alpha: 0.1,
            beta: 0.1,
            gamma: 0.1,
            period: 52,
        };
        assert!(weekly.to_string().contains("period=52"));
    }

    // ========== LinearRegression Tests ==========

    #[test]
    fn test_linear_regression_construction() {
        let model = SelectedModel::LinearRegression;
        assert!(matches!(model, SelectedModel::LinearRegression));
    }

    #[test]
    fn test_linear_regression_display() {
        let model = SelectedModel::LinearRegression;
        assert_eq!(model.to_string(), "LinearRegression");
    }

    // ========== KNN Tests ==========

    #[test]
    fn test_knn_construction() {
        let model = SelectedModel::KNN { k: 5, window: 10 };
        if let SelectedModel::KNN { k, window } = model {
            assert_eq!(k, 5);
            assert_eq!(window, 10);
        } else {
            panic!("Expected KNN variant");
        }
    }

    #[test]
    fn test_knn_display() {
        let model = SelectedModel::KNN { k: 5, window: 10 };
        assert_eq!(model.to_string(), "KNN(k=5, window=10)");
    }

    #[test]
    fn test_knn_display_various_values() {
        let model_small = SelectedModel::KNN { k: 1, window: 1 };
        assert_eq!(model_small.to_string(), "KNN(k=1, window=1)");

        let model_large = SelectedModel::KNN { k: 100, window: 500 };
        assert_eq!(model_large.to_string(), "KNN(k=100, window=500)");
    }

    // ========== Clone Tests ==========

    #[test]
    fn test_clone_arima() {
        let model = SelectedModel::Arima { p: 1, d: 2, q: 3 };
        let cloned = model.clone();
        assert_eq!(model.to_string(), cloned.to_string());
    }

    #[test]
    fn test_clone_ses() {
        let model = SelectedModel::SES { alpha: 0.5 };
        let cloned = model.clone();
        assert_eq!(model.to_string(), cloned.to_string());
    }

    #[test]
    fn test_clone_holt() {
        let model = SelectedModel::Holt {
            alpha: 0.5,
            beta: 0.3,
        };
        let cloned = model.clone();
        assert_eq!(model.to_string(), cloned.to_string());
    }

    #[test]
    fn test_clone_holtwinters() {
        let model = SelectedModel::HoltWinters {
            alpha: 0.5,
            beta: 0.3,
            gamma: 0.2,
            period: 12,
        };
        let cloned = model.clone();
        assert_eq!(model.to_string(), cloned.to_string());
    }

    #[test]
    fn test_clone_linear_regression() {
        let model = SelectedModel::LinearRegression;
        let cloned = model.clone();
        assert_eq!(model.to_string(), cloned.to_string());
    }

    #[test]
    fn test_clone_knn() {
        let model = SelectedModel::KNN { k: 5, window: 10 };
        let cloned = model.clone();
        assert_eq!(model.to_string(), cloned.to_string());
    }

    // ========== Debug Tests ==========

    #[test]
    fn test_debug_format() {
        let model = SelectedModel::Arima { p: 1, d: 1, q: 1 };
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("Arima"));
        assert!(debug_str.contains("p: 1"));
        assert!(debug_str.contains("d: 1"));
        assert!(debug_str.contains("q: 1"));
    }

    #[test]
    fn test_debug_all_variants() {
        let models: Vec<SelectedModel> = vec![
            SelectedModel::Arima { p: 1, d: 1, q: 1 },
            SelectedModel::SES { alpha: 0.5 },
            SelectedModel::Holt {
                alpha: 0.5,
                beta: 0.3,
            },
            SelectedModel::HoltWinters {
                alpha: 0.5,
                beta: 0.3,
                gamma: 0.2,
                period: 12,
            },
            SelectedModel::LinearRegression,
            SelectedModel::KNN { k: 5, window: 10 },
        ];

        for model in models {
            let debug_str = format!("{:?}", model);
            assert!(!debug_str.is_empty());
        }
    }

    // ========== ModelType Alias Tests ==========

    #[test]
    fn test_model_type_alias() {
        let model: ModelType = SelectedModel::SES { alpha: 0.5 };
        assert!(matches!(model, SelectedModel::SES { .. }));
    }

    #[test]
    fn test_model_type_alias_all_variants() {
        let _: ModelType = SelectedModel::Arima { p: 1, d: 1, q: 1 };
        let _: ModelType = SelectedModel::SES { alpha: 0.5 };
        let _: ModelType = SelectedModel::Holt {
            alpha: 0.5,
            beta: 0.3,
        };
        let _: ModelType = SelectedModel::HoltWinters {
            alpha: 0.5,
            beta: 0.3,
            gamma: 0.2,
            period: 12,
        };
        let _: ModelType = SelectedModel::LinearRegression;
        let _: ModelType = SelectedModel::KNN { k: 5, window: 10 };
    }

    // ========== Pattern Matching Tests ==========

    #[test]
    fn test_pattern_matching_exhaustive() {
        let models: Vec<SelectedModel> = vec![
            SelectedModel::Arima { p: 1, d: 1, q: 1 },
            SelectedModel::SES { alpha: 0.5 },
            SelectedModel::Holt {
                alpha: 0.5,
                beta: 0.3,
            },
            SelectedModel::HoltWinters {
                alpha: 0.5,
                beta: 0.3,
                gamma: 0.2,
                period: 12,
            },
            SelectedModel::LinearRegression,
            SelectedModel::KNN { k: 5, window: 10 },
        ];

        for model in models {
            let name = match &model {
                SelectedModel::Arima { .. } => "arima",
                SelectedModel::SES { .. } => "ses",
                SelectedModel::Holt { .. } => "holt",
                SelectedModel::HoltWinters { .. } => "holtwinters",
                SelectedModel::LinearRegression => "linear",
                SelectedModel::KNN { .. } => "knn",
            };
            assert!(!name.is_empty());
        }
    }
}
