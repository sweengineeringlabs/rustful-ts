//! Risk calculator trait.

/// Risk metrics calculator trait.
pub trait RiskCalculator: Send + Sync {
    /// Calculate Value at Risk (historical method).
    fn var_historical(&self, returns: &[f64], confidence: f64) -> f64;

    /// Calculate Sharpe ratio.
    fn sharpe_ratio(&self, returns: &[f64], risk_free_rate: f64) -> f64;

    /// Calculate maximum drawdown.
    fn max_drawdown(&self, equity_curve: &[f64]) -> f64;

    /// Calculate Sortino ratio (downside deviation only).
    fn sortino_ratio(&self, returns: &[f64], risk_free_rate: f64) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Mock risk calculator implementation for testing
    struct MockRiskCalculator;

    impl RiskCalculator for MockRiskCalculator {
        fn var_historical(&self, returns: &[f64], confidence: f64) -> f64 {
            if returns.is_empty() {
                return 0.0;
            }
            let mut sorted_returns: Vec<f64> = returns.to_vec();
            sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let index = ((1.0 - confidence) * returns.len() as f64).floor() as usize;
            let index = index.min(returns.len() - 1);
            -sorted_returns[index] // VaR is typically reported as positive
        }

        fn sharpe_ratio(&self, returns: &[f64], risk_free_rate: f64) -> f64 {
            if returns.is_empty() {
                return 0.0;
            }
            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns
                .iter()
                .map(|r| (r - mean).powi(2))
                .sum::<f64>()
                / returns.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev == 0.0 {
                return 0.0;
            }
            (mean - risk_free_rate) / std_dev
        }

        fn max_drawdown(&self, equity_curve: &[f64]) -> f64 {
            if equity_curve.is_empty() {
                return 0.0;
            }

            let mut max_drawdown = 0.0;
            let mut peak = equity_curve[0];

            for &value in equity_curve {
                if value > peak {
                    peak = value;
                }
                let drawdown = (peak - value) / peak;
                if drawdown > max_drawdown {
                    max_drawdown = drawdown;
                }
            }
            max_drawdown
        }

        fn sortino_ratio(&self, returns: &[f64], risk_free_rate: f64) -> f64 {
            if returns.is_empty() {
                return 0.0;
            }
            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;

            // Calculate downside deviation (only negative returns)
            let downside_returns: Vec<f64> = returns
                .iter()
                .filter(|&&r| r < risk_free_rate)
                .map(|&r| (r - risk_free_rate).powi(2))
                .collect();

            if downside_returns.is_empty() {
                return f64::INFINITY; // No downside risk
            }

            let downside_variance: f64 =
                downside_returns.iter().sum::<f64>() / downside_returns.len() as f64;
            let downside_dev = downside_variance.sqrt();

            if downside_dev == 0.0 {
                return 0.0;
            }
            (mean - risk_free_rate) / downside_dev
        }
    }

    /// Fixed result calculator for testing trait contract
    struct FixedRiskCalculator {
        var: f64,
        sharpe: f64,
        max_dd: f64,
        sortino: f64,
    }

    impl FixedRiskCalculator {
        fn new(var: f64, sharpe: f64, max_dd: f64, sortino: f64) -> Self {
            Self {
                var,
                sharpe,
                max_dd,
                sortino,
            }
        }
    }

    impl RiskCalculator for FixedRiskCalculator {
        fn var_historical(&self, _returns: &[f64], _confidence: f64) -> f64 {
            self.var
        }

        fn sharpe_ratio(&self, _returns: &[f64], _risk_free_rate: f64) -> f64 {
            self.sharpe
        }

        fn max_drawdown(&self, _equity_curve: &[f64]) -> f64 {
            self.max_dd
        }

        fn sortino_ratio(&self, _returns: &[f64], _risk_free_rate: f64) -> f64 {
            self.sortino
        }
    }

    // VaR tests
    #[test]
    fn test_var_historical_basic() {
        let calc = MockRiskCalculator;
        let returns = vec![-0.05, -0.02, 0.01, 0.03, 0.02, -0.01, 0.04, -0.03, 0.02, 0.01];
        let var = calc.var_historical(&returns, 0.95);
        assert!(var > 0.0); // VaR should be positive (loss)
    }

    #[test]
    fn test_var_historical_empty() {
        let calc = MockRiskCalculator;
        let returns: Vec<f64> = vec![];
        let var = calc.var_historical(&returns, 0.95);
        assert_eq!(var, 0.0);
    }

    #[test]
    fn test_var_historical_all_positive() {
        let calc = MockRiskCalculator;
        let returns = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let var = calc.var_historical(&returns, 0.95);
        // With all positive returns, worst loss would be the smallest gain (opportunity cost view)
        // or negative VaR depending on implementation
        assert!(var <= 0.0 || var >= 0.0); // Just verify it returns a number
    }

    #[test]
    fn test_var_historical_all_negative() {
        let calc = MockRiskCalculator;
        let returns = vec![-0.01, -0.02, -0.03, -0.04, -0.05];
        let var = calc.var_historical(&returns, 0.95);
        assert!(var > 0.0); // Should be positive (representing a loss)
    }

    #[test]
    fn test_var_historical_different_confidences() {
        let calc = MockRiskCalculator;
        let returns = vec![-0.10, -0.05, -0.02, 0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10];

        let var_90 = calc.var_historical(&returns, 0.90);
        let var_95 = calc.var_historical(&returns, 0.95);
        let var_99 = calc.var_historical(&returns, 0.99);

        // Higher confidence = looking at more extreme losses
        assert!(var_99 >= var_95 || (var_99 - var_95).abs() < 0.01);
        assert!(var_95 >= var_90 || (var_95 - var_90).abs() < 0.01);
    }

    // Sharpe ratio tests
    #[test]
    fn test_sharpe_ratio_positive() {
        let calc = MockRiskCalculator;
        let returns = vec![0.01, 0.02, 0.015, 0.025, 0.02]; // Good positive returns
        let sharpe = calc.sharpe_ratio(&returns, 0.001);
        assert!(sharpe > 0.0);
    }

    #[test]
    fn test_sharpe_ratio_negative() {
        let calc = MockRiskCalculator;
        let returns = vec![-0.01, -0.02, -0.015, -0.025, -0.02]; // Negative returns
        let sharpe = calc.sharpe_ratio(&returns, 0.001);
        assert!(sharpe < 0.0);
    }

    #[test]
    fn test_sharpe_ratio_empty() {
        let calc = MockRiskCalculator;
        let returns: Vec<f64> = vec![];
        let sharpe = calc.sharpe_ratio(&returns, 0.01);
        assert_eq!(sharpe, 0.0);
    }

    #[test]
    fn test_sharpe_ratio_constant_returns() {
        let calc = MockRiskCalculator;
        let returns = vec![0.01, 0.01, 0.01, 0.01, 0.01]; // Zero volatility
        let sharpe = calc.sharpe_ratio(&returns, 0.005);
        // With zero std dev, should return 0 or handle gracefully
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_sharpe_ratio_risk_free_rate_effect() {
        let calc = MockRiskCalculator;
        let returns = vec![0.02, 0.03, 0.025, 0.035, 0.03];

        let sharpe_low_rf = calc.sharpe_ratio(&returns, 0.001);
        let sharpe_high_rf = calc.sharpe_ratio(&returns, 0.02);

        // Higher risk-free rate = lower Sharpe
        assert!(sharpe_low_rf > sharpe_high_rf);
    }

    // Max drawdown tests
    #[test]
    fn test_max_drawdown_basic() {
        let calc = MockRiskCalculator;
        let equity = vec![100.0, 110.0, 105.0, 115.0, 108.0, 120.0];
        let mdd = calc.max_drawdown(&equity);
        // Max drawdown: peak 115 to trough 108 = (115-108)/115 = 0.0609
        assert!(mdd > 0.0);
        assert!(mdd < 1.0);
    }

    #[test]
    fn test_max_drawdown_no_drawdown() {
        let calc = MockRiskCalculator;
        let equity = vec![100.0, 110.0, 120.0, 130.0, 140.0]; // Monotonic increase
        let mdd = calc.max_drawdown(&equity);
        assert_eq!(mdd, 0.0);
    }

    #[test]
    fn test_max_drawdown_total_loss() {
        let calc = MockRiskCalculator;
        let equity = vec![100.0, 50.0, 25.0, 10.0, 0.0];
        let mdd = calc.max_drawdown(&equity);
        assert_eq!(mdd, 1.0); // 100% drawdown
    }

    #[test]
    fn test_max_drawdown_empty() {
        let calc = MockRiskCalculator;
        let equity: Vec<f64> = vec![];
        let mdd = calc.max_drawdown(&equity);
        assert_eq!(mdd, 0.0);
    }

    #[test]
    fn test_max_drawdown_single_point() {
        let calc = MockRiskCalculator;
        let equity = vec![100.0];
        let mdd = calc.max_drawdown(&equity);
        assert_eq!(mdd, 0.0);
    }

    #[test]
    fn test_max_drawdown_recovery() {
        let calc = MockRiskCalculator;
        // Drawdown and recovery
        let equity = vec![100.0, 120.0, 90.0, 110.0, 130.0];
        let mdd = calc.max_drawdown(&equity);
        // Peak 120, trough 90: (120-90)/120 = 0.25
        assert!((mdd - 0.25).abs() < 0.01);
    }

    // Sortino ratio tests
    #[test]
    fn test_sortino_ratio_positive_returns() {
        let calc = MockRiskCalculator;
        let returns = vec![0.02, 0.03, 0.01, 0.04, 0.02]; // All positive
        let sortino = calc.sortino_ratio(&returns, 0.0);
        // With no negative returns, should be infinity or very high
        assert!(sortino.is_infinite() || sortino > 10.0);
    }

    #[test]
    fn test_sortino_ratio_mixed_returns() {
        let calc = MockRiskCalculator;
        let returns = vec![0.02, -0.01, 0.03, -0.02, 0.01];
        let sortino = calc.sortino_ratio(&returns, 0.0);
        assert!(sortino.is_finite());
    }

    #[test]
    fn test_sortino_ratio_empty() {
        let calc = MockRiskCalculator;
        let returns: Vec<f64> = vec![];
        let sortino = calc.sortino_ratio(&returns, 0.0);
        assert_eq!(sortino, 0.0);
    }

    #[test]
    fn test_sortino_ratio_all_negative() {
        let calc = MockRiskCalculator;
        let returns = vec![-0.01, -0.02, -0.03, -0.01, -0.02];
        let sortino = calc.sortino_ratio(&returns, 0.0);
        assert!(sortino < 0.0);
    }

    #[test]
    fn test_sortino_vs_sharpe() {
        let calc = MockRiskCalculator;
        // Returns with large upside volatility but little downside
        let returns = vec![0.01, 0.05, 0.02, 0.08, -0.01, 0.03];

        let sharpe = calc.sharpe_ratio(&returns, 0.0);
        let sortino = calc.sortino_ratio(&returns, 0.0);

        // Sortino should typically be higher when upside volatility dominates
        // (since it only penalizes downside)
        assert!(sortino > sharpe || sortino.is_infinite());
    }

    // Fixed calculator tests
    #[test]
    fn test_fixed_risk_calculator() {
        let calc = FixedRiskCalculator::new(0.05, 1.5, 0.15, 2.0);

        assert_eq!(calc.var_historical(&[], 0.95), 0.05);
        assert_eq!(calc.sharpe_ratio(&[], 0.0), 1.5);
        assert_eq!(calc.max_drawdown(&[]), 0.15);
        assert_eq!(calc.sortino_ratio(&[], 0.0), 2.0);
    }

    // Trait bounds tests
    #[test]
    fn test_risk_calculator_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<MockRiskCalculator>();
        assert_send::<FixedRiskCalculator>();
    }

    #[test]
    fn test_risk_calculator_is_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<MockRiskCalculator>();
        assert_sync::<FixedRiskCalculator>();
    }

    #[test]
    fn test_risk_calculator_trait_object() {
        let calc: Box<dyn RiskCalculator> = Box::new(MockRiskCalculator);
        let returns = vec![0.01, 0.02, -0.01, 0.03];

        let var = calc.var_historical(&returns, 0.95);
        let sharpe = calc.sharpe_ratio(&returns, 0.0);
        let mdd = calc.max_drawdown(&[100.0, 110.0, 105.0]);
        let sortino = calc.sortino_ratio(&returns, 0.0);

        assert!(var.is_finite());
        assert!(sharpe.is_finite());
        assert!(mdd.is_finite());
        assert!(sortino.is_finite() || sortino.is_infinite());
    }

    #[test]
    fn test_risk_calculator_in_arc() {
        let calc: Arc<dyn RiskCalculator> = Arc::new(MockRiskCalculator);

        let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02];
        let sharpe = calc.sharpe_ratio(&returns, 0.0);
        assert!(sharpe.is_finite());
    }

    #[test]
    fn test_risk_calculator_multiple_implementations() {
        let calculators: Vec<Box<dyn RiskCalculator>> = vec![
            Box::new(MockRiskCalculator),
            Box::new(FixedRiskCalculator::new(0.03, 1.2, 0.10, 1.8)),
        ];

        let returns = vec![0.01, 0.02, -0.01];

        for calc in &calculators {
            let sharpe = calc.sharpe_ratio(&returns, 0.0);
            assert!(sharpe.is_finite());
        }
    }

    // Edge case tests
    #[test]
    fn test_var_with_single_return() {
        let calc = MockRiskCalculator;
        let returns = vec![-0.05];
        let var = calc.var_historical(&returns, 0.95);
        assert_eq!(var, 0.05);
    }

    #[test]
    fn test_max_drawdown_with_zeros() {
        let calc = MockRiskCalculator;
        let equity = vec![100.0, 100.0, 100.0];
        let mdd = calc.max_drawdown(&equity);
        assert_eq!(mdd, 0.0);
    }

    #[test]
    fn test_sharpe_with_single_return() {
        let calc = MockRiskCalculator;
        let returns = vec![0.05];
        let sharpe = calc.sharpe_ratio(&returns, 0.0);
        // With single value, std dev is 0
        assert_eq!(sharpe, 0.0);
    }

    #[test]
    fn test_large_returns_dataset() {
        let calc = MockRiskCalculator;
        let returns: Vec<f64> = (0..1000)
            .map(|i| ((i as f64) * 0.1).sin() * 0.05)
            .collect();

        let var = calc.var_historical(&returns, 0.95);
        let sharpe = calc.sharpe_ratio(&returns, 0.0);
        let sortino = calc.sortino_ratio(&returns, 0.0);

        assert!(var.is_finite());
        assert!(sharpe.is_finite());
        assert!(sortino.is_finite() || sortino.is_infinite());
    }

    #[test]
    fn test_large_equity_curve() {
        let calc = MockRiskCalculator;
        let equity: Vec<f64> = (0..1000)
            .map(|i| 1000.0 + (i as f64) + ((i as f64) * 0.1).sin() * 50.0)
            .collect();

        let mdd = calc.max_drawdown(&equity);
        assert!(mdd >= 0.0 && mdd <= 1.0);
    }
}
