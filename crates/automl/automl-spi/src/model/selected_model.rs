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
