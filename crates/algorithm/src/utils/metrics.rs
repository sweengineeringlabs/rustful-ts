//! Forecast accuracy metrics
//!
//! Provides standard metrics for evaluating time series forecasts.

/// Mean Absolute Error (MAE)
///
/// Average of absolute differences between predictions and actual values.
/// Lower is better. Same scale as the data.
///
/// # Example
///
/// ```rust
/// use algorithm::utils::metrics::mae;
///
/// let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let predicted = vec![1.1, 2.2, 2.9, 4.1, 5.0];
/// let error = mae(&actual, &predicted);
/// ```
pub fn mae(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return f64::NAN;
    }

    let sum: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).abs())
        .sum();

    sum / actual.len() as f64
}

/// Mean Squared Error (MSE)
///
/// Average of squared differences. Penalizes large errors more heavily.
/// Lower is better.
///
/// # Example
///
/// ```rust
/// use algorithm::utils::metrics::mse;
///
/// let actual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let predicted = vec![1.1, 2.2, 2.9, 4.1, 5.0];
/// let error = mse(&actual, &predicted);
/// ```
pub fn mse(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return f64::NAN;
    }

    let sum: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum();

    sum / actual.len() as f64
}

/// Root Mean Squared Error (RMSE)
///
/// Square root of MSE. Same scale as the data.
/// Lower is better.
pub fn rmse(actual: &[f64], predicted: &[f64]) -> f64 {
    mse(actual, predicted).sqrt()
}

/// Mean Absolute Percentage Error (MAPE)
///
/// Average of absolute percentage errors. Scale-independent.
/// Lower is better. Undefined when actual values are zero.
///
/// # Returns
///
/// Value between 0 and infinity (as a decimal, not percentage).
pub fn mape(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return f64::NAN;
    }

    let sum: f64 = actual
        .iter()
        .zip(predicted.iter())
        .filter(|(a, _)| a.abs() > 1e-10)
        .map(|(a, p)| ((a - p) / a).abs())
        .sum();

    sum / actual.len() as f64
}

/// Symmetric Mean Absolute Percentage Error (sMAPE)
///
/// Symmetric version of MAPE that handles zero values better.
/// Returns value between 0 and 2 (as a decimal).
pub fn smape(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return f64::NAN;
    }

    let sum: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| {
            let denom = a.abs() + p.abs();
            if denom > 1e-10 {
                2.0 * (a - p).abs() / denom
            } else {
                0.0
            }
        })
        .sum();

    sum / actual.len() as f64
}

/// Mean Absolute Scaled Error (MASE)
///
/// Scale-free metric that compares forecast error to naive forecast error.
/// Values < 1 indicate better than naive forecast.
///
/// # Arguments
///
/// * `actual` - Actual values
/// * `predicted` - Predicted values
/// * `training_data` - Historical data used to compute naive error
/// * `seasonality` - Seasonal period for naive forecast (1 for non-seasonal)
pub fn mase(
    actual: &[f64],
    predicted: &[f64],
    training_data: &[f64],
    seasonality: usize,
) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return f64::NAN;
    }

    let seasonality = seasonality.max(1);

    // Compute naive forecast error on training data
    if training_data.len() <= seasonality {
        return f64::NAN;
    }

    let naive_errors: f64 = training_data
        .iter()
        .skip(seasonality)
        .zip(training_data.iter())
        .map(|(curr, prev)| (curr - prev).abs())
        .sum();

    let naive_mae = naive_errors / (training_data.len() - seasonality) as f64;

    if naive_mae < 1e-10 {
        return f64::NAN;
    }

    mae(actual, predicted) / naive_mae
}

/// R-squared (Coefficient of Determination)
///
/// Measures how well predictions explain variance in actual values.
/// 1.0 = perfect, 0.0 = same as mean prediction, negative = worse than mean.
pub fn r_squared(actual: &[f64], predicted: &[f64]) -> f64 {
    if actual.len() != predicted.len() || actual.is_empty() {
        return f64::NAN;
    }

    let mean = actual.iter().sum::<f64>() / actual.len() as f64;

    let ss_tot: f64 = actual.iter().map(|a| (a - mean).powi(2)).sum();
    let ss_res: f64 = actual
        .iter()
        .zip(predicted.iter())
        .map(|(a, p)| (a - p).powi(2))
        .sum();

    if ss_tot < 1e-10 {
        return 1.0;
    }

    1.0 - ss_res / ss_tot
}

/// Compute all common metrics at once
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub mae: f64,
    pub mse: f64,
    pub rmse: f64,
    pub mape: f64,
    pub smape: f64,
    pub r_squared: f64,
}

impl MetricsSummary {
    /// Compute all metrics for a forecast
    pub fn compute(actual: &[f64], predicted: &[f64]) -> Self {
        Self {
            mae: mae(actual, predicted),
            mse: mse(actual, predicted),
            rmse: rmse(actual, predicted),
            mape: mape(actual, predicted),
            smape: smape(actual, predicted),
            r_squared: r_squared(actual, predicted),
        }
    }
}

