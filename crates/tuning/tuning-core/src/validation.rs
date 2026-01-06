//! Validation strategy implementations.

use tuning_spi::{Validator, ValidationStrategy, ValidationSplit, Result, TuningError};

/// Train/Test split validator.
#[derive(Debug, Clone)]
pub struct TrainTestValidator {
    train_ratio: f64,
}

impl TrainTestValidator {
    pub fn new(train_ratio: f64) -> Self {
        Self { train_ratio }
    }
}

impl Validator for TrainTestValidator {
    fn splits(&self, data_len: usize) -> Result<Vec<ValidationSplit>> {
        let train_end = (data_len as f64 * self.train_ratio) as usize;

        if train_end < 2 || train_end >= data_len - 1 {
            return Err(TuningError::InsufficientData {
                required: 4,
                got: data_len,
            });
        }

        Ok(vec![ValidationSplit {
            train_start: 0,
            train_end,
            test_start: train_end,
            test_end: data_len,
        }])
    }

    fn strategy(&self) -> ValidationStrategy {
        ValidationStrategy::TrainTest { train_ratio: self.train_ratio }
    }
}

/// Walk-forward validator.
#[derive(Debug, Clone)]
pub struct WalkForwardValidator {
    windows: usize,
    train_ratio: f64,
}

impl WalkForwardValidator {
    pub fn new(windows: usize, train_ratio: f64) -> Self {
        Self { windows, train_ratio }
    }
}

impl Validator for WalkForwardValidator {
    fn splits(&self, data_len: usize) -> Result<Vec<ValidationSplit>> {
        if self.windows == 0 {
            return Err(TuningError::InvalidConfig("windows must be > 0".into()));
        }

        let window_size = data_len / self.windows;
        if window_size < 2 {
            return Err(TuningError::InsufficientData {
                required: self.windows * 2,
                got: data_len,
            });
        }

        let mut splits = Vec::new();

        for i in 0..self.windows {
            let window_end = (i + 1) * window_size;
            let train_size = (window_size as f64 * self.train_ratio) as usize;
            let train_start = i * window_size;
            let train_end = train_start + train_size;
            let test_start = train_end;
            let test_end = window_end.min(data_len);

            if train_end > test_start || test_start >= test_end {
                continue;
            }

            splits.push(ValidationSplit {
                train_start,
                train_end,
                test_start,
                test_end,
            });
        }

        if splits.is_empty() {
            return Err(TuningError::InsufficientData {
                required: self.windows * 4,
                got: data_len,
            });
        }

        Ok(splits)
    }

    fn strategy(&self) -> ValidationStrategy {
        ValidationStrategy::WalkForward {
            windows: self.windows,
            train_ratio: self.train_ratio,
        }
    }
}

/// K-Fold cross-validator.
#[derive(Debug, Clone)]
pub struct KFoldValidator {
    folds: usize,
}

impl KFoldValidator {
    pub fn new(folds: usize) -> Self {
        Self { folds }
    }
}

impl Validator for KFoldValidator {
    fn splits(&self, data_len: usize) -> Result<Vec<ValidationSplit>> {
        if self.folds < 2 {
            return Err(TuningError::InvalidConfig("folds must be >= 2".into()));
        }

        let fold_size = data_len / self.folds;
        if fold_size < 1 {
            return Err(TuningError::InsufficientData {
                required: self.folds,
                got: data_len,
            });
        }

        let mut splits = Vec::new();

        for i in 0..self.folds {
            let test_start = i * fold_size;
            let test_end = if i == self.folds - 1 {
                data_len
            } else {
                (i + 1) * fold_size
            };

            // Training is everything except the test fold
            // For time series, we typically use data before test
            let train_start = 0;
            let train_end = test_start;

            if train_end <= train_start {
                continue; // First fold has no training data
            }

            splits.push(ValidationSplit {
                train_start,
                train_end,
                test_start,
                test_end,
            });
        }

        Ok(splits)
    }

    fn strategy(&self) -> ValidationStrategy {
        ValidationStrategy::KFold { folds: self.folds }
    }
}

/// Time series cross-validator (expanding window).
#[derive(Debug, Clone)]
pub struct TimeSeriesCVValidator {
    n_splits: usize,
    test_size: usize,
}

impl TimeSeriesCVValidator {
    pub fn new(n_splits: usize, test_size: usize) -> Self {
        Self { n_splits, test_size }
    }
}

impl Validator for TimeSeriesCVValidator {
    fn splits(&self, data_len: usize) -> Result<Vec<ValidationSplit>> {
        let min_train = data_len - self.test_size * self.n_splits;

        if min_train < self.test_size {
            return Err(TuningError::InsufficientData {
                required: self.test_size * (self.n_splits + 1),
                got: data_len,
            });
        }

        let mut splits = Vec::new();

        for i in 0..self.n_splits {
            let train_end = min_train + i * self.test_size;
            let test_start = train_end;
            let test_end = test_start + self.test_size;

            splits.push(ValidationSplit {
                train_start: 0,
                train_end,
                test_start,
                test_end,
            });
        }

        Ok(splits)
    }

    fn strategy(&self) -> ValidationStrategy {
        ValidationStrategy::TimeSeriesCV {
            n_splits: self.n_splits,
            test_size: self.test_size,
        }
    }
}

/// No validation (in-sample only).
#[derive(Debug, Clone, Default)]
pub struct NoValidator;

impl Validator for NoValidator {
    fn splits(&self, data_len: usize) -> Result<Vec<ValidationSplit>> {
        Ok(vec![ValidationSplit {
            train_start: 0,
            train_end: data_len,
            test_start: 0,
            test_end: data_len,
        }])
    }

    fn strategy(&self) -> ValidationStrategy {
        ValidationStrategy::None
    }
}

/// Create validator from strategy.
pub fn create_validator(strategy: &ValidationStrategy) -> Box<dyn Validator> {
    match strategy {
        ValidationStrategy::None => Box::new(NoValidator),
        ValidationStrategy::TrainTest { train_ratio } => {
            Box::new(TrainTestValidator::new(*train_ratio))
        }
        ValidationStrategy::WalkForward { windows, train_ratio } => {
            Box::new(WalkForwardValidator::new(*windows, *train_ratio))
        }
        ValidationStrategy::KFold { folds } => {
            Box::new(KFoldValidator::new(*folds))
        }
        ValidationStrategy::TimeSeriesCV { n_splits, test_size } => {
            Box::new(TimeSeriesCVValidator::new(*n_splits, *test_size))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_test_validator() {
        let validator = TrainTestValidator::new(0.7);
        let splits = validator.splits(100).unwrap();

        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].train_start, 0);
        assert_eq!(splits[0].train_end, 70);
        assert_eq!(splits[0].test_start, 70);
        assert_eq!(splits[0].test_end, 100);
    }

    #[test]
    fn test_walk_forward_validator() {
        let validator = WalkForwardValidator::new(5, 0.8);
        let splits = validator.splits(100).unwrap();

        assert!(!splits.is_empty());
        // Each split should have train before test
        for split in &splits {
            assert!(split.train_end <= split.test_start);
        }
    }
}
