//! FFI Bindings for rustful-ts
//!
//! C-compatible bindings for the time series library.
//! Enables integration with Python, C/C++, Java, and other languages.

use optimizer_core::{
    IndicatorOptimizer, IndicatorConfig, ParamRange,
    Objective, OptimizationMethod, ValidationStrategy,
};

// ============================================================================
// Indicator Optimizer FFI
// ============================================================================

/// Opaque handle for IndicatorOptimizer
#[repr(C)]
pub struct IndicatorOptimizerHandle {
    ptr: *mut IndicatorOptimizer,
}

/// FFI-safe optimization result
#[repr(C)]
pub struct IndicatorOptResultFFI {
    pub best_score: f64,
    pub oos_score: f64,
    pub has_oos: bool,
    pub robustness: f64,
    pub has_robustness: bool,
    pub evaluations: i64,
}

/// Create a new optimizer
#[no_mangle]
pub extern "C" fn ts_indicator_optimizer_new() -> IndicatorOptimizerHandle {
    let optimizer = Box::new(IndicatorOptimizer::new());
    IndicatorOptimizerHandle {
        ptr: Box::into_raw(optimizer),
    }
}

/// Free an optimizer
#[no_mangle]
pub extern "C" fn ts_indicator_optimizer_free(handle: IndicatorOptimizerHandle) {
    if !handle.ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(handle.ptr);
        }
    }
}

/// Add SMA range
#[no_mangle]
pub extern "C" fn ts_indicator_optimizer_add_sma(
    handle: IndicatorOptimizerHandle,
    min: i64,
    max: i64,
    step: i64,
) -> IndicatorOptimizerHandle {
    if handle.ptr.is_null() {
        return handle;
    }
    unsafe {
        let opt = &mut *handle.ptr;
        opt.indicators.push(IndicatorConfig::SMA {
            period: ParamRange::new(min as usize, max as usize, step as usize),
        });
    }
    handle
}

/// Add RSI range
#[no_mangle]
pub extern "C" fn ts_indicator_optimizer_add_rsi(
    handle: IndicatorOptimizerHandle,
    min: i64,
    max: i64,
    step: i64,
) -> IndicatorOptimizerHandle {
    if handle.ptr.is_null() {
        return handle;
    }
    unsafe {
        let opt = &mut *handle.ptr;
        opt.indicators.push(IndicatorConfig::RSI {
            period: ParamRange::new(min as usize, max as usize, step as usize),
        });
    }
    handle
}

/// Set objective (0=DirectionalAccuracy, 1=SharpeRatio, 2=TotalReturn, 3=IC, 4=MaxDD, 5=Sortino)
#[no_mangle]
pub extern "C" fn ts_indicator_optimizer_set_objective(
    handle: IndicatorOptimizerHandle,
    objective: i32,
) -> IndicatorOptimizerHandle {
    if handle.ptr.is_null() {
        return handle;
    }
    unsafe {
        let opt = &mut *handle.ptr;
        opt.objective = match objective {
            0 => Objective::DirectionalAccuracy,
            1 => Objective::SharpeRatio,
            2 => Objective::TotalReturn,
            3 => Objective::InformationCoefficient,
            4 => Objective::MaxDrawdown,
            5 => Objective::SortinoRatio,
            _ => Objective::SharpeRatio,
        };
    }
    handle
}

/// Set method (0=Grid, 1=ParallelGrid, 2=Random, 3=Genetic, 4=Bayesian)
#[no_mangle]
pub extern "C" fn ts_indicator_optimizer_set_method(
    handle: IndicatorOptimizerHandle,
    method: i32,
    param1: i64,
    param2: i64,
) -> IndicatorOptimizerHandle {
    if handle.ptr.is_null() {
        return handle;
    }
    unsafe {
        let opt = &mut *handle.ptr;
        opt.method = match method {
            0 => OptimizationMethod::GridSearch,
            1 => OptimizationMethod::ParallelGrid,
            2 => OptimizationMethod::RandomSearch {
                iterations: param1 as usize,
            },
            3 => OptimizationMethod::GeneticAlgorithm {
                population: param1 as usize,
                generations: param2 as usize,
                mutation_rate: 0.1,
                crossover_rate: 0.8,
            },
            4 => OptimizationMethod::Bayesian {
                iterations: param1 as usize,
            },
            _ => OptimizationMethod::GridSearch,
        };
    }
    handle
}

/// Set validation (0=None, 1=TrainTest, 2=WalkForward, 3=KFold)
#[no_mangle]
pub extern "C" fn ts_indicator_optimizer_set_validation(
    handle: IndicatorOptimizerHandle,
    validation: i32,
    param1: f64,
    param2: i64,
) -> IndicatorOptimizerHandle {
    if handle.ptr.is_null() {
        return handle;
    }
    unsafe {
        let opt = &mut *handle.ptr;
        opt.validation = match validation {
            0 => ValidationStrategy::None,
            1 => ValidationStrategy::TrainTest {
                train_ratio: param1,
            },
            2 => ValidationStrategy::WalkForward {
                windows: param2 as usize,
                train_ratio: param1,
            },
            3 => ValidationStrategy::KFold {
                folds: param2 as usize,
            },
            _ => ValidationStrategy::TrainTest { train_ratio: 0.7 },
        };
    }
    handle
}

/// Run optimization
#[no_mangle]
pub extern "C" fn ts_indicator_optimizer_optimize(
    handle: IndicatorOptimizerHandle,
    prices: *const f64,
    len: i64,
) -> IndicatorOptResultFFI {
    if handle.ptr.is_null() || prices.is_null() || len <= 0 {
        return IndicatorOptResultFFI {
            best_score: f64::NEG_INFINITY,
            oos_score: 0.0,
            has_oos: false,
            robustness: 0.0,
            has_robustness: false,
            evaluations: 0,
        };
    }

    let prices_slice = unsafe { std::slice::from_raw_parts(prices, len as usize) };
    let opt = unsafe { &*handle.ptr };
    let result = opt.optimize(prices_slice);

    IndicatorOptResultFFI {
        best_score: result.best_score,
        oos_score: result.oos_score.unwrap_or(0.0),
        has_oos: result.oos_score.is_some(),
        robustness: result.robustness.unwrap_or(0.0),
        has_robustness: result.robustness.is_some(),
        evaluations: result.evaluations as i64,
    }
}
