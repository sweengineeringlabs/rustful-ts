# Changelog

All notable changes to rustful-ts will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- WASM bindings completion in progress

### Changed
- Documentation restructured to SEA framework

## [0.2.0] - 2024-12-27

### Added
- Multi-crate workspace structure:
  - `rustful-core` - Core algorithms (ARIMA, SES, Holt, Holt-Winters, SMA, KNN)
  - `rustful-forecast` - Pipeline infrastructure and decomposition
  - `rustful-financial` - Portfolio management, backtesting, risk metrics
  - `rustful-automl` - Model selection, grid search, ensembles
  - `rustful-anomaly` - Z-score, IQR detectors, monitoring, alerting
  - `rustful-wasm` - Unified WASM bindings
  - `rustful-server` - REST API (Axum)
  - `rustful-cli` - Command-line tool (Clap)
- TypeScript package structure:
  - `ts/src/core/` - Algorithm wrappers
  - `ts/src/pipeline/` - Builder API
  - `ts/src/financial/` - Financial module
  - `ts/src/automl/` - AutoML module
  - `ts/src/anomaly/` - Anomaly detection
- Documentation framework (SEA):
  - Architecture diagrams (block, workflow)
  - Developer guide
  - Crate overview documentation
- Pipeline builder API for composing forecasts
- `Predictor` trait for all algorithms
- `PipelineStep` trait for transformations
- `AnomalyDetector` trait for detection algorithms

### Changed
- Migrated from single crate to Cargo workspace
- Moved src/ to crates/rustful-core/src/
- Updated WASM bindings location to rustful-wasm

## [0.1.0] - 2024-12-20

### Added
- Initial release
- Core forecasting algorithms:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - SES (Simple Exponential Smoothing)
  - Holt (Double Exponential Smoothing)
  - Holt-Winters (Triple Exponential Smoothing)
  - SMA (Simple Moving Average)
  - Linear Regression
  - TimeSeriesKNN (K-Nearest Neighbors)
- WASM bindings via wasm-bindgen
- TypeScript type definitions
- Basic metrics (MAE, MSE, RMSE, MAPE)
- Yahoo Finance data fetching

[Unreleased]: https://github.com/sweengineeringlabs/rustful-ts/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/sweengineeringlabs/rustful-ts/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/sweengineeringlabs/rustful-ts/releases/tag/v0.1.0
