# Framework Backlog

Cross-cutting work items for rustful-ts framework.

## In Progress

_None_

## Planned

### High Priority

- [ ] Implement streaming predictions
- [ ] Add more anomaly detection methods (Isolation Forest)
- [ ] REST API authentication

### Medium Priority

- [ ] Web Workers support for browser
- [ ] SIMD optimization for WASM
- [ ] Batch prediction API
- [ ] Model serialization/deserialization

### Low Priority

- [ ] Python bindings (PyO3)
- [ ] R bindings
- [ ] GPU acceleration
- [ ] Distributed computing support

## Completed

- [x] Workspace structure migration
- [x] Core algorithms (ARIMA, SES, Holt, Holt-Winters, SMA, KNN)
- [x] TypeScript package structure
- [x] Pipeline builder API
- [x] Financial analytics module
- [x] Anomaly detection module
- [x] AutoML module
- [x] REST API structure
- [x] CLI structure
- [x] Documentation framework (SEA)
- [x] WASM bindings for all algorithms
- [x] Confidence intervals for forecasts
- [x] examples/basic.rs for all crates
- [x] tests/integration.rs for all crates

## Documentation Debt

- [ ] Complete API reference documentation
- [ ] Add algorithm theory documentation

## Module-Specific Backlogs

See individual crate backlogs:
- [rustful-core](../crates/rustful-core/backlog.md)
- [rustful-financial](../crates/rustful-financial/backlog.md)
- [rustful-anomaly](../crates/rustful-anomaly/backlog.md)
