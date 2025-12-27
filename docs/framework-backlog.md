# Framework Backlog

Cross-cutting work items for rustful-ts framework.

## In Progress

| Item | Status | Priority |
|------|--------|----------|
| Complete WASM bindings for all algorithms | In Progress | High |

## Planned

### High Priority

- [ ] Add confidence intervals to forecasts
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

## Documentation Debt

- [ ] Add examples/basic.rs to each crate
- [ ] Add tests/integration.rs to each crate
- [ ] Complete API reference documentation
- [ ] Add algorithm theory documentation

## Module-Specific Backlogs

See individual crate backlogs:
- [rustful-core](../crates/rustful-core/backlog.md)
- [rustful-financial](../crates/rustful-financial/backlog.md)
- [rustful-anomaly](../crates/rustful-anomaly/backlog.md)
