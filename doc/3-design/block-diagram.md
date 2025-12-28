# Block Diagram

**Audience**: Architects, Technical Leadership, Developers

Visual block diagram of the rustful-ts framework architecture.

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                 rustful-ts                                     ║
║                    High-Performance Time Series Framework                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              INTERFACE LAYER                                     │
│                                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐   │
│  │                      │  │                      │  │                      │   │
│  │   TypeScript/WASM    │  │      REST API        │  │        CLI           │   │
│  │                      │  │                      │  │                      │   │
│  │  ┌────────────────┐  │  │  ┌────────────────┐  │  │  ┌────────────────┐  │   │
│  │  │ ts/src/core    │  │  │  │ POST /forecast │  │  │  │ rustful        │  │   │
│  │  │ ts/src/pipeline│  │  │  │ POST /detect   │  │  │  │   forecast     │  │   │
│  │  │ ts/src/financial│ │  │  │ POST /backtest │  │  │  │   detect       │  │   │
│  │  │ ts/src/automl  │  │  │  │ WS   /stream   │  │  │  │   backtest     │  │   │
│  │  │ ts/src/anomaly │  │  │  │ GET  /health   │  │  │  │   serve        │  │   │
│  │  └────────────────┘  │  │  └────────────────┘  │  │  └────────────────┘  │   │
│  │                      │  │                      │  │                      │   │
│  │    rustful-wasm      │  │   rustful-server     │  │    rustful-cli       │   │
│  │    (wasm-bindgen)    │  │   (axum + tokio)     │  │    (clap)            │   │
│  └──────────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘   │
│             │                         │                         │               │
└─────────────┼─────────────────────────┼─────────────────────────┼───────────────┘
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               DOMAIN LAYER                                       │
│                                                                                  │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐            │
│  │  rustful-forecast │  │ rustful-financial │  │  rustful-anomaly  │            │
│  ├───────────────────┤  ├───────────────────┤  ├───────────────────┤            │
│  │ ● Pipeline        │  │ ● Portfolio       │  │ ● ZScoreDetector  │            │
│  │ ● PipelineStep    │  │ ● Position        │  │ ● IQRDetector     │            │
│  │ ● NormalizeStep   │  │ ● Backtester      │  │ ● Monitor         │            │
│  │ ● DifferenceStep  │  │ ● SignalGenerator │  │ ● Alert           │            │
│  │ ● Decomposition   │  │ ● Risk Metrics    │  │ ● AlertSeverity   │            │
│  │ ● Seasonality     │  │   - VaR           │  │                   │            │
│  │                   │  │   - Sharpe        │  │                   │            │
│  │                   │  │   - MaxDrawdown   │  │                   │            │
│  └─────────┬─────────┘  └─────────┬─────────┘  └─────────┬─────────┘            │
│            │                      │                      │                       │
│            └──────────────────────┼──────────────────────┘                       │
│                                   │                                              │
│                                   ▼                                              │
│                      ┌───────────────────────┐                                   │
│                      │    rustful-automl     │                                   │
│                      ├───────────────────────┤                                   │
│                      │ ● AutoSelector        │                                   │
│                      │ ● GridSearch          │                                   │
│                      │ ● RandomSearch        │                                   │
│                      │ ● EnsembleForecaster  │                                   │
│                      │ ● CrossValidation     │                                   │
│                      └───────────┬───────────┘                                   │
│                                  │                                               │
└──────────────────────────────────┼───────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            FOUNDATION LAYER                                      │
│                                                                                  │
│                         ┌───────────────────────┐                                │
│                         │     rustful-core      │                                │
│                         ├───────────────────────┤                                │
│                         │                       │                                │
│                         │  ┌─────────────────┐  │                                │
│                         │  │   algorithms/   │  │                                │
│                         │  ├─────────────────┤  │                                │
│                         │  │ ● Arima         │  │                                │
│                         │  │ ● SES           │  │                                │
│                         │  │ ● Holt          │  │                                │
│                         │  │ ● HoltWinters   │  │                                │
│                         │  │ ● SMA           │  │                                │
│                         │  │ ● LinearRegress │  │                                │
│                         │  │ ● TimeSeriesKNN │  │                                │
│                         │  └─────────────────┘  │                                │
│                         │                       │                                │
│                         │  ┌─────────────────┐  │                                │
│                         │  │     utils/      │  │                                │
│                         │  ├─────────────────┤  │                                │
│                         │  │ ● metrics       │  │                                │
│                         │  │ ● preprocessing │  │                                │
│                         │  │ ● validation    │  │                                │
│                         │  └─────────────────┘  │                                │
│                         │                       │                                │
│                         │  ┌─────────────────┐  │                                │
│                         │  │     data/       │  │                                │
│                         │  ├─────────────────┤  │                                │
│                         │  │ ● yahoo         │  │                                │
│                         │  │   (fetch)       │  │                                │
│                         │  └─────────────────┘  │                                │
│                         │                       │                                │
│                         └───────────────────────┘                                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CORE TRAITS                                         │
│                                                                                  │
│    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│    │    Predictor    │    │  PipelineStep   │    │ AnomalyDetector │            │
│    ├─────────────────┤    ├─────────────────┤    ├─────────────────┤            │
│    │ fit()           │    │ transform()     │    │ fit()           │            │
│    │ predict()       │    │ inverse_        │    │ detect()        │            │
│    │ is_fitted()     │    │   transform()   │    │ score()         │            │
│    └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Layer Summary

| Layer | Crates | Purpose |
|-------|--------|---------|
| Interface | rustful-wasm, rustful-server, rustful-cli | External access |
| Domain | rustful-forecast, rustful-financial, rustful-anomaly, rustful-automl | Business logic |
| Foundation | rustful-core | Core algorithms |

## Summary

The rustful-ts framework uses a three-layer architecture with clear separation of concerns. Each layer has specific responsibilities and dependencies flow downward.

**Key Takeaways**:
1. Interface layer provides TypeScript/WASM, REST API, and CLI access
2. Domain layer contains business logic for forecasting, finance, and anomaly detection
3. Foundation layer (rustful-core) provides all prediction algorithms

---

**Related Documentation**:
- [Architecture](architecture.md) - Detailed architecture documentation
- [Workflow Diagrams](workflow-diagrams.md) - Data flow and process workflows
- [ADRs](adr/README.md) - Architecture decisions

**Last Updated**: 2025-12-28
