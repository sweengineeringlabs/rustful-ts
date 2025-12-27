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

---

**Related Documentation**:
- [Architecture](architecture.md) - Detailed architecture documentation
- [ADRs](adr/README.md) - Architecture decisions

**Last Updated**: 2024-12-27
