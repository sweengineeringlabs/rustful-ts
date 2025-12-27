# anomaly Overview

## Audience

DevOps engineers and data scientists who need real-time outlier detection and monitoring.

## WHAT

Anomaly detection for time series:
- **Detectors** - Z-Score, IQR, Median Absolute Deviation
- **Monitoring** - Real-time anomaly tracking
- **Alerting** - Configurable thresholds and severity levels

## WHY

| Problem | Solution |
|---------|----------|
| Anomalies corrupt prediction model training | Detect and filter outliers before training |
| Real-time monitoring needs streaming detection | Incremental detectors for streaming data |
| Alert systems require configurable thresholds | Flexible threshold and severity configuration |

## HOW

```rust
use anomaly::{ZScoreDetector, AnomalyDetector};

let mut detector = ZScoreDetector::new(3.0);
detector.fit(&historical_data)?;

let result = detector.detect(&new_data)?;
for (i, &is_anomaly) in result.is_anomaly.iter().enumerate() {
    if is_anomaly {
        println!("Anomaly at index {}: score {:.2}", i, result.scores[i]);
    }
}
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../doc/4-development/developer-guide.md) | Build, test, contribute |
| [Backlog](../backlog.md) | Planned features |

---

**Status**: Beta
