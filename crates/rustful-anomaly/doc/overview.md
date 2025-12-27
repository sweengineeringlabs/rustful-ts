# rustful-anomaly Overview

## WHAT: Anomaly Detection Module

rustful-anomaly provides outlier detection, real-time monitoring, and alerting for time series data.

Key capabilities:
- **Detectors** - Z-Score, IQR, Median Absolute Deviation
- **Monitoring** - Real-time anomaly tracking
- **Alerting** - Configurable alert system with severity levels

## WHY: Outlier Detection

**Problems Solved**:
1. Anomalies corrupt prediction model training
2. Real-time monitoring needs streaming detection
3. Alert systems require configurable thresholds

**When to Use**: DevOps monitoring, IoT sensor data, fraud detection

**When NOT to Use**: Well-behaved data without outliers

## HOW: Usage Guide

### Anomaly Detection

```rust
use rustful_anomaly::detectors::{ZScoreDetector, AnomalyDetector};

let mut detector = ZScoreDetector::new(3.0);
detector.fit(&historical_data)?;

let result = detector.detect(&new_data)?;
for (i, is_anomaly) in result.anomalies.iter().enumerate() {
    if *is_anomaly {
        println!("Anomaly at index {}", i);
    }
}
```

### Detector Types

| Detector | Use Case |
|----------|----------|
| `ZScoreDetector` | Normally distributed data |
| `IQRDetector` | Robust to existing outliers |

### Real-Time Monitoring

```rust
use rustful_anomaly::monitor::Monitor;
use rustful_anomaly::alert::AlertHandler;

let detector = ZScoreDetector::new(3.0);
let mut monitor = Monitor::new(detector, 100);

// Process streaming data
for value in stream {
    if let Some(alert) = monitor.update(value)? {
        println!("Alert: {:?}", alert);
    }
}
```

### Alert System

```rust
use rustful_anomaly::alert::{Alert, AlertSeverity};

// Alerts include:
// - timestamp
// - value
// - score (deviation measure)
// - severity (Low, Medium, High, Critical)
```

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| rustful-core | Independent (no dependency) |
| rustful-server | Exposes via REST + WebSocket |
| rustful-cli | Available via CLI commands |

**Integration Points**:
- Standalone module for anomaly detection
- REST API provides streaming WebSocket endpoint

## Examples and Tests

### Examples

**Location**: [`examples/`](../examples/)

- `basic.rs` - Simple detection example

### Tests

**Location**: [`tests/`](../tests/)

- `integration.rs` - Public API tests

### Testing

```bash
cargo test -p rustful-anomaly
```

---

**Status**: Beta
**Roadmap**: See [backlog.md](../backlog.md) | [Framework Backlog](../../../doc/framework-backlog.md)
