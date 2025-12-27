# server Overview

## Audience

Backend developers who need HTTP API access to time series functionality from any language.

## WHAT

REST API server for rustful-ts:
- **REST Endpoints** - JSON-based forecasting, anomaly detection, backtesting
- **WebSocket** - Real-time anomaly streaming
- **Health Checks** - Monitoring and load balancer support

## WHY

| Problem | Solution |
|---------|----------|
| Non-JavaScript/Rust applications need access | Language-agnostic HTTP API |
| Microservice architectures require HTTP APIs | Standard REST endpoints |
| Real-time monitoring needs streaming | WebSocket endpoint for anomaly alerts |

## HOW

```bash
# Start server
cargo run -p server -- --port 8080

# Forecast request
curl -X POST http://localhost:8080/api/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"data": [1,2,3,4,5], "model": "arima", "steps": 3}'
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../doc/4-development/developer-guide.md) | Build, test, contribute |
| [Backlog](../backlog.md) | Planned features |

---

**Status**: Beta
