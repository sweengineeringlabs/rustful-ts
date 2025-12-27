# rustful-server Overview

## WHAT: REST API Server

rustful-server provides an HTTP API for accessing rustful-ts functionality from any language or platform.

Key capabilities:
- **REST Endpoints** - JSON-based forecasting, anomaly detection, backtesting
- **WebSocket** - Real-time anomaly streaming
- **Health Checks** - Monitoring and load balancer support

## WHY: Language-Agnostic Access

**Problems Solved**:
1. Non-JavaScript/Rust applications need access
2. Microservice architectures require HTTP APIs
3. Real-time monitoring needs streaming

**When to Use**: Microservices, polyglot environments, real-time monitoring

**When NOT to Use**: Direct TypeScript/Rust integration is simpler

## HOW: Usage Guide

### Starting the Server

```bash
cargo run -p rustful-server -- --port 8080
```

Or via CLI:
```bash
rustful serve --port 8080
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/forecast` | Generate forecasts |
| POST | `/api/v1/automl/select` | Auto model selection |
| POST | `/api/v1/financial/backtest` | Run backtest |
| POST | `/api/v1/anomaly/detect` | Detect anomalies |
| WS | `/api/v1/anomaly/stream` | Real-time anomaly stream |

### Forecast Request

```bash
curl -X POST http://localhost:8080/api/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "model": "arima",
    "params": {"p": 1, "d": 1, "q": 1},
    "steps": 5
  }'
```

### Response

```json
{
  "forecast": [11.2, 12.1, 13.0, 13.9, 14.8],
  "model": "arima",
  "params": {"p": 1, "d": 1, "q": 1}
}
```

### Anomaly Detection

```bash
curl -X POST http://localhost:8080/api/v1/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{
    "data": [1, 2, 3, 100, 4, 5],
    "method": "zscore",
    "threshold": 3.0
  }'
```

### WebSocket Streaming

```javascript
const ws = new WebSocket('ws://localhost:8080/api/v1/anomaly/stream');
ws.onmessage = (event) => {
  const alert = JSON.parse(event.data);
  console.log('Anomaly:', alert);
};
```

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| rustful-core | Uses for predictions |
| rustful-anomaly | Uses for detection |
| rustful-financial | Uses for backtesting |

**Integration Points**:
- HTTP interface to all functionality
- WebSocket for streaming anomalies

## Examples and Tests

### Running

```bash
# Development
cargo run -p rustful-server

# Production
cargo build --release -p rustful-server
./target/release/rustful-server --port 8080
```

### Testing

```bash
cargo test -p rustful-server
```

---

**Status**: Beta
**Roadmap**: See [framework-backlog.md](../../../docs/framework-backlog.md)
