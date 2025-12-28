# tui Overview

> **Scope**: High-level overview only. Implementation details belong in [Developer Guide](../../../doc/4-development/developer-guide.md).

## Audience

Developers and users who need interactive terminal-based time series analysis.

## WHAT

Terminal User Interface for rustful-ts:
- **Dashboard** - Real-time data visualization in terminal
- **Charts** - ASCII/Unicode time series charts
- **Interactive Controls** - Keyboard-driven navigation

## WHY

| Problem | Solution |
|---------|----------|
| Need quick data visualization without GUI | Terminal-based charts and dashboards |
| SSH/headless environments need visualization | Works in any terminal |
| Real-time monitoring needs live updates | Streaming data display |

## HOW

```rust
use tui::Dashboard;

let mut dashboard = Dashboard::new()?;
dashboard.add_chart("Stock Price", &prices);
dashboard.run()?;
```

## Documentation

| Document | Description |
|----------|-------------|
| [Developer Guide](../../../doc/4-development/developer-guide.md) | Build, test, API reference |

---

**Status**: Alpha
