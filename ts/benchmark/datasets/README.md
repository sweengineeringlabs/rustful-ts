# Benchmark Datasets

Real-world datasets for performance benchmarking.

## Datasets

| File | Source | Description | Points | Characteristics |
|------|--------|-------------|--------|-----------------|
| `stock-spy-daily.csv` | Yahoo Finance | S&P 500 ETF (SPY) daily closing prices | 1,259 | Trends, volatility clustering, fat tails |
| `weather-melbourne-daily-temp.csv` | Brownlee/BoM | Melbourne daily minimum temperatures (1981-1990) | 3,650 | Strong seasonality, trends |
| `sensor-sunspots.csv` | Brownlee/SIDC | Monthly sunspot observations (1749-1983) | 2,820 | Cyclical patterns (~11 year cycle) |
| `sensor-airline-passengers.csv` | Brownlee | Monthly airline passengers (1949-1960) | 144 | Trend + seasonality |

## Sources

1. **Stock Data**: Yahoo Finance API - `https://query1.finance.yahoo.com/v8/finance/chart/SPY`
2. **Weather Data**: Jason Brownlee's ML Datasets - `https://github.com/jbrownlee/Datasets`
3. **Sunspot Data**: Solar Influences Data Center (SIDC) via Brownlee
4. **Airline Data**: Box & Jenkins (1976) via Brownlee

## Data Characteristics Comparison

| Aspect | Stock | Weather | Sunspots | Airline |
|--------|-------|---------|----------|---------|
| Seasonality | Weak | Strong (annual) | Strong (~11yr) | Strong (annual) |
| Trend | Bull market | None | None | Upward |
| Volatility | Clustering | Low | Medium | Low |
| Outliers | Yes (crashes) | Few | Few | None |
| Stationarity | Non-stationary | Seasonal | Cyclical | Non-stationary |

## License

These datasets are publicly available for research and educational purposes.
