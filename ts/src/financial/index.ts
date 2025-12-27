/**
 * Financial analytics module
 */

export { Portfolio, Position } from './portfolio';
export { BacktestResult, backtest, SimpleStrategy } from './backtesting';
export {
  sharpeRatio,
  sharpeRatioSync,
  sortinoRatio,
  sortinoRatioSync,
  maxDrawdown,
  maxDrawdownSync,
  drawdownSeries,
  varHistorical,
  varHistoricalSync,
  cvar,
  dailyReturns,
  cumulativeReturns,
  annualizedReturn,
  annualizedVolatility,
} from './risk';
export { Signal, SignalGenerator, SMACrossover, RSIStrategy } from './signals';
