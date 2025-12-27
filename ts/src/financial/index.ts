/**
 * Financial analytics module
 */

export { Portfolio, Position } from './portfolio';
export { BacktestResult, backtest, SimpleStrategy } from './backtesting';
export {
  sharpeRatio,
  sortinoRatio,
  maxDrawdown,
  drawdownSeries,
  varHistorical,
  cvar,
  dailyReturns,
  cumulativeReturns,
  annualizedReturn,
  annualizedVolatility,
} from './risk';
export { Signal, SignalGenerator, SMACrossover, RSIStrategy } from './signals';
