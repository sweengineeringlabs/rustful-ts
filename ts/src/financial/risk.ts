/**
 * Risk metrics for financial analysis - WASM-backed implementations
 */

import { getWasmModule, ensureWasm, isWasmReady } from '../wasm-loader';

/**
 * Calculate daily returns from prices
 */
export function dailyReturns(prices: number[]): number[] {
  if (prices.length < 2) return [];
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }
  return returns;
}

/**
 * Calculate cumulative returns from prices
 */
export function cumulativeReturns(prices: number[]): number[] {
  if (prices.length === 0) return [];
  const initial = prices[0];
  return prices.map((p) => (p - initial) / initial);
}

/**
 * Calculate annualized return
 * @param returns - Daily returns
 * @param tradingDays - Trading days per year (default: 252)
 */
export function annualizedReturn(returns: number[], tradingDays: number = 252): number {
  if (returns.length === 0) return 0;
  const meanDaily = returns.reduce((a, b) => a + b, 0) / returns.length;
  return meanDaily * tradingDays;
}

/**
 * Calculate annualized volatility
 * @param returns - Daily returns
 * @param tradingDays - Trading days per year (default: 252)
 */
export function annualizedVolatility(returns: number[], tradingDays: number = 252): number {
  if (returns.length < 2) return 0;
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + (r - mean) ** 2, 0) / returns.length;
  return Math.sqrt(variance * tradingDays);
}

/**
 * Calculate Sharpe ratio (WASM-backed)
 * @param returns - Array of returns
 * @param riskFreeRate - Risk-free rate (default: 0)
 */
export async function sharpeRatio(
  returns: number[],
  riskFreeRate: number = 0
): Promise<number> {
  await ensureWasm();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasm = getWasmModule() as any;
  return wasm.compute_sharpe_ratio(new Float64Array(returns), riskFreeRate);
}

/**
 * Calculate Sharpe ratio synchronously (requires WASM to be initialized)
 * @param returns - Array of returns
 * @param riskFreeRate - Risk-free rate (default: 0)
 */
export function sharpeRatioSync(
  returns: number[],
  riskFreeRate: number = 0
): number {
  if (!isWasmReady()) {
    throw new Error('WASM not initialized. Call initWasm() first or use async sharpeRatio()');
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasm = getWasmModule() as any;
  return wasm.compute_sharpe_ratio(new Float64Array(returns), riskFreeRate);
}

/**
 * Calculate Sortino ratio (WASM-backed)
 * @param returns - Array of returns
 * @param riskFreeRate - Risk-free rate (default: 0)
 */
export async function sortinoRatio(
  returns: number[],
  riskFreeRate: number = 0
): Promise<number> {
  await ensureWasm();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasm = getWasmModule() as any;
  return wasm.compute_sortino_ratio(new Float64Array(returns), riskFreeRate);
}

/**
 * Calculate Sortino ratio synchronously (requires WASM to be initialized)
 */
export function sortinoRatioSync(
  returns: number[],
  riskFreeRate: number = 0
): number {
  if (!isWasmReady()) {
    throw new Error('WASM not initialized. Call initWasm() first or use async sortinoRatio()');
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasm = getWasmModule() as any;
  return wasm.compute_sortino_ratio(new Float64Array(returns), riskFreeRate);
}

/**
 * Calculate maximum drawdown (WASM-backed)
 * @param prices - Price series or equity curve
 */
export async function maxDrawdown(prices: number[]): Promise<number> {
  await ensureWasm();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasm = getWasmModule() as any;
  return wasm.compute_max_drawdown(new Float64Array(prices));
}

/**
 * Calculate maximum drawdown synchronously (requires WASM to be initialized)
 */
export function maxDrawdownSync(prices: number[]): number {
  if (!isWasmReady()) {
    throw new Error('WASM not initialized. Call initWasm() first or use async maxDrawdown()');
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasm = getWasmModule() as any;
  return wasm.compute_max_drawdown(new Float64Array(prices));
}

/**
 * Calculate drawdown series
 * @param prices - Price series or equity curve
 */
export function drawdownSeries(prices: number[]): number[] {
  if (prices.length === 0) return [];

  const drawdowns: number[] = [];
  let peak = prices[0];

  for (const price of prices) {
    if (price > peak) {
      peak = price;
    }
    drawdowns.push((peak - price) / peak);
  }

  return drawdowns;
}

/**
 * Calculate Value at Risk using historical method (WASM-backed)
 * @param returns - Array of returns
 * @param confidence - Confidence level (e.g., 0.95 for 95%)
 */
export async function varHistorical(
  returns: number[],
  confidence: number = 0.95
): Promise<number> {
  await ensureWasm();
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasm = getWasmModule() as any;
  return Math.abs(wasm.compute_var(new Float64Array(returns), confidence));
}

/**
 * Calculate Value at Risk synchronously (requires WASM to be initialized)
 */
export function varHistoricalSync(
  returns: number[],
  confidence: number = 0.95
): number {
  if (!isWasmReady()) {
    throw new Error('WASM not initialized. Call initWasm() first or use async varHistorical()');
  }
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const wasm = getWasmModule() as any;
  return Math.abs(wasm.compute_var(new Float64Array(returns), confidence));
}

/**
 * Calculate Conditional VaR (Expected Shortfall)
 * Note: Pure TypeScript implementation
 * @param returns - Array of returns
 * @param confidence - Confidence level (e.g., 0.95 for 95%)
 */
export function cvar(returns: number[], confidence: number = 0.95): number {
  if (returns.length === 0) return 0;

  const sorted = [...returns].sort((a, b) => a - b);
  const cutoffIndex = Math.floor((1 - confidence) * sorted.length);
  const tail = sorted.slice(0, cutoffIndex + 1);

  if (tail.length === 0) return 0;
  const avgLoss = tail.reduce((a, b) => a + b, 0) / tail.length;
  return -avgLoss; // Return as positive number
}
