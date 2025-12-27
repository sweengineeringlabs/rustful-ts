/**
 * Risk metrics for financial analysis
 */

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
 * Calculate Sharpe ratio
 * @param returns - Array of returns
 * @param riskFreeRate - Risk-free rate (annualized, default: 0)
 * @param tradingDays - Trading days per year (default: 252)
 */
export function sharpeRatio(
  returns: number[],
  riskFreeRate: number = 0,
  tradingDays: number = 252
): number {
  if (returns.length < 2) return 0;

  const annReturn = annualizedReturn(returns, tradingDays);
  const annVol = annualizedVolatility(returns, tradingDays);

  if (annVol === 0) return 0;
  return (annReturn - riskFreeRate) / annVol;
}

/**
 * Calculate Sortino ratio (uses downside deviation)
 * @param returns - Array of returns
 * @param riskFreeRate - Risk-free rate (annualized, default: 0)
 * @param tradingDays - Trading days per year (default: 252)
 */
export function sortinoRatio(
  returns: number[],
  riskFreeRate: number = 0,
  tradingDays: number = 252
): number {
  if (returns.length < 2) return 0;

  const annReturn = annualizedReturn(returns, tradingDays);
  const downsideReturns = returns.filter((r) => r < 0);

  if (downsideReturns.length === 0) return Infinity;

  const downsideVariance =
    downsideReturns.reduce((sum, r) => sum + r ** 2, 0) / downsideReturns.length;
  const downsideDeviation = Math.sqrt(downsideVariance * tradingDays);

  if (downsideDeviation === 0) return Infinity;
  return (annReturn - riskFreeRate) / downsideDeviation;
}

/**
 * Calculate maximum drawdown
 * @param prices - Price series or equity curve
 */
export function maxDrawdown(prices: number[]): number {
  if (prices.length === 0) return 0;

  let maxDD = 0;
  let peak = prices[0];

  for (const price of prices) {
    if (price > peak) {
      peak = price;
    }
    const dd = (peak - price) / peak;
    if (dd > maxDD) {
      maxDD = dd;
    }
  }

  return maxDD;
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
 * Calculate Value at Risk (historical method)
 * @param returns - Array of returns
 * @param confidence - Confidence level (e.g., 0.95 for 95%)
 */
export function varHistorical(returns: number[], confidence: number = 0.95): number {
  if (returns.length === 0) return 0;

  const sorted = [...returns].sort((a, b) => a - b);
  const index = Math.floor((1 - confidence) * sorted.length);
  return -sorted[index]; // Return as positive number
}

/**
 * Calculate Conditional VaR (Expected Shortfall)
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
