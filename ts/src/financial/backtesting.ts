/**
 * Backtesting engine
 */

import { Portfolio } from './portfolio';
import { Signal, SignalGenerator, SMACrossover } from './signals';
import { sharpeRatioSync, maxDrawdownSync, dailyReturns } from './risk';
import { isWasmReady } from '../wasm-loader';

/**
 * Result of a backtest run
 */
export interface BacktestResult {
  /** Total return percentage */
  totalReturn: number;
  /** Sharpe ratio */
  sharpeRatio: number;
  /** Maximum drawdown */
  maxDrawdown: number;
  /** Win rate (profitable trades / total trades) */
  winRate: number;
  /** Number of trades executed */
  numTrades: number;
  /** Equity curve (portfolio value over time) */
  equityCurve: number[];
  /** List of trades */
  trades: Trade[];
}

/**
 * A single trade record
 */
export interface Trade {
  type: 'BUY' | 'SELL';
  price: number;
  quantity: number;
  timestamp: number;
  pnl?: number;
}

/**
 * Simple buy-and-hold strategy for comparison
 */
export class SimpleStrategy implements SignalGenerator {
  readonly name = 'Buy and Hold';
  private bought = false;

  generate(prices: number[]): Signal {
    if (!this.bought && prices.length > 0) {
      this.bought = true;
      return Signal.Buy;
    }
    return Signal.Hold;
  }

  generateSeries(prices: number[]): Signal[] {
    return prices.map((_, i) => (i === 0 ? Signal.Buy : Signal.Hold));
  }

  reset(): void {
    this.bought = false;
  }
}

/**
 * Run a backtest on historical price data
 */
export function backtest(
  prices: number[],
  strategy: SignalGenerator,
  options: {
    initialCapital?: number;
    positionSize?: number; // Fraction of capital per trade (0-1)
    commission?: number; // Commission per trade
  } = {}
): BacktestResult {
  const { initialCapital = 10000, positionSize = 1.0, commission = 0 } = options;

  const portfolio = new Portfolio(initialCapital);
  const equityCurve: number[] = [];
  const trades: Trade[] = [];
  let position = 0;
  let entryPrice = 0;

  for (let i = 0; i < prices.length; i++) {
    const price = prices[i];
    const signal = strategy.generate(prices.slice(0, i + 1));

    // Execute trades based on signals
    if (signal === Signal.Buy && position === 0) {
      const capital = portfolio.cash * positionSize;
      const quantity = Math.floor((capital - commission) / price);

      if (quantity > 0) {
        portfolio.buy('ASSET', quantity, price);
        position = quantity;
        entryPrice = price;
        trades.push({
          type: 'BUY',
          price,
          quantity,
          timestamp: i,
        });
      }
    } else if (signal === Signal.Sell && position > 0) {
      const pnl = (price - entryPrice) * position - commission;
      portfolio.sell('ASSET', position, price);
      trades.push({
        type: 'SELL',
        price,
        quantity: position,
        timestamp: i,
        pnl,
      });
      position = 0;
    }

    // Record equity
    const currentValue = portfolio.value({ ASSET: price });
    equityCurve.push(currentValue);
  }

  // Close any remaining position at the end
  if (position > 0) {
    const finalPrice = prices[prices.length - 1];
    const pnl = (finalPrice - entryPrice) * position;
    portfolio.sell('ASSET', position, finalPrice);
    trades.push({
      type: 'SELL',
      price: finalPrice,
      quantity: position,
      timestamp: prices.length - 1,
      pnl,
    });
  }

  // Calculate metrics
  const returns = dailyReturns(equityCurve);
  const sellTrades = trades.filter((t) => t.type === 'SELL');
  const winningTrades = sellTrades.filter((t) => (t.pnl ?? 0) > 0);
  const winRate = sellTrades.length > 0 ? winningTrades.length / sellTrades.length : 0;

  // Calculate performance metrics
  // Note: sharpeRatio and maxDrawdown use WASM - ensure initWasm() was called
  let sharpe = 0;
  let mdd = 0;

  if (isWasmReady()) {
    sharpe = sharpeRatioSync(returns);
    mdd = maxDrawdownSync(equityCurve);
  } else {
    // Fallback to simple calculations if WASM not ready
    const meanReturn = returns.length > 0 ? returns.reduce((a, b) => a + b, 0) / returns.length : 0;
    const variance = returns.length > 0 ? returns.reduce((sum, r) => sum + (r - meanReturn) ** 2, 0) / returns.length : 0;
    const stdDev = Math.sqrt(variance);
    sharpe = stdDev !== 0 ? meanReturn / stdDev : 0;

    // Calculate max drawdown
    let peak = equityCurve[0] || 0;
    for (const val of equityCurve) {
      if (val > peak) peak = val;
      const dd = (peak - val) / peak;
      if (dd > mdd) mdd = dd;
    }
  }

  return {
    totalReturn: (equityCurve[equityCurve.length - 1] - initialCapital) / initialCapital,
    sharpeRatio: sharpe,
    maxDrawdown: mdd,
    winRate,
    numTrades: trades.length,
    equityCurve,
    trades,
  };
}
