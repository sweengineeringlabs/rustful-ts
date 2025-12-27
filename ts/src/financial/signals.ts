/**
 * Trading signals
 */

/**
 * Trading signal types
 */
export enum Signal {
  Buy = 'BUY',
  Sell = 'SELL',
  Hold = 'HOLD',
}

/**
 * Interface for signal generators
 */
export interface SignalGenerator {
  /** Name of the strategy */
  readonly name: string;

  /** Generate a signal based on price data */
  generate(prices: number[]): Signal;

  /** Generate signals for each point in the series */
  generateSeries(prices: number[]): Signal[];
}

/**
 * Simple Moving Average Crossover Strategy
 */
export class SMACrossover implements SignalGenerator {
  readonly name = 'SMA Crossover';
  private shortPeriod: number;
  private longPeriod: number;

  constructor(shortPeriod: number = 10, longPeriod: number = 20) {
    this.shortPeriod = shortPeriod;
    this.longPeriod = longPeriod;
  }

  private sma(data: number[], period: number): number {
    if (data.length < period) return NaN;
    const slice = data.slice(-period);
    return slice.reduce((a, b) => a + b, 0) / period;
  }

  generate(prices: number[]): Signal {
    if (prices.length < this.longPeriod + 1) {
      return Signal.Hold;
    }

    const shortSMA = this.sma(prices, this.shortPeriod);
    const longSMA = this.sma(prices, this.longPeriod);
    const prevShortSMA = this.sma(prices.slice(0, -1), this.shortPeriod);
    const prevLongSMA = this.sma(prices.slice(0, -1), this.longPeriod);

    // Crossover detection
    if (prevShortSMA <= prevLongSMA && shortSMA > longSMA) {
      return Signal.Buy;
    }
    if (prevShortSMA >= prevLongSMA && shortSMA < longSMA) {
      return Signal.Sell;
    }

    return Signal.Hold;
  }

  generateSeries(prices: number[]): Signal[] {
    const signals: Signal[] = [];
    for (let i = 1; i <= prices.length; i++) {
      signals.push(this.generate(prices.slice(0, i)));
    }
    return signals;
  }
}

/**
 * RSI-based Strategy
 */
export class RSIStrategy implements SignalGenerator {
  readonly name = 'RSI Strategy';
  private period: number;
  private overbought: number;
  private oversold: number;

  constructor(period: number = 14, overbought: number = 70, oversold: number = 30) {
    this.period = period;
    this.overbought = overbought;
    this.oversold = oversold;
  }

  private calculateRSI(prices: number[]): number {
    if (prices.length < this.period + 1) {
      return 50; // Neutral
    }

    let gains = 0;
    let losses = 0;

    for (let i = prices.length - this.period; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      if (change > 0) {
        gains += change;
      } else {
        losses -= change;
      }
    }

    const avgGain = gains / this.period;
    const avgLoss = losses / this.period;

    if (avgLoss === 0) return 100;

    const rs = avgGain / avgLoss;
    return 100 - 100 / (1 + rs);
  }

  generate(prices: number[]): Signal {
    const rsi = this.calculateRSI(prices);

    if (rsi < this.oversold) {
      return Signal.Buy;
    }
    if (rsi > this.overbought) {
      return Signal.Sell;
    }

    return Signal.Hold;
  }

  generateSeries(prices: number[]): Signal[] {
    const signals: Signal[] = [];
    for (let i = 1; i <= prices.length; i++) {
      signals.push(this.generate(prices.slice(0, i)));
    }
    return signals;
  }
}
