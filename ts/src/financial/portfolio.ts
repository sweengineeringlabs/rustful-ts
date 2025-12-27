/**
 * Portfolio management
 */

/**
 * A single position in the portfolio
 */
export interface Position {
  symbol: string;
  quantity: number;
  entryPrice: number;
  entryTimestamp?: number;
}

/**
 * Portfolio of positions with cash management
 */
export class Portfolio {
  private positions: Map<string, Position> = new Map();
  private _cash: number;
  private _initialCash: number;

  constructor(initialCash: number) {
    this._cash = initialCash;
    this._initialCash = initialCash;
  }

  /**
   * Get current cash balance
   */
  get cash(): number {
    return this._cash;
  }

  /**
   * Get all positions
   */
  getPositions(): Position[] {
    return Array.from(this.positions.values());
  }

  /**
   * Get position for a symbol
   */
  getPosition(symbol: string): Position | undefined {
    return this.positions.get(symbol);
  }

  /**
   * Buy shares of a symbol
   */
  buy(symbol: string, quantity: number, price: number): boolean {
    const cost = quantity * price;
    if (cost > this._cash) {
      return false; // Insufficient funds
    }

    this._cash -= cost;

    const existing = this.positions.get(symbol);
    if (existing) {
      // Average in
      const totalQuantity = existing.quantity + quantity;
      const avgPrice =
        (existing.quantity * existing.entryPrice + quantity * price) / totalQuantity;
      existing.quantity = totalQuantity;
      existing.entryPrice = avgPrice;
    } else {
      this.positions.set(symbol, {
        symbol,
        quantity,
        entryPrice: price,
        entryTimestamp: Date.now(),
      });
    }

    return true;
  }

  /**
   * Sell shares of a symbol
   */
  sell(symbol: string, quantity: number, price: number): boolean {
    const position = this.positions.get(symbol);
    if (!position || position.quantity < quantity) {
      return false; // Insufficient shares
    }

    this._cash += quantity * price;
    position.quantity -= quantity;

    if (position.quantity === 0) {
      this.positions.delete(symbol);
    }

    return true;
  }

  /**
   * Get total portfolio value given current prices
   */
  value(prices: Map<string, number> | Record<string, number>): number {
    const priceMap = prices instanceof Map ? prices : new Map(Object.entries(prices));

    let positionsValue = 0;
    for (const position of this.positions.values()) {
      const currentPrice = priceMap.get(position.symbol) ?? position.entryPrice;
      positionsValue += position.quantity * currentPrice;
    }

    return this._cash + positionsValue;
  }

  /**
   * Get total return percentage
   */
  totalReturn(prices: Map<string, number> | Record<string, number>): number {
    const currentValue = this.value(prices);
    return (currentValue - this._initialCash) / this._initialCash;
  }

  /**
   * Get unrealized P&L for a position
   */
  unrealizedPnL(symbol: string, currentPrice: number): number {
    const position = this.positions.get(symbol);
    if (!position) return 0;
    return (currentPrice - position.entryPrice) * position.quantity;
  }

  /**
   * Reset portfolio to initial state
   */
  reset(): void {
    this._cash = this._initialCash;
    this.positions.clear();
  }
}
