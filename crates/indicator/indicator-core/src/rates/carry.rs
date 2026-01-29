//! Carry indicator (IND-304).
//!
//! Measures the expected return from holding a bond assuming unchanged yields.
//! Includes both coupon income and price appreciation from roll-down.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Carry configuration.
#[derive(Debug, Clone)]
pub struct CarryConfig {
    /// Frequency of coupon payments per year (default: 2).
    pub frequency: usize,
    /// Holding period in days (default: 30).
    pub holding_period_days: usize,
    /// Trading days per year (default: 252).
    pub trading_days_year: usize,
}

impl CarryConfig {
    pub fn new(frequency: usize, holding_period_days: usize) -> Self {
        Self {
            frequency,
            holding_period_days,
            trading_days_year: 252,
        }
    }

    /// Monthly holding period.
    pub fn monthly() -> Self {
        Self {
            frequency: 2,
            holding_period_days: 21,
            trading_days_year: 252,
        }
    }

    /// Quarterly holding period.
    pub fn quarterly() -> Self {
        Self {
            frequency: 2,
            holding_period_days: 63,
            trading_days_year: 252,
        }
    }
}

impl Default for CarryConfig {
    fn default() -> Self {
        Self {
            frequency: 2,
            holding_period_days: 30,
            trading_days_year: 252,
        }
    }
}

/// Carry output containing components.
#[derive(Debug, Clone)]
pub struct CarryOutput {
    /// Total carry (annualized)
    pub total_carry: f64,
    /// Income component (coupon yield)
    pub income_carry: f64,
    /// Roll-down component (price appreciation)
    pub rolldown_carry: f64,
}

/// Carry (IND-304).
///
/// The expected return from holding a fixed income position assuming:
/// 1. The yield curve remains unchanged
/// 2. The bond "rolls down" the curve as time passes
///
/// Carry = Income Carry + Roll-Down Return
///
/// Income Carry = Coupon / Price (current yield)
/// Roll-Down = (Expected Price After Roll) - (Current Price)
///
/// Use cases:
/// - Identifying relative value opportunities
/// - Comparing bonds with different maturities
/// - Evaluating carry vs hedging costs
/// - Factor in carry & roll strategy decisions
#[derive(Debug, Clone)]
pub struct Carry {
    config: CarryConfig,
}

impl Carry {
    pub fn new(frequency: usize, holding_period_days: usize) -> Self {
        Self {
            config: CarryConfig::new(frequency, holding_period_days),
        }
    }

    pub fn from_config(config: CarryConfig) -> Self {
        Self { config }
    }

    /// Calculate bond price given parameters.
    fn bond_price(
        &self,
        coupon_rate: f64,
        yield_to_maturity: f64,
        years_to_maturity: f64,
        face_value: f64,
    ) -> f64 {
        let freq = self.config.frequency as f64;
        let periods = (years_to_maturity * freq).ceil() as usize;

        if periods == 0 {
            return face_value;
        }

        let coupon_payment = (coupon_rate * face_value) / freq;
        let periodic_yield = yield_to_maturity / freq;

        let mut price = 0.0;

        for t in 1..=periods {
            let discount = 1.0 / (1.0 + periodic_yield).powi(t as i32);
            price += coupon_payment * discount;
        }

        let final_discount = 1.0 / (1.0 + periodic_yield).powi(periods as i32);
        price += face_value * final_discount;

        price
    }

    /// Calculate income carry (current yield).
    pub fn income_carry(&self, coupon_rate: f64, price: f64) -> f64 {
        if price <= 0.0 {
            return f64::NAN;
        }
        (coupon_rate * 100.0) / price // Annualized
    }

    /// Calculate roll-down return assuming unchanged yield curve.
    ///
    /// # Arguments
    /// * `current_yield` - Current yield at bond's maturity
    /// * `rolled_yield` - Yield at shorter maturity (after roll)
    /// * `coupon_rate` - Bond coupon rate
    /// * `years_to_maturity` - Current years to maturity
    pub fn rolldown_return(
        &self,
        current_yield: f64,
        rolled_yield: f64,
        coupon_rate: f64,
        years_to_maturity: f64,
    ) -> f64 {
        let holding_years = self.config.holding_period_days as f64
            / self.config.trading_days_year as f64;

        let current_price = self.bond_price(coupon_rate, current_yield, years_to_maturity, 100.0);
        let new_maturity = years_to_maturity - holding_years;

        if new_maturity <= 0.0 {
            return 0.0;
        }

        let future_price = self.bond_price(coupon_rate, rolled_yield, new_maturity, 100.0);

        if current_price <= 0.0 {
            return f64::NAN;
        }

        // Annualized return
        let period_return = (future_price - current_price) / current_price;
        period_return * (self.config.trading_days_year as f64 / self.config.holding_period_days as f64)
    }

    /// Calculate total carry (income + roll-down).
    pub fn total_carry(
        &self,
        current_yield: f64,
        rolled_yield: f64,
        coupon_rate: f64,
        years_to_maturity: f64,
    ) -> CarryOutput {
        let current_price = self.bond_price(coupon_rate, current_yield, years_to_maturity, 100.0);

        let income = self.income_carry(coupon_rate, current_price);
        let rolldown = self.rolldown_return(current_yield, rolled_yield, coupon_rate, years_to_maturity);

        CarryOutput {
            total_carry: income + rolldown,
            income_carry: income,
            rolldown_carry: rolldown,
        }
    }

    /// Calculate carry from yield curve series.
    /// Assumes yields are ordered by maturity and interpolates.
    ///
    /// # Arguments
    /// * `yields` - Series of yields (close prices represent yields)
    /// * `coupon_rate` - Bond coupon rate
    /// * `initial_maturity` - Starting maturity in years
    pub fn calculate(&self, yields: &[f64], coupon_rate: f64, initial_maturity: f64) -> Vec<f64> {
        let n = yields.len();
        if n < 2 {
            return vec![f64::NAN; n];
        }

        let holding_years = self.config.holding_period_days as f64
            / self.config.trading_days_year as f64;
        let days_per_year = self.config.trading_days_year as f64;

        yields
            .iter()
            .enumerate()
            .map(|(i, &current_yield)| {
                let years_remaining = initial_maturity - (i as f64 / days_per_year);

                if years_remaining <= holding_years {
                    return f64::NAN;
                }

                // Estimate rolled yield (simplified: use next available yield as proxy)
                let rolled_yield = if i + self.config.holding_period_days < n {
                    yields[i + self.config.holding_period_days]
                } else {
                    current_yield // No roll-down data available
                };

                let output = self.total_carry(current_yield, rolled_yield, coupon_rate, years_remaining);
                output.total_carry
            })
            .collect()
    }

    /// Calculate break-even yield change.
    /// How much yields can rise before carry is offset by capital loss.
    pub fn breakeven_yield_change(&self, carry: f64, duration: f64) -> f64 {
        if duration <= 0.0 {
            return f64::NAN;
        }
        // Carry (annualized) / Duration = breakeven yield change
        // Scale by holding period
        let holding_years = self.config.holding_period_days as f64
            / self.config.trading_days_year as f64;
        (carry * holding_years) / duration
    }

    /// Calculate carry-to-duration ratio (risk-adjusted carry).
    pub fn carry_to_duration(carry: f64, duration: f64) -> f64 {
        if duration <= 0.0 {
            f64::NAN
        } else {
            carry / duration
        }
    }
}

impl TechnicalIndicator for Carry {
    fn name(&self) -> &str {
        "Carry"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.len() < 2 {
            return Err(IndicatorError::InsufficientData {
                required: 2,
                got: data.close.len(),
            });
        }

        // Default: assume 5% coupon, 10-year initial maturity
        let values = self.calculate(&data.close, 0.05, 10.0);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_income_carry() {
        let indicator = Carry::new(2, 30);

        // 5% coupon, price at par
        let income = indicator.income_carry(0.05, 100.0);
        assert!((income - 0.05).abs() < 0.001);

        // 5% coupon, price below par (higher current yield)
        let income_discount = indicator.income_carry(0.05, 95.0);
        assert!(income_discount > 0.05);
    }

    #[test]
    fn test_rolldown_return_upward_sloping() {
        let indicator = Carry::new(2, 21);

        // Upward sloping curve: 10Y at 5%, rolling to 9.9Y at 4.9%
        let rolldown = indicator.rolldown_return(0.05, 0.049, 0.05, 10.0);

        // Should be positive (rolling into lower yield = price gain)
        assert!(rolldown > 0.0);
    }

    #[test]
    fn test_rolldown_return_flat_curve() {
        let indicator = Carry::new(2, 21);

        // Flat curve: yield unchanged after roll
        let rolldown = indicator.rolldown_return(0.05, 0.05, 0.05, 10.0);

        // Should be approximately zero (only time value change)
        assert!(rolldown.abs() < 0.01);
    }

    #[test]
    fn test_total_carry() {
        let indicator = Carry::new(2, 30);

        let output = indicator.total_carry(0.05, 0.048, 0.05, 10.0);

        // Total should be sum of components
        let expected_total = output.income_carry + output.rolldown_carry;
        assert!((output.total_carry - expected_total).abs() < 0.0001);

        // Income carry should be around 5%
        assert!((output.income_carry - 0.05).abs() < 0.01);

        // Rolldown should be positive for normal curve
        assert!(output.rolldown_carry > 0.0);
    }

    #[test]
    fn test_breakeven_yield_change() {
        let indicator = Carry::new(2, 252); // 1 year holding

        // 5% carry, duration 7
        let breakeven = indicator.breakeven_yield_change(0.05, 7.0);

        // Should be approximately 0.05 / 7 = 0.714% for 1 year
        assert!((breakeven - 0.00714).abs() < 0.001);
    }

    #[test]
    fn test_carry_to_duration() {
        // 6% carry, duration 5 = 1.2 carry per unit of duration
        let ratio = Carry::carry_to_duration(0.06, 5.0);
        assert!((ratio - 0.012).abs() < 0.001);
    }

    #[test]
    fn test_rolling_carry() {
        let indicator = Carry::new(2, 21);

        // Slightly declining yields (normal curve roll-down scenario)
        let yields = vec![0.050, 0.049, 0.048, 0.047, 0.046];
        let carries = indicator.calculate(&yields, 0.05, 10.0);

        assert_eq!(carries.len(), 5);
        // First few should have valid carry values
        assert!(!carries[0].is_nan());
    }

    #[test]
    fn test_carry_inverted_curve() {
        let indicator = Carry::new(2, 21);

        // Inverted curve: rolling into higher yield = negative rolldown
        let rolldown = indicator.rolldown_return(0.04, 0.05, 0.05, 10.0);

        // Should be negative
        assert!(rolldown < 0.0);
    }

    #[test]
    fn test_carry_configs() {
        let monthly = CarryConfig::monthly();
        assert_eq!(monthly.holding_period_days, 21);

        let quarterly = CarryConfig::quarterly();
        assert_eq!(quarterly.holding_period_days, 63);
    }
}
