//! Convexity indicator (IND-303).
//!
//! Measures the rate of change of duration as yields change.
//! Provides a second-order approximation of bond price sensitivity.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Convexity configuration.
#[derive(Debug, Clone)]
pub struct ConvexityConfig {
    /// Frequency of coupon payments per year (default: 2 for semi-annual).
    pub frequency: usize,
    /// Yield change for numerical calculation (default: 0.0001 = 1bp).
    pub yield_delta: f64,
}

impl ConvexityConfig {
    pub fn new(frequency: usize) -> Self {
        Self {
            frequency,
            yield_delta: 0.0001,
        }
    }

    pub fn with_delta(frequency: usize, yield_delta: f64) -> Self {
        Self { frequency, yield_delta }
    }
}

impl Default for ConvexityConfig {
    fn default() -> Self {
        Self {
            frequency: 2,
            yield_delta: 0.0001,
        }
    }
}

/// Convexity (IND-303).
///
/// Measures the curvature of the price-yield relationship.
/// Duration gives a linear approximation; convexity improves accuracy.
///
/// Price Change = -Duration * ΔY + (1/2) * Convexity * (ΔY)²
///
/// Properties:
/// - All option-free bonds have positive convexity
/// - Higher convexity = better price performance in rate moves
/// - Convexity increases with maturity and decreases with coupon
/// - Callable bonds can have negative convexity
///
/// The convexity adjustment improves duration estimates for large yield changes.
#[derive(Debug, Clone)]
pub struct Convexity {
    config: ConvexityConfig,
}

impl Convexity {
    pub fn new(frequency: usize) -> Self {
        Self {
            config: ConvexityConfig::new(frequency),
        }
    }

    pub fn from_config(config: ConvexityConfig) -> Self {
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

        // PV of coupon payments
        for t in 1..=periods {
            let discount = 1.0 / (1.0 + periodic_yield).powi(t as i32);
            price += coupon_payment * discount;
        }

        // PV of face value
        let final_discount = 1.0 / (1.0 + periodic_yield).powi(periods as i32);
        price += face_value * final_discount;

        price
    }

    /// Calculate Macaulay Convexity analytically.
    pub fn macaulay_convexity(
        &self,
        coupon_rate: f64,
        yield_to_maturity: f64,
        years_to_maturity: f64,
        face_value: f64,
    ) -> f64 {
        let freq = self.config.frequency as f64;
        let periods = (years_to_maturity * freq).ceil() as usize;

        if periods == 0 {
            return 0.0;
        }

        let coupon_payment = (coupon_rate * face_value) / freq;
        let periodic_yield = yield_to_maturity / freq;

        let mut price = 0.0;
        let mut weighted_sum = 0.0;

        // Calculate weighted sum of t*(t+1) * PV(CF)
        for t in 1..=periods {
            let discount = 1.0 / (1.0 + periodic_yield).powi(t as i32);
            let pv = coupon_payment * discount;
            price += pv;

            let t_years = t as f64 / freq;
            let t_plus_1 = (t + 1) as f64 / freq;
            weighted_sum += t_years * t_plus_1 * pv;
        }

        // Face value at maturity
        let final_discount = 1.0 / (1.0 + periodic_yield).powi(periods as i32);
        let pv_face = face_value * final_discount;
        price += pv_face;
        weighted_sum += years_to_maturity * (years_to_maturity + 1.0 / freq) * pv_face;

        if price > 0.0 {
            weighted_sum / (price * (1.0 + periodic_yield).powi(2))
        } else {
            f64::NAN
        }
    }

    /// Calculate Modified Convexity using numerical differentiation.
    /// More robust and handles complex bonds.
    pub fn modified_convexity(
        &self,
        coupon_rate: f64,
        yield_to_maturity: f64,
        years_to_maturity: f64,
        face_value: f64,
    ) -> f64 {
        let delta = self.config.yield_delta;

        let price_center = self.bond_price(coupon_rate, yield_to_maturity, years_to_maturity, face_value);
        let price_up = self.bond_price(coupon_rate, yield_to_maturity + delta, years_to_maturity, face_value);
        let price_down = self.bond_price(coupon_rate, yield_to_maturity - delta, years_to_maturity, face_value);

        if price_center <= 0.0 {
            return f64::NAN;
        }

        // Second derivative approximation
        (price_up + price_down - 2.0 * price_center) / (price_center * delta * delta)
    }

    /// Calculate effective convexity (works for bonds with options).
    pub fn effective_convexity(
        price_up: f64,
        price_down: f64,
        price_center: f64,
        yield_change: f64,
    ) -> f64 {
        if price_center <= 0.0 || yield_change <= 0.0 {
            return f64::NAN;
        }
        (price_up + price_down - 2.0 * price_center) / (price_center * yield_change * yield_change)
    }

    /// Calculate rolling convexity from yield series.
    pub fn calculate(
        &self,
        yields: &[f64],
        coupon_rate: f64,
        initial_maturity: f64,
    ) -> Vec<f64> {
        let n = yields.len();
        if n == 0 {
            return vec![];
        }

        let days_per_year = 252.0;

        yields
            .iter()
            .enumerate()
            .map(|(i, &ytm)| {
                let years_remaining = initial_maturity - (i as f64 / days_per_year);
                if years_remaining <= 0.0 {
                    0.0
                } else {
                    self.modified_convexity(coupon_rate, ytm, years_remaining, 100.0)
                }
            })
            .collect()
    }

    /// Calculate convexity adjustment to duration-based price estimate.
    ///
    /// # Arguments
    /// * `convexity` - Bond convexity
    /// * `yield_change` - Change in yield as decimal
    ///
    /// # Returns
    /// Convexity adjustment as percentage of price
    pub fn convexity_adjustment(convexity: f64, yield_change: f64) -> f64 {
        0.5 * convexity * yield_change * yield_change * 100.0
    }

    /// Calculate full price change with duration and convexity.
    pub fn price_change_full(
        duration: f64,
        convexity: f64,
        yield_change: f64,
    ) -> f64 {
        let duration_effect = -duration * yield_change * 100.0;
        let convexity_effect = Self::convexity_adjustment(convexity, yield_change);
        duration_effect + convexity_effect
    }

    /// Calculate dollar convexity.
    pub fn dollar_convexity(convexity: f64, price: f64) -> f64 {
        convexity * price / 100.0
    }
}

impl TechnicalIndicator for Convexity {
    fn name(&self) -> &str {
        "Convexity"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        // Default: assume 5% coupon, 10-year initial maturity
        let values = self.calculate(&data.close, 0.05, 10.0);
        Ok(IndicatorOutput::single(values))
    }

    fn min_periods(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convexity_positive() {
        let indicator = Convexity::new(2);

        // All option-free bonds should have positive convexity
        let convexity = indicator.modified_convexity(0.05, 0.05, 10.0, 100.0);

        assert!(convexity > 0.0);
    }

    #[test]
    fn test_convexity_increases_with_maturity() {
        let indicator = Convexity::new(2);

        let conv_5yr = indicator.modified_convexity(0.05, 0.05, 5.0, 100.0);
        let conv_10yr = indicator.modified_convexity(0.05, 0.05, 10.0, 100.0);
        let conv_30yr = indicator.modified_convexity(0.05, 0.05, 30.0, 100.0);

        assert!(conv_10yr > conv_5yr);
        assert!(conv_30yr > conv_10yr);
    }

    #[test]
    fn test_convexity_decreases_with_coupon() {
        let indicator = Convexity::new(2);

        let conv_low = indicator.modified_convexity(0.02, 0.05, 10.0, 100.0);
        let conv_high = indicator.modified_convexity(0.08, 0.05, 10.0, 100.0);

        assert!(conv_low > conv_high);
    }

    #[test]
    fn test_convexity_adjustment() {
        // Convexity of 100, yield change of 2%
        let adj = Convexity::convexity_adjustment(100.0, 0.02);

        // 0.5 * 100 * 0.02^2 * 100 = 2.0%
        assert!((adj - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_full_price_change() {
        // Duration 7, Convexity 50, yield increase 1%
        let price_change = Convexity::price_change_full(7.0, 50.0, 0.01);

        // Duration effect: -7 * 0.01 * 100 = -7%
        // Convexity effect: 0.5 * 50 * 0.0001 * 100 = 0.25%
        // Total: -6.75%
        let expected = -7.0 + 0.25;
        assert!((price_change - expected).abs() < 0.001);
    }

    #[test]
    fn test_effective_convexity() {
        // Price up = 102, Price down = 101.5, Price center = 100, yield change = 1%
        let eff_conv = Convexity::effective_convexity(102.0, 101.5, 100.0, 0.01);

        // (102 + 101.5 - 200) / (100 * 0.0001) = 3.5 / 0.01 = 350
        assert!((eff_conv - 350.0).abs() < 0.1);
    }

    #[test]
    fn test_rolling_convexity() {
        let indicator = Convexity::new(2);

        let yields = vec![0.05, 0.052, 0.048, 0.051, 0.049];
        let convexities = indicator.calculate(&yields, 0.05, 10.0);

        assert_eq!(convexities.len(), 5);
        for c in &convexities {
            assert!(*c > 0.0);
        }
    }

    #[test]
    fn test_dollar_convexity() {
        // Convexity = 50, Price = 100
        let dollar_conv = Convexity::dollar_convexity(50.0, 100.0);

        assert!((dollar_conv - 50.0).abs() < 0.001);
    }

    #[test]
    fn test_zero_coupon_high_convexity() {
        let indicator = Convexity::new(1);

        // Zero coupon bonds have higher convexity than coupon bonds
        let conv_zero = indicator.modified_convexity(0.0, 0.05, 10.0, 100.0);
        let conv_coupon = indicator.modified_convexity(0.05, 0.05, 10.0, 100.0);

        assert!(conv_zero > conv_coupon);
    }
}
