//! Duration indicator (IND-302).
//!
//! Measures a bond's sensitivity to interest rate changes.
//! A fundamental fixed income risk metric.

use indicator_spi::{
    TechnicalIndicator, IndicatorOutput, IndicatorError, Result, OHLCVSeries,
};

/// Duration configuration.
#[derive(Debug, Clone)]
pub struct DurationConfig {
    /// Frequency of coupon payments per year (default: 2 for semi-annual).
    pub frequency: usize,
    /// Whether to calculate Modified Duration (default: true).
    pub modified: bool,
}

impl DurationConfig {
    pub fn new(frequency: usize, modified: bool) -> Self {
        Self { frequency, modified }
    }

    /// Create config for semi-annual coupon bonds.
    pub fn semi_annual() -> Self {
        Self { frequency: 2, modified: true }
    }

    /// Create config for annual coupon bonds.
    pub fn annual() -> Self {
        Self { frequency: 1, modified: true }
    }

    /// Create config for Macaulay duration.
    pub fn macaulay(frequency: usize) -> Self {
        Self { frequency, modified: false }
    }
}

impl Default for DurationConfig {
    fn default() -> Self {
        Self {
            frequency: 2,
            modified: true,
        }
    }
}

/// Duration (IND-302).
///
/// Measures a bond's price sensitivity to interest rate changes.
/// Two main types:
///
/// Macaulay Duration: Weighted average time to receive cash flows
/// Modified Duration: Percentage price change for 1% yield change
///
/// Modified Duration = Macaulay Duration / (1 + yield/frequency)
///
/// Interpretation:
/// - Higher duration = more sensitive to rate changes
/// - Duration of 5 means ~5% price change for 1% yield change
/// - Zero-coupon bonds have duration equal to maturity
/// - Coupon-bearing bonds have duration less than maturity
#[derive(Debug, Clone)]
pub struct Duration {
    config: DurationConfig,
}

impl Duration {
    pub fn new(frequency: usize) -> Self {
        Self {
            config: DurationConfig::new(frequency, true),
        }
    }

    pub fn from_config(config: DurationConfig) -> Self {
        Self { config }
    }

    /// Calculate Macaulay Duration for a single bond.
    ///
    /// # Arguments
    /// * `coupon_rate` - Annual coupon rate as decimal (e.g., 0.05 for 5%)
    /// * `yield_to_maturity` - YTM as decimal
    /// * `years_to_maturity` - Time to maturity in years
    /// * `face_value` - Face value of bond (default: 100)
    pub fn macaulay_duration(
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

        let mut pv_sum = 0.0;
        let mut weighted_sum = 0.0;

        // Calculate PV of coupon payments
        for t in 1..=periods {
            let discount_factor = 1.0 / (1.0 + periodic_yield).powi(t as i32);
            let pv = coupon_payment * discount_factor;
            pv_sum += pv;
            weighted_sum += (t as f64 / freq) * pv;
        }

        // Add PV of face value at maturity
        let final_discount = 1.0 / (1.0 + periodic_yield).powi(periods as i32);
        let pv_face = face_value * final_discount;
        pv_sum += pv_face;
        weighted_sum += years_to_maturity * pv_face;

        if pv_sum > 0.0 {
            weighted_sum / pv_sum
        } else {
            f64::NAN
        }
    }

    /// Calculate Modified Duration.
    pub fn modified_duration(
        &self,
        coupon_rate: f64,
        yield_to_maturity: f64,
        years_to_maturity: f64,
        face_value: f64,
    ) -> f64 {
        let mac_duration = self.macaulay_duration(
            coupon_rate,
            yield_to_maturity,
            years_to_maturity,
            face_value,
        );

        let freq = self.config.frequency as f64;
        mac_duration / (1.0 + yield_to_maturity / freq)
    }

    /// Calculate duration based on config setting.
    pub fn calculate_single(
        &self,
        coupon_rate: f64,
        yield_to_maturity: f64,
        years_to_maturity: f64,
    ) -> f64 {
        if self.config.modified {
            self.modified_duration(coupon_rate, yield_to_maturity, years_to_maturity, 100.0)
        } else {
            self.macaulay_duration(coupon_rate, yield_to_maturity, years_to_maturity, 100.0)
        }
    }

    /// Calculate rolling duration from yield series.
    /// Assumes constant coupon rate and decreasing maturity over time.
    ///
    /// # Arguments
    /// * `yields` - Series of yield to maturity values
    /// * `coupon_rate` - Fixed coupon rate
    /// * `initial_maturity` - Initial years to maturity
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

        // Assuming daily data, 252 trading days per year
        let days_per_year = 252.0;

        yields
            .iter()
            .enumerate()
            .map(|(i, &ytm)| {
                let years_remaining = initial_maturity - (i as f64 / days_per_year);
                if years_remaining <= 0.0 {
                    0.0
                } else {
                    self.calculate_single(coupon_rate, ytm, years_remaining)
                }
            })
            .collect()
    }

    /// Calculate effective duration using numerical differentiation.
    /// More accurate for bonds with embedded options.
    ///
    /// # Arguments
    /// * `price_down` - Bond price if yield decreases
    /// * `price_up` - Bond price if yield increases
    /// * `initial_price` - Current bond price
    /// * `yield_change` - Size of yield change (e.g., 0.01 for 1%)
    pub fn effective_duration(
        price_down: f64,
        price_up: f64,
        initial_price: f64,
        yield_change: f64,
    ) -> f64 {
        if initial_price <= 0.0 || yield_change <= 0.0 {
            return f64::NAN;
        }
        (price_down - price_up) / (2.0 * initial_price * yield_change)
    }

    /// Estimate price change based on duration.
    ///
    /// # Arguments
    /// * `duration` - Modified duration
    /// * `yield_change` - Change in yield as decimal (e.g., 0.01 for +1%)
    pub fn price_change_estimate(duration: f64, yield_change: f64) -> f64 {
        -duration * yield_change * 100.0 // Returns percentage change
    }

    /// Calculate dollar duration (DV01 * 100).
    /// Represents dollar change in price for 1% yield change.
    pub fn dollar_duration(duration: f64, price: f64) -> f64 {
        duration * price / 100.0
    }
}

impl TechnicalIndicator for Duration {
    fn name(&self) -> &str {
        "Duration"
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        if data.close.is_empty() {
            return Err(IndicatorError::InsufficientData {
                required: 1,
                got: 0,
            });
        }

        // Default: assume 5% coupon, 10-year initial maturity
        // In practice, these would be configured or passed via custom series
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
    fn test_macaulay_duration_zero_coupon() {
        let indicator = Duration::from_config(DurationConfig::macaulay(1));

        // Zero coupon bond: duration = maturity
        let duration = indicator.macaulay_duration(0.0, 0.05, 5.0, 100.0);

        assert!((duration - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_macaulay_duration_coupon_bond() {
        let indicator = Duration::from_config(DurationConfig::macaulay(2));

        // 5% coupon, 5% yield, 10 year bond
        let duration = indicator.macaulay_duration(0.05, 0.05, 10.0, 100.0);

        // Duration should be less than maturity
        assert!(duration < 10.0);
        assert!(duration > 0.0);
        // Approximately 8 years for this bond
        assert!((duration - 8.0).abs() < 0.5);
    }

    #[test]
    fn test_modified_duration() {
        let indicator = Duration::new(2);

        let mac_duration = indicator.macaulay_duration(0.05, 0.06, 10.0, 100.0);
        let mod_duration = indicator.modified_duration(0.05, 0.06, 10.0, 100.0);

        // Modified < Macaulay
        assert!(mod_duration < mac_duration);

        // Verify relationship: ModD = MacD / (1 + y/f)
        let expected = mac_duration / (1.0 + 0.06 / 2.0);
        assert!((mod_duration - expected).abs() < 0.001);
    }

    #[test]
    fn test_effective_duration() {
        // Example: price at yield-1% = 105, price at yield+1% = 95, current = 100
        let eff_dur = Duration::effective_duration(105.0, 95.0, 100.0, 0.01);

        assert!((eff_dur - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_price_change_estimate() {
        // Duration of 7, yield increase of 0.5%
        let price_change = Duration::price_change_estimate(7.0, 0.005);

        // Should be approximately -3.5%
        assert!((price_change - (-3.5)).abs() < 0.001);
    }

    #[test]
    fn test_rolling_duration() {
        let indicator = Duration::new(2);

        // Yields over 5 days
        let yields = vec![0.05, 0.052, 0.048, 0.051, 0.049];
        let durations = indicator.calculate(&yields, 0.05, 10.0);

        assert_eq!(durations.len(), 5);
        // All durations should be positive
        for d in &durations {
            assert!(*d > 0.0);
        }
    }

    #[test]
    fn test_dollar_duration() {
        // Duration = 7, Price = 100
        let dv01 = Duration::dollar_duration(7.0, 100.0);

        assert!((dv01 - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_higher_coupon_lower_duration() {
        let indicator = Duration::from_config(DurationConfig::macaulay(2));

        // Higher coupon = lower duration
        let dur_low_coupon = indicator.macaulay_duration(0.02, 0.05, 10.0, 100.0);
        let dur_high_coupon = indicator.macaulay_duration(0.08, 0.05, 10.0, 100.0);

        assert!(dur_high_coupon < dur_low_coupon);
    }
}
