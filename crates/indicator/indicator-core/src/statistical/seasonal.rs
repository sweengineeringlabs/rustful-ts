//! Seasonal Indicators
//!
//! Indicators for analyzing seasonal and cyclical patterns.

use indicator_spi::{
    IndicatorError, IndicatorOutput, OHLCVSeries, Result,
    TechnicalIndicator,
};

/// Seasonal Strength - Monthly/weekly pattern strength
#[derive(Debug, Clone)]
pub struct SeasonalStrength {
    period: usize,
    cycle_length: usize,
}

impl SeasonalStrength {
    pub fn new(period: usize, cycle_length: usize) -> Result<Self> {
        if period < cycle_length * 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "period".to_string(),
                reason: "must be at least 2x cycle_length".to_string(),
            });
        }
        Ok(Self { period, cycle_length })
    }

    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.period..n {
            let start = i.saturating_sub(self.period);
            let position = i % self.cycle_length;

            // Calculate average returns at this position in cycle
            let mut sum = 0.0;
            let mut count = 0;

            let mut j = start + position;
            while j < i {
                if j > 0 && close[j - 1] > 0.0 {
                    sum += (close[j] / close[j - 1] - 1.0) * 100.0;
                    count += 1;
                }
                j += self.cycle_length;
            }

            if count > 0 {
                result[i] = sum / count as f64;
            }
        }
        result
    }
}

impl TechnicalIndicator for SeasonalStrength {
    fn name(&self) -> &str {
        "Seasonal Strength"
    }

    fn min_periods(&self) -> usize {
        self.period + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Day of Week Effect - Weekly seasonality
#[derive(Debug, Clone)]
pub struct DayOfWeekEffect {
    lookback_weeks: usize,
}

impl DayOfWeekEffect {
    pub fn new(lookback_weeks: usize) -> Result<Self> {
        if lookback_weeks < 4 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback_weeks".to_string(),
                reason: "must be at least 4 weeks".to_string(),
            });
        }
        Ok(Self { lookback_weeks })
    }

    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let period = self.lookback_weeks * 5; // Trading days
        let mut result = vec![0.0; n];

        for i in period..n {
            let start = i.saturating_sub(period);
            let day_pos = i % 5;

            let mut sum = 0.0;
            let mut count = 0;

            let mut j = start + day_pos;
            while j < i {
                if j > 0 && close[j - 1] > 0.0 {
                    sum += (close[j] / close[j - 1] - 1.0) * 100.0;
                    count += 1;
                }
                j += 5;
            }

            if count > 0 {
                result[i] = sum / count as f64;
            }
        }
        result
    }
}

impl TechnicalIndicator for DayOfWeekEffect {
    fn name(&self) -> &str {
        "Day of Week Effect"
    }

    fn min_periods(&self) -> usize {
        self.lookback_weeks * 5 + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Turn of Month Effect - Month-end seasonality
#[derive(Debug, Clone)]
pub struct TurnOfMonthEffect {
    lookback: usize,
    trading_days_per_month: usize,
}

impl TurnOfMonthEffect {
    pub fn new(lookback: usize) -> Result<Self> {
        if lookback < 60 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 60 days".to_string(),
            });
        }
        Ok(Self { lookback, trading_days_per_month: 21 })
    }

    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);
            let day_in_month = i % self.trading_days_per_month;

            // Last 3 days and first 3 days of month
            let is_turn_of_month = day_in_month <= 2 || day_in_month >= self.trading_days_per_month - 3;

            if is_turn_of_month {
                // Calculate historical performance during turn-of-month
                let mut sum = 0.0;
                let mut count = 0;

                for j in start..i {
                    let j_day = j % self.trading_days_per_month;
                    let j_is_turn = j_day <= 2 || j_day >= self.trading_days_per_month - 3;

                    if j_is_turn && j > 0 && close[j - 1] > 0.0 {
                        sum += (close[j] / close[j - 1] - 1.0) * 100.0;
                        count += 1;
                    }
                }

                if count > 0 {
                    result[i] = sum / count as f64;
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for TurnOfMonthEffect {
    fn name(&self) -> &str {
        "Turn of Month Effect"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// Holiday Effect - Pre-holiday returns
#[derive(Debug, Clone)]
pub struct HolidayEffect {
    lookback: usize,
}

impl HolidayEffect {
    pub fn new(lookback: usize) -> Result<Self> {
        if lookback < 20 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 20".to_string(),
            });
        }
        Ok(Self { lookback })
    }

    /// Returns average pre-holiday returns proxy
    pub fn calculate(&self, close: &[f64], volume: &[f64]) -> Vec<f64> {
        let n = close.len().min(volume.len());
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);

            // Detect low-volume days (proxy for holiday/pre-holiday)
            let avg_vol = volume[start..i].iter().sum::<f64>() / (i - start) as f64;

            if avg_vol > 0.0 && volume[i] < avg_vol * 0.5 {
                // This is likely a pre-holiday day
                // Calculate historical pre-holiday performance
                let mut sum = 0.0;
                let mut count = 0;

                for j in start..i {
                    if volume[j] < avg_vol * 0.5 && j > 0 && close[j - 1] > 0.0 {
                        sum += (close[j] / close[j - 1] - 1.0) * 100.0;
                        count += 1;
                    }
                }

                if count > 0 {
                    result[i] = sum / count as f64;
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for HolidayEffect {
    fn name(&self) -> &str {
        "Holiday Effect"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close, &data.volume)))
    }
}

/// Quarterly Effect - Quarter-end patterns
#[derive(Debug, Clone)]
pub struct QuarterlyEffect {
    lookback: usize,
    trading_days_per_quarter: usize,
}

impl QuarterlyEffect {
    pub fn new(lookback: usize) -> Result<Self> {
        if lookback < 126 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback".to_string(),
                reason: "must be at least 126 days (2 quarters)".to_string(),
            });
        }
        Ok(Self { lookback, trading_days_per_quarter: 63 })
    }

    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let mut result = vec![0.0; n];

        for i in self.lookback..n {
            let start = i.saturating_sub(self.lookback);
            let day_in_quarter = i % self.trading_days_per_quarter;

            // Last 5 days of quarter
            let is_quarter_end = day_in_quarter >= self.trading_days_per_quarter - 5;

            if is_quarter_end {
                let mut sum = 0.0;
                let mut count = 0;

                for j in start..i {
                    let j_day = j % self.trading_days_per_quarter;
                    if j_day >= self.trading_days_per_quarter - 5 && j > 0 && close[j - 1] > 0.0 {
                        sum += (close[j] / close[j - 1] - 1.0) * 100.0;
                        count += 1;
                    }
                }

                if count > 0 {
                    result[i] = sum / count as f64;
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for QuarterlyEffect {
    fn name(&self) -> &str {
        "Quarterly Effect"
    }

    fn min_periods(&self) -> usize {
        self.lookback + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

/// January Effect - New year seasonality
#[derive(Debug, Clone)]
pub struct JanuaryEffect {
    lookback_years: usize,
    trading_days_per_year: usize,
}

impl JanuaryEffect {
    pub fn new(lookback_years: usize) -> Result<Self> {
        if lookback_years < 2 {
            return Err(IndicatorError::InvalidParameter {
                name: "lookback_years".to_string(),
                reason: "must be at least 2 years".to_string(),
            });
        }
        Ok(Self { lookback_years, trading_days_per_year: 252 })
    }

    pub fn calculate(&self, close: &[f64]) -> Vec<f64> {
        let n = close.len();
        let period = self.lookback_years * self.trading_days_per_year;
        let mut result = vec![0.0; n];

        if n < period {
            return result;
        }

        for i in period..n {
            let start = i.saturating_sub(period);
            let day_in_year = i % self.trading_days_per_year;

            // First 21 trading days of year (approximately January)
            let is_january = day_in_year < 21;

            if is_january {
                let mut sum = 0.0;
                let mut count = 0;

                for j in start..i {
                    let j_day = j % self.trading_days_per_year;
                    if j_day < 21 && j > 0 && close[j - 1] > 0.0 {
                        sum += (close[j] / close[j - 1] - 1.0) * 100.0;
                        count += 1;
                    }
                }

                if count > 0 {
                    result[i] = sum / count as f64;
                }
            }
        }
        result
    }
}

impl TechnicalIndicator for JanuaryEffect {
    fn name(&self) -> &str {
        "January Effect"
    }

    fn min_periods(&self) -> usize {
        self.lookback_years * self.trading_days_per_year + 1
    }

    fn compute(&self, data: &OHLCVSeries) -> Result<IndicatorOutput> {
        Ok(IndicatorOutput::single(self.calculate(&data.close)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>) {
        // 600 days of data for seasonal tests
        let mut close = Vec::with_capacity(600);
        let mut volume = Vec::with_capacity(600);
        let mut price = 100.0;

        for i in 0..600 {
            // Add some seasonal variation
            let seasonal = (i as f64 * std::f64::consts::PI / 21.0).sin() * 0.01;
            price *= 1.0 + seasonal + 0.001;
            close.push(price);

            // Volume with some variation
            let base_vol = 1000.0;
            let vol_var = (i % 10) as f64 * 100.0;
            volume.push(base_vol + vol_var);
        }
        (close, volume)
    }

    #[test]
    fn test_seasonal_strength() {
        let (close, _) = make_test_data();
        let ss = SeasonalStrength::new(100, 21).unwrap();
        let result = ss.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_day_of_week_effect() {
        let (close, _) = make_test_data();
        let dow = DayOfWeekEffect::new(10).unwrap();
        let result = dow.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_turn_of_month_effect() {
        let (close, _) = make_test_data();
        let tom = TurnOfMonthEffect::new(100).unwrap();
        let result = tom.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_holiday_effect() {
        let (close, volume) = make_test_data();
        let he = HolidayEffect::new(50).unwrap();
        let result = he.calculate(&close, &volume);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_quarterly_effect() {
        let (close, _) = make_test_data();
        let qe = QuarterlyEffect::new(200).unwrap();
        let result = qe.calculate(&close);

        assert_eq!(result.len(), close.len());
    }

    #[test]
    fn test_january_effect() {
        let (close, _) = make_test_data();
        let je = JanuaryEffect::new(2).unwrap();
        let result = je.calculate(&close);

        assert_eq!(result.len(), close.len());
    }
}
