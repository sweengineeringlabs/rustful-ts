//! Utility functions for working with quotes and prices.

use super::Quote;

/// Extract closing prices from quotes.
pub fn closing_prices(quotes: &[Quote]) -> Vec<f64> {
    quotes.iter().map(|q| q.close).collect()
}

/// Extract adjusted closing prices from quotes.
pub fn adj_closing_prices(quotes: &[Quote]) -> Vec<f64> {
    quotes.iter().map(|q| q.adj_close).collect()
}

/// Extract volumes from quotes.
pub fn volumes(quotes: &[Quote]) -> Vec<f64> {
    quotes.iter().map(|q| q.volume as f64).collect()
}

/// Calculate daily returns from prices.
pub fn daily_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return Vec::new();
    }

    prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect()
}

/// Calculate log returns from prices.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    if prices.len() < 2 {
        return Vec::new();
    }

    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_quotes() -> Vec<Quote> {
        vec![
            Quote::new(1704067200, 100.0, 105.0, 99.0, 100.0, 99.5, 1000),
            Quote::new(1704153600, 100.0, 108.0, 98.0, 105.0, 104.5, 1200),
            Quote::new(1704240000, 105.0, 110.0, 104.0, 110.0, 109.5, 1100),
            Quote::new(1704326400, 110.0, 112.0, 108.0, 108.0, 107.5, 900),
        ]
    }

    // closing_prices tests
    #[test]
    fn test_closing_prices_basic() {
        let quotes = sample_quotes();
        let closes = closing_prices(&quotes);

        assert_eq!(closes.len(), 4);
        assert_eq!(closes[0], 100.0);
        assert_eq!(closes[1], 105.0);
        assert_eq!(closes[2], 110.0);
        assert_eq!(closes[3], 108.0);
    }

    #[test]
    fn test_closing_prices_empty() {
        let quotes: Vec<Quote> = vec![];
        let closes = closing_prices(&quotes);
        assert!(closes.is_empty());
    }

    #[test]
    fn test_closing_prices_single() {
        let quotes = vec![Quote::new(1704067200, 100.0, 105.0, 99.0, 103.0, 102.5, 1000)];
        let closes = closing_prices(&quotes);

        assert_eq!(closes.len(), 1);
        assert_eq!(closes[0], 103.0);
    }

    // adj_closing_prices tests
    #[test]
    fn test_adj_closing_prices_basic() {
        let quotes = sample_quotes();
        let adj_closes = adj_closing_prices(&quotes);

        assert_eq!(adj_closes.len(), 4);
        assert_eq!(adj_closes[0], 99.5);
        assert_eq!(adj_closes[1], 104.5);
        assert_eq!(adj_closes[2], 109.5);
        assert_eq!(adj_closes[3], 107.5);
    }

    #[test]
    fn test_adj_closing_prices_empty() {
        let quotes: Vec<Quote> = vec![];
        let adj_closes = adj_closing_prices(&quotes);
        assert!(adj_closes.is_empty());
    }

    #[test]
    fn test_adj_closing_prices_single() {
        let quotes = vec![Quote::new(1704067200, 100.0, 105.0, 99.0, 103.0, 102.5, 1000)];
        let adj_closes = adj_closing_prices(&quotes);

        assert_eq!(adj_closes.len(), 1);
        assert_eq!(adj_closes[0], 102.5);
    }

    // volumes tests
    #[test]
    fn test_volumes_basic() {
        let quotes = sample_quotes();
        let vols = volumes(&quotes);

        assert_eq!(vols.len(), 4);
        assert_eq!(vols[0], 1000.0);
        assert_eq!(vols[1], 1200.0);
        assert_eq!(vols[2], 1100.0);
        assert_eq!(vols[3], 900.0);
    }

    #[test]
    fn test_volumes_empty() {
        let quotes: Vec<Quote> = vec![];
        let vols = volumes(&quotes);
        assert!(vols.is_empty());
    }

    #[test]
    fn test_volumes_converts_to_f64() {
        let quotes = vec![Quote::new(1704067200, 100.0, 105.0, 99.0, 103.0, 102.5, u64::MAX)];
        let vols = volumes(&quotes);

        assert_eq!(vols.len(), 1);
        assert_eq!(vols[0], u64::MAX as f64);
    }

    // daily_returns tests
    #[test]
    fn test_daily_returns_basic() {
        let prices = vec![100.0, 105.0, 110.0, 108.0];
        let returns = daily_returns(&prices);

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - 0.05).abs() < 1e-10); // (105-100)/100 = 0.05
        assert!((returns[1] - 0.047619047619047616).abs() < 1e-10); // (110-105)/105
        assert!((returns[2] - (-0.01818181818181818)).abs() < 1e-10); // (108-110)/110
    }

    #[test]
    fn test_daily_returns_empty() {
        let prices: Vec<f64> = vec![];
        let returns = daily_returns(&prices);
        assert!(returns.is_empty());
    }

    #[test]
    fn test_daily_returns_single_price() {
        let prices = vec![100.0];
        let returns = daily_returns(&prices);
        assert!(returns.is_empty());
    }

    #[test]
    fn test_daily_returns_two_prices() {
        let prices = vec![100.0, 110.0];
        let returns = daily_returns(&prices);

        assert_eq!(returns.len(), 1);
        assert!((returns[0] - 0.1).abs() < 1e-10); // 10% return
    }

    #[test]
    fn test_daily_returns_negative_return() {
        let prices = vec![100.0, 90.0];
        let returns = daily_returns(&prices);

        assert_eq!(returns.len(), 1);
        assert!((returns[0] - (-0.1)).abs() < 1e-10); // -10% return
    }

    #[test]
    fn test_daily_returns_zero_return() {
        let prices = vec![100.0, 100.0];
        let returns = daily_returns(&prices);

        assert_eq!(returns.len(), 1);
        assert_eq!(returns[0], 0.0);
    }

    #[test]
    fn test_daily_returns_large_gain() {
        let prices = vec![100.0, 200.0];
        let returns = daily_returns(&prices);

        assert_eq!(returns.len(), 1);
        assert!((returns[0] - 1.0).abs() < 1e-10); // 100% return
    }

    // log_returns tests
    #[test]
    fn test_log_returns_basic() {
        let prices = vec![100.0, 105.0, 110.0, 108.0];
        let returns = log_returns(&prices);

        assert_eq!(returns.len(), 3);
        assert!((returns[0] - (105.0_f64 / 100.0).ln()).abs() < 1e-10);
        assert!((returns[1] - (110.0_f64 / 105.0).ln()).abs() < 1e-10);
        assert!((returns[2] - (108.0_f64 / 110.0).ln()).abs() < 1e-10);
    }

    #[test]
    fn test_log_returns_empty() {
        let prices: Vec<f64> = vec![];
        let returns = log_returns(&prices);
        assert!(returns.is_empty());
    }

    #[test]
    fn test_log_returns_single_price() {
        let prices = vec![100.0];
        let returns = log_returns(&prices);
        assert!(returns.is_empty());
    }

    #[test]
    fn test_log_returns_two_prices() {
        let prices = vec![100.0, 110.0];
        let returns = log_returns(&prices);

        assert_eq!(returns.len(), 1);
        let expected = (110.0_f64 / 100.0).ln();
        assert!((returns[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_returns_negative_return() {
        let prices = vec![100.0, 90.0];
        let returns = log_returns(&prices);

        assert_eq!(returns.len(), 1);
        let expected = (90.0_f64 / 100.0).ln();
        assert!((returns[0] - expected).abs() < 1e-10);
        assert!(returns[0] < 0.0);
    }

    #[test]
    fn test_log_returns_zero_return() {
        let prices = vec![100.0, 100.0];
        let returns = log_returns(&prices);

        assert_eq!(returns.len(), 1);
        assert_eq!(returns[0], 0.0);
    }

    #[test]
    fn test_log_returns_double() {
        let prices = vec![100.0, 200.0];
        let returns = log_returns(&prices);

        assert_eq!(returns.len(), 1);
        let expected = 2.0_f64.ln(); // ln(2) ~ 0.693
        assert!((returns[0] - expected).abs() < 1e-10);
    }

    // Integration tests combining functions
    #[test]
    fn test_daily_returns_from_quotes() {
        let quotes = sample_quotes();
        let closes = closing_prices(&quotes);
        let returns = daily_returns(&closes);

        assert_eq!(returns.len(), 3);
        // First return: (105-100)/100 = 0.05
        assert!((returns[0] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_log_returns_from_quotes() {
        let quotes = sample_quotes();
        let closes = closing_prices(&quotes);
        let returns = log_returns(&closes);

        assert_eq!(returns.len(), 3);
        // First return: ln(105/100)
        let expected = (105.0_f64 / 100.0).ln();
        assert!((returns[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_adj_closing_daily_returns() {
        let quotes = sample_quotes();
        let adj_closes = adj_closing_prices(&quotes);
        let returns = daily_returns(&adj_closes);

        assert_eq!(returns.len(), 3);
        // First return: (104.5-99.5)/99.5
        let expected = (104.5 - 99.5) / 99.5;
        assert!((returns[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_vs_daily_returns_approximation() {
        // For small returns, log returns approximate daily returns
        let prices = vec![100.0, 101.0]; // 1% return
        let daily = daily_returns(&prices);
        let log = log_returns(&prices);

        // They should be close for small returns
        assert!((daily[0] - log[0]).abs() < 0.001);
    }

    #[test]
    fn test_log_returns_additive_property() {
        let prices = vec![100.0, 110.0, 121.0];
        let returns = log_returns(&prices);

        // ln(121/100) should equal ln(110/100) + ln(121/110)
        let total_return = (121.0_f64 / 100.0).ln();
        let sum_of_returns = returns[0] + returns[1];

        assert!((total_return - sum_of_returns).abs() < 1e-10);
    }
}
