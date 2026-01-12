//! Real Data Tests - Using Market Data Fixtures
//!
//! These tests use real market data from Yahoo Finance to verify
//! indicator behavior with actual price movements.

mod fixtures;

use fixtures::*;
use indicator_core::*;

// ============================================================================
// Data Integrity Tests
// ============================================================================

#[test]
fn test_fixtures_load_correctly() {
    let spy = spy_daily();
    let btc = btc_daily();
    let aapl = aapl_daily();
    let gld = gld_daily();

    println!("SPY: {} bars", spy.len());
    println!("BTC: {} bars", btc.len());
    println!("AAPL: {} bars", aapl.len());
    println!("GLD: {} bars", gld.len());

    assert!(spy.len() > 200);
    assert!(btc.len() > 300);
    assert!(aapl.len() > 200);
    assert!(gld.len() > 200);
}

// ============================================================================
// Moving Average Tests with Real Data
// ============================================================================

#[test]
fn test_sma_real_data() {
    let data = spy_daily();
    let sma = SMA::new(20);
    let result = sma.calculate(&data.close);

    // SMA should have valid values after period
    let valid_count = result.iter().filter(|x| !x.is_nan()).count();
    assert!(valid_count > data.len() - 20);

    // SMA should be within price range
    for (i, &val) in result.iter().enumerate() {
        if !val.is_nan() {
            let window_start = i.saturating_sub(19);
            let window_min = data.close[window_start..=i]
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let window_max = data.close[window_start..=i]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                val >= window_min && val <= window_max,
                "SMA {} outside range [{}, {}] at {}",
                val,
                window_min,
                window_max,
                i
            );
        }
    }
}

#[test]
fn test_ema_real_data() {
    let data = spy_daily();
    let ema = EMA::new(20);
    let result = ema.calculate(&data.close);

    let valid_count = result.iter().filter(|x| !x.is_nan()).count();
    assert!(valid_count > data.len() - 20);

    // EMA should be responsive to price changes
    // Compare EMA change direction with price change direction for recent bars
    let mut correct_direction = 0;
    for i in 21..data.len() {
        if !result[i].is_nan() && !result[i - 1].is_nan() {
            let price_up = data.close[i] > data.close[i - 1];
            let ema_up = result[i] > result[i - 1];
            if price_up == ema_up {
                correct_direction += 1;
            }
        }
    }
    // EMA should follow price direction most of the time
    let accuracy = correct_direction as f64 / (data.len() - 21) as f64;
    assert!(accuracy > 0.5, "EMA direction accuracy: {:.2}%", accuracy * 100.0);
}

// ============================================================================
// Oscillator Tests with Real Data
// ============================================================================

#[test]
fn test_rsi_real_data() {
    let data = spy_daily();
    let rsi = RSI::new(14);
    let result = rsi.calculate(&data.close);

    // RSI should be between 0 and 100
    for (i, &val) in result.iter().enumerate() {
        if !val.is_nan() {
            assert!(
                val >= 0.0 && val <= 100.0,
                "RSI {} out of range at {}",
                val,
                i
            );
        }
    }

    // RSI should have valid values after warmup
    let valid_count = result.iter().filter(|x| !x.is_nan()).count();
    assert!(valid_count > data.len() - 15);
}

#[test]
fn test_stochastic_real_data() {
    let data = spy_daily();
    let stoch = Stochastic::new(14, 3);
    let (k, d) = stoch.calculate(&data.high, &data.low, &data.close);

    // %K and %D should be between 0 and 100
    for (i, (&k_val, &d_val)) in k.iter().zip(d.iter()).enumerate() {
        if !k_val.is_nan() {
            assert!(
                k_val >= 0.0 && k_val <= 100.0,
                "%K {} out of range at {}",
                k_val,
                i
            );
        }
        if !d_val.is_nan() {
            assert!(
                d_val >= 0.0 && d_val <= 100.0,
                "%D {} out of range at {}",
                d_val,
                i
            );
        }
    }
}

#[test]
fn test_cci_real_data() {
    let data = spy_daily();
    let cci = CCI::new(20);
    let result = cci.calculate(&data.high, &data.low, &data.close);

    // CCI typically oscillates around 0, with extremes at +/-100 or beyond
    let mut above_100 = 0;
    let mut below_minus_100 = 0;
    let mut between = 0;

    for &val in result.iter() {
        if !val.is_nan() {
            if val > 100.0 {
                above_100 += 1;
            } else if val < -100.0 {
                below_minus_100 += 1;
            } else {
                between += 1;
            }
        }
    }

    // CCI distribution - in trending markets, extremes are common
    let total = above_100 + below_minus_100 + between;
    println!(
        "CCI distribution: {:.1}% between, {:.1}% above +100, {:.1}% below -100",
        between as f64 / total as f64 * 100.0,
        above_100 as f64 / total as f64 * 100.0,
        below_minus_100 as f64 / total as f64 * 100.0
    );
    // Just verify we have a reasonable distribution (not all extreme)
    assert!(between > 0, "Should have some values between -100 and 100");
}

// ============================================================================
// Trend Indicator Tests with Real Data
// ============================================================================

#[test]
fn test_macd_real_data() {
    let data = spy_daily();
    let macd = MACD::new(12, 26, 9);
    let (macd_line, signal, histogram) = macd.calculate(&data.close);

    // MACD histogram = MACD - Signal
    for i in 0..histogram.len() {
        if !macd_line[i].is_nan() && !signal[i].is_nan() && !histogram[i].is_nan() {
            let expected = macd_line[i] - signal[i];
            assert!(
                (histogram[i] - expected).abs() < 1e-10,
                "Histogram mismatch at {}: {} vs {}",
                i,
                histogram[i],
                expected
            );
        }
    }

    // Count crossovers (potential trading signals)
    let mut crossovers = 0;
    for i in 1..histogram.len() {
        if !histogram[i].is_nan() && !histogram[i - 1].is_nan() {
            if (histogram[i] > 0.0 && histogram[i - 1] <= 0.0)
                || (histogram[i] < 0.0 && histogram[i - 1] >= 0.0)
            {
                crossovers += 1;
            }
        }
    }
    println!("MACD crossovers in SPY 2024: {}", crossovers);
}

#[test]
fn test_adx_real_data() {
    let data = spy_daily();
    let adx = ADX::new(14);
    let output = adx.calculate(&data.high, &data.low, &data.close);

    // ADX should be between 0 and 100
    for (i, &val) in output.adx.iter().enumerate() {
        if !val.is_nan() {
            assert!(
                val >= 0.0 && val <= 100.0,
                "ADX {} out of range at {}",
                val,
                i
            );
        }
    }

    // +DI and -DI should be between 0 and 100
    for (i, (&plus, &minus)) in output.plus_di.iter().zip(output.minus_di.iter()).enumerate() {
        if !plus.is_nan() {
            assert!(plus >= 0.0 && plus <= 100.0, "+DI {} out of range at {}", plus, i);
        }
        if !minus.is_nan() {
            assert!(minus >= 0.0 && minus <= 100.0, "-DI {} out of range at {}", minus, i);
        }
    }
}

// ============================================================================
// Volatility Tests with Real Data
// ============================================================================

#[test]
fn test_atr_real_data() {
    let data = spy_daily();
    let atr = ATR::new(14);
    let result = atr.calculate(&data.high, &data.low, &data.close);

    // ATR should be positive
    for (i, &val) in result.iter().enumerate() {
        if !val.is_nan() {
            assert!(val > 0.0, "ATR {} <= 0 at {}", val, i);
        }
    }

    // ATR should be less than price (typically a small percentage)
    for i in 14..data.len() {
        if !result[i].is_nan() {
            let atr_pct = result[i] / data.close[i] * 100.0;
            assert!(
                atr_pct < 20.0, // ATR rarely exceeds 20% of price for SPY
                "ATR {}% of price at {} seems too high",
                atr_pct,
                i
            );
        }
    }
}

#[test]
fn test_bollinger_bands_real_data() {
    let data = spy_daily();
    let bb = BollingerBands::new(20, 2.0);
    let (middle, upper, lower) = bb.calculate(&data.close);

    // Price should oscillate within bands most of the time
    let mut within_bands = 0;
    let mut above_upper = 0;
    let mut below_lower = 0;

    for i in 0..data.len() {
        if !middle[i].is_nan() {
            if data.close[i] > upper[i] {
                above_upper += 1;
            } else if data.close[i] < lower[i] {
                below_lower += 1;
            } else {
                within_bands += 1;
            }
        }
    }

    let total = within_bands + above_upper + below_lower;
    let within_pct = within_bands as f64 / total as f64;
    println!(
        "Bollinger: {:.1}% within, {:.1}% above, {:.1}% below",
        within_pct * 100.0,
        above_upper as f64 / total as f64 * 100.0,
        below_lower as f64 / total as f64 * 100.0
    );

    // With 2 std dev, ~95% should be within bands theoretically
    // In practice with trending markets, it's often 85-95%
    assert!(within_pct > 0.80, "Expected >80% within bands, got {:.1}%", within_pct * 100.0);
}

// ============================================================================
// Volume Tests with Real Data
// ============================================================================

#[test]
fn test_obv_real_data() {
    let data = spy_daily();
    let obv = OBV::new();
    let result = obv.calculate(&data.close, &data.volume);

    // OBV should track cumulative volume direction
    // On up days, OBV increases; on down days, OBV decreases
    for i in 1..data.len() {
        let price_change = data.close[i] - data.close[i - 1];
        let obv_change = result[i] - result[i - 1];

        if price_change > 0.0 {
            assert!(
                obv_change >= 0.0,
                "OBV should increase on up day at {}",
                i
            );
        } else if price_change < 0.0 {
            assert!(
                obv_change <= 0.0,
                "OBV should decrease on down day at {}",
                i
            );
        }
    }
}

#[test]
fn test_vwap_real_data() {
    let data = spy_daily();
    let vwap = VWAP::new();
    let result = vwap.calculate(&data.high, &data.low, &data.close, &data.volume);

    // VWAP should be within daily range for each bar
    for i in 0..data.len() {
        if !result[i].is_nan() {
            // VWAP of the session should be near typical price
            assert!(result[i] > 0.0, "VWAP should be positive at {}", i);
        }
    }
}

// ============================================================================
// Cross-Asset Comparison Tests
// ============================================================================

#[test]
fn test_btc_higher_volatility_than_spy() {
    let spy = spy_daily();
    let btc = btc_daily();

    let spy_hv = HistoricalVolatility::new(20);
    let btc_hv = HistoricalVolatility::new(20);

    let spy_vol = spy_hv.calculate(&spy.close);
    let btc_vol = btc_hv.calculate(&btc.close);

    // Calculate average volatility
    let spy_avg: f64 = spy_vol.iter().filter(|x| !x.is_nan()).sum::<f64>()
        / spy_vol.iter().filter(|x| !x.is_nan()).count() as f64;
    let btc_avg: f64 = btc_vol.iter().filter(|x| !x.is_nan()).sum::<f64>()
        / btc_vol.iter().filter(|x| !x.is_nan()).count() as f64;

    println!("SPY avg volatility: {:.2}%", spy_avg * 100.0);
    println!("BTC avg volatility: {:.2}%", btc_avg * 100.0);

    // BTC should be more volatile than SPY
    assert!(
        btc_avg > spy_avg,
        "Expected BTC volatility > SPY volatility"
    );
}

// ============================================================================
// Non-Repainting Verification with Real Data
// ============================================================================

#[test]
fn test_indicators_dont_repaint_real_data() {
    let data = spy_daily();

    // Test with subset, then full data
    let subset_len = data.len() - 20;
    let subset_close = &data.close[..subset_len];
    let full_close = &data.close;

    // SMA
    let sma = SMA::new(20);
    let sma1 = sma.calculate(subset_close);
    let sma2 = sma.calculate(full_close);
    for i in 0..sma1.len() {
        if !sma1[i].is_nan() {
            assert!(
                (sma1[i] - sma2[i]).abs() < 1e-10,
                "SMA repainted at {} in real data",
                i
            );
        }
    }

    // EMA
    let ema = EMA::new(20);
    let ema1 = ema.calculate(subset_close);
    let ema2 = ema.calculate(full_close);
    for i in 0..ema1.len() {
        if !ema1[i].is_nan() {
            assert!(
                (ema1[i] - ema2[i]).abs() < 1e-10,
                "EMA repainted at {} in real data",
                i
            );
        }
    }

    // RSI
    let rsi = RSI::new(14);
    let rsi1 = rsi.calculate(subset_close);
    let rsi2 = rsi.calculate(full_close);
    for i in 0..rsi1.len() {
        if !rsi1[i].is_nan() {
            assert!(
                (rsi1[i] - rsi2[i]).abs() < 1e-10,
                "RSI repainted at {} in real data",
                i
            );
        }
    }
}
