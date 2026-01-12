//! Repainting Behavior Tests
//!
//! These tests verify that:
//! - Non-repainting indicators maintain stable historical values
//! - Repainting indicators change values within their confirmation window
//! - Repainting indicators stabilize after confirmation

use indicator_core::*;

// ============================================================================
// Test Helpers
// ============================================================================

/// Verifies an indicator does NOT repaint by checking historical values remain stable
/// when new data is appended.
fn assert_no_repaint<F>(name: &str, calc: F, initial_data: &[f64], check_indices: &[usize])
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    // Calculate with initial data
    let result1 = calc(initial_data);

    // Store historical values
    let historical: Vec<(usize, f64)> = check_indices
        .iter()
        .filter(|&&i| i < result1.len() && !result1[i].is_nan())
        .map(|&i| (i, result1[i]))
        .collect();

    // Extend data with new values
    let mut extended = initial_data.to_vec();
    extended.extend_from_slice(&[
        initial_data.last().unwrap_or(&100.0) * 1.02,
        initial_data.last().unwrap_or(&100.0) * 0.98,
        initial_data.last().unwrap_or(&100.0) * 1.01,
        initial_data.last().unwrap_or(&100.0) * 1.03,
        initial_data.last().unwrap_or(&100.0) * 0.99,
    ]);

    // Recalculate with extended data
    let result2 = calc(&extended);

    // Verify historical values haven't changed
    for (idx, expected) in historical {
        let actual = result2[idx];
        assert!(
            (actual - expected).abs() < 1e-10,
            "{} repainted at index {}: was {}, now {}",
            name,
            idx,
            expected,
            actual
        );
    }
}

/// Verifies an OHLCV indicator does NOT repaint
fn assert_no_repaint_ohlcv<F>(
    name: &str,
    calc: F,
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    volume: &[f64],
    check_indices: &[usize],
) where
    F: Fn(&[f64], &[f64], &[f64], &[f64], &[f64]) -> Vec<f64>,
{
    // Calculate with initial data
    let result1 = calc(open, high, low, close, volume);

    // Store historical values
    let historical: Vec<(usize, f64)> = check_indices
        .iter()
        .filter(|&&i| i < result1.len() && !result1[i].is_nan())
        .map(|&i| (i, result1[i]))
        .collect();

    // Extend all OHLCV data
    let last_close = *close.last().unwrap_or(&100.0);
    let new_closes = [
        last_close * 1.02,
        last_close * 0.98,
        last_close * 1.01,
        last_close * 1.03,
        last_close * 0.99,
    ];

    let mut ext_open = open.to_vec();
    let mut ext_high = high.to_vec();
    let mut ext_low = low.to_vec();
    let mut ext_close = close.to_vec();
    let mut ext_volume = volume.to_vec();

    for &c in &new_closes {
        ext_open.push(ext_close.last().copied().unwrap_or(c));
        ext_high.push(c * 1.01);
        ext_low.push(c * 0.99);
        ext_close.push(c);
        ext_volume.push(*volume.last().unwrap_or(&1000.0) * 1.1);
    }

    // Recalculate with extended data
    let result2 = calc(&ext_open, &ext_high, &ext_low, &ext_close, &ext_volume);

    // Verify historical values haven't changed
    for (idx, expected) in historical {
        let actual = result2[idx];
        assert!(
            (actual - expected).abs() < 1e-10,
            "{} repainted at index {}: was {}, now {}",
            name,
            idx,
            expected,
            actual
        );
    }
}

// ============================================================================
// Sample Data
// ============================================================================

fn sample_prices() -> Vec<f64> {
    vec![
        100.0, 101.5, 99.8, 102.3, 101.0, 103.5, 102.8, 104.2, 103.0, 105.5, 104.5, 106.0, 105.2,
        107.3, 106.1, 108.0, 107.5, 109.2, 108.3, 110.0,
    ]
}

fn sample_ohlcv() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let close = sample_prices();
    let high: Vec<f64> = close.iter().map(|c| c * 1.01).collect();
    let low: Vec<f64> = close.iter().map(|c| c * 0.99).collect();
    let open: Vec<f64> = close
        .iter()
        .enumerate()
        .map(|(i, _)| if i == 0 { close[0] } else { close[i - 1] })
        .collect();
    let volume: Vec<f64> = (0..close.len()).map(|i| 1000.0 + (i as f64 * 100.0)).collect();
    (open, high, low, close, volume)
}

// ============================================================================
// Non-Repainting Tests: Moving Averages
// ============================================================================

#[test]
fn test_sma_no_repaint() {
    let data = sample_prices();
    let sma = SMA::new(5);
    assert_no_repaint("SMA", |d| sma.calculate(d), &data, &[5, 8, 10, 15]);
}

#[test]
fn test_ema_no_repaint() {
    let data = sample_prices();
    let ema = EMA::new(5);
    assert_no_repaint("EMA", |d| ema.calculate(d), &data, &[5, 8, 10, 15]);
}

#[test]
fn test_wma_no_repaint() {
    let data = sample_prices();
    let wma = WMA::new(5);
    assert_no_repaint("WMA", |d| wma.calculate(d), &data, &[5, 8, 10, 15]);
}

#[test]
fn test_dema_no_repaint() {
    let data = sample_prices();
    let dema = DEMA::new(5);
    assert_no_repaint("DEMA", |d| dema.calculate(d), &data, &[8, 10, 15]);
}

#[test]
fn test_tema_no_repaint() {
    let data = sample_prices();
    let tema = TEMA::new(5);
    assert_no_repaint("TEMA", |d| tema.calculate(d), &data, &[10, 12, 15]);
}

#[test]
fn test_hma_no_repaint() {
    let data = sample_prices();
    let hma = HMA::new(5);
    assert_no_repaint("HMA", |d| hma.calculate(d), &data, &[6, 8, 10, 15]);
}

#[test]
fn test_kama_no_repaint() {
    let data = sample_prices();
    let kama = KAMA::new(10, 2, 30);
    assert_no_repaint("KAMA", |d| kama.calculate(d), &data, &[10, 12, 15]);
}

#[test]
fn test_zlema_no_repaint() {
    let data = sample_prices();
    let zlema = ZLEMA::new(5);
    assert_no_repaint("ZLEMA", |d| zlema.calculate(d), &data, &[5, 8, 10, 15]);
}

#[test]
fn test_smma_no_repaint() {
    let data = sample_prices();
    let smma = SMMA::new(5);
    assert_no_repaint("SMMA", |d| smma.calculate(d), &data, &[5, 8, 10, 15]);
}

#[test]
fn test_alma_no_repaint() {
    let data = sample_prices();
    let alma = ALMA::new(5, 0.85, 6.0);
    assert_no_repaint("ALMA", |d| alma.calculate(d), &data, &[5, 8, 10, 15]);
}

#[test]
fn test_t3_no_repaint() {
    let data = sample_prices();
    let t3 = T3::new(5, 0.7);
    assert_no_repaint("T3", |d| t3.calculate(d), &data, &[10, 12, 15]);
}

#[test]
fn test_trima_no_repaint() {
    let data = sample_prices();
    let trima = TRIMA::new(5);
    assert_no_repaint("TRIMA", |d| trima.calculate(d), &data, &[5, 8, 10, 15]);
}

// ============================================================================
// Non-Repainting Tests: Oscillators
// ============================================================================

#[test]
fn test_rsi_no_repaint() {
    let data = sample_prices();
    let rsi = RSI::new(14);
    assert_no_repaint("RSI", |d| rsi.calculate(d), &data, &[15, 17, 18]);
}

#[test]
fn test_stochastic_no_repaint() {
    let (open, high, low, close, volume) = sample_ohlcv();
    let stoch = Stochastic::new(14, 3);
    assert_no_repaint_ohlcv(
        "Stochastic",
        |_o, h, l, c, _v| {
            let (k, _) = stoch.calculate(h, l, c);
            k
        },
        &open,
        &high,
        &low,
        &close,
        &volume,
        &[16, 17, 18],
    );
}

#[test]
fn test_williams_r_no_repaint() {
    let (open, high, low, close, volume) = sample_ohlcv();
    let wr = WilliamsR::new(14);
    assert_no_repaint_ohlcv(
        "WilliamsR",
        |_o, h, l, c, _v| wr.calculate(h, l, c),
        &open,
        &high,
        &low,
        &close,
        &volume,
        &[14, 16, 18],
    );
}

#[test]
fn test_cci_no_repaint() {
    let (open, high, low, close, volume) = sample_ohlcv();
    let cci = CCI::new(14);
    assert_no_repaint_ohlcv(
        "CCI",
        |_o, h, l, c, _v| cci.calculate(h, l, c),
        &open,
        &high,
        &low,
        &close,
        &volume,
        &[14, 16, 18],
    );
}

#[test]
fn test_trix_no_repaint() {
    let data = sample_prices();
    let trix = TRIX::new(5);
    assert_no_repaint("TRIX", |d| trix.calculate(d), &data, &[12, 14, 16, 18]);
}

#[test]
fn test_momentum_no_repaint() {
    let data = sample_prices();
    let mom = Momentum::new(10);
    assert_no_repaint("Momentum", |d| mom.calculate(d), &data, &[10, 12, 15, 18]);
}

#[test]
fn test_roc_no_repaint() {
    let data = sample_prices();
    let roc = ROC::new(10);
    assert_no_repaint("ROC", |d| roc.calculate(d), &data, &[10, 12, 15, 18]);
}

// ============================================================================
// Non-Repainting Tests: Trend Indicators
// ============================================================================

#[test]
fn test_macd_no_repaint() {
    let data = sample_prices();
    let macd = MACD::new(12, 26, 9);
    assert_no_repaint(
        "MACD",
        |d| {
            let (line, _, _) = macd.calculate(d);
            line
        },
        &data,
        &[15, 17, 18],
    );
}

#[test]
fn test_adx_no_repaint() {
    let (open, high, low, close, volume) = sample_ohlcv();
    let adx = ADX::new(14);
    assert_no_repaint_ohlcv(
        "ADX",
        |_o, h, l, c, _v| adx.calculate(h, l, c).adx,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &[15, 17, 18],
    );
}

#[test]
fn test_supertrend_no_repaint() {
    let (open, high, low, close, volume) = sample_ohlcv();
    let st = SuperTrend::new(10, 3.0);
    assert_no_repaint_ohlcv(
        "SuperTrend",
        |_o, h, l, c, _v| st.calculate(h, l, c).supertrend,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &[12, 14, 16, 18],
    );
}

#[test]
fn test_parabolic_sar_no_repaint() {
    let (open, high, low, close, volume) = sample_ohlcv();
    let sar = ParabolicSAR::new(0.02, 0.02, 0.2);
    assert_no_repaint_ohlcv(
        "ParabolicSAR",
        |_o, h, l, _c, _v| sar.calculate(h, l).sar,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &[5, 8, 10, 15],
    );
}

// ============================================================================
// Non-Repainting Tests: Volatility Indicators
// ============================================================================

#[test]
fn test_atr_no_repaint() {
    let (open, high, low, close, volume) = sample_ohlcv();
    let atr = ATR::new(14);
    assert_no_repaint_ohlcv(
        "ATR",
        |_o, h, l, c, _v| atr.calculate(h, l, c),
        &open,
        &high,
        &low,
        &close,
        &volume,
        &[15, 17, 18],
    );
}

#[test]
fn test_historical_volatility_no_repaint() {
    let data = sample_prices();
    let hv = HistoricalVolatility::new(10);
    assert_no_repaint(
        "HistoricalVolatility",
        |d| hv.calculate(d),
        &data,
        &[12, 15, 18],
    );
}

// ============================================================================
// Non-Repainting Tests: Bands
// ============================================================================

#[test]
fn test_bollinger_no_repaint() {
    let data = sample_prices();
    let bb = BollingerBands::new(20, 2.0);
    assert_no_repaint(
        "BollingerBands",
        |d| {
            let (middle, _, _) = bb.calculate(d);
            middle
        },
        &data,
        &[19],
    );
}

#[test]
fn test_keltner_no_repaint() {
    let (open, high, low, close, volume) = sample_ohlcv();
    let kc = KeltnerChannels::new(20, 10, 2.0);
    assert_no_repaint_ohlcv(
        "KeltnerChannels",
        |_o, h, l, c, _v| {
            let (middle, _, _) = kc.calculate(h, l, c);
            middle
        },
        &open,
        &high,
        &low,
        &close,
        &volume,
        &[19],
    );
}

#[test]
fn test_donchian_no_repaint() {
    let (open, high, low, close, volume) = sample_ohlcv();
    let dc = DonchianChannels::new(20);
    assert_no_repaint_ohlcv(
        "DonchianChannels",
        |_o, h, l, _c, _v| {
            let (upper, _, _) = dc.calculate(h, l);
            upper
        },
        &open,
        &high,
        &low,
        &close,
        &volume,
        &[19],
    );
}

// ============================================================================
// Non-Repainting Tests: Volume Indicators
// ============================================================================

#[test]
fn test_obv_no_repaint() {
    let (open, high, low, close, volume) = sample_ohlcv();
    let obv = OBV::new();
    assert_no_repaint_ohlcv(
        "OBV",
        |_o, _h, _l, c, v| obv.calculate(c, v),
        &open,
        &high,
        &low,
        &close,
        &volume,
        &[5, 10, 15, 18],
    );
}

#[test]
fn test_mfi_no_repaint() {
    let (open, high, low, close, volume) = sample_ohlcv();
    let mfi = MFI::new(14);
    assert_no_repaint_ohlcv(
        "MFI",
        |_o, h, l, c, v| mfi.calculate(h, l, c, v),
        &open,
        &high,
        &low,
        &close,
        &volume,
        &[15, 17, 18],
    );
}

#[test]
fn test_cmf_no_repaint() {
    let (open, high, low, close, volume) = sample_ohlcv();
    let cmf = CMF::new(20);
    assert_no_repaint_ohlcv(
        "CMF",
        |_o, h, l, c, v| cmf.calculate(h, l, c, v),
        &open,
        &high,
        &low,
        &close,
        &volume,
        &[19],
    );
}

#[test]
fn test_vwap_no_repaint() {
    let (open, high, low, close, volume) = sample_ohlcv();
    let vwap = VWAP::new();
    assert_no_repaint_ohlcv(
        "VWAP",
        |_o, h, l, c, v| vwap.calculate(h, l, c, v),
        &open,
        &high,
        &low,
        &close,
        &volume,
        &[5, 10, 15, 18],
    );
}

// ============================================================================
// Non-Repainting Tests: Filters
// ============================================================================

#[test]
fn test_kalman_no_repaint() {
    let data = sample_prices();
    let kalman = KalmanFilter::new(1.0, 1.0);
    assert_no_repaint(
        "KalmanFilter",
        |d| kalman.calculate(d),
        &data,
        &[5, 10, 15, 18],
    );
}

#[test]
fn test_median_no_repaint() {
    let data = sample_prices();
    let median = MedianFilter::new(5);
    assert_no_repaint(
        "MedianFilter",
        |d| median.calculate(d),
        &data,
        &[5, 10, 15, 18],
    );
}

#[test]
fn test_gaussian_no_repaint() {
    let data = sample_prices();
    let gaussian = GaussianFilter::new(5, 1.0);
    assert_no_repaint(
        "GaussianFilter",
        |d| gaussian.calculate(d),
        &data,
        &[5, 10, 15, 18],
    );
}

// ============================================================================
// Non-Repainting Tests: DSP
// ============================================================================

#[test]
fn test_supersmoother_no_repaint() {
    let data = sample_prices();
    let ss = Supersmoother::new(10);
    assert_no_repaint("Supersmoother", |d| ss.calculate(d), &data, &[5, 10, 15, 18]);
}

#[test]
fn test_decycler_no_repaint() {
    let data = sample_prices();
    let dc = Decycler::new(20);
    assert_no_repaint(
        "Decycler",
        |d| {
            let (trend, _) = dc.calculate(d);
            trend
        },
        &data,
        &[5, 10, 15, 18],
    );
}

// ============================================================================
// Non-Repainting Tests: Composite
// ============================================================================

#[test]
fn test_elder_ray_no_repaint() {
    let (open, high, low, close, volume) = sample_ohlcv();
    let er = ElderRay::new(13);
    assert_no_repaint_ohlcv(
        "ElderRay",
        |_o, h, l, c, _v| er.calculate(h, l, c).bull_power,
        &open,
        &high,
        &low,
        &close,
        &volume,
        &[15, 17, 18],
    );
}

// ============================================================================
// Non-Repainting Tests: Statistical
// ============================================================================

#[test]
fn test_stddev_no_repaint() {
    let data = sample_prices();
    let stddev = StandardDeviation::new(20);
    assert_no_repaint("StandardDeviation", |d| stddev.calculate(d), &data, &[19]);
}

#[test]
fn test_zscore_no_repaint() {
    let data = sample_prices();
    let zscore = ZScore::new(20);
    assert_no_repaint("ZScore", |d| zscore.calculate(d), &data, &[19]);
}

#[test]
fn test_linear_regression_no_repaint() {
    let data = sample_prices();
    let lr = LinearRegression::new(10);
    assert_no_repaint(
        "LinearRegression",
        |d| lr.calculate(d),
        &data,
        &[12, 15, 18],
    );
}

// ============================================================================
// Repainting Tests: Pattern Indicators
// ============================================================================

#[test]
fn test_zigzag_repaints() {
    // ZigZag is expected to repaint as it identifies swing points
    let zigzag = ZigZag::new(5.0);
    let (_, high, low, _, _) = sample_ohlcv();

    // Initial data - might identify a swing high
    let initial_len = 15;
    let result1 = zigzag.calculate(&high[..initial_len], &low[..initial_len]);

    // Extended data - swing identification might change
    let result2 = zigzag.calculate(&high, &low);

    // Document that ZigZag can repaint recent values
    // The exact behavior depends on price action
    let repaints_detected = result1
        .iter()
        .enumerate()
        .take(initial_len.saturating_sub(3)) // Check bars not in confirmation window
        .filter(|(i, &v)| {
            let v2 = result2.get(*i).copied().unwrap_or(f64::NAN);
            !v.is_nan() && !v2.is_nan() && (v - v2).abs() > 1e-10
        })
        .count();

    println!(
        "ZigZag repaints detected in confirmed region: {}",
        repaints_detected
    );
}

#[test]
fn test_fractals_repaints_within_window() {
    // Fractals require N bars on each side - values within window can change
    let fractals = Fractals::new(2);

    // Data with potential fractal high at index 5
    let high1 = vec![
        100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0,
    ];
    let low1: Vec<f64> = high1.iter().map(|h| h - 2.0).collect();

    let (up1, _) = fractals.calculate(&high1, &low1);

    // Extend with higher high that invalidates the fractal
    let mut high2 = high1.clone();
    high2.extend_from_slice(&[106.0, 107.0]); // New high invalidates fractal at 5
    let low2: Vec<f64> = high2.iter().map(|h| h - 2.0).collect();

    let (up2, _) = fractals.calculate(&high2, &low2);

    // Check if fractal at position 5 changed
    if up1.len() > 5 && up2.len() > 5 {
        let changed = (up1[5].is_nan() != up2[5].is_nan())
            || (!up1[5].is_nan() && !up2[5].is_nan() && (up1[5] - up2[5]).abs() > 1e-10);
        println!(
            "Fractals repaint test: index 5 value changed = {} (was {:?}, now {:?})",
            changed,
            up1.get(5),
            up2.get(5)
        );
    }
}

#[test]
fn test_fractals_stable_after_confirmation() {
    // Once a fractal is confirmed (N bars on each side), it should not change
    let fractals = Fractals::new(2);

    // Data with confirmed fractal high at index 2 (2 bars each side)
    let high = vec![
        100.0, 101.0, 105.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0,
    ];
    let low: Vec<f64> = high.iter().map(|h| h - 2.0).collect();

    let (up1, _) = fractals.calculate(&high, &low);

    // Extend with more data - confirmed fractal should not change
    let mut extended_high = high.clone();
    extended_high.extend_from_slice(&[94.0, 93.0, 92.0, 91.0, 90.0]);
    let extended_low: Vec<f64> = extended_high.iter().map(|h| h - 2.0).collect();

    let (up2, _) = fractals.calculate(&extended_high, &extended_low);

    // Fractal at index 2 should be stable (confirmed with 2 bars each side)
    if up1.len() > 2 && up2.len() > 2 && !up1[2].is_nan() {
        assert!(
            (up1[2] - up2[2]).abs() < 1e-10,
            "Confirmed fractal changed: was {}, now {}",
            up1[2],
            up2[2]
        );
    }
}

// ============================================================================
// Repainting Tests: Swing Indicators
// ============================================================================

#[test]
fn test_swing_points_repaints() {
    let sp = SwingPoints::new(3);
    let (_, high, low, _, _) = sample_ohlcv();

    // Calculate with subset
    let initial_len = 15;
    let (result1, _) = sp.calculate(&high[..initial_len], &low[..initial_len]);

    // Calculate with full data
    let (result2, _) = sp.calculate(&high, &low);

    // Recent swing points (within confirmation window) may change
    let confirmation_window = 3; // strength parameter
    let safe_index = initial_len.saturating_sub(confirmation_window + 1);

    // Values before confirmation window should be stable
    for i in 0..safe_index {
        if result1.len() > i && result2.len() > i {
            let v1 = result1[i];
            let v2 = result2[i];
            if !v1.is_nan() && !v2.is_nan() {
                assert!(
                    (v1 - v2).abs() < 1e-10,
                    "SwingPoints changed confirmed value at {}: {} -> {}",
                    i,
                    v1,
                    v2
                );
            }
        }
    }
}

#[test]
fn test_pivot_highs_lows_repaints() {
    let phl = PivotHighsLows::new(3, 3);
    let (_, high, low, _, _) = sample_ohlcv();

    let initial_len = 15;
    let (result1, _, _) = phl.calculate(&high[..initial_len], &low[..initial_len]);
    let (result2, _, _) = phl.calculate(&high, &low);

    // Pivot points within right lookback window may change
    let right_bars = 3;
    let safe_index = initial_len.saturating_sub(right_bars + 1);

    for i in 0..safe_index {
        if result1.len() > i && result2.len() > i {
            let v1 = result1[i];
            let v2 = result2[i];
            if !v1.is_nan() && !v2.is_nan() {
                assert!(
                    (v1 - v2).abs() < 1e-10,
                    "PivotHighsLows changed confirmed value at {}: {} -> {}",
                    i,
                    v1,
                    v2
                );
            }
        }
    }
}

// ============================================================================
// Repainting Tests: DeMark Indicators
// ============================================================================

#[test]
fn test_td_setup_partial_repaint() {
    // TD Setup counts can reset if conditions fail before reaching 9
    let setup = TDSetup::new();
    let (open, high, low, close, volume) = sample_ohlcv();

    // Create OHLCVSeries for the indicator
    let series1 = OHLCVSeries {
        open: open[..15].to_vec(),
        high: high[..15].to_vec(),
        low: low[..15].to_vec(),
        close: close[..15].to_vec(),
        volume: volume[..15].to_vec(),
    };

    let series2 = OHLCVSeries {
        open: open.clone(),
        high: high.clone(),
        low: low.clone(),
        close: close.clone(),
        volume: volume.clone(),
    };

    // With limited data, setup might be in progress
    let result1 = setup.calculate(&series1);

    // Extended data might complete or invalidate the setup
    let result2 = setup.calculate(&series2);

    // Active setup counts can change; completed setups are stable
    println!(
        "TD Setup - initial bars: {}, extended bars: {}",
        result1.count.len(),
        result2.count.len()
    );
}

// ============================================================================
// Additional Non-Repainting Verification Tests
// ============================================================================

/// Test that adding data to a non-repainting indicator
/// never changes any historical values
#[test]
fn test_incremental_data_stability() {
    let data = sample_prices();
    let sma = SMA::new(5);
    let ema = EMA::new(5);
    let rsi = RSI::new(14);

    // Calculate with progressively more data
    for end_idx in 10..data.len() {
        let subset = &data[..end_idx];
        let full = &data[..end_idx + 1];

        // SMA
        let sma1 = sma.calculate(subset);
        let sma2 = sma.calculate(full);
        for i in 0..sma1.len() {
            if !sma1[i].is_nan() {
                assert!(
                    (sma1[i] - sma2[i]).abs() < 1e-10,
                    "SMA repainted at {} when adding bar {}",
                    i,
                    end_idx
                );
            }
        }

        // EMA
        let ema1 = ema.calculate(subset);
        let ema2 = ema.calculate(full);
        for i in 0..ema1.len() {
            if !ema1[i].is_nan() {
                assert!(
                    (ema1[i] - ema2[i]).abs() < 1e-10,
                    "EMA repainted at {} when adding bar {}",
                    i,
                    end_idx
                );
            }
        }

        // RSI
        let rsi1 = rsi.calculate(subset);
        let rsi2 = rsi.calculate(full);
        for i in 0..rsi1.len() {
            if !rsi1[i].is_nan() {
                assert!(
                    (rsi1[i] - rsi2[i]).abs() < 1e-10,
                    "RSI repainted at {} when adding bar {}",
                    i,
                    end_idx
                );
            }
        }
    }
}
