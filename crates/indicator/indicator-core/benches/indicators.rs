//! Indicator benchmarks.
//!
//! Run with: cargo bench -p indicator-core

use std::time::Instant;

// Simple benchmark helper (criterion would be better for real benchmarks)
fn bench<F: Fn()>(name: &str, iterations: usize, f: F) {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations as u32;
    println!("{}: {:?} per iteration ({} iterations)", name, per_iter, iterations);
}

fn generate_data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let high: Vec<f64> = (0..n).map(|i| 105.0 + (i as f64 * 0.1).sin() * 10.0).collect();
    let low: Vec<f64> = (0..n).map(|i| 95.0 + (i as f64 * 0.1).sin() * 10.0).collect();
    let close: Vec<f64> = (0..n).map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0).collect();
    let volume: Vec<f64> = (0..n).map(|i| 1000.0 + (i as f64 * 0.2).sin() * 500.0).collect();
    (high, low, close, volume)
}

fn main() {
    use indicator_core::*;

    let sizes = [100, 1000, 10000];

    for &size in &sizes {
        println!("\n=== Data size: {} ===\n", size);

        let (high, low, close, volume) = generate_data(size);
        let iterations = if size < 1000 { 10000 } else if size < 10000 { 1000 } else { 100 };

        // ============================================
        // Moving Averages
        // ============================================
        println!("--- Moving Averages ---");

        bench(&format!("SMA(20) n={}", size), iterations, || {
            let sma = SMA::new(20);
            let _ = sma.calculate(&close);
        });

        bench(&format!("EMA(20) n={}", size), iterations, || {
            let ema = EMA::new(20);
            let _ = ema.calculate(&close);
        });

        bench(&format!("WMA(20) n={}", size), iterations, || {
            let wma = WMA::new(20);
            let _ = wma.calculate(&close);
        });

        bench(&format!("DEMA(20) n={}", size), iterations, || {
            let dema = DEMA::new(20);
            let _ = dema.calculate(&close);
        });

        bench(&format!("TEMA(20) n={}", size), iterations, || {
            let tema = TEMA::new(20);
            let _ = tema.calculate(&close);
        });

        bench(&format!("HMA(20) n={}", size), iterations, || {
            let hma = HMA::new(20);
            let _ = hma.calculate(&close);
        });

        bench(&format!("KAMA(10,2,30) n={}", size), iterations, || {
            let kama = KAMA::default();
            let _ = kama.calculate(&close);
        });

        bench(&format!("ZLEMA(20) n={}", size), iterations, || {
            let zlema = ZLEMA::new(20);
            let _ = zlema.calculate(&close);
        });

        // ============================================
        // Filters
        // ============================================
        println!("\n--- Filters ---");

        bench(&format!("Kalman n={}", size), iterations, || {
            let kalman = KalmanFilter::default();
            let _ = kalman.calculate(&close);
        });

        bench(&format!("Median(5) n={}", size), iterations, || {
            let median = MedianFilter::new(5);
            let _ = median.calculate(&close);
        });

        bench(&format!("Gaussian(5,1.0) n={}", size), iterations, || {
            let gaussian = GaussianFilter::new(5, 1.0);
            let _ = gaussian.calculate(&close);
        });

        // ============================================
        // Oscillators
        // ============================================
        println!("\n--- Oscillators ---");

        bench(&format!("RSI(14) n={}", size), iterations, || {
            let rsi = RSI::new(14);
            let _ = rsi.calculate(&close);
        });

        bench(&format!("Stochastic(14,3) n={}", size), iterations, || {
            let stoch = Stochastic::new(14, 3);
            let _ = stoch.calculate(&high, &low, &close);
        });

        bench(&format!("WilliamsR(14) n={}", size), iterations, || {
            let wr = WilliamsR::new(14);
            let _ = wr.calculate(&high, &low, &close);
        });

        bench(&format!("CCI(20) n={}", size), iterations, || {
            let cci = CCI::new(20);
            let _ = cci.calculate(&high, &low, &close);
        });

        bench(&format!("TRIX(15) n={}", size), iterations, || {
            let trix = TRIX::new(15);
            let _ = trix.calculate(&close);
        });

        bench(&format!("UltimateOsc(7,14,28) n={}", size), iterations, || {
            let uo = UltimateOscillator::default();
            let _ = uo.calculate(&high, &low, &close);
        });

        // ============================================
        // Trend Indicators
        // ============================================
        println!("\n--- Trend Indicators ---");

        bench(&format!("MACD(12,26,9) n={}", size), iterations, || {
            let macd = MACD::new(12, 26, 9);
            let _ = macd.calculate(&close);
        });

        bench(&format!("ADX(14) n={}", size), iterations, || {
            let adx = ADX::new(14);
            let _ = adx.calculate(&high, &low, &close);
        });

        bench(&format!("SuperTrend(10,3) n={}", size), iterations, || {
            let st = SuperTrend::default();
            let _ = st.calculate(&high, &low, &close);
        });

        bench(&format!("ParabolicSAR n={}", size), iterations, || {
            let sar = ParabolicSAR::default();
            let _ = sar.calculate(&high, &low);
        });

        bench(&format!("Ichimoku(9,26,52) n={}", size), iterations, || {
            let ichimoku = Ichimoku::default();
            let _ = ichimoku.calculate(&high, &low, &close);
        });

        // ============================================
        // Volatility Indicators
        // ============================================
        println!("\n--- Volatility Indicators ---");

        bench(&format!("Bollinger(20,2) n={}", size), iterations, || {
            let bb = BollingerBands::new(20, 2.0);
            let _ = bb.calculate(&close);
        });

        bench(&format!("ATR(14) n={}", size), iterations, || {
            let atr = ATR::new(14);
            let _ = atr.calculate(&high, &low, &close);
        });

        bench(&format!("Keltner(20,10,2) n={}", size), iterations, || {
            let kc = KeltnerChannels::default();
            let _ = kc.calculate(&high, &low, &close);
        });

        bench(&format!("Donchian(20) n={}", size), iterations, || {
            let dc = DonchianChannels::default();
            let _ = dc.calculate(&high, &low);
        });

        // ============================================
        // Volume Indicators
        // ============================================
        println!("\n--- Volume Indicators ---");

        bench(&format!("VWAP n={}", size), iterations, || {
            let vwap = VWAP::new();
            let _ = vwap.calculate(&high, &low, &close, &volume);
        });

        bench(&format!("OBV n={}", size), iterations, || {
            let obv = OBV::new();
            let _ = obv.calculate(&close, &volume);
        });

        bench(&format!("MFI(14) n={}", size), iterations, || {
            let mfi = MFI::new(14);
            let _ = mfi.calculate(&high, &low, &close, &volume);
        });

        bench(&format!("CMF(20) n={}", size), iterations, || {
            let cmf = CMF::new(20);
            let _ = cmf.calculate(&high, &low, &close, &volume);
        });

        // ============================================
        // Other
        // ============================================
        println!("\n--- Other ---");

        bench(&format!("ElderRay(13) n={}", size), iterations, || {
            let er = ElderRay::default();
            let _ = er.calculate(&high, &low, &close);
        });
    }
}
