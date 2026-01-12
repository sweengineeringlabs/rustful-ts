//! NAS100 optimization with 50 indicators - parallel optimization comparison.

use optimizer_core::{
    optimizer::{ParallelOptimizerRunner, GridSearchConfig, GeneticConfig, BayesianConfig},
    evaluators::*,
    objective::SharpeRatio,
    datasource::FixtureDataSource,
};
use optimizer_spi::{DataSource, ParamRange, FloatParamRange, Timeframe, ValidationSplit, IndicatorEvaluator};
use std::time::Instant;

fn main() {
    println!("=== NAS100 50-Indicator Optimization Suite ===\n");

    let ds = FixtureDataSource::new();
    let data = ds.load("NAS100", Timeframe::D1).expect("Failed to load NAS100 data");

    // Use last 5 years
    let five_years = 1260;
    let data = if data.close.len() > five_years {
        data.slice(data.close.len() - five_years, data.close.len())
    } else {
        data
    };

    println!("Data: {} bars (~5 years)\n", data.close.len());

    let objective = SharpeRatio::new();

    // 70/30 train/test split
    let train_size = (data.close.len() as f64 * 0.7) as usize;
    let splits = vec![ValidationSplit {
        train_start: 0,
        train_end: train_size,
        test_start: train_size,
        test_end: data.close.len(),
    }];

    // Build all 50 evaluators
    let evaluators: Vec<(&str, Box<dyn IndicatorEvaluator>)> = vec![
        // === CORE 7 ===
        ("RSI", Box::new(RSIEvaluator::new(ParamRange::new(5, 25, 5)))),
        ("SMA", Box::new(SMAEvaluator::new(ParamRange::new(5, 15, 5), ParamRange::new(20, 40, 10)))),
        ("EMA", Box::new(EMAEvaluator::new(ParamRange::new(5, 15, 5), ParamRange::new(20, 40, 10)))),
        ("MACD", Box::new(MACDEvaluator::new(ParamRange::new(8, 14, 3), ParamRange::new(20, 28, 4), ParamRange::new(7, 11, 2)))),
        ("Bollinger", Box::new(BollingerEvaluator::new(ParamRange::new(15, 25, 5), FloatParamRange::new(1.5, 2.5, 0.5)))),
        ("Stochastic", Box::new(StochasticEvaluator::new(ParamRange::new(10, 20, 5), ParamRange::new(3, 5, 1)))),
        ("ATR", Box::new(ATREvaluator::new(ParamRange::new(10, 20, 5)))),

        // === OSCILLATORS ===
        ("WilliamsR", Box::new(WilliamsREvaluator::new(ParamRange::new(10, 20, 5)))),
        ("CCI", Box::new(CCIEvaluator::new(ParamRange::new(14, 26, 6)))),
        ("ROC", Box::new(ROCEvaluator::new(ParamRange::new(9, 15, 3)))),
        ("Momentum", Box::new(MomentumEvaluator::new(ParamRange::new(8, 16, 4)))),
        ("TRIX", Box::new(TRIXEvaluator::new(ParamRange::new(12, 18, 3)))),
        ("UltimateOsc", Box::new(UltimateOscillatorEvaluator::new(ParamRange::new(5, 9, 2), ParamRange::new(12, 16, 2), ParamRange::new(24, 32, 4)))),
        ("CMO", Box::new(CMOEvaluator::new(ParamRange::new(10, 20, 5)))),
        ("StochRSI", Box::new(StochRSIEvaluator::new(ParamRange::new(12, 16, 2), ParamRange::new(12, 16, 2)))),
        ("TSI", Box::new(TSIEvaluator::new(ParamRange::new(20, 28, 4), ParamRange::new(10, 16, 3)))),
        ("DeMarker", Box::new(DeMarkerEvaluator::new(ParamRange::new(10, 18, 4)))),
        ("FisherTransform", Box::new(FisherTransformEvaluator::new(ParamRange::new(8, 14, 3)))),
        ("AwesomeOsc", Box::new(AwesomeOscillatorEvaluator::new(ParamRange::new(4, 6, 1), ParamRange::new(30, 38, 4)))),
        ("PPO", Box::new(PPOEvaluator::new(ParamRange::new(10, 14, 2), ParamRange::new(24, 28, 2)))),
        ("KST", Box::new(KSTEvaluator::new(ParamRange::new(8, 12, 2)))),

        // === TREND ===
        ("ADX", Box::new(ADXEvaluator::new(ParamRange::new(10, 18, 4)))),
        ("SuperTrend", Box::new(SuperTrendEvaluator::new(ParamRange::new(8, 14, 3), FloatParamRange::new(2.0, 4.0, 1.0)))),
        ("ParabolicSAR", Box::new(ParabolicSAREvaluator::new(FloatParamRange::new(0.01, 0.03, 0.01), FloatParamRange::new(0.15, 0.25, 0.05)))),
        ("Aroon", Box::new(AroonEvaluator::new(ParamRange::new(20, 30, 5)))),
        ("Coppock", Box::new(CoppockEvaluator::new(ParamRange::new(8, 12, 2), ParamRange::new(12, 16, 2), ParamRange::new(9, 13, 2)))),
        ("DPO", Box::new(DPOEvaluator::new(ParamRange::new(18, 26, 4)))),
        ("Vortex", Box::new(VortexEvaluator::new(ParamRange::new(12, 18, 3)))),

        // === VOLUME ===
        ("OBV", Box::new(OBVEvaluator::new(ParamRange::new(15, 25, 5)))),
        ("MFI", Box::new(MFIEvaluator::new(ParamRange::new(12, 18, 3)))),
        ("CMF", Box::new(CMFEvaluator::new(ParamRange::new(18, 26, 4)))),
        ("ForceIndex", Box::new(ForceIndexEvaluator::new(ParamRange::new(10, 16, 3)))),
        ("VROC", Box::new(VROCEvaluator::new(ParamRange::new(12, 18, 3)))),

        // === BANDS/CHANNELS ===
        ("Keltner", Box::new(KeltnerEvaluator::new(ParamRange::new(18, 24, 3), ParamRange::new(8, 12, 2), FloatParamRange::new(1.5, 2.5, 0.5)))),
        ("Donchian", Box::new(DonchianEvaluator::new(ParamRange::new(18, 26, 4)))),

        // === MOVING AVERAGES ===
        ("WMA", Box::new(WMAEvaluator::new(ParamRange::new(8, 14, 3), ParamRange::new(26, 34, 4)))),
        ("DEMA", Box::new(DEMAEvaluator::new(ParamRange::new(10, 14, 2), ParamRange::new(24, 28, 2)))),
        ("TEMA", Box::new(TEMAEvaluator::new(ParamRange::new(10, 14, 2), ParamRange::new(24, 28, 2)))),
        ("HMA", Box::new(HMAEvaluator::new(ParamRange::new(7, 11, 2), ParamRange::new(18, 24, 3)))),
        ("KAMA", Box::new(KAMAEvaluator::new(ParamRange::new(8, 12, 2), ParamRange::new(2, 3, 1), ParamRange::new(28, 34, 3)))),
        ("PriceSMA", Box::new(PriceSMAEvaluator::new(ParamRange::new(40, 60, 10)))),
        ("PriceEMA", Box::new(PriceEMAEvaluator::new(ParamRange::new(16, 24, 4)))),
        ("ZLEMA", Box::new(ZLEMAEvaluator::new(ParamRange::new(10, 14, 2), ParamRange::new(24, 28, 2)))),

        // === VOLATILITY ===
        ("HistVol", Box::new(HistVolEvaluator::new(ParamRange::new(16, 24, 4)))),
        ("Choppiness", Box::new(ChoppinessEvaluator::new(ParamRange::new(12, 18, 3)))),

        // === DSP ===
        ("LaguerreRSI", Box::new(LaguerreRSIEvaluator::new(FloatParamRange::new(0.3, 0.7, 0.2)))),
        ("CGOscillator", Box::new(CGOscillatorEvaluator::new(ParamRange::new(8, 14, 3)))),

        // === COMPOSITE ===
        ("TTMSqueeze", Box::new(TTMSqueezeEvaluator::new(ParamRange::new(18, 24, 3), ParamRange::new(18, 24, 3)))),
        ("Schaff", Box::new(SchaffEvaluator::new(ParamRange::new(20, 26, 3), ParamRange::new(46, 54, 4), ParamRange::new(8, 12, 2)))),
        ("ElderRay", Box::new(ElderRayEvaluator::new(ParamRange::new(11, 15, 2)))),
    ];

    println!("Running {} indicators...\n", evaluators.len());
    println!("{:<20} {:>10} {:>10} {:>10} {:>8}", "Indicator", "IS Sharpe", "OOS Sharpe", "Robust", "Time");
    println!("{}", "-".repeat(62));

    let mut results: Vec<(&str, f64, f64, f64, String)> = Vec::new();
    let total_start = Instant::now();

    for (name, evaluator) in &evaluators {
        let start = Instant::now();

        // Use grid search for speed (small param spaces)
        let config = GridSearchConfig { parallel: true, top_n: 3 };
        let optimizer = optimizer_core::optimizer::IndicatorGridSearch::with_config(config);

        match optimizer.optimize(evaluator.as_ref(), &data, &objective, &splits) {
            Ok(result) => {
                let oos = result.oos_score.unwrap_or(0.0);
                let robust = result.robustness.unwrap_or(0.0);
                let elapsed = format!("{:.1}s", start.elapsed().as_secs_f64());

                println!("{:<20} {:>10.4} {:>10.4} {:>10.2} {:>8}",
                    name, result.best_score, oos, robust, elapsed);

                results.push((name, result.best_score, oos, robust, elapsed));
            }
            Err(e) => {
                println!("{:<20} ERROR: {}", name, e);
            }
        }
    }

    let total_elapsed = total_start.elapsed();

    // Sort by OOS Sharpe (descending)
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n{}", "=".repeat(62));
    println!("TOP 10 BY OUT-OF-SAMPLE SHARPE:");
    println!("{}", "=".repeat(62));
    println!("{:<5} {:<20} {:>10} {:>10} {:>10}", "Rank", "Indicator", "IS Sharpe", "OOS Sharpe", "Robust");
    println!("{}", "-".repeat(62));

    for (i, (name, is_sharpe, oos_sharpe, robust, _)) in results.iter().take(10).enumerate() {
        println!("{:<5} {:<20} {:>10.4} {:>10.4} {:>10.2}",
            i + 1, name, is_sharpe, oos_sharpe, robust);
    }

    // Summary stats
    let avg_oos: f64 = results.iter().map(|(_, _, oos, _, _)| oos).sum::<f64>() / results.len() as f64;
    let positive_oos = results.iter().filter(|(_, _, oos, _, _)| *oos > 0.0).count();

    println!("\n{}", "=".repeat(62));
    println!("SUMMARY:");
    println!("  Total indicators: {}", results.len());
    println!("  Positive OOS: {} ({:.1}%)", positive_oos, positive_oos as f64 / results.len() as f64 * 100.0);
    println!("  Average OOS Sharpe: {:.4}", avg_oos);
    println!("  Best OOS: {} ({:.4})", results[0].0, results[0].2);
    println!("  Total time: {:.1}s", total_elapsed.as_secs_f64());
    println!("{}", "=".repeat(62));
}
