//! Basic example demonstrating forecast pipeline
//!
//! Run with: cargo run --example basic -p rustful-forecast

use forecast::{Pipeline, NormalizeStep, DifferenceStep, StandardizeStep, PipelineStep};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== rustful-forecast Basic Examples ===\n");

    // Sample time series data
    let data = vec![
        100.0, 102.0, 105.0, 103.0, 108.0,
        110.0, 107.0, 112.0, 115.0, 113.0,
    ];

    println!("Original data: {:?}\n", data);

    // 1. Normalization
    println!("1. Normalize Step");
    let mut normalize = NormalizeStep::new();
    normalize.fit(&data);
    let normalized = normalize.transform(&data)?;
    println!("   Normalized: {:?}", normalized.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>());
    let denormalized = normalize.inverse_transform(&normalized)?;
    println!("   Denormalized: {:?}\n", denormalized.iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());

    // 2. Standardization
    println!("2. Standardize Step");
    let mut standardize = StandardizeStep::new();
    standardize.fit(&data);
    let standardized = standardize.transform(&data)?;
    println!("   Standardized: {:?}", standardized.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>());
    let destandardized = standardize.inverse_transform(&standardized)?;
    println!("   Destandardized: {:?}\n", destandardized.iter().map(|x| format!("{:.1}", x)).collect::<Vec<_>>());

    // 3. Differencing
    println!("3. Difference Step (order=1)");
    let difference = DifferenceStep::new(1);
    let differenced = difference.transform(&data)?;
    println!("   Differenced: {:?}\n", differenced);

    // 4. Pipeline (combining multiple steps)
    println!("4. Pipeline (Normalize -> Difference)");
    let mut pipeline = Pipeline::new();
    pipeline.add_step(Box::new(NormalizeStep::new()));
    pipeline.add_step(Box::new(DifferenceStep::new(1)));

    let transformed = pipeline.fit_transform(&data)?;
    println!("   Transformed: {:?}", transformed.iter().map(|x| format!("{:.3}", x)).collect::<Vec<_>>());

    println!("\n=== Examples Complete ===");
    Ok(())
}
