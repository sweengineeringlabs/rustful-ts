//! Basic example demonstrating data crate functionality
//!
//! Run with: cargo run --example basic -p data
//!
//! Note: Real data fetching requires network access and the "fetch" feature.
//! This example uses mock data to demonstrate the API.

use data::{closing_prices, adj_closing_prices, volumes, daily_returns, log_returns, Quote};

fn main() {
    println!("=== Data Crate Example ===\n");

    // Create sample quote data (simulating AAPL stock)
    let quotes = vec![
        Quote {
            timestamp: 1704067200,
            open: 185.33,
            high: 186.12,
            low: 184.89,
            close: 185.64,
            adj_close: 184.50,
            volume: 45678901,
        },
        Quote {
            timestamp: 1704153600,
            open: 185.75,
            high: 187.45,
            low: 185.50,
            close: 186.89,
            adj_close: 185.74,
            volume: 52341890,
        },
        Quote {
            timestamp: 1704240000,
            open: 187.00,
            high: 188.50,
            low: 186.75,
            close: 187.95,
            adj_close: 186.79,
            volume: 48923456,
        },
        Quote {
            timestamp: 1704326400,
            open: 188.25,
            high: 190.00,
            low: 187.50,
            close: 189.45,
            adj_close: 188.28,
            volume: 61234567,
        },
        Quote {
            timestamp: 1704412800,
            open: 189.50,
            high: 191.25,
            low: 188.75,
            close: 190.75,
            adj_close: 189.57,
            volume: 55678234,
        },
    ];

    println!("Sample OHLCV data ({} quotes):", quotes.len());
    println!("Date          Open      High      Low       Close     Volume");
    println!("{}", "-".repeat(65));
    for q in &quotes {
        println!(
            "{:13} {:9.2} {:9.2} {:9.2} {:9.2} {:>10}",
            format!("T+{}", q.timestamp / 86400 - 19722), // Days since 2024-01-01
            q.open,
            q.high,
            q.low,
            q.close,
            q.volume
        );
    }

    // Extract price series
    println!("\n--- Price Extraction ---");
    let close = closing_prices(&quotes);
    let adj_close = adj_closing_prices(&quotes);
    let vols = volumes(&quotes);

    println!("Closing prices:     {:?}", close);
    println!("Adj closing prices: {:?}", adj_close);
    println!(
        "Volumes (M):        {:?}",
        vols.iter().map(|v| v / 1_000_000.0).collect::<Vec<_>>()
    );

    // Calculate returns
    println!("\n--- Returns Calculation ---");
    let simple_ret = daily_returns(&close);
    let log_ret = log_returns(&close);

    println!("Daily returns:");
    for (i, (sr, lr)) in simple_ret.iter().zip(log_ret.iter()).enumerate() {
        println!(
            "  Day {}: Simple = {:+.4} ({:+.2}%), Log = {:+.4}",
            i + 1,
            sr,
            sr * 100.0,
            lr
        );
    }

    // Summary statistics
    println!("\n--- Summary Statistics ---");
    let avg_return = simple_ret.iter().sum::<f64>() / simple_ret.len() as f64;
    let total_return = close.last().unwrap() / close.first().unwrap() - 1.0;
    let avg_volume = vols.iter().sum::<f64>() / vols.len() as f64;

    println!("Average daily return: {:+.4} ({:+.2}%)", avg_return, avg_return * 100.0);
    println!("Total period return:  {:+.4} ({:+.2}%)", total_return, total_return * 100.0);
    println!("Average volume:       {:.2}M", avg_volume / 1_000_000.0);

    // Price range analysis
    let high = quotes.iter().map(|q| q.high).fold(f64::NEG_INFINITY, f64::max);
    let low = quotes.iter().map(|q| q.low).fold(f64::INFINITY, f64::min);
    println!("Price range:          ${:.2} - ${:.2}", low, high);

    println!("\n=== Example Complete ===");
}
