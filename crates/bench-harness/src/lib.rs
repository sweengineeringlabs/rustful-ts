//! Benchmark harness and code metrics utilities.
//!
//! # Benchmarking
//!
//! ```ignore
//! use bench_harness::{bench_print, header, section, footer};
//!
//! fn main() {
//!     header("My Benchmarks");
//!     section("Algorithm A");
//!     bench_print("operation", 100, || {
//!         // computation
//!         result
//!     });
//!     footer();
//! }
//! ```
//!
//! # Code Metrics
//!
//! ```ignore
//! use bench_harness::metrics::{count_lines, print_metrics};
//!
//! // Count lines in a directory
//! let stats = count_lines("./src").unwrap();
//! print_metrics(&stats);
//! ```

pub mod metrics;

use std::hint::black_box;
use std::time::{Duration, Instant};

/// Benchmark result containing timing statistics.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub name: String,
    pub iterations: u32,
    pub total: Duration,
    pub per_iter: Duration,
}

impl BenchResult {
    /// Format result as a table row.
    pub fn as_row(&self) -> String {
        format!(
            "{:30} {:>10.2?} total, {:>10.2?}/iter ({} iters)",
            self.name, self.total, self.per_iter, self.iterations
        )
    }
}

/// Run a benchmark with warmup and return results.
///
/// The closure must return a value to prevent the compiler from
/// optimizing away the computation via dead code elimination.
///
/// # Arguments
///
/// * `name` - Benchmark name for display
/// * `iterations` - Number of timed iterations
/// * `f` - Closure to benchmark (must return a value)
///
/// # Example
///
/// ```ignore
/// let result = bench("fit model", 100, || {
///     let mut model = Model::new();
///     model.fit(&data).unwrap();
///     model  // return to prevent elimination
/// });
/// println!("{}", result.as_row());
/// ```
pub fn bench<F, R>(name: &str, iterations: u32, mut f: F) -> BenchResult
where
    F: FnMut() -> R,
{
    // Warmup: prime caches, trigger any lazy initialization
    for _ in 0..3 {
        black_box(f());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(f());
    }
    let total = start.elapsed();
    let per_iter = total / iterations;

    BenchResult {
        name: name.to_string(),
        iterations,
        total,
        per_iter,
    }
}

/// Run a benchmark and print results immediately.
///
/// Convenience wrapper around [`bench`] that prints the result.
pub fn bench_print<F, R>(name: &str, iterations: u32, f: F)
where
    F: FnMut() -> R,
{
    let result = bench(name, iterations, f);
    println!("{}", result.as_row());
}

/// Print a section header for organizing benchmark output.
pub fn section(name: &str) {
    println!("\n--- {} ---", name);
}

/// Print a benchmark suite header.
pub fn header(name: &str) {
    println!("=== {} ===\n", name);
}

/// Print a benchmark suite footer.
pub fn footer() {
    println!("\n=== Benchmark Complete ===");
}
