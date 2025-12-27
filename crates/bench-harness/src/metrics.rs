//! Code metrics using tokei for accurate line counting.
//!
//! Provides accurate code, comment, and blank line counts by
//! understanding language syntax.

use std::collections::BTreeMap;
use std::path::Path;
use tokei::{Config, Languages};

/// Statistics for a single language.
#[derive(Debug, Clone, Default)]
pub struct LanguageStats {
    pub files: usize,
    pub code: usize,
    pub comments: usize,
    pub blanks: usize,
}

impl LanguageStats {
    /// Total lines (code + comments + blanks)
    pub fn total(&self) -> usize {
        self.code + self.comments + self.blanks
    }
}

/// Aggregated code metrics for a project.
#[derive(Debug, Clone, Default)]
pub struct CodeMetrics {
    pub languages: BTreeMap<String, LanguageStats>,
    pub total: LanguageStats,
}

impl CodeMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get stats for a specific language.
    pub fn get(&self, language: &str) -> Option<&LanguageStats> {
        self.languages.get(language)
    }

    /// Get Rust-specific stats.
    pub fn rust(&self) -> Option<&LanguageStats> {
        self.get("Rust")
    }
}

/// Count lines of code in a directory.
///
/// Uses tokei to accurately count code, comments, and blank lines
/// by understanding language syntax.
///
/// # Example
///
/// ```ignore
/// let metrics = count_lines("./src")?;
/// println!("Rust code: {} lines", metrics.rust().unwrap().code);
/// ```
pub fn count_lines<P: AsRef<Path>>(path: P) -> Result<CodeMetrics, String> {
    let paths = &[path.as_ref()];
    let excluded = &[];
    let config = Config::default();

    let mut languages = Languages::new();
    languages.get_statistics(paths, excluded, &config);

    let mut metrics = CodeMetrics::new();

    for (lang_type, language) in languages {
        let name = format!("{:?}", lang_type);
        let stats = LanguageStats {
            files: language.reports.len(),
            code: language.code,
            comments: language.comments,
            blanks: language.blanks,
        };

        metrics.total.files += stats.files;
        metrics.total.code += stats.code;
        metrics.total.comments += stats.comments;
        metrics.total.blanks += stats.blanks;

        metrics.languages.insert(name, stats);
    }

    Ok(metrics)
}

/// Count lines for multiple paths and aggregate.
pub fn count_lines_multi<P: AsRef<Path>>(paths: &[P]) -> Result<CodeMetrics, String> {
    let path_refs: Vec<&Path> = paths.iter().map(|p| p.as_ref()).collect();
    let excluded = &[];
    let config = Config::default();

    let mut languages = Languages::new();
    languages.get_statistics(&path_refs, excluded, &config);

    let mut metrics = CodeMetrics::new();

    for (lang_type, language) in languages {
        let name = format!("{:?}", lang_type);
        let stats = LanguageStats {
            files: language.reports.len(),
            code: language.code,
            comments: language.comments,
            blanks: language.blanks,
        };

        metrics.total.files += stats.files;
        metrics.total.code += stats.code;
        metrics.total.comments += stats.comments;
        metrics.total.blanks += stats.blanks;

        metrics.languages.insert(name, stats);
    }

    Ok(metrics)
}

/// Print metrics in a formatted table.
pub fn print_metrics(metrics: &CodeMetrics) {
    println!("─────────────────────────────────────────────────────────────────");
    println!(
        "{:15} {:>8} {:>10} {:>10} {:>10}",
        "Language", "Files", "Code", "Comments", "Blanks"
    );
    println!("─────────────────────────────────────────────────────────────────");

    for (name, stats) in &metrics.languages {
        println!(
            "{:15} {:>8} {:>10} {:>10} {:>10}",
            name, stats.files, stats.code, stats.comments, stats.blanks
        );
    }

    println!("─────────────────────────────────────────────────────────────────");
    println!(
        "{:15} {:>8} {:>10} {:>10} {:>10}",
        "Total",
        metrics.total.files,
        metrics.total.code,
        metrics.total.comments,
        metrics.total.blanks
    );
    println!("─────────────────────────────────────────────────────────────────");
}

/// Print a summary (total lines only).
pub fn print_summary(metrics: &CodeMetrics) {
    println!(
        "Files: {} | Code: {} | Comments: {} | Blanks: {} | Total: {}",
        metrics.total.files,
        metrics.total.code,
        metrics.total.comments,
        metrics.total.blanks,
        metrics.total.total()
    );
}
