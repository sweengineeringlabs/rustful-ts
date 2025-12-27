//! Lightweight code metrics without heavy dependencies.
//!
//! Counts code, comments, and blank lines for Rust and TOML files.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

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

    fn add(&mut self, other: &LanguageStats) {
        self.files += other.files;
        self.code += other.code;
        self.comments += other.comments;
        self.blanks += other.blanks;
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

/// Count lines in a single file.
fn count_file(path: &Path) -> Option<(&'static str, LanguageStats)> {
    let ext = path.extension()?.to_str()?;
    let lang = match ext {
        "rs" => "Rust",
        "toml" => "Toml",
        "md" => "Markdown",
        _ => return None,
    };

    let content = fs::read_to_string(path).ok()?;
    let mut stats = LanguageStats { files: 1, ..Default::default() };
    let mut in_block_comment = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            stats.blanks += 1;
            continue;
        }

        match lang {
            "Rust" => {
                if in_block_comment {
                    stats.comments += 1;
                    if trimmed.contains("*/") {
                        in_block_comment = false;
                    }
                } else if trimmed.starts_with("/*") {
                    stats.comments += 1;
                    if !trimmed.contains("*/") {
                        in_block_comment = true;
                    }
                } else if trimmed.starts_with("//") {
                    stats.comments += 1;
                } else {
                    stats.code += 1;
                }
            }
            "Toml" => {
                if trimmed.starts_with('#') {
                    stats.comments += 1;
                } else {
                    stats.code += 1;
                }
            }
            "Markdown" => {
                stats.comments += 1; // All markdown is "comments"
            }
            _ => stats.code += 1,
        }
    }

    Some((lang, stats))
}

/// Recursively walk directory and collect files.
fn walk_dir(path: &Path, files: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                // Skip hidden dirs, target, node_modules
                if !name.starts_with('.') && name != "target" && name != "node_modules" {
                    walk_dir(&path, files);
                }
            } else if path.is_file() {
                files.push(path);
            }
        }
    }
}

/// Count lines of code in a directory.
pub fn count_lines<P: AsRef<Path>>(path: P) -> Result<CodeMetrics, String> {
    let mut files = Vec::new();
    walk_dir(path.as_ref(), &mut files);

    let mut metrics = CodeMetrics::new();

    for file_path in files {
        if let Some((lang, stats)) = count_file(&file_path) {
            metrics.total.add(&stats);
            metrics
                .languages
                .entry(lang.to_string())
                .or_default()
                .add(&stats);
        }
    }

    Ok(metrics)
}

/// Count lines for multiple paths and aggregate.
pub fn count_lines_multi<P: AsRef<Path>>(paths: &[P]) -> Result<CodeMetrics, String> {
    let mut metrics = CodeMetrics::new();
    for path in paths {
        let m = count_lines(path)?;
        for (lang, stats) in m.languages {
            metrics.total.add(&stats);
            metrics.languages.entry(lang).or_default().add(&stats);
        }
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
