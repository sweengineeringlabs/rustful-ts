//! loc - Fast, lightweight lines of code counter
//!
//! Zero dependencies. Just std.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let args: Vec<String> = env::args().collect();

    let (paths, show_files) = parse_args(&args);

    if paths.is_empty() {
        print_usage();
        return;
    }

    let mut stats = Stats::new();

    for path in &paths {
        count_path(Path::new(path), &mut stats);
    }

    print_results(&stats, show_files);
}

fn parse_args(args: &[String]) -> (Vec<String>, bool) {
    let mut paths = Vec::new();
    let mut show_files = false;

    for arg in args.iter().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "-f" | "--files" => show_files = true,
            "-v" | "--version" => {
                println!("loc {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            _ if !arg.starts_with('-') => paths.push(arg.clone()),
            _ => {
                eprintln!("Unknown option: {}", arg);
                std::process::exit(1);
            }
        }
    }

    if paths.is_empty() {
        paths.push(".".to_string());
    }

    (paths, show_files)
}

fn print_usage() {
    println!("loc - Fast, lightweight lines of code counter");
    println!();
    println!("USAGE:");
    println!("    loc [OPTIONS] [PATH...]");
    println!();
    println!("OPTIONS:");
    println!("    -f, --files     Show per-file breakdown");
    println!("    -h, --help      Show this help");
    println!("    -v, --version   Show version");
    println!();
    println!("EXAMPLES:");
    println!("    loc .                  Count lines in current directory");
    println!("    loc src lib            Count lines in src and lib directories");
    println!("    loc -f src             Show per-file breakdown");
}

// Language definitions
struct Language {
    name: &'static str,
    extensions: &'static [&'static str],
    line_comment: Option<&'static str>,
    block_comment: Option<(&'static str, &'static str)>,
}

const LANGUAGES: &[Language] = &[
    Language {
        name: "Rust",
        extensions: &["rs"],
        line_comment: Some("//"),
        block_comment: Some(("/*", "*/")),
    },
    Language {
        name: "TypeScript",
        extensions: &["ts", "tsx"],
        line_comment: Some("//"),
        block_comment: Some(("/*", "*/")),
    },
    Language {
        name: "JavaScript",
        extensions: &["js", "jsx", "mjs", "cjs"],
        line_comment: Some("//"),
        block_comment: Some(("/*", "*/")),
    },
    Language {
        name: "Python",
        extensions: &["py", "pyi"],
        line_comment: Some("#"),
        block_comment: Some(("\"\"\"", "\"\"\"")),
    },
    Language {
        name: "Go",
        extensions: &["go"],
        line_comment: Some("//"),
        block_comment: Some(("/*", "*/")),
    },
    Language {
        name: "Java",
        extensions: &["java"],
        line_comment: Some("//"),
        block_comment: Some(("/*", "*/")),
    },
    Language {
        name: "C",
        extensions: &["c", "h"],
        line_comment: Some("//"),
        block_comment: Some(("/*", "*/")),
    },
    Language {
        name: "C++",
        extensions: &["cpp", "cc", "cxx", "hpp", "hh", "hxx"],
        line_comment: Some("//"),
        block_comment: Some(("/*", "*/")),
    },
    Language {
        name: "C#",
        extensions: &["cs"],
        line_comment: Some("//"),
        block_comment: Some(("/*", "*/")),
    },
    Language {
        name: "Ruby",
        extensions: &["rb"],
        line_comment: Some("#"),
        block_comment: Some(("=begin", "=end")),
    },
    Language {
        name: "Shell",
        extensions: &["sh", "bash", "zsh"],
        line_comment: Some("#"),
        block_comment: None,
    },
    Language {
        name: "YAML",
        extensions: &["yml", "yaml"],
        line_comment: Some("#"),
        block_comment: None,
    },
    Language {
        name: "TOML",
        extensions: &["toml"],
        line_comment: Some("#"),
        block_comment: None,
    },
    Language {
        name: "JSON",
        extensions: &["json"],
        line_comment: None,
        block_comment: None,
    },
    Language {
        name: "Markdown",
        extensions: &["md", "markdown"],
        line_comment: None,
        block_comment: None,
    },
    Language {
        name: "HTML",
        extensions: &["html", "htm"],
        line_comment: None,
        block_comment: Some(("<!--", "-->")),
    },
    Language {
        name: "CSS",
        extensions: &["css"],
        line_comment: None,
        block_comment: Some(("/*", "*/")),
    },
    Language {
        name: "SQL",
        extensions: &["sql"],
        line_comment: Some("--"),
        block_comment: Some(("/*", "*/")),
    },
    Language {
        name: "Kotlin",
        extensions: &["kt", "kts"],
        line_comment: Some("//"),
        block_comment: Some(("/*", "*/")),
    },
    Language {
        name: "Swift",
        extensions: &["swift"],
        line_comment: Some("//"),
        block_comment: Some(("/*", "*/")),
    },
    Language {
        name: "Zig",
        extensions: &["zig"],
        line_comment: Some("//"),
        block_comment: None,
    },
    Language {
        name: "Lua",
        extensions: &["lua"],
        line_comment: Some("--"),
        block_comment: Some(("--[[", "]]")),
    },
    Language {
        name: "Dockerfile",
        extensions: &["dockerfile"],
        line_comment: Some("#"),
        block_comment: None,
    },
];

fn get_language(path: &Path) -> Option<&'static Language> {
    // Special case for Dockerfile
    if path.file_name().map(|n| n.to_str()) == Some(Some("Dockerfile")) {
        return LANGUAGES.iter().find(|l| l.name == "Dockerfile");
    }

    let ext = path.extension()?.to_str()?;
    LANGUAGES.iter().find(|l| l.extensions.contains(&ext))
}

#[derive(Default, Clone)]
struct FileStats {
    code: usize,
    comments: usize,
    blanks: usize,
}

impl FileStats {
    fn total(&self) -> usize {
        self.code + self.comments + self.blanks
    }

    fn add(&mut self, other: &FileStats) {
        self.code += other.code;
        self.comments += other.comments;
        self.blanks += other.blanks;
    }
}

#[derive(Default)]
struct LangStats {
    files: usize,
    stats: FileStats,
}

struct Stats {
    by_lang: BTreeMap<&'static str, LangStats>,
    by_file: Vec<(PathBuf, &'static str, FileStats)>,
}

impl Stats {
    fn new() -> Self {
        Self {
            by_lang: BTreeMap::new(),
            by_file: Vec::new(),
        }
    }

    fn add(&mut self, path: PathBuf, lang: &'static str, stats: FileStats) {
        let entry = self.by_lang.entry(lang).or_default();
        entry.files += 1;
        entry.stats.add(&stats);
        self.by_file.push((path, lang, stats));
    }

    fn total(&self) -> (usize, FileStats) {
        let mut files = 0;
        let mut stats = FileStats::default();
        for lang_stats in self.by_lang.values() {
            files += lang_stats.files;
            stats.add(&lang_stats.stats);
        }
        (files, stats)
    }
}

fn count_file(path: &Path, lang: &Language) -> Option<FileStats> {
    let content = fs::read_to_string(path).ok()?;
    let mut stats = FileStats::default();
    let mut in_block = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.is_empty() {
            stats.blanks += 1;
            continue;
        }

        // Handle block comments
        if let Some((start, end)) = lang.block_comment {
            if in_block {
                stats.comments += 1;
                if trimmed.contains(end) {
                    in_block = false;
                }
                continue;
            }

            if trimmed.starts_with(start) {
                stats.comments += 1;
                if !trimmed[start.len()..].contains(end) {
                    in_block = true;
                }
                continue;
            }
        }

        // Handle line comments
        if let Some(comment) = lang.line_comment {
            if trimmed.starts_with(comment) {
                stats.comments += 1;
                continue;
            }
        }

        // Markdown special case: all content is prose, count as comments
        if lang.name == "Markdown" {
            stats.comments += 1;
            continue;
        }

        stats.code += 1;
    }

    Some(stats)
}

fn count_path(path: &Path, stats: &mut Stats) {
    if path.is_file() {
        if let Some(lang) = get_language(path) {
            if let Some(file_stats) = count_file(path, lang) {
                stats.add(path.to_path_buf(), lang.name, file_stats);
            }
        }
    } else if path.is_dir() {
        walk_dir(path, stats);
    }
}

fn walk_dir(dir: &Path, stats: &mut Stats) {
    let Ok(entries) = fs::read_dir(dir) else { return };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Skip hidden, target, node_modules, vendor, .git
        if name.starts_with('.')
            || name == "target"
            || name == "node_modules"
            || name == "vendor"
            || name == "dist"
            || name == "build"
        {
            continue;
        }

        if path.is_dir() {
            walk_dir(&path, stats);
        } else if let Some(lang) = get_language(&path) {
            if let Some(file_stats) = count_file(&path, lang) {
                stats.add(path.clone(), lang.name, file_stats);
            }
        }
    }
}

fn print_results(stats: &Stats, show_files: bool) {
    let (total_files, total_stats) = stats.total();

    if total_files == 0 {
        println!("No source files found.");
        return;
    }

    let sep = "─".repeat(65);

    // Per-file breakdown if requested
    if show_files {
        println!("{}", sep);
        println!("{:40} {:>8} {:>8} {:>6}", "File", "Code", "Comment", "Blank");
        println!("{}", sep);

        for (path, _lang, file_stats) in &stats.by_file {
            let display = path.display().to_string();
            let name = if display.len() > 40 {
                format!("...{}", &display[display.len() - 37..])
            } else {
                display
            };
            println!(
                "{:40} {:>8} {:>8} {:>6}",
                name, file_stats.code, file_stats.comments, file_stats.blanks
            );
        }
        println!("{}", sep);
        println!();
    }

    // Language summary
    println!("{}", sep);
    println!(
        "{:15} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "Language", "Files", "Code", "Comments", "Blanks", "Total"
    );
    println!("{}", sep);

    for (lang, lang_stats) in &stats.by_lang {
        println!(
            "{:15} {:>6} {:>10} {:>10} {:>10} {:>10}",
            lang,
            lang_stats.files,
            lang_stats.stats.code,
            lang_stats.stats.comments,
            lang_stats.stats.blanks,
            lang_stats.stats.total()
        );
    }

    println!("{}", sep);
    println!(
        "{:15} {:>6} {:>10} {:>10} {:>10} {:>10}",
        "Total",
        total_files,
        total_stats.code,
        total_stats.comments,
        total_stats.blanks,
        total_stats.total()
    );
    println!("{}", sep);
}
