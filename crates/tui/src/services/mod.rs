//! Service layer for TUI operations.

mod data_loader;

// These will be used when file loading is fully implemented
#[allow(unused_imports)]
pub use data_loader::{load_csv_file, load_json_file, LoadError};
