//! Data loading service for CSV and JSON files.

#![allow(dead_code)] // Will be used when file loading UI is implemented

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use crate::app::TimeSeriesData;

/// Error type for data loading operations.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Failed to read file: {0}")]
    ReadError(String),

    #[error("Failed to parse CSV: {0}")]
    CsvError(String),

    #[error("Failed to parse JSON: {0}")]
    JsonError(String),

    #[error("No numeric data found in file")]
    NoNumericData,
}

/// Load time series data from a CSV file.
pub fn load_csv_file(path: &Path, column: Option<&str>) -> Result<TimeSeriesData, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::FileNotFound(e.to_string()))?;
    let mut reader = csv::Reader::from_reader(BufReader::new(file));

    let headers = reader
        .headers()
        .map_err(|e| LoadError::CsvError(e.to_string()))?
        .clone();

    // Find the target column
    let col_idx = if let Some(col_name) = column {
        // Try to find by name
        headers
            .iter()
            .position(|h| h == col_name)
            .or_else(|| col_name.parse::<usize>().ok())
            .ok_or_else(|| LoadError::CsvError(format!("Column '{}' not found", col_name)))?
    } else {
        // Find first numeric column
        find_first_numeric_column(&mut reader, &headers)?
    };

    let column_name = headers.get(col_idx).map(String::from);

    // Re-read the file to get values
    let file = File::open(path).map_err(|e| LoadError::FileNotFound(e.to_string()))?;
    let mut reader = csv::Reader::from_reader(BufReader::new(file));

    let mut values = Vec::new();
    let mut timestamps = Vec::new();

    for result in reader.records() {
        let record = result.map_err(|e| LoadError::CsvError(e.to_string()))?;
        if let Some(field) = record.get(col_idx) {
            if let Ok(val) = field.parse::<f64>() {
                values.push(val);
                // Try to get timestamp from first column if it's different
                if col_idx > 0 {
                    if let Some(ts) = record.get(0) {
                        timestamps.push(ts.to_string());
                    }
                }
            }
        }
    }

    if values.is_empty() {
        return Err(LoadError::NoNumericData);
    }

    let mut data = TimeSeriesData::new(values, path.to_path_buf());
    if let Some(name) = column_name {
        data = data.with_column_name(name);
    }
    if !timestamps.is_empty() && timestamps.len() == data.values.len() {
        data = data.with_timestamps(timestamps);
    }

    Ok(data)
}

/// Load time series data from a JSON file.
pub fn load_json_file(path: &Path, column: Option<&str>) -> Result<TimeSeriesData, LoadError> {
    let file = File::open(path).map_err(|e| LoadError::FileNotFound(e.to_string()))?;
    let reader = BufReader::new(file);

    let json: serde_json::Value =
        serde_json::from_reader(reader).map_err(|e| LoadError::JsonError(e.to_string()))?;

    let values = extract_values_from_json(&json, column)?;

    if values.is_empty() {
        return Err(LoadError::NoNumericData);
    }

    let mut data = TimeSeriesData::new(values, path.to_path_buf());
    if let Some(col) = column {
        data = data.with_column_name(col.to_string());
    }

    Ok(data)
}

fn find_first_numeric_column(
    reader: &mut csv::Reader<BufReader<File>>,
    _headers: &csv::StringRecord,
) -> Result<usize, LoadError> {
    // Read first data row to determine types
    if let Some(result) = reader.records().next() {
        let record = result.map_err(|e| LoadError::CsvError(e.to_string()))?;
        for (i, field) in record.iter().enumerate() {
            if field.parse::<f64>().is_ok() {
                return Ok(i);
            }
        }
    }
    Err(LoadError::NoNumericData)
}

fn extract_values_from_json(
    json: &serde_json::Value,
    column: Option<&str>,
) -> Result<Vec<f64>, LoadError> {
    // Try different JSON structures
    let values = if let Some(col) = column {
        // Look for specific key
        json.get(col)
            .or_else(|| json.get("data").and_then(|d| d.get(col)))
    } else {
        // Try common key names
        json.get("values")
            .or_else(|| json.get("data"))
            .or_else(|| json.get("y"))
            .or_else(|| json.get("series"))
            .or_else(|| {
                // If it's an array at root level, use it directly
                if json.is_array() {
                    Some(json)
                } else {
                    None
                }
            })
    };

    match values {
        Some(serde_json::Value::Array(arr)) => {
            let nums: Vec<f64> = arr
                .iter()
                .filter_map(|v| {
                    v.as_f64()
                        .or_else(|| v.as_i64().map(|i| i as f64))
                        .or_else(|| v.get("value").and_then(|vv| vv.as_f64()))
                        .or_else(|| v.get("y").and_then(|vv| vv.as_f64()))
                })
                .collect();
            if nums.is_empty() {
                Err(LoadError::NoNumericData)
            } else {
                Ok(nums)
            }
        }
        _ => Err(LoadError::NoNumericData),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_load_csv() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "date,value").unwrap();
        writeln!(file, "2024-01-01,100.5").unwrap();
        writeln!(file, "2024-01-02,101.2").unwrap();
        writeln!(file, "2024-01-03,99.8").unwrap();

        let data = load_csv_file(file.path(), None).unwrap();
        assert_eq!(data.values.len(), 3);
        assert!((data.values[0] - 100.5).abs() < 0.01);
    }

    #[test]
    fn test_load_json_array() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"values": [1.0, 2.0, 3.0, 4.0, 5.0]}}"#).unwrap();

        let data = load_json_file(file.path(), None).unwrap();
        assert_eq!(data.values.len(), 5);
    }
}
