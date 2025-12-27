//! Real-time monitoring

use super::detectors::{AnomalyDetector, AnomalyResult};
use super::alerting::Alert;
use rustful_core::Result;

/// Real-time monitor for streaming data
pub struct Monitor<D: AnomalyDetector> {
    detector: D,
    buffer: Vec<f64>,
    buffer_size: usize,
}

impl<D: AnomalyDetector> Monitor<D> {
    pub fn new(detector: D, buffer_size: usize) -> Self {
        Self {
            detector,
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
        }
    }

    /// Push a new value and check for anomalies
    pub fn push(&mut self, value: f64) -> Result<Option<Alert>> {
        self.buffer.push(value);
        if self.buffer.len() > self.buffer_size {
            self.buffer.remove(0);
        }

        if self.buffer.len() >= self.buffer_size {
            let result = self.detector.detect(&self.buffer)?;
            if let Some(&is_anomaly) = result.is_anomaly.last() {
                if is_anomaly {
                    let score = result.scores.last().copied().unwrap_or(0.0);
                    return Ok(Some(Alert::new(value, score)));
                }
            }
        }
        Ok(None)
    }

    /// Get current buffer contents
    pub fn buffer(&self) -> &[f64] {
        &self.buffer
    }

    /// Clear the buffer
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}
