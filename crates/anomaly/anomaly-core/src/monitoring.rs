//! Real-time monitoring implementation.

use anomaly_api::MonitorConfig;
use anomaly_spi::{Alert, AnomalyDetector, MonitoringStream, Result};

use super::alerting::create_alert;

/// Real-time monitor for streaming anomaly detection.
pub struct Monitor<D: AnomalyDetector> {
    detector: D,
    buffer: Vec<f64>,
    buffer_size: usize,
}

impl<D: AnomalyDetector> Monitor<D> {
    /// Create a new monitor with the given detector and buffer size.
    pub fn new(detector: D, buffer_size: usize) -> Self {
        Self {
            detector,
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
        }
    }

    /// Create from configuration.
    pub fn from_config(detector: D, config: MonitorConfig) -> Self {
        Self::new(detector, config.buffer_size)
    }

    /// Get the underlying detector.
    pub fn detector(&self) -> &D {
        &self.detector
    }

    /// Get mutable reference to the detector.
    pub fn detector_mut(&mut self) -> &mut D {
        &mut self.detector
    }
}

impl<D: AnomalyDetector> MonitoringStream<D> for Monitor<D> {
    fn push(&mut self, value: f64) -> Result<Option<Alert>> {
        self.buffer.push(value);
        if self.buffer.len() > self.buffer_size {
            self.buffer.remove(0);
        }

        if self.buffer.len() >= self.buffer_size {
            let result = self.detector.detect(&self.buffer)?;
            if let Some(&is_anomaly) = result.is_anomaly.last() {
                if is_anomaly {
                    let score = result.scores.last().copied().unwrap_or(0.0);
                    return Ok(Some(create_alert(value, score)));
                }
            }
        }
        Ok(None)
    }

    fn buffer(&self) -> &[f64] {
        &self.buffer
    }

    fn reset(&mut self) {
        self.buffer.clear();
    }
}
