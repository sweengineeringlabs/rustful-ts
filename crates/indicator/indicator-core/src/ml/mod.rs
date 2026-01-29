//! Machine Learning-Based Indicators
//!
//! Simplified/proxy versions of ML-based indicators that capture the essence
//! of machine learning approaches using technical indicators. These are not
//! full ML implementations but rather heuristic approximations that mimic
//! the behavior patterns of their namesake algorithms.

pub mod svm_trend;
pub mod autoencoder_anomaly;
pub mod hmm_regime;
pub mod rl_signal;

// Re-exports
pub use svm_trend::{SVMTrendClassifier, SVMTrendClassifierConfig, SVMTrendOutput, SVMTrendClass};
pub use autoencoder_anomaly::AutoencoderAnomaly;
pub use hmm_regime::{HMMRegime, MarketRegimeState};
pub use rl_signal::{RLSignal, RLAction};
