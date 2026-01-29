//! Wyckoff Method Indicators
//!
//! Technical analysis indicators based on the Wyckoff Method, which analyzes
//! price and volume relationships to identify market phases (accumulation,
//! distribution, markup, markdown) and supply/demand dynamics.

pub mod wyckoff_wave;
pub mod optimism_pessimism;
pub mod force_index;
pub mod technometer;
pub mod selling_climax;
pub mod automatic_rally;
pub mod spring_upthrust;
pub mod sign_of_strength;

// Re-exports
pub use wyckoff_wave::{WyckoffWave, WyckoffWaveConfig, WyckoffWaveOutput};
pub use optimism_pessimism::{OptimismPessimismIndex, OptimismPessimismConfig};
pub use force_index::{WyckoffForceIndex, WyckoffForceIndexConfig};
pub use technometer::{Technometer, TechnometerConfig};
pub use selling_climax::{SellingClimaxDetector, SellingClimaxConfig, ClimaxEvent};
pub use automatic_rally::{AutomaticRallyReaction, AutomaticRallyConfig, RallyReactionEvent};
pub use spring_upthrust::{SpringUpthrust, SpringUpthrustConfig, SpringUpthrustOutput, SpringUpthrustEvent, FalseBreakoutType};
pub use sign_of_strength::{SignOfStrengthWeakness, SignOfStrengthConfig, SignOfStrengthOutput, SOSSOWType};
