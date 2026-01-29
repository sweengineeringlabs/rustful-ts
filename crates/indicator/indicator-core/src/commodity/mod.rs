//! Commodity Indicators
//!
//! Indicators for analyzing commodity market dynamics including futures curve shape,
//! calendar spreads, storage economics, and processing margins.

pub mod contango_backwardation;
pub mod roll_yield;
pub mod convenience_yield;
pub mod inventory_levels;
pub mod crack_spread;
pub mod crush_spread;

// Re-exports
pub use contango_backwardation::{ContangoBackwardation, ContangoBackwardationOutput, CurveShape};
pub use roll_yield::{RollYield, RollYieldOutput};
pub use convenience_yield::{ConvenienceYield, ConvenienceYieldOutput};
pub use inventory_levels::{InventoryLevels, InventoryLevelsOutput, InventorySignal};
pub use crack_spread::{CrackSpread, CrackSpreadOutput};
pub use crush_spread::{CrushSpread, CrushSpreadOutput};
