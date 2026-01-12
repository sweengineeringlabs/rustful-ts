//! Swing Trading and Price Structure Indicators
//!
//! Indicators for swing trading analysis and market structure identification.
//! Includes swing point detection, order blocks, fair value gaps, and structural analysis.

pub mod swing_index;
pub mod asi;
pub mod gann_swing;
pub mod market_structure;
pub mod order_blocks;
pub mod fair_value_gap;
pub mod liquidity_voids;
pub mod break_of_structure;
pub mod swing_points;
pub mod pivot_highs_lows;

// Re-exports
pub use swing_index::SwingIndex;
pub use asi::AccumulativeSwingIndex;
pub use gann_swing::GannSwing;
pub use market_structure::MarketStructure;
pub use order_blocks::OrderBlocks;
pub use fair_value_gap::FairValueGap;
pub use liquidity_voids::LiquidityVoids;
pub use break_of_structure::BreakOfStructure;
pub use swing_points::SwingPoints;
pub use pivot_highs_lows::PivotHighsLows;

// Re-export SPI types
pub use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCV, OHLCVSeries,
};
