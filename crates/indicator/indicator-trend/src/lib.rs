//! Technical Trend Indicators
//!
//! Indicators for identifying and measuring trend direction and strength.

pub mod aroon;
pub mod vortex;
pub mod alligator;
pub mod gator;
pub mod dpo;
pub mod coppock;
pub mod vhf;
pub mod rwi;
pub mod tii;
pub mod tdi;
pub mod rainbow;
pub mod mcginley;

// Re-exports
pub use aroon::{Aroon, AroonOutput};
pub use vortex::VortexIndicator;
pub use alligator::{Alligator, AlligatorOutput};
pub use gator::GatorOscillator;
pub use dpo::DPO;
pub use coppock::CoppockCurve;
pub use vhf::VerticalHorizontalFilter;
pub use rwi::RandomWalkIndex;
pub use tii::TrendIntensityIndex;
pub use tdi::TrendDetectionIndex;
pub use rainbow::RainbowMA;
pub use mcginley::McGinleyDynamic;

// Re-export SPI types
pub use indicator_spi::{
    TechnicalIndicator, SignalIndicator, IndicatorOutput, IndicatorSignal,
    IndicatorError, Result, OHLCV, OHLCVSeries,
};
