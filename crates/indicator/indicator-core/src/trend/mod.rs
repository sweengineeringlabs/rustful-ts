//! Technical Trend Indicators
//!
//! Indicators for identifying and measuring trend direction and strength.

pub mod alligator;
pub mod aroon;
pub mod coppock;
pub mod dpo;
pub mod gator;
pub mod mcginley;
pub mod rainbow;
pub mod rwi;
pub mod tdi;
pub mod tii;
pub mod vhf;
pub mod vortex;

// Re-exports
pub use alligator::{Alligator, AlligatorOutput};
pub use aroon::{Aroon, AroonOutput};
pub use coppock::CoppockCurve;
pub use dpo::DPO;
pub use gator::GatorOscillator;
pub use mcginley::McGinleyDynamic;
pub use rainbow::RainbowMA;
pub use rwi::RandomWalkIndex;
pub use tdi::TrendDetectionIndex;
pub use tii::TrendIntensityIndex;
pub use vhf::VerticalHorizontalFilter;
pub use vortex::VortexIndicator;
