//! Fixed Income & Rates Indicators
//!
//! Indicators for analyzing interest rates, yield curves, and fixed income markets.
//! These indicators help identify economic conditions and potential regime changes
//! through yield curve analysis.

pub mod yield_curve_slope;
pub mod yield_curve_curvature;
pub mod real_yield;
pub mod credit_spread;
pub mod ted_spread;
pub mod swap_spread;
pub mod duration;
pub mod convexity;
pub mod carry;

// Re-exports
pub use yield_curve_slope::YieldCurveSlope;
pub use yield_curve_curvature::{YieldCurveCurvature, CurveShape};
pub use real_yield::{RealYield, RealYieldEnvironment};
pub use credit_spread::{CreditSpread, CreditSpreadConfig};
pub use ted_spread::{TEDSpread, TEDSpreadConfig, TEDSpreadLevel};
pub use swap_spread::{SwapSpread, SwapSpreadConfig, SwapSpreadCondition};
pub use duration::{Duration, DurationConfig};
pub use convexity::{Convexity, ConvexityConfig};
pub use carry::{Carry, CarryConfig, CarryOutput};
