//! Support and Resistance Indicators
//!
//! Price level identification including pivot points and Fibonacci levels.

pub mod pivot_points;
pub mod fibonacci;
pub mod extended;
pub mod fib_fans;
pub mod fib_channels;
pub mod fib_clusters;
pub mod auto_fibonacci;
pub mod fib_speed_resistance;

// Re-exports
pub use pivot_points::PivotPoints;
pub use fibonacci::{Fibonacci, FibonacciLevels};
pub use extended::{
    DynamicSupportResistance, PriceClusters, VolumeSupportResistance,
    SwingLevelDetector, TrendlineBreak, PsychologicalLevels,
};
pub use fib_fans::{FibonacciFans, FibFanLevels};
pub use fib_channels::{FibonacciChannels, FibChannelLevels, ChannelZone};
pub use fib_clusters::{FibonacciClusters, ClusterZone};
pub use auto_fibonacci::{AutoFibonacci, AutoFibLevels, FibRetraceLevels, FibExtensionLevels, SwingPoint as FibSwingPoint, SwingType as FibSwingType};
pub use fib_speed_resistance::{FibonacciSpeedResistance, SpeedResistanceArc, SpeedFanLines, SpeedZone};
