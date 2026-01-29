//! Harmonic Pattern Indicators
//!
//! Harmonic patterns use Fibonacci ratios to identify potential reversal points.
//! These patterns are based on geometric price structures that repeat in the markets.

pub mod butterfly;
pub mod bat;
pub mod crab;
pub mod shark;
pub mod cypher;
pub mod abcd;

// Re-exports
pub use butterfly::{ButterflyPattern, ButterflyPatternConfig, ButterflyPatternOutput};
pub use bat::{BatPattern, BatPatternConfig, BatPatternOutput};
pub use crab::{CrabPattern, CrabPatternConfig, CrabPatternOutput};
pub use shark::{SharkPattern, SharkPatternConfig, SharkPatternOutput};
pub use cypher::{CypherPattern, CypherPatternConfig, CypherPatternOutput};
pub use abcd::{ABCDPattern, ABCDPatternConfig, ABCDPatternOutput};
