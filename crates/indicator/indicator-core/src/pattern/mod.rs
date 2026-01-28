//! Pattern Recognition Indicators
//!
//! Pattern detection and candlestick analysis.

pub mod zigzag;
pub mod heikin_ashi;
pub mod darvas;
pub mod fractals;
pub mod doji;
pub mod hammer;
pub mod engulfing;
pub mod harami;
pub mod morning_star;
pub mod three_soldiers;
pub mod marubozu;
pub mod piercing;
pub mod spinning_top;
pub mod tweezer;
pub mod three_inside;
pub mod three_outside;
pub mod abandoned_baby;
pub mod belt_hold;
pub mod kicking;
pub mod three_line_strike;
pub mod tasuki_gap;
pub mod rising_falling_methods;
pub mod kase_bars;
pub mod minervini;
pub mod price_patterns;
pub mod extended;
pub mod r#final;
pub mod advanced;

// Re-exports
pub use zigzag::ZigZag;
pub use heikin_ashi::{HeikinAshi, HeikinAshiOutput};
pub use darvas::DarvasBox;
pub use fractals::Fractals;
pub use doji::Doji;
pub use hammer::Hammer;
pub use engulfing::Engulfing;
pub use harami::Harami;
pub use morning_star::MorningStar;
pub use three_soldiers::ThreeSoldiers;
pub use marubozu::Marubozu;
pub use piercing::Piercing;
pub use spinning_top::SpinningTop;
pub use tweezer::Tweezer;
pub use three_inside::ThreeInside;
pub use three_outside::ThreeOutside;
pub use abandoned_baby::AbandonedBaby;
pub use belt_hold::BeltHold;
pub use kicking::Kicking;
pub use three_line_strike::ThreeLineStrike;
pub use tasuki_gap::TasukiGap;
pub use rising_falling_methods::RisingFallingMethods;
pub use kase_bars::{KaseBars, KaseBarsOutput, KaseBarsStats};
pub use minervini::{
    TrendTemplate, VolatilityContractionPattern, PocketPivot,
    PowerPlay, BullFlag, CupPattern,
};
pub use price_patterns::{
    DoubleTop, DoubleBottom, HeadShoulders, Triangle, Channel, Wedge,
};
pub use extended::{
    GapAnalysis, InsideBar, OutsideBar, NarrowRange,
    WideRangeBar, TrendBar, ConsolidationPattern,
};
pub use r#final::{
    PriceMomentumPattern, RangeContractionExpansion,
};
pub use advanced::{
    TrendContinuationPattern, ReversalCandlePattern, VolumePricePattern,
    MomentumPattern, BreakoutPattern, ConsolidationBreak,
    PatternStrength, PatternProbability, MultiTimeframePattern,
    PatternCluster, SequentialPattern, PatternBreakoutStrength,
};
