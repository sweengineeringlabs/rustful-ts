//! Band and Channel Indicators
//!
//! Price band and channel indicators.

pub mod bollinger;
pub mod keltner;
pub mod donchian;
pub mod acceleration_bands;
pub mod chandelier;
pub mod envelope;
pub mod high_low_bands;
pub mod price_channel;
pub mod projection_bands;
pub mod starc;
pub mod std_error_bands;
pub mod tirone;
pub mod percentage_bands;
pub mod vwap_bands;
pub mod auto_envelope;
pub mod elder_auto_envelope;
pub mod fractal_chaos_bands;
pub mod prime_number_bands;
pub mod donchian_ext;
pub mod extended;
pub mod advanced;

// Re-exports
pub use bollinger::BollingerBands;
pub use keltner::KeltnerChannels;
pub use donchian::DonchianChannels;
pub use acceleration_bands::AccelerationBands;
pub use chandelier::ChandelierExit;
pub use envelope::Envelope;
pub use high_low_bands::HighLowBands;
pub use price_channel::PriceChannel;
pub use projection_bands::ProjectionBands;
pub use starc::STARCBands;
pub use std_error_bands::StandardErrorBands;
pub use tirone::{TironeLevels, TironeLevelsOutput};
pub use percentage_bands::PercentageBands;
pub use vwap_bands::VWAPBands;
pub use auto_envelope::AutoEnvelope;
pub use elder_auto_envelope::ElderAutoEnvelope;
pub use fractal_chaos_bands::FractalChaosBands;
pub use prime_number_bands::PrimeNumberBands;
pub use donchian_ext::{
    DonchianWidth, FourWeekRule, TurtleEntry, TurtleEntryOutput,
    TurtleExit, DualBreakout, DonchianMiddleCross,
};
pub use extended::{
    AdaptiveBands, FixedPercentageEnvelope, MomentumBands,
    VolumeWeightedBands, DynamicChannel, LinearRegressionChannel,
};
pub use advanced::{
    VolatilityBands, TrendBands, MomentumBandsAdvanced,
    PriceEnvelope, DynamicPriceChannel, RangeBands,
    AdaptiveBandsSystem, TrendAwareBands, VolatilityAdjustedBands,
    CycleBands, DynamicEnvelope,
    AdaptiveKeltnerChannels, VolatilityWeightedBands, TrendFollowingChannel,
    DynamicSupportResistanceBands, MomentumBandwidth, PriceEnvelopeOscillator,
    // New band indicators (Jan 2026)
    VolatilityBandwidth, BandBreakoutStrength, DynamicPriceBands,
    TrendAlignedBands, MomentumDrivenBands, AdaptiveEnvelopeBands,
    // New band indicators (Jan 2026 - Phase 2)
    PricePercentileBands, VolumeBands, ATRBands,
    AdaptiveChannelBands, RegressionBands, QuantileBands,
};
