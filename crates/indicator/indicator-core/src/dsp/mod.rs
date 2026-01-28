//! Ehlers DSP-Based Technical Indicators
//!
//! Digital Signal Processing (DSP) indicators developed by John Ehlers.
//! These indicators apply signal processing techniques to financial data,
//! including adaptive moving averages, cycle detection, and noise filtering.

pub mod mesa;
pub mod mama;
pub mod sine_wave;
pub mod hilbert;
pub mod cyber_cycle;
pub mod cg_oscillator;
pub mod laguerre_rsi;
pub mod roofing;
pub mod supersmoother;
pub mod decycler;
pub mod cycles;
pub mod extended;

// Re-exports
pub use mesa::MESA;
pub use mama::MAMA;
pub use sine_wave::SineWave;
pub use hilbert::HilbertTransform;
pub use cyber_cycle::CyberCycle;
pub use cg_oscillator::CGOscillator;
pub use laguerre_rsi::LaguerreRSI;
pub use roofing::RoofingFilter;
pub use supersmoother::Supersmoother;
pub use decycler::Decycler;
pub use cycles::{
    DominantCyclePeriod, CycleAmplitude, CyclePhase,
    TrendCycleDecomposition, CycleMomentum, CycleTurningPoint,
};
pub use extended::{
    SpectralDensity, PhaseIndicator, InstantaneousFrequency,
    AdaptiveBandwidthFilter, ZeroLagIndicator, SignalToNoiseRatio,
};
