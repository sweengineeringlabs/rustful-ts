//! Filter Indicators
//!
//! Signal filtering and noise reduction indicators including step-based filters.

pub mod kalman;
pub mod median;
pub mod gaussian;
pub mod svhma;
pub mod filtered_average;
pub mod step_vhf_vma;
pub mod extended;
pub mod final_pair;
pub mod advanced;

// Re-exports
pub use kalman::KalmanFilter;
pub use median::MedianFilter;
pub use gaussian::GaussianFilter;
pub use svhma::SVHMA;
pub use filtered_average::DeviationFilteredAverage;
pub use step_vhf_vma::StepVhfAdaptiveVMA;
pub use extended::{
    ExponentialSmoothingFilter, ButterworthFilter, HighPassFilter,
    BandPassFilter, AdaptiveNoiseFilter, TrendFilter,
};
pub use final_pair::{RecursiveFilter, NormalizedPriceFilter};
pub use advanced::{
    AdaptiveLowPassFilter, NoiseReductionFilter, TrendExtractionFilter,
    CycleExtractionFilter, AdaptiveHighPassFilter, BandwidthAdaptiveFilter,
    ButterworthBandpassFilter, ChebyshevFilter, WeightedMedianFilter,
    DoubleExponentialFilter, AdaptiveBandpassFilter, HodrickPrescottFilter,
    // 6 NEW filter indicators
    GaussianAdaptiveFilter, SavitzkyGolayFilter, TriangularFilter,
    HammingFilter, SuperSmootherFilter, DecyclerFilter,
    // 6 ADDITIONAL NEW filter indicators
    WienerFilter, KalmanSmoother, MovingMedianFilter,
    ExponentialSmoother, AdaptiveNoiseFilterLMS, TrendSeparationFilter,
};
