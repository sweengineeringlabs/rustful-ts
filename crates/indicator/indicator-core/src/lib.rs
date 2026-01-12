//! Technical Indicator Core Implementations
//!
//! SIMD-optimized technical analysis indicators.

// ============================================================================
// Moving Averages
// ============================================================================
pub mod sma;
pub mod ema;
pub mod wma;
pub mod dema;
pub mod tema;
pub mod hma;
pub mod kama;
pub mod zlema;
pub mod smma;
pub mod alma;
pub mod frama;
pub mod vidya;
pub mod t3;
pub mod triangular;
pub mod gmma;
pub mod sine_wma;

// ============================================================================
// Filters
// ============================================================================
pub mod kalman;
pub mod median;
pub mod gaussian;

// ============================================================================
// Momentum/Oscillators
// ============================================================================
pub mod rsi;
pub mod stochastic;
pub mod williams_r;
pub mod cci;
pub mod trix;
pub mod ultimate_oscillator;

// ============================================================================
// Trend Indicators
// ============================================================================
pub mod macd;
pub mod adx;
pub mod ichimoku;
pub mod supertrend;
pub mod parabolic_sar;

// ============================================================================
// Volatility Indicators
// ============================================================================
pub mod bollinger;
pub mod atr;
pub mod donchian;
pub mod keltner;

// ============================================================================
// Volume Indicators
// ============================================================================
pub mod vwap;
pub mod obv;
pub mod mfi;
pub mod cmf;

// ============================================================================
// Support/Resistance
// ============================================================================
pub mod pivot_points;
pub mod fibonacci;

// ============================================================================
// Other
// ============================================================================
pub mod elder_ray;

#[cfg(feature = "simd")]
pub mod simd;

// ============================================================================
// Extended Indicator Categories
// ============================================================================
pub mod oscillators;
pub mod trend;
pub mod volatility;
pub mod volume;
pub mod statistical;
pub mod pattern;
pub mod risk;
pub mod bands;
pub mod dsp;
pub mod composite;
pub mod breadth;
pub mod swing;
pub mod demark;

// ============================================================================
// Moving Average Exports
// ============================================================================
pub use sma::SMA;
pub use ema::EMA;
pub use wma::WMA;
pub use dema::DEMA;
pub use tema::TEMA;
pub use hma::HMA;
pub use kama::KAMA;
pub use zlema::ZLEMA;
pub use smma::SMMA;
pub use alma::ALMA;
pub use frama::FRAMA;
pub use vidya::VIDYA;
pub use t3::T3;
pub use triangular::TRIMA;
pub use gmma::{GMMA, GMMAOutput};
pub use sine_wma::SineWMA;

// ============================================================================
// Filter Exports
// ============================================================================
pub use kalman::KalmanFilter;
pub use median::MedianFilter;
pub use gaussian::GaussianFilter;

// ============================================================================
// Oscillator Exports
// ============================================================================
pub use rsi::RSI;
pub use stochastic::Stochastic;
pub use williams_r::WilliamsR;
pub use cci::CCI;
pub use trix::TRIX;
pub use ultimate_oscillator::UltimateOscillator;

// ============================================================================
// Trend Exports
// ============================================================================
pub use macd::MACD;
pub use adx::{ADX, ADXOutput};
pub use ichimoku::{Ichimoku, IchimokuOutput};
pub use supertrend::{SuperTrend, SuperTrendOutput};
pub use parabolic_sar::{ParabolicSAR, ParabolicSAROutput};

// ============================================================================
// Volatility Exports
// ============================================================================
pub use bollinger::BollingerBands;
pub use atr::ATR;
pub use donchian::DonchianChannels;
pub use keltner::KeltnerChannels;

// ============================================================================
// Volume Exports
// ============================================================================
pub use vwap::VWAP;
pub use obv::OBV;
pub use mfi::MFI;
pub use cmf::CMF;

// ============================================================================
// Support/Resistance Exports
// ============================================================================
pub use pivot_points::{PivotPoints, PivotPointsResult};
pub use fibonacci::{Fibonacci, FibonacciLevels};

// ============================================================================
// Other Exports
// ============================================================================
pub use elder_ray::{ElderRay, ElderRayOutput};

// Re-export SPI types
pub use indicator_spi::{
    TechnicalIndicator, StreamingIndicator, SignalIndicator,
    IndicatorOutput, IndicatorSignal, IndicatorError, Result,
    OHLCV, OHLCVSeries,
};

// Re-export API configs
pub use indicator_api::{
    // Moving Averages
    SMAConfig, EMAConfig, WMAConfig, DEMAConfig, TEMAConfig,
    HMAConfig, KAMAConfig, ZLEMAConfig,
    SMMAConfig, ALMAConfig, FRAMAConfig, VIDYAConfig,
    T3Config, TRIMAConfig, GMMAConfig, SineWMAConfig,
    // Filters
    KalmanConfig, MedianConfig, GaussianConfig,
    // Oscillators
    RSIConfig, StochasticConfig, WilliamsRConfig, CCIConfig,
    TRIXConfig, UltimateOscillatorConfig,
    // Trend
    MACDConfig, ADXConfig, IchimokuConfig, SuperTrendConfig, ParabolicSARConfig,
    // Volatility
    BollingerConfig, ATRConfig, DonchianConfig, KeltnerConfig,
    // Volume
    VWAPConfig, OBVConfig, MFIConfig, CMFConfig,
    // Other
    ElderRayConfig, PivotType,
};

// ============================================================================
// Extended Category Re-exports
// ============================================================================

// Oscillators
pub use oscillators::{
    ROC, Momentum, ChandeMomentum, DeMarker, AwesomeOscillator, AcceleratorOscillator,
    KST, PPO, RVI, StochasticRSI, ConnorsRSI, TSI, SMI, RMI, FisherTransform,
    InverseFisherTransform, Qstick, PMO, SpecialK, DisparityIndex, PrettyGoodOscillator,
    APO, ErgodicOscillator, PolarizedFractalEfficiency, IntradayMomentumIndex,
    RelativeVolatilityIndex, DoubleStochastic,
};

// Trend
pub use trend::{
    Alligator, AlligatorOutput, Aroon, AroonOutput, CoppockCurve, DPO, GatorOscillator,
    McGinleyDynamic, RainbowMA, RandomWalkIndex, TrendDetectionIndex, TrendIntensityIndex,
    VerticalHorizontalFilter, VortexIndicator,
};

// Volatility
pub use volatility::{
    HistoricalVolatility, ChaikinVolatility, MassIndex, ParkinsonVolatility,
    GarmanKlassVolatility, RogersSatchellVolatility, YangZhangVolatility,
    RealizedVolatility, NormalizedATR, ChoppinessIndex, UlcerIndex,
};

// Volume
pub use volume::{
    VWMA, ADLine, ForceIndex, KlingerOscillator, BalanceOfPower, EaseOfMovement,
    VROC, PVT, NVI, PVI, WilliamsAD, TwiggsMoneyFlow, VolumeOscillator, NetVolume,
    ChaikinOscillator, TWAP,
};

// Statistical
pub use statistical::{
    StandardDeviation, Variance, ZScore, LinearRegression, LinearRegressionOutput,
    Correlation, Spread, Ratio, ZScoreSpread, Autocorrelation, Skewness, Kurtosis,
};

// Pattern
pub use pattern::{
    ZigZag, HeikinAshi, HeikinAshiOutput, DarvasBox, Fractals, Doji, Hammer, Engulfing,
    Harami, MorningStar, ThreeSoldiers, Marubozu, Piercing, SpinningTop, Tweezer,
    ThreeInside, ThreeOutside, AbandonedBaby, BeltHold, Kicking, ThreeLineStrike,
    TasukiGap, RisingFallingMethods,
};

// Risk
pub use risk::{
    SharpeRatio, SortinoRatio, CalmarRatio, MaxDrawdown, ValueAtRisk, VaRMethod,
    ConditionalVaR, Beta, Alpha, TreynorRatio, InformationRatio, OmegaRatio, GainLossRatio,
};

// Bands
pub use bands::{
    AccelerationBands, ChandelierExit, Envelope, HighLowBands, PriceChannel,
    ProjectionBands, STARCBands, StandardErrorBands, TironeLevels, TironeLevelsOutput,
};

// DSP
pub use dsp::{
    MESA, MAMA, SineWave, HilbertTransform, CyberCycle, CGOscillator,
    LaguerreRSI, RoofingFilter, Supersmoother, Decycler,
};

// Composite
pub use composite::{
    TTMSqueeze, TTMSqueezeConfig, TTMSqueezeOutput,
    ElderImpulse, ElderImpulseConfig, ElderImpulseOutput,
    SchaffTrendCycle, SchaffConfig, SchaffOutput,
    ElderTripleScreen, ElderTripleScreenConfig, ElderTripleScreenOutput,
    CommoditySelectionIndex, CommoditySelectionConfig, CommoditySelectionOutput,
    SqueezeMomentum, SqueezeMomentumConfig, SqueezeMomentumOutput,
    TrendStrengthIndex, TrendStrengthConfig, TrendStrengthOutput, TrendComponents,
    RegimeDetector, RegimeDetectorConfig, RegimeDetectorOutput, MarketRegime,
};

// Breadth
pub use breadth::{
    AdvanceDeclineLine, BreadthThrust, CumulativeVolumeIndex, UpDownVolume,
    HighLowData, HighLowIndex, HighLowMethod, HighLowSeries,
    McClellanOscillator, McClellanSummationIndex,
    MarketCondition, PercentAboveMA, PercentAboveMASeries,
    ContrarianSignal, PutCallRatio, PutCallSeries, PutCallSignal,
    TickBias, TickIndex, TickSeries, TickSignal, TickStats,
    TRINSignal, TRIN, BreadthData, BreadthSeries, BreadthIndicator,
};

// Swing
pub use swing::{
    SwingIndex, AccumulativeSwingIndex, GannSwing, GannSwingState,
    MarketStructure, MarketTrend, StructurePoint,
    OrderBlocks, OrderBlock, OrderBlockType,
    FairValueGap, FVGType, FVGZone,
    LiquidityVoids, LiquidityVoid, LiquidityVoidType,
    BreakOfStructure, BOSType, BOSEvent, CHoCHType,
    SwingPoints, SwingPoint, SwingPointType,
    PivotHighsLows, PivotPoint, PivotType as SwingPivotType,
};

// DeMark
pub use demark::{
    TDSetup, TDSetupOutput, TDSetupConfig, SetupPhase,
    TDCountdown, TDCountdownOutput, TDCountdownConfig, CountdownPhase,
    TDSequential, TDSequentialOutput, TDSequentialConfig, SequentialState, SignalStrength,
    TDCombo, TDComboOutput, TDComboConfig, ComboPhase, ComboState,
    TDREI, TDREIOutput, TDREIConfig,
    TDPOQ, TDPOQOutput, TDPOQConfig,
    TDPressure, TDPressureOutput, TDPressureConfig,
    TDDWave, TDDWaveOutput, TDDWaveConfig, WaveDirection, DWavePhase, PivotType as DWavePivotType,
    TDTrendFactor, TDTrendFactorOutput, TDTrendFactorConfig, TrendState,
};
