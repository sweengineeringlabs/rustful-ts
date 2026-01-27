//! Technical Indicator Core Implementations
//!
//! SIMD-optimized technical analysis indicators organized by category.

// ============================================================================
// Category Modules
// ============================================================================
pub mod moving_averages;
pub mod filters;
pub mod oscillators;
pub mod trend;
pub mod volatility;
pub mod volume;
pub mod bands;
pub mod support_resistance;
pub mod statistical;
pub mod pattern;
pub mod risk;
pub mod dsp;
pub mod composite;
pub mod breadth;
pub mod swing;
pub mod demark;

#[cfg(feature = "simd")]
pub mod simd;

// ============================================================================
// Moving Average Exports
// ============================================================================
pub use moving_averages::{
    SMA, EMA, WMA, DEMA, TEMA, HMA, KAMA, ZLEMA, SMMA, ALMA,
    FRAMA, VIDYA, T3, TRIMA, GMMA, SineWMA,
};

// ============================================================================
// Filter Exports
// ============================================================================
pub use filters::{
    KalmanFilter, MedianFilter, GaussianFilter, SVHMA,
    DeviationFilteredAverage, StepVhfAdaptiveVMA,
};

// ============================================================================
// Oscillator Exports
// ============================================================================
pub use oscillators::{
    RSI, Stochastic, WilliamsR, CCI, TRIX, UltimateOscillator,
    ROC, Momentum, ChandeMomentum, DeMarker, AwesomeOscillator, AcceleratorOscillator,
    KST, PPO, RVI, StochasticRSI, ConnorsRSI, TSI, SMI, RMI, FisherTransform,
    InverseFisherTransform, Qstick, PMO, SpecialK, DisparityIndex, PrettyGoodOscillator,
    APO, ErgodicOscillator, PolarizedFractalEfficiency, IntradayMomentumIndex,
    RelativeVolatilityIndex, DoubleStochastic,
};

// ============================================================================
// Trend Exports
// ============================================================================
pub use trend::{
    MACD, ADX, Ichimoku, IchimokuOutput, SuperTrend, ParabolicSAR,
    Alligator, AlligatorOutput, Aroon, AroonOutput, CoppockCurve, DPO,
    EfficiencyRatio, GatorOscillator, KaseCD,
    McGinleyDynamic, RainbowMA, RandomWalkIndex, TrendDetectionIndex, TrendIntensityIndex,
    VerticalHorizontalFilter, VortexIndicator,
};

// ============================================================================
// Volatility Exports
// ============================================================================
pub use volatility::{
    ATR, HistoricalVolatility, ChaikinVolatility, MassIndex, ParkinsonVolatility,
    GarmanKlassVolatility, RogersSatchellVolatility, YangZhangVolatility,
    RealizedVolatility, NormalizedATR, ChoppinessIndex, UlcerIndex,
};

// ============================================================================
// Volume Exports
// ============================================================================
pub use volume::{
    VWAP, OBV, MFI, CMF, VWMA, ADLine, ForceIndex, KlingerOscillator,
    BalanceOfPower, EaseOfMovement, VROC, PVT, NVI, PVI, WilliamsAD,
    TwiggsMoneyFlow, VolumeOscillator, NetVolume, ChaikinOscillator, TWAP,
};

// ============================================================================
// Bands Exports
// ============================================================================
pub use bands::{
    BollingerBands, KeltnerChannels, DonchianChannels,
    AccelerationBands, ChandelierExit, Envelope, HighLowBands, PriceChannel,
    ProjectionBands, STARCBands, StandardErrorBands, TironeLevels, TironeLevelsOutput,
};

// ============================================================================
// Support/Resistance Exports
// ============================================================================
pub use support_resistance::{PivotPoints, Fibonacci, FibonacciLevels};

// ============================================================================
// Statistical Exports
// ============================================================================
pub use statistical::{
    StandardDeviation, Variance, ZScore, LinearRegression, LinearRegressionOutput,
    Correlation, Spread, Ratio, ZScoreSpread, Autocorrelation, Skewness, Kurtosis,
};

// ============================================================================
// Pattern Exports
// ============================================================================
pub use pattern::{
    ZigZag, HeikinAshi, HeikinAshiOutput, DarvasBox, Fractals, Doji, Hammer, Engulfing,
    Harami, MorningStar, ThreeSoldiers, Marubozu, Piercing, SpinningTop, Tweezer,
    ThreeInside, ThreeOutside, AbandonedBaby, BeltHold, Kicking, ThreeLineStrike,
    TasukiGap, RisingFallingMethods,
};

// ============================================================================
// Risk Exports
// ============================================================================
pub use risk::{
    SharpeRatio, SortinoRatio, CalmarRatio, MaxDrawdown, ValueAtRisk, VaRMethod,
    ConditionalVaR, Beta, Alpha, TreynorRatio, InformationRatio, OmegaRatio, GainLossRatio,
};

// ============================================================================
// DSP Exports
// ============================================================================
pub use dsp::{
    MESA, MAMA, SineWave, HilbertTransform, CyberCycle, CGOscillator,
    LaguerreRSI, RoofingFilter, Supersmoother, Decycler,
};

// ============================================================================
// Composite Exports
// ============================================================================
pub use composite::{
    TTMSqueeze, TTMSqueezeConfig, TTMSqueezeOutput,
    ElderImpulse, ElderImpulseConfig, ElderImpulseOutput,
    ElderRay,
    SchaffTrendCycle, SchaffConfig, SchaffOutput,
    ElderTripleScreen, ElderTripleScreenConfig, ElderTripleScreenOutput,
    CommoditySelectionIndex, CommoditySelectionConfig, CommoditySelectionOutput,
    SqueezeMomentum, SqueezeMomentumConfig, SqueezeMomentumOutput,
    TrendStrengthIndex, TrendStrengthConfig, TrendStrengthOutput, TrendComponents,
    RegimeDetector, RegimeDetectorConfig, RegimeDetectorOutput, MarketRegime,
};

// ============================================================================
// Breadth Exports
// ============================================================================
pub use breadth::{
    AdvanceDeclineLine, BreadthThrust, CumulativeVolumeIndex, UpDownVolume,
    HighLowData, HighLowIndex, HighLowMethod, HighLowSeries,
    McClellanOscillator, McClellanSummationIndex,
    MarketCondition, PercentAboveMA, PercentAboveMASeries,
    ContrarianSignal, PutCallRatio, PutCallSeries, PutCallSignal,
    TickBias, TickIndex, TickSeries, TickSignal, TickStats,
    TRINSignal, TRIN, BreadthData, BreadthSeries, BreadthIndicator,
};

// ============================================================================
// Swing Exports
// ============================================================================
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

// ============================================================================
// DeMark Exports
// ============================================================================
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

// ============================================================================
// Re-export SPI types
// ============================================================================
pub use indicator_spi::{
    TechnicalIndicator, StreamingIndicator, SignalIndicator,
    IndicatorOutput, IndicatorSignal, IndicatorError, Result,
    OHLCV, OHLCVSeries,
};

// ============================================================================
// Re-export API configs
// ============================================================================
pub use indicator_api::{
    // Moving Averages
    SMAConfig, EMAConfig, WMAConfig, DEMAConfig, TEMAConfig,
    HMAConfig, KAMAConfig, ZLEMAConfig,
    SMMAConfig, ALMAConfig, FRAMAConfig, VIDYAConfig,
    T3Config, TRIMAConfig, GMMAConfig, SineWMAConfig,
    // Filters
    KalmanConfig, MedianConfig, GaussianConfig,
    SVHMAConfig, SVHMAThresholdMode, SVHMAUpdateMode,
    DeviationFilteredAverageConfig, StepVhfAdaptiveVMAConfig,
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
