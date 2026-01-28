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
pub mod intermarket;
pub mod crypto;
pub mod sentiment;
pub mod momentum;

#[cfg(feature = "simd")]
pub mod simd;

// ============================================================================
// Moving Average Exports
// ============================================================================
pub use moving_averages::{
    SMA, EMA, WMA, DEMA, TEMA, HMA, KAMA, ZLEMA, SMMA, ALMA,
    FRAMA, VIDYA, T3, TRIMA, GMMA, SineWMA, JurikMA, EVWMA,
    VolumeAdjustedMA, RangeWeightedMA, MomentumWeightedMA,
    AdaptiveMA, DoubleSmoothedMA, TripleSmoothedMA,
    FractalAdaptiveMA, VolumeAdaptiveMA, TrendAdaptiveMA,
    NoiseAdaptiveMA, MomentumAdaptiveMA, EfficiencyAdaptiveMA,
    VolatilityAdaptiveMA, CycleAdaptiveMA,
};

// ============================================================================
// Filter Exports
// ============================================================================
pub use filters::{
    KalmanFilter, MedianFilter, GaussianFilter, SVHMA,
    DeviationFilteredAverage, StepVhfAdaptiveVMA,
    ExponentialSmoothingFilter, ButterworthFilter, HighPassFilter,
    BandPassFilter, AdaptiveNoiseFilter, TrendFilter,
    RecursiveFilter, NormalizedPriceFilter,
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
    RelativeVolatilityIndex, DoubleStochastic, PriceOscillator, RainbowOscillator,
    KasePermissionStochastic, ProjectionOscillator, KasePeakOscillator,
    ElderPower, ElderPowerOutput,
    FastStochasticK, FastStochasticD, SlowStochasticK, SlowStochasticD,
    FullStochastic, FullStochasticOutput, LaneStochastic,
    StochasticDivergence, DivergenceType, DivergenceOutput,
    StochasticPop, StochasticDrop, StochasticCrossover, CrossoverOutput,
    PriceVelocity, PriceAcceleration, MomentumDivergence, SmoothedROC,
    ComparativeMomentum, DynamicMomentumIndex, MomentumQuality, NormalizedMomentum,
    AdaptiveRSI, VolumeWeightedRSI, StochasticMomentum,
    TrendIntensityOscillator, RangeOscillator, MomentumDivergenceOscillator,
    CycleOscillator, VolatilityOscillator, TrendOscillator,
    PriceActionOscillator, VolumeFlowOscillator, MomentumFlowOscillator,
    AdaptiveCCI, SmoothedTSI, VolatilityAdjustedRSI,
    CycleTunedOscillator, MomentumAccumulator, PriceStrengthIndex,
    DynamicOscillator, TrendOptimizedOscillator, RangeOptimizedOscillator,
    CompositeOscillator, AdaptiveStochastic, MomentumWaveOscillator,
    UltimateOscillatorEnhanced, PercentRankOscillator, ZScoreOscillator,
    VelocityOscillator, AccelerationOscillator, HarmonicOscillator,
    WilliamsREnhanced, StochasticMomentumIndex, PercentagePriceOscillator,
    VolumeOscillatorEnhanced, TrendOscillatorIndex, CycleMomentumOscillator,
};

// ============================================================================
// Trend Exports
// ============================================================================
pub use trend::{
    MACD, ADX, Ichimoku, IchimokuOutput, SuperTrend, ParabolicSAR,
    Alligator, AlligatorOutput, Aroon, AroonOutput, CoppockCurve, DPO,
    EfficiencyRatio, GatorOscillator, KaseCD,
    McGinleyDynamic, RainbowMA, RandomWalkIndex, TrendDetectionIndex, TrendIntensityIndex,
    VerticalHorizontalFilter, VortexIndicator, SafeZoneStop, SafeZoneStopOutput,
    CompositeTrendScore, TrendPersistence, PriceChannelPosition,
    TrendExhaustion, DirectionalMovementQuality, MultiTimeframeTrend,
    TrendAcceleration, TrendConsistency, AdaptiveTrendLine,
    TrendStrengthMeter, TrendChangeDetector, MultiScaleTrend,
    TrendContinuity, TrendMomentumConvergence, AdaptiveTrendStrength,
    TrendDirectionIndex, TrendMaturity, MultiPeriodTrendAlignment,
};

// ============================================================================
// Volatility Exports
// ============================================================================
pub use volatility::{
    ATR, HistoricalVolatility, ChaikinVolatility, MassIndex, ParkinsonVolatility,
    GarmanKlassVolatility, RogersSatchellVolatility, YangZhangVolatility,
    RealizedVolatility, NormalizedATR, ChoppinessIndex, UlcerIndex,
    VolatilityCone, VolatilityConeOutput, CloseToCloseVolatility,
    KaseDevStops, KaseDevStopsOutput, MarketThermometer,
    VolatilityRatio, RangeExpansionIndex, IntradayIntensityVolatility,
    NormalizedVolatility, VolatilityBreakout, VolatilityRegimeClassifier,
    VolatilityTrend, VolatilityMomentum, RelativeVolatility,
    PriceVolatilitySkew, ImpliedVolatilityProxy, VolatilityPersistence,
    VolatilityAcceleration, VolatilityMeanReversionDistance, VolatilitySpread,
    NormalizedRangeVolatility, VolatilitySkewIndicator, AdaptiveVolatilityBands,
};

// ============================================================================
// Volume Exports
// ============================================================================
pub use volume::{
    VWAP, OBV, MFI, CMF, VWMA, ADLine, ForceIndex, KlingerOscillator,
    BalanceOfPower, EaseOfMovement, VROC, PVT, NVI, PVI, WilliamsAD,
    TwiggsMoneyFlow, VolumeOscillator, NetVolume, ChaikinOscillator, TWAP,
    VolumeProfile, VolumeProfileOutput, MarketProfile, MarketProfileOutput,
    VolumeMomentum, RelativeVolume, VolumeWeightedPriceMomentum,
    VPTExtended, VolumeBuyingPressure, PriceVolumeRank,
    VolumeAccumulation, VolumeBreakout, RelativeVolumeStrength,
    VolumeClimaxDetector, SmartMoneyVolume, VolumeEfficiency,
    VolumeDistribution, VolumeIntensity, VolumeTrend,
    VolumeAnomaly, VolumePriceConfirmation, VolumeExhaustion,
    VolumeWeightedMomentum, VolumeForceIndex, CumulativeVolumeOscillator,
    VolumeRateOfChange, RelativeVolumeProfile, VolumeImpulse,
};

// ============================================================================
// Bands Exports
// ============================================================================
pub use bands::{
    BollingerBands, KeltnerChannels, DonchianChannels,
    AccelerationBands, ChandelierExit, ElderAutoEnvelope, Envelope, HighLowBands, PriceChannel,
    ProjectionBands, STARCBands, StandardErrorBands, TironeLevels, TironeLevelsOutput,
    AdaptiveBands, FixedPercentageEnvelope, MomentumBands,
    VolumeWeightedBands, DynamicChannel, LinearRegressionChannel,
    VolatilityBands, TrendBands, MomentumBandsAdvanced,
    PriceEnvelope, DynamicPriceChannel, RangeBands,
    AdaptiveKeltnerChannels, VolatilityWeightedBands, TrendFollowingChannel,
    DynamicSupportResistanceBands, MomentumBandwidth, PriceEnvelopeOscillator,
};

// ============================================================================
// Support/Resistance Exports
// ============================================================================
pub use support_resistance::{
    PivotPoints, Fibonacci, FibonacciLevels,
    DynamicSupportResistance, PriceClusters, VolumeSupportResistance,
    SwingLevelDetector, TrendlineBreak, PsychologicalLevels,
};

// ============================================================================
// Statistical Exports
// ============================================================================
pub use statistical::{
    StandardDeviation, Variance, ZScore, LinearRegression, LinearRegressionOutput,
    Correlation, Spread, Ratio, ZScoreSpread, Autocorrelation, Skewness, Kurtosis,
    FractalDimension, FractalDimensionMethod,
    HurstExponent, HurstMethod,
    DetrendedFluctuationAnalysis,
    MarketEntropy, EntropyMethod,
    RollingVariance, RollingSkewness, RollingKurtosis,
    PriceDistribution, ReturnDistribution, TailRiskIndicator,
    RollingCovariance, SerialCorrelation, RunsTest,
    MeanReversionStrength, DistributionMoments, OutlierDetector,
};

// ============================================================================
// Pattern Exports
// ============================================================================
pub use pattern::{
    ZigZag, HeikinAshi, HeikinAshiOutput, DarvasBox, Fractals, Doji, Hammer, Engulfing,
    Harami, MorningStar, ThreeSoldiers, Marubozu, Piercing, SpinningTop, Tweezer,
    ThreeInside, ThreeOutside, AbandonedBaby, BeltHold, Kicking, ThreeLineStrike,
    TasukiGap, RisingFallingMethods, KaseBars, KaseBarsOutput, KaseBarsStats,
    TrendTemplate, VolatilityContractionPattern, PocketPivot,
    PowerPlay, BullFlag, CupPattern,
    DoubleTop, DoubleBottom, HeadShoulders, Triangle, Channel, Wedge,
    GapAnalysis, InsideBar, OutsideBar, NarrowRange,
    WideRangeBar, TrendBar, ConsolidationPattern,
    PriceMomentumPattern, RangeContractionExpansion,
    TrendContinuationPattern, ReversalCandlePattern, VolumePricePattern,
    MomentumPattern, BreakoutPattern, ConsolidationBreak,
};

// ============================================================================
// Risk Exports
// ============================================================================
pub use risk::{
    SharpeRatio, SortinoRatio, CalmarRatio, MaxDrawdown, ValueAtRisk, VaRMethod,
    ConditionalVaR, Beta, Alpha, TreynorRatio, InformationRatio, OmegaRatio, GainLossRatio,
    SterlingRatio, BurkeRatio, UlcerPerformanceIndex, PainIndex, RecoveryFactor, TailRatio,
    ConditionalDrawdown, RiskAdjustedReturn, ReturnVariance,
    DrawdownDuration, RecoveryRatio, VolatilityRiskRatio,
    DownsideDeviation, UpsidePotentialRatio, KappaRatio,
    WinRate, ProfitFactor, Expectancy,
};

// ============================================================================
// DSP Exports
// ============================================================================
pub use dsp::{
    MESA, MAMA, SineWave, HilbertTransform, CyberCycle, CGOscillator,
    LaguerreRSI, RoofingFilter, Supersmoother, Decycler,
    DominantCyclePeriod, CycleAmplitude, CyclePhase,
    TrendCycleDecomposition, CycleMomentum, CycleTurningPoint,
    SpectralDensity, PhaseIndicator, InstantaneousFrequency,
    AdaptiveBandwidthFilter, ZeroLagIndicator, SignalToNoiseRatio,
    AutoCorrelationPeriod, TrendStrengthFFT, CycleDeviationAmplitude,
    PhaseAccumulator, SpectralNoiseRatio, AdaptiveCycleFilter,
    AdaptiveFrequencyFilter, TrendPhaseAnalysis, CycleModeIndicator,
    SignalStrengthMeter, FrequencyResponseIndicator, PhaseSynchronization,
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
    TrendMomentumScore, VolatilityTrendCombo, MultiPeriodMomentum,
    MomentumStrengthIndex, MarketConditionScore, PriceActionScore,
    QualityMomentumFactor, ValueMomentumComposite, RiskAdjustedTrend,
    BreakoutStrengthIndex, TrendReversalProbability, MultiFactorAlphaScore,
    TrendVolatilityIndex, TrendVolatilityIndexConfig, TrendVolatilityIndexOutput,
    MomentumQualityScore, MomentumQualityScoreConfig, MomentumQualityScoreOutput,
    MarketPhaseIndicator, MarketPhaseIndicatorConfig, MarketPhaseIndicatorOutput, MarketPhase,
    PriceTrendStrength, PriceTrendStrengthConfig, PriceTrendStrengthOutput,
    AdaptiveMarketIndicator, AdaptiveMarketIndicatorConfig, AdaptiveMarketIndicatorOutput,
    CompositeSignalStrength, CompositeSignalStrengthConfig, CompositeSignalStrengthOutput,
};

// ============================================================================
// Breadth Exports
// ============================================================================
pub use breadth::{
    AdvanceDeclineLine, BreadthThrust, CumulativeVolumeIndex, UpDownVolume,
    HighLowData, HighLowIndex, HighLowMethod, HighLowSeries,
    McClellanOscillator, McClellanSummationIndex,
    MarketCondition, PercentAboveMA, PercentAboveMASeries,
    NewHighsLows, NewHighsLowsMode,
    ContrarianSignal, PutCallRatio, PutCallSeries, PutCallSignal,
    TickBias, TickIndex, TickSeries, TickSignal, TickStats,
    TRINSignal, TRIN, BreadthData, BreadthSeries, BreadthIndicator,
    BullishPercent, BPISeries, BPISignal, BPIStatus,
    MarketMomentumBreadth, BreadthOscillator, CumulativeBreadthIndex,
    VolumeBreadthRatio, BreadthDivergence, ParticipationRate,
    BreadthMomentumIndex, CumulativeBreadthMomentum, BreadthVolatility,
    BreadthTrendStrength, BreadthExtremeDetector, BreadthDivergenceIndex,
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
    SwingMomentum, SwingRange, SwingDirection,
    SwingVelocity, SwingStrength, SwingFailurePattern,
    SwingTrendStrength, SwingReversal, SwingVolatility,
    SwingMomentumAdvanced, SwingTargetLevels, SwingDuration,
    SwingProjection, SwingConfirmation, SwingRangeAnalysis,
    SwingBreakout, SwingMomentumFlow, SwingTrendAnalysis,
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
    TDCamouflage, TDCLOP, TDMovingAverageQualifier,
    TDRiskLevel, TDMomentum, TDDifferential,
    TDLine, TDLineOutput, TDRange, TDRangeOutput,
    TDChannel, TDChannelOutput, TDQualifier, TDQualifierOutput,
    TDAlignment, TDAlignmentOutput, TDExhaustion, TDExhaustionOutput,
};

// ============================================================================
// Intermarket Exports
// ============================================================================
pub use intermarket::{
    DualSeries, MultiSeries,
    Cointegration, CointegrationOutput, CointegrationSignal,
    CurrencyStrength, CurrencyPair, CurrencyStrengthOutput,
    RelativeStrength, RelativeStrengthOutput, RelativeStrengthSignal,
    SectorRotation, SectorRank, SectorRotationOutput,
    CrossMarketMomentum, BetaCoefficient, MarketRegimeIndicator,
    SectorRelativePerformance, CorrelationMomentum, RiskAppetiteIndex, DivergenceIndex,
    LeadLagIndicator, PriceSpreadMomentum, CorrelationTrend,
    RelativeValueIndex, SpreadMeanReversion, PairsTradingSignal,
    CrossMarketCorrelation, RelativeStrengthMomentum, IntermarketDivergence,
    SectorMomentumRank, CrossAssetVolatility, MarketLeadLag,
};

// ============================================================================
// Crypto / On-Chain Exports
// ============================================================================
pub use crypto::{
    NVTRatio, NVTRatioOutput, NVTSignal,
    MVRVRatio, MVRVOutput, MVRVSignal,
    SOPR, SOPROutput, SOPRSignal,
    HashRibbons, HashRibbonsOutput, HashRibbonsPhase,
    FearGreedIndex, FearGreedOutput, FearGreedLevel, FearGreedWeights,
    HashRateMomentum, MinerCapitulation, WhaleAccumulation,
    RetailSentimentProxy, InstitutionalFlowProxy, NetworkActivityProxy,
    ActiveAddressesProxy, ExchangeFlowProxy, HODLBehaviorProxy,
    NetworkValueMomentum, TransactionVelocityProxy, CryptoMomentumScore,
};

// ============================================================================
// Sentiment Exports
// ============================================================================
pub use sentiment::{
    FearIndex, GreedIndex, CrowdPsychology, MarketEuphoria,
    Capitulation, SmartMoneyConfidence,
    MarketMomentumSentiment, VolatilitySentiment, TrendSentiment,
    ReversalSentiment, ExtremeReadings, SentimentOscillator,
    PriceActionSentiment, VolumeBasedSentiment, MomentumSentiment,
    ExtremeSentiment, SentimentDivergence, CompositeSentimentScore,
    SentimentMomentum, SentimentExtremeDetector, SentimentTrendFollower,
    SentimentContrarianSignal, SentimentVolatility, SentimentCycle,
    SentimentStrength, SentimentAcceleration, SentimentMeanReversion,
    CrowdBehaviorIndex, SentimentRegimeDetector, ContraSentimentSignal,
};

// ============================================================================
// Momentum Exports
// ============================================================================
pub use momentum::{
    MomentumDivergenceIndex, MomentumPersistence, MomentumRegime,
    RelativeMomentumIndex, MomentumAccelerator, AdaptiveMomentumFilter,
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
    T3Config, TRIMAConfig, GMMAConfig, SineWMAConfig, JurikMAConfig, EVWMAConfig,
    // Filters
    KalmanConfig, MedianConfig, GaussianConfig,
    SVHMAConfig, SVHMAThresholdMode, SVHMAUpdateMode,
    DeviationFilteredAverageConfig, StepVhfAdaptiveVMAConfig,
    // Oscillators
    RSIConfig, StochasticConfig, WilliamsRConfig, CCIConfig,
    TRIXConfig, UltimateOscillatorConfig,
    // Trend
    MACDConfig, ADXConfig, IchimokuConfig, SuperTrendConfig, ParabolicSARConfig, SafeZoneStopConfig,
    // Volatility
    BollingerConfig, ElderAutoEnvelopeConfig, ATRConfig, DonchianConfig, KeltnerConfig,
    VolatilityConeConfig, CloseToCloseVolatilityConfig, ElderThermometerConfig, KaseDevStopsConfig,
    // Volume
    VWAPConfig, OBVConfig, MFIConfig, CMFConfig,
    VolumeProfileConfig, MarketProfileConfig,
    // Breadth
    BullishPercentConfig, NewHighsLowsConfig, NewHighsLowsOutputMode,
    // Intermarket
    RelativeStrengthConfig,
    // Statistical
    FractalDimensionConfig, DFAConfig, EntropyConfig, EntropyMethodConfig,
    // Pattern
    KaseBarsConfig,
    // Crypto
    NVTRatioConfig,
    // Other
    ElderRayConfig, ElderPowerConfig, PivotType,
};
