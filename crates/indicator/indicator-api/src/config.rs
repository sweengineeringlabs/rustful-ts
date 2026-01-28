//! Indicator configuration types.

use serde::{Deserialize, Serialize};

// ============================================================================
// Moving Averages
// ============================================================================

/// Simple Moving Average configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SMAConfig {
    pub period: usize,
}

impl SMAConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for SMAConfig {
    fn default() -> Self {
        Self { period: 20 }
    }
}

/// Exponential Moving Average configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EMAConfig {
    pub period: usize,
    /// Optional custom smoothing factor (default: 2 / (period + 1)).
    pub alpha: Option<f64>,
}

impl EMAConfig {
    pub fn new(period: usize) -> Self {
        Self { period, alpha: None }
    }

    pub fn with_alpha(period: usize, alpha: f64) -> Self {
        Self { period, alpha: Some(alpha) }
    }

    /// Get the smoothing factor.
    pub fn smoothing_factor(&self) -> f64 {
        self.alpha.unwrap_or_else(|| 2.0 / (self.period as f64 + 1.0))
    }
}

impl Default for EMAConfig {
    fn default() -> Self {
        Self { period: 20, alpha: None }
    }
}

// ============================================================================
// Oscillators
// ============================================================================

/// Relative Strength Index configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RSIConfig {
    pub period: usize,
    /// Overbought threshold (default: 70).
    pub overbought: f64,
    /// Oversold threshold (default: 30).
    pub oversold: f64,
}

impl RSIConfig {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            overbought: 70.0,
            oversold: 30.0,
        }
    }

    pub fn with_thresholds(period: usize, overbought: f64, oversold: f64) -> Self {
        Self { period, overbought, oversold }
    }
}

impl Default for RSIConfig {
    fn default() -> Self {
        Self {
            period: 14,
            overbought: 70.0,
            oversold: 30.0,
        }
    }
}

/// Stochastic Oscillator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticConfig {
    pub k_period: usize,
    pub d_period: usize,
    pub overbought: f64,
    pub oversold: f64,
}

impl StochasticConfig {
    pub fn new(k_period: usize, d_period: usize) -> Self {
        Self {
            k_period,
            d_period,
            overbought: 80.0,
            oversold: 20.0,
        }
    }
}

impl Default for StochasticConfig {
    fn default() -> Self {
        Self {
            k_period: 14,
            d_period: 3,
            overbought: 80.0,
            oversold: 20.0,
        }
    }
}

// ============================================================================
// Trend Indicators
// ============================================================================

/// MACD (Moving Average Convergence Divergence) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MACDConfig {
    pub fast_period: usize,
    pub slow_period: usize,
    pub signal_period: usize,
}

impl MACDConfig {
    pub fn new(fast: usize, slow: usize, signal: usize) -> Self {
        Self {
            fast_period: fast,
            slow_period: slow,
            signal_period: signal,
        }
    }
}

impl Default for MACDConfig {
    fn default() -> Self {
        Self {
            fast_period: 12,
            slow_period: 26,
            signal_period: 9,
        }
    }
}

// ============================================================================
// Volatility Indicators
// ============================================================================

/// Bollinger Bands configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BollingerConfig {
    pub period: usize,
    pub std_dev: f64,
}

impl BollingerConfig {
    pub fn new(period: usize, std_dev: f64) -> Self {
        Self { period, std_dev }
    }
}

impl Default for BollingerConfig {
    fn default() -> Self {
        Self {
            period: 20,
            std_dev: 2.0,
        }
    }
}

/// Elder's AutoEnvelope configuration.
///
/// Adaptive envelope bands that automatically adjust width based on recent volatility
/// using standard deviation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElderAutoEnvelopeConfig {
    /// Period for EMA calculation.
    pub ema_period: usize,
    /// Period for standard deviation calculation.
    pub std_period: usize,
    /// Multiplier for standard deviation to determine band width.
    pub multiplier: f64,
}

impl ElderAutoEnvelopeConfig {
    pub fn new(ema_period: usize, std_period: usize, multiplier: f64) -> Self {
        Self { ema_period, std_period, multiplier }
    }
}

impl Default for ElderAutoEnvelopeConfig {
    fn default() -> Self {
        Self {
            ema_period: 13,
            std_period: 13,
            multiplier: 2.7,
        }
    }
}

/// Average True Range configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATRConfig {
    pub period: usize,
}

impl ATRConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for ATRConfig {
    fn default() -> Self {
        Self { period: 14 }
    }
}

/// Volatility Cone configuration.
///
/// Shows percentile bands of historical volatility over different lookback periods.
/// Used to compare current volatility to historical ranges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityConeConfig {
    /// Lookback periods for volatility calculation (e.g., [20, 40, 60, 120, 252]).
    pub periods: Vec<usize>,
    /// Percentile levels to compute (e.g., [0.1, 0.25, 0.5, 0.75, 0.9]).
    pub percentiles: Vec<f64>,
}

impl VolatilityConeConfig {
    /// Create a new Volatility Cone configuration.
    pub fn new(periods: Vec<usize>, percentiles: Vec<f64>) -> Self {
        Self { periods, percentiles }
    }

    /// Create with standard periods and custom percentiles.
    pub fn with_percentiles(percentiles: Vec<f64>) -> Self {
        Self {
            periods: vec![20, 40, 60, 120, 252],
            percentiles,
        }
    }

    /// Create with custom periods and standard percentiles.
    pub fn with_periods(periods: Vec<usize>) -> Self {
        Self {
            periods,
            percentiles: vec![0.1, 0.25, 0.5, 0.75, 0.9],
        }
    }
}

impl Default for VolatilityConeConfig {
    fn default() -> Self {
        Self {
            periods: vec![20, 40, 60, 120, 252],
            percentiles: vec![0.1, 0.25, 0.5, 0.75, 0.9],
        }
    }
}

/// Close-to-Close Volatility configuration.
///
/// Calculates volatility using the standard deviation of logarithmic returns
/// from closing prices only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloseToCloseVolatilityConfig {
    /// Lookback period for volatility calculation.
    pub period: usize,
    /// Whether to annualize the volatility.
    pub annualize: bool,
    /// Number of trading days per year for annualization (default: 252).
    pub trading_days: usize,
}

impl CloseToCloseVolatilityConfig {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            annualize: true,
            trading_days: 252,
        }
    }

    pub fn with_annualization(period: usize, annualize: bool, trading_days: usize) -> Self {
        Self {
            period,
            annualize,
            trading_days,
        }
    }

    pub fn without_annualization(period: usize) -> Self {
        Self {
            period,
            annualize: false,
            trading_days: 252,
        }
    }
}

impl Default for CloseToCloseVolatilityConfig {
    fn default() -> Self {
        Self {
            period: 20,
            annualize: true,
            trading_days: 252,
        }
    }
}

/// Elder's Market Thermometer configuration (IND-178).
///
/// Measures intraday volatility by comparing current bar's range to previous bar.
/// High readings indicate increased volatility.
///
/// Algorithm:
/// 1. Calculate absolute difference: |high - previous_high| and |low - previous_low|
/// 2. Thermometer = max(|H - prev_H|, |L - prev_L|)
/// 3. Calculate EMA of thermometer values for smoothing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElderThermometerConfig {
    /// EMA period for smoothing (default: 22).
    pub period: usize,
}

impl ElderThermometerConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for ElderThermometerConfig {
    fn default() -> Self {
        Self { period: 22 }
    }
}

/// Kase Dev Stops configuration (IND-182).
///
/// Deviation-based trailing stops using True Range and standard deviation.
/// Provides adaptive stop levels that adjust to market volatility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaseDevStopsConfig {
    /// Lookback period for ATR and StdDev calculations.
    pub period: usize,
    /// Number of standard deviations for stop calculation.
    pub num_devs: f64,
}

impl KaseDevStopsConfig {
    /// Create a new Kase Dev Stops configuration.
    pub fn new(period: usize, num_devs: f64) -> Self {
        Self { period, num_devs }
    }

    /// Create with default num_devs (2.0).
    pub fn with_period(period: usize) -> Self {
        Self { period, num_devs: 2.0 }
    }
}

impl Default for KaseDevStopsConfig {
    fn default() -> Self {
        Self {
            period: 30,
            num_devs: 2.0,
        }
    }
}

// ============================================================================
// Other Indicators
// ============================================================================

/// Rate of Change configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ROCConfig {
    pub period: usize,
}

impl ROCConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for ROCConfig {
    fn default() -> Self {
        Self { period: 10 }
    }
}

/// Standard Deviation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StdDevConfig {
    pub period: usize,
    /// Degrees of freedom (0 for population, 1 for sample).
    pub ddof: usize,
}

impl StdDevConfig {
    pub fn new(period: usize) -> Self {
        Self { period, ddof: 1 }
    }

    pub fn population(period: usize) -> Self {
        Self { period, ddof: 0 }
    }
}

impl Default for StdDevConfig {
    fn default() -> Self {
        Self { period: 20, ddof: 1 }
    }
}

// ============================================================================
// Advanced Indicators
// ============================================================================

/// Ichimoku Cloud configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IchimokuConfig {
    pub tenkan_period: usize,
    pub kijun_period: usize,
    pub senkou_b_period: usize,
}

impl IchimokuConfig {
    pub fn new(tenkan: usize, kijun: usize, senkou_b: usize) -> Self {
        Self {
            tenkan_period: tenkan,
            kijun_period: kijun,
            senkou_b_period: senkou_b,
        }
    }
}

impl Default for IchimokuConfig {
    fn default() -> Self {
        Self {
            tenkan_period: 9,
            kijun_period: 26,
            senkou_b_period: 52,
        }
    }
}

/// SuperTrend configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperTrendConfig {
    pub period: usize,
    pub multiplier: f64,
}

impl SuperTrendConfig {
    pub fn new(period: usize, multiplier: f64) -> Self {
        Self { period, multiplier }
    }
}

impl Default for SuperTrendConfig {
    fn default() -> Self {
        Self {
            period: 10,
            multiplier: 3.0,
        }
    }
}

/// Elder's SafeZone Stop configuration.
///
/// A directional stop loss indicator based on recent price penetrations.
/// Provides separate stops for long and short positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeZoneStopConfig {
    /// Lookback period for averaging penetrations.
    pub period: usize,
    /// Multiplier for average penetration distance.
    pub coefficient: f64,
}

impl SafeZoneStopConfig {
    pub fn new(period: usize, coefficient: f64) -> Self {
        Self { period, coefficient }
    }
}

impl Default for SafeZoneStopConfig {
    fn default() -> Self {
        Self {
            period: 10,
            coefficient: 2.5,
        }
    }
}

/// Parabolic SAR configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParabolicSARConfig {
    pub af_start: f64,
    pub af_step: f64,
    pub af_max: f64,
}

impl ParabolicSARConfig {
    pub fn new(af_start: f64, af_step: f64, af_max: f64) -> Self {
        Self { af_start, af_step, af_max }
    }
}

impl Default for ParabolicSARConfig {
    fn default() -> Self {
        Self {
            af_start: 0.02,
            af_step: 0.02,
            af_max: 0.2,
        }
    }
}

/// Donchian Channels configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DonchianConfig {
    pub period: usize,
}

impl DonchianConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for DonchianConfig {
    fn default() -> Self {
        Self { period: 20 }
    }
}

/// Keltner Channels configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeltnerConfig {
    pub ema_period: usize,
    pub atr_period: usize,
    pub multiplier: f64,
}

impl KeltnerConfig {
    pub fn new(ema_period: usize, atr_period: usize, multiplier: f64) -> Self {
        Self { ema_period, atr_period, multiplier }
    }
}

impl Default for KeltnerConfig {
    fn default() -> Self {
        Self {
            ema_period: 20,
            atr_period: 10,
            multiplier: 2.0,
        }
    }
}

/// Elder Ray configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElderRayConfig {
    pub period: usize,
}

impl ElderRayConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for ElderRayConfig {
    fn default() -> Self {
        Self { period: 13 }
    }
}

/// Elder's Bull/Bear Power (Enhanced) configuration.
///
/// Enhanced version of Elder Ray with combined bull/bear power signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElderPowerConfig {
    /// EMA period (default: 13).
    pub period: usize,
}

impl ElderPowerConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for ElderPowerConfig {
    fn default() -> Self {
        Self { period: 13 }
    }
}

/// Pivot Points type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum PivotType {
    #[default]
    Standard,
    Fibonacci,
    Woodie,
    Camarilla,
}

// ============================================================================
// Additional Moving Averages
// ============================================================================

/// Weighted Moving Average configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WMAConfig {
    pub period: usize,
}

impl WMAConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for WMAConfig {
    fn default() -> Self {
        Self { period: 20 }
    }
}

/// Double Exponential Moving Average configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DEMAConfig {
    pub period: usize,
}

impl DEMAConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for DEMAConfig {
    fn default() -> Self {
        Self { period: 20 }
    }
}

/// Triple Exponential Moving Average configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TEMAConfig {
    pub period: usize,
}

impl TEMAConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for TEMAConfig {
    fn default() -> Self {
        Self { period: 20 }
    }
}

/// Hull Moving Average configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HMAConfig {
    pub period: usize,
}

impl HMAConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for HMAConfig {
    fn default() -> Self {
        Self { period: 20 }
    }
}

/// Kaufman Adaptive Moving Average configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KAMAConfig {
    pub period: usize,
    pub fast_period: usize,
    pub slow_period: usize,
}

impl KAMAConfig {
    pub fn new(period: usize, fast: usize, slow: usize) -> Self {
        Self {
            period,
            fast_period: fast,
            slow_period: slow,
        }
    }
}

impl Default for KAMAConfig {
    fn default() -> Self {
        Self {
            period: 10,
            fast_period: 2,
            slow_period: 30,
        }
    }
}

/// Zero-Lag Exponential Moving Average configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZLEMAConfig {
    pub period: usize,
}

impl ZLEMAConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for ZLEMAConfig {
    fn default() -> Self {
        Self { period: 20 }
    }
}

// ============================================================================
// Filters
// ============================================================================

/// Kalman Filter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanConfig {
    pub process_noise: f64,
    pub measurement_noise: f64,
}

impl KalmanConfig {
    pub fn new(process_noise: f64, measurement_noise: f64) -> Self {
        Self { process_noise, measurement_noise }
    }
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            process_noise: 0.01,
            measurement_noise: 0.1,
        }
    }
}

/// Median Filter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedianConfig {
    pub period: usize,
}

impl MedianConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for MedianConfig {
    fn default() -> Self {
        Self { period: 5 }
    }
}

/// Gaussian Filter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianConfig {
    pub period: usize,
    pub sigma: f64,
}

impl GaussianConfig {
    pub fn new(period: usize, sigma: f64) -> Self {
        Self { period, sigma }
    }
}

impl Default for GaussianConfig {
    fn default() -> Self {
        Self { period: 5, sigma: 1.0 }
    }
}

// ============================================================================
// Additional Oscillators
// ============================================================================

/// Williams %R configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WilliamsRConfig {
    pub period: usize,
    pub overbought: f64,
    pub oversold: f64,
}

impl WilliamsRConfig {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            overbought: -20.0,
            oversold: -80.0,
        }
    }
}

impl Default for WilliamsRConfig {
    fn default() -> Self {
        Self {
            period: 14,
            overbought: -20.0,
            oversold: -80.0,
        }
    }
}

/// Commodity Channel Index configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CCIConfig {
    pub period: usize,
    pub overbought: f64,
    pub oversold: f64,
}

impl CCIConfig {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            overbought: 100.0,
            oversold: -100.0,
        }
    }
}

impl Default for CCIConfig {
    fn default() -> Self {
        Self {
            period: 20,
            overbought: 100.0,
            oversold: -100.0,
        }
    }
}

/// TRIX configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TRIXConfig {
    pub period: usize,
}

impl TRIXConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for TRIXConfig {
    fn default() -> Self {
        Self { period: 15 }
    }
}

/// Ultimate Oscillator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UltimateOscillatorConfig {
    pub period1: usize,  // Short
    pub period2: usize,  // Medium
    pub period3: usize,  // Long
}

impl UltimateOscillatorConfig {
    pub fn new(short: usize, medium: usize, long: usize) -> Self {
        Self {
            period1: short,
            period2: medium,
            period3: long,
        }
    }
}

impl Default for UltimateOscillatorConfig {
    fn default() -> Self {
        Self {
            period1: 7,
            period2: 14,
            period3: 28,
        }
    }
}

// ============================================================================
// Trend Indicators
// ============================================================================

/// Average Directional Index configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ADXConfig {
    pub period: usize,
    pub strong_trend: f64,
}

impl ADXConfig {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            strong_trend: 25.0,
        }
    }
}

impl Default for ADXConfig {
    fn default() -> Self {
        Self {
            period: 14,
            strong_trend: 25.0,
        }
    }
}

// ============================================================================
// Volume Indicators
// ============================================================================

/// Volume Weighted Average Price configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VWAPConfig {
    pub reset_daily: bool,
}

impl VWAPConfig {
    pub fn new() -> Self {
        Self { reset_daily: true }
    }

    pub fn cumulative() -> Self {
        Self { reset_daily: false }
    }
}

impl Default for VWAPConfig {
    fn default() -> Self {
        Self { reset_daily: false }
    }
}

/// On-Balance Volume configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OBVConfig {
    pub signal_period: Option<usize>,
}

impl OBVConfig {
    pub fn new() -> Self {
        Self { signal_period: None }
    }

    pub fn with_signal(period: usize) -> Self {
        Self { signal_period: Some(period) }
    }
}

impl Default for OBVConfig {
    fn default() -> Self {
        Self { signal_period: None }
    }
}

/// Money Flow Index configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MFIConfig {
    pub period: usize,
    pub overbought: f64,
    pub oversold: f64,
}

impl MFIConfig {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            overbought: 80.0,
            oversold: 20.0,
        }
    }
}

impl Default for MFIConfig {
    fn default() -> Self {
        Self {
            period: 14,
            overbought: 80.0,
            oversold: 20.0,
        }
    }
}

/// Chaikin Money Flow configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CMFConfig {
    pub period: usize,
}

impl CMFConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for CMFConfig {
    fn default() -> Self {
        Self { period: 20 }
    }
}

// ============================================================================
// Parameter Ranges for Optimization
// ============================================================================

/// Integer parameter range for optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamRange {
    pub min: usize,
    pub max: usize,
    pub step: usize,
}

impl ParamRange {
    pub fn new(min: usize, max: usize, step: usize) -> Self {
        Self { min, max, step }
    }

    /// Get all values in this range.
    pub fn values(&self) -> Vec<usize> {
        (self.min..=self.max).step_by(self.step).collect()
    }

    /// Number of discrete values.
    pub fn count(&self) -> usize {
        if self.step == 0 {
            return 1;
        }
        (self.max - self.min) / self.step + 1
    }
}

/// Float parameter range for optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatParamRange {
    pub min: f64,
    pub max: f64,
    pub step: f64,
}

impl FloatParamRange {
    pub fn new(min: f64, max: f64, step: f64) -> Self {
        Self { min, max, step }
    }

    /// Get all values in this range.
    pub fn values(&self) -> Vec<f64> {
        let mut result = Vec::new();
        let mut val = self.min;
        while val <= self.max + 1e-10 {
            result.push(val);
            val += self.step;
        }
        result
    }

    /// Number of discrete values.
    pub fn count(&self) -> usize {
        if self.step <= 0.0 {
            return 1;
        }
        ((self.max - self.min) / self.step + 1.0) as usize
    }
}

// ============================================================================
// Indicator Configuration Enum
// ============================================================================

/// Unified indicator configuration for optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndicatorType {
    SMA { period: ParamRange },
    EMA { period: ParamRange },
    RSI { period: ParamRange },
    MACD {
        fast: ParamRange,
        slow: ParamRange,
        signal: ParamRange,
    },
    Bollinger {
        period: ParamRange,
        std_dev: FloatParamRange,
    },
    ATR { period: ParamRange },
    ROC { period: ParamRange },
    StdDev { period: ParamRange },
    Stochastic {
        k_period: ParamRange,
        d_period: ParamRange,
    },
}

impl IndicatorType {
    pub fn name(&self) -> &'static str {
        match self {
            IndicatorType::SMA { .. } => "SMA",
            IndicatorType::EMA { .. } => "EMA",
            IndicatorType::RSI { .. } => "RSI",
            IndicatorType::MACD { .. } => "MACD",
            IndicatorType::Bollinger { .. } => "Bollinger",
            IndicatorType::ATR { .. } => "ATR",
            IndicatorType::ROC { .. } => "ROC",
            IndicatorType::StdDev { .. } => "StdDev",
            IndicatorType::Stochastic { .. } => "Stochastic",
        }
    }

    /// Total number of parameter combinations.
    pub fn combinations(&self) -> usize {
        match self {
            IndicatorType::SMA { period } => period.count(),
            IndicatorType::EMA { period } => period.count(),
            IndicatorType::RSI { period } => period.count(),
            IndicatorType::MACD { fast, slow, signal } => {
                fast.count() * slow.count() * signal.count()
            }
            IndicatorType::Bollinger { period, std_dev } => {
                period.count() * std_dev.count()
            }
            IndicatorType::ATR { period } => period.count(),
            IndicatorType::ROC { period } => period.count(),
            IndicatorType::StdDev { period } => period.count(),
            IndicatorType::Stochastic { k_period, d_period } => {
                k_period.count() * d_period.count()
            }
        }
    }
}

// ============================================================================
// Advanced Moving Averages
// ============================================================================

/// Smoothed Moving Average (SMMA/RMA) configuration.
/// Also known as Wilder's Smoothing Method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SMMAConfig {
    pub period: usize,
}

impl SMMAConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for SMMAConfig {
    fn default() -> Self {
        Self { period: 14 }
    }
}

/// Arnaud Legoux Moving Average (ALMA) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ALMAConfig {
    pub period: usize,
    /// Offset controls the Gaussian curve position (0.85 = closer to recent prices).
    pub offset: f64,
    /// Sigma controls the width of the Gaussian curve.
    pub sigma: f64,
}

impl ALMAConfig {
    pub fn new(period: usize, offset: f64, sigma: f64) -> Self {
        Self { period, offset, sigma }
    }
}

impl Default for ALMAConfig {
    fn default() -> Self {
        Self {
            period: 9,
            offset: 0.85,
            sigma: 6.0,
        }
    }
}

/// Fractal Adaptive Moving Average (FRAMA) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FRAMAConfig {
    pub period: usize,
    /// Fast smoothing constant (optional, default: 1.0).
    pub fast_sc: Option<f64>,
    /// Slow smoothing constant (optional, default: 0.01).
    pub slow_sc: Option<f64>,
}

impl FRAMAConfig {
    pub fn new(period: usize) -> Self {
        Self { period, fast_sc: None, slow_sc: None }
    }

    pub fn with_smoothing(period: usize, fast_sc: f64, slow_sc: f64) -> Self {
        Self { period, fast_sc: Some(fast_sc), slow_sc: Some(slow_sc) }
    }
}

impl Default for FRAMAConfig {
    fn default() -> Self {
        Self { period: 16, fast_sc: None, slow_sc: None }
    }
}

/// Variable Index Dynamic Average (VIDYA) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VIDYAConfig {
    pub period: usize,
    /// Chande Momentum Oscillator period.
    pub cmo_period: usize,
}

impl VIDYAConfig {
    pub fn new(period: usize, cmo_period: usize) -> Self {
        Self { period, cmo_period }
    }
}

impl Default for VIDYAConfig {
    fn default() -> Self {
        Self { period: 14, cmo_period: 9 }
    }
}

/// Tillson T3 Moving Average configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T3Config {
    pub period: usize,
    /// Volume factor (0.0 to 1.0, default: 0.7).
    pub volume_factor: f64,
}

impl T3Config {
    pub fn new(period: usize, volume_factor: f64) -> Self {
        Self { period, volume_factor }
    }
}

impl Default for T3Config {
    fn default() -> Self {
        Self { period: 5, volume_factor: 0.7 }
    }
}

/// Triangular Moving Average (TRIMA) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TRIMAConfig {
    pub period: usize,
}

impl TRIMAConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for TRIMAConfig {
    fn default() -> Self {
        Self { period: 20 }
    }
}

/// Guppy Multiple Moving Average (GMMA) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GMMAConfig {
    /// Short-term EMA periods.
    pub short_periods: Vec<usize>,
    /// Long-term EMA periods.
    pub long_periods: Vec<usize>,
}

impl GMMAConfig {
    pub fn new(short_periods: Vec<usize>, long_periods: Vec<usize>) -> Self {
        Self { short_periods, long_periods }
    }
}

impl Default for GMMAConfig {
    fn default() -> Self {
        Self {
            short_periods: vec![3, 5, 8, 10, 12, 15],
            long_periods: vec![30, 35, 40, 45, 50, 60],
        }
    }
}

/// Sine Weighted Moving Average configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SineWMAConfig {
    pub period: usize,
}

impl SineWMAConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for SineWMAConfig {
    fn default() -> Self {
        Self { period: 14 }
    }
}

/// Elastic Volume Weighted Moving Average configuration.
///
/// EVWMA uses volume to dynamically adjust its smoothing factor.
/// Higher volume gives more weight to the current price.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EVWMAConfig {
    pub period: usize,
}

impl EVWMAConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for EVWMAConfig {
    fn default() -> Self {
        Self { period: 20 }
    }
}

/// Jurik Moving Average (JMA) approximation configuration.
///
/// A low-lag, smooth adaptive moving average. Since the original JMA
/// is proprietary, this implements an approximation using adaptive smoothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JurikMAConfig {
    /// Smoothing period (similar to EMA period).
    pub period: usize,
    /// Phase adjustment (-100 to +100).
    /// Negative values increase smoothness, positive values reduce lag.
    pub phase: f64,
    /// Smoothing power factor (typically 1-3).
    /// Higher values increase responsiveness to price changes.
    pub power: f64,
}

impl JurikMAConfig {
    pub fn new(period: usize, phase: f64, power: f64) -> Self {
        Self { period, phase, power }
    }

    pub fn with_period(period: usize) -> Self {
        Self {
            period,
            phase: 0.0,
            power: 2.0,
        }
    }
}

impl Default for JurikMAConfig {
    fn default() -> Self {
        Self {
            period: 14,
            phase: 0.0,
            power: 2.0,
        }
    }
}

// ============================================================================
// Step-Based Filters (SVHMA Framework)
// ============================================================================

/// Threshold mode for SVHMA - determines when to update the filter.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SVHMAThresholdMode {
    /// Fixed threshold value.
    Fixed,
    /// Percentage of current price.
    Percentage,
    /// ATR-based threshold: θ = k × ATR.
    Atr,
    /// Standard deviation of prices: θ = k × σ(x).
    StdDev,
    /// Standard deviation of price changes: θ = k × σ(Δx).
    ChangeVolatility,
    /// Donchian range: θ = k × (H_n - L_n).
    Donchian,
    /// VHF-based: θ = k × (1 - VHF) × ATR.
    Vhf,
}

impl Default for SVHMAThresholdMode {
    fn default() -> Self {
        Self::Atr
    }
}

/// Update function for SVHMA - determines the new value on update.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SVHMAUpdateMode {
    /// Snap directly to current price.
    SnapToPrice,
    /// Snap to moving average (default).
    SnapToMA,
    /// Damped step: y + α(x - y).
    Damped,
    /// Quantized step: y + ⌊δ/s⌋ × s.
    Quantized,
    /// VHF-adaptive VMA with dynamic smoothing.
    VhfAdaptive,
}

impl Default for SVHMAUpdateMode {
    fn default() -> Self {
        Self::SnapToMA
    }
}

/// Step Variable Horizontal Moving Average (SVHMA) configuration.
///
/// A unified framework for step-based moving averages that only update
/// when price deviations exceed a configurable threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVHMAConfig {
    /// Period for MA and threshold calculations.
    pub period: usize,
    /// Threshold mode (how to calculate θ).
    pub threshold_mode: SVHMAThresholdMode,
    /// Threshold multiplier (k).
    pub threshold_multiplier: f64,
    /// Fixed threshold value (for Fixed mode).
    pub fixed_threshold: f64,
    /// Update mode (how to calculate new value).
    pub update_mode: SVHMAUpdateMode,
    /// Step size for Quantized mode.
    pub step_size: f64,
    /// Damping factor for Damped mode (α).
    pub damping_factor: f64,
    /// Enable directional constraint (only move in trend direction).
    pub directional: bool,
}

impl SVHMAConfig {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            threshold_mode: SVHMAThresholdMode::default(),
            threshold_multiplier: 2.0,
            fixed_threshold: 0.0,
            update_mode: SVHMAUpdateMode::default(),
            step_size: 1.0,
            damping_factor: 0.5,
            directional: false,
        }
    }

    /// Create config matching filtered_averages.mq5 behavior.
    /// Uses Change Volatility threshold with Snap-to-MA update.
    pub fn filtered_average(period: usize, filter: f64) -> Self {
        Self {
            period,
            threshold_mode: SVHMAThresholdMode::ChangeVolatility,
            threshold_multiplier: filter,
            fixed_threshold: 0.0,
            update_mode: SVHMAUpdateMode::SnapToMA,
            step_size: 1.0,
            damping_factor: 0.5,
            directional: false,
        }
    }

    /// Create config matching step_vhf_adaptive_vma.mq5 behavior.
    /// Uses Fixed threshold with VHF-adaptive + quantized update.
    pub fn step_vhf_adaptive(period: usize, step_size: f64) -> Self {
        Self {
            period,
            threshold_mode: SVHMAThresholdMode::Fixed,
            threshold_multiplier: 1.0,
            fixed_threshold: step_size,
            update_mode: SVHMAUpdateMode::VhfAdaptive,
            step_size,
            damping_factor: 0.5,
            directional: false,
        }
    }

    pub fn with_threshold_mode(mut self, mode: SVHMAThresholdMode) -> Self {
        self.threshold_mode = mode;
        self
    }

    pub fn with_threshold_multiplier(mut self, k: f64) -> Self {
        self.threshold_multiplier = k;
        self
    }

    pub fn with_update_mode(mut self, mode: SVHMAUpdateMode) -> Self {
        self.update_mode = mode;
        self
    }

    pub fn with_directional(mut self, directional: bool) -> Self {
        self.directional = directional;
        self
    }
}

impl Default for SVHMAConfig {
    fn default() -> Self {
        Self::new(14)
    }
}

/// Deviation Filtered Average configuration.
///
/// A step-based moving average that only updates when price changes exceed
/// a threshold based on the standard deviation of recent price changes.
/// Based on: filtered_averages.mq5 by mladen (2018)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationFilteredAverageConfig {
    /// Period for MA and filter calculations.
    pub period: usize,
    /// Filter multiplier (k in θ = k × σ(Δ)).
    pub filter: f64,
    /// Use EMA (true) or SMA (false) as base.
    pub use_ema: bool,
}

impl DeviationFilteredAverageConfig {
    pub fn new(period: usize, filter: f64) -> Self {
        Self {
            period,
            filter,
            use_ema: true,
        }
    }

    pub fn with_sma(mut self) -> Self {
        self.use_ema = false;
        self
    }
}

impl Default for DeviationFilteredAverageConfig {
    fn default() -> Self {
        Self {
            period: 14,
            filter: 2.5,
            use_ema: true,
        }
    }
}

/// Step VHF Adaptive VMA configuration.
///
/// A step-based variable moving average that uses the Vertical Horizontal Filter
/// to modulate the smoothing factor, with quantized step output.
/// Based on: step_vhf_adaptive_vma.mq5 by mladen (2018)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepVhfAdaptiveVMAConfig {
    /// VMA period (smoothing base).
    pub vma_period: usize,
    /// VHF period (can differ from VMA period).
    pub vhf_period: usize,
    /// Step size for quantization (0 = no stepping).
    pub step_size: f64,
}

impl StepVhfAdaptiveVMAConfig {
    pub fn new(period: usize, step_size: f64) -> Self {
        Self {
            vma_period: period,
            vhf_period: period,
            step_size,
        }
    }

    pub fn with_vhf_period(mut self, vhf_period: usize) -> Self {
        self.vhf_period = vhf_period;
        self
    }
}

impl Default for StepVhfAdaptiveVMAConfig {
    fn default() -> Self {
        Self {
            vma_period: 14,
            vhf_period: 14,
            step_size: 1.0,
        }
    }
}

// ============================================================================
// Volume Profile Indicators
// ============================================================================

/// Volume Profile configuration.
///
/// Shows volume traded at each price level as a histogram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeProfileConfig {
    /// Price bin size (auto-calculated if None).
    pub tick_size: Option<f64>,
    /// Number of bins if tick_size is None.
    pub num_bins: usize,
    /// Value area percentage (default: 0.70).
    pub value_area_pct: f64,
    /// Profile period in bars (0 = entire series).
    pub period: usize,
    /// Use close only vs full OHLC range.
    pub close_only: bool,
}

impl VolumeProfileConfig {
    pub fn new(num_bins: usize) -> Self {
        Self {
            tick_size: None,
            num_bins,
            value_area_pct: 0.70,
            period: 0,
            close_only: false,
        }
    }

    pub fn with_tick_size(tick_size: f64) -> Self {
        Self {
            tick_size: Some(tick_size),
            num_bins: 50,
            value_area_pct: 0.70,
            period: 0,
            close_only: false,
        }
    }

    pub fn with_period(mut self, period: usize) -> Self {
        self.period = period;
        self
    }

    pub fn with_value_area_pct(mut self, pct: f64) -> Self {
        self.value_area_pct = pct;
        self
    }
}

impl Default for VolumeProfileConfig {
    fn default() -> Self {
        Self {
            tick_size: None,
            num_bins: 50,
            value_area_pct: 0.70,
            period: 0,
            close_only: false,
        }
    }
}

/// Market Profile (TPO) configuration.
///
/// Time Price Opportunity based analysis showing time spent at each price level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketProfileConfig {
    /// Price bin size (auto-calculated if None).
    pub tick_size: Option<f64>,
    /// Number of bins if tick_size is None.
    pub num_bins: usize,
    /// Value area percentage (default: 0.70).
    pub value_area_pct: f64,
    /// Bars per TPO period.
    pub tpo_period: usize,
    /// Initial Balance TPO periods.
    pub ib_periods: usize,
    /// Session length in bars (0 = entire series).
    pub session_bars: usize,
}

impl MarketProfileConfig {
    pub fn new(num_bins: usize) -> Self {
        Self {
            tick_size: None,
            num_bins,
            value_area_pct: 0.70,
            tpo_period: 1,
            ib_periods: 2,
            session_bars: 0,
        }
    }

    pub fn with_tick_size(tick_size: f64) -> Self {
        Self {
            tick_size: Some(tick_size),
            num_bins: 50,
            value_area_pct: 0.70,
            tpo_period: 1,
            ib_periods: 2,
            session_bars: 0,
        }
    }

    pub fn with_tpo_period(mut self, period: usize) -> Self {
        self.tpo_period = period;
        self
    }

    pub fn with_ib_periods(mut self, periods: usize) -> Self {
        self.ib_periods = periods;
        self
    }

    pub fn with_session_bars(mut self, bars: usize) -> Self {
        self.session_bars = bars;
        self
    }
}

impl Default for MarketProfileConfig {
    fn default() -> Self {
        Self {
            tick_size: None,
            num_bins: 50,
            value_area_pct: 0.70,
            tpo_period: 1,
            ib_periods: 2,
            session_bars: 0,
        }
    }
}

// ============================================================================
// Breadth Indicators
// ============================================================================

/// Bullish Percent Index configuration.
///
/// Measures the percentage of stocks in an index showing bullish technical conditions.
/// Uses a proxy of percentage above MA since Point & Figure data is typically unavailable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BullishPercentConfig {
    /// Moving average period for bullish condition check (default: 20).
    pub ma_period: usize,
    /// Overbought threshold (default: 70).
    pub overbought: f64,
    /// Oversold threshold (default: 30).
    pub oversold: f64,
}

impl BullishPercentConfig {
    pub fn new(ma_period: usize) -> Self {
        Self {
            ma_period,
            overbought: 70.0,
            oversold: 30.0,
        }
    }

    pub fn with_thresholds(ma_period: usize, overbought: f64, oversold: f64) -> Self {
        Self {
            ma_period,
            overbought,
            oversold,
        }
    }
}

impl Default for BullishPercentConfig {
    fn default() -> Self {
        Self {
            ma_period: 20,
            overbought: 70.0,
            oversold: 30.0,
        }
    }
}

/// Output mode for New Highs/New Lows indicator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum NewHighsLowsOutputMode {
    /// New Highs - New Lows (raw difference)
    #[default]
    Difference,
    /// New Highs / New Lows ratio
    Ratio,
    /// (New Highs - New Lows) / (New Highs + New Lows) * 100
    Percent,
}

/// New Highs/New Lows indicator configuration.
///
/// Tracks the difference between stocks making new highs versus new lows
/// over a configurable lookback period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewHighsLowsConfig {
    /// Lookback period for determining new highs/lows (default: 252 for 52-week)
    pub period: usize,
    /// Output calculation mode
    pub output_mode: NewHighsLowsOutputMode,
}

impl NewHighsLowsConfig {
    pub fn new(period: usize) -> Self {
        Self {
            period,
            output_mode: NewHighsLowsOutputMode::Difference,
        }
    }

    pub fn with_mode(period: usize, output_mode: NewHighsLowsOutputMode) -> Self {
        Self { period, output_mode }
    }
}

impl Default for NewHighsLowsConfig {
    fn default() -> Self {
        Self {
            period: 252,
            output_mode: NewHighsLowsOutputMode::Difference,
        }
    }
}

// ============================================================================
// Intermarket Indicators
// ============================================================================

/// Relative Strength (Comparative) configuration.
///
/// Compares asset performance versus a benchmark by calculating the ratio
/// of their prices over time. Useful for identifying outperformance/underperformance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativeStrengthConfig {
    /// Whether to normalize the ratio to start at 100.
    pub normalize: bool,
    /// Period for rate of change calculation (None = no ROC).
    pub roc_period: Option<usize>,
}

impl RelativeStrengthConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self {
            normalize: false,
            roc_period: None,
        }
    }

    /// Create a config with normalization enabled.
    pub fn normalized() -> Self {
        Self {
            normalize: true,
            roc_period: None,
        }
    }

    /// Create a config with ROC momentum calculation.
    pub fn with_momentum(roc_period: usize) -> Self {
        Self {
            normalize: false,
            roc_period: Some(roc_period),
        }
    }

    /// Create a full-featured config with normalization and momentum.
    pub fn full(roc_period: usize) -> Self {
        Self {
            normalize: true,
            roc_period: Some(roc_period),
        }
    }

    /// Set whether to normalize the ratio.
    pub fn set_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set the ROC period for momentum calculation.
    pub fn set_roc_period(mut self, period: Option<usize>) -> Self {
        self.roc_period = period;
        self
    }
}

impl Default for RelativeStrengthConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Statistical Indicators
// ============================================================================

/// Fractal Dimension indicator configuration.
///
/// Measures market complexity/roughness using fractal analysis.
/// - Values near 1.5 indicate random walk
/// - Values < 1.5 indicate trending market
/// - Values > 1.5 indicate mean-reverting market
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractalDimensionConfig {
    /// Lookback period for fractal dimension calculation.
    pub period: usize,
}

impl FractalDimensionConfig {
    pub fn new(period: usize) -> Self {
        Self { period }
    }
}

impl Default for FractalDimensionConfig {
    fn default() -> Self {
        Self { period: 30 }
    }
}

// ============================================================================
// Pattern / Transform Indicators
// ============================================================================

/// Kase Bars configuration.
///
/// Volatility-normalized OHLC bars developed by Cynthia Kase.
/// Normalizes price data by ATR to create bars that account for
/// varying volatility conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaseBarsConfig {
    /// Period for ATR calculation (default: 30).
    pub period: usize,
    /// Smoothing period for baseline calculation (default: same as period).
    pub smoothing: Option<usize>,
}

impl KaseBarsConfig {
    /// Create a new Kase Bars configuration.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            smoothing: None,
        }
    }

    /// Create with custom smoothing period.
    pub fn with_smoothing(period: usize, smoothing: usize) -> Self {
        Self {
            period,
            smoothing: Some(smoothing),
        }
    }

    /// Get the smoothing period (defaults to period if not specified).
    pub fn smoothing_period(&self) -> usize {
        self.smoothing.unwrap_or(self.period)
    }
}

impl Default for KaseBarsConfig {
    fn default() -> Self {
        Self {
            period: 30,
            smoothing: None,
        }
    }
}
