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
