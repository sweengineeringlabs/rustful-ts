//! Bulk indicator evaluators - additional indicators beyond the core 7.

use optimizer_spi::{
    IndicatorEvaluator, IndicatorParams, MarketData, EvaluationResult,
    ParamRange, FloatParamRange, Result, OptimizerError, Signal,
};
use indicator_core::*;

// ============================================================================
// Oscillator Evaluators
// ============================================================================

/// Williams %R Evaluator
pub struct WilliamsREvaluator {
    period_range: ParamRange,
    overbought: f64,
    oversold: f64,
}

impl WilliamsREvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range, overbought: -20.0, oversold: -80.0 }
    }
}

impl IndicatorEvaluator for WilliamsREvaluator {
    fn name(&self) -> &str { "WilliamsR" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(14.0) as usize;
        let wr = WilliamsR::new(period);
        let values = wr.calculate(&data.high, &data.low, &data.close);

        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v > self.overbought { Signal::Sell }
            else if v < self.oversold { Signal::Buy }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// CCI (Commodity Channel Index) Evaluator
pub struct CCIEvaluator {
    period_range: ParamRange,
    overbought: f64,
    oversold: f64,
}

impl CCIEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range, overbought: 100.0, oversold: -100.0 }
    }
}

impl IndicatorEvaluator for CCIEvaluator {
    fn name(&self) -> &str { "CCI" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(20.0) as usize;
        let cci = CCI::new(period);
        let values = cci.calculate(&data.high, &data.low, &data.close);

        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v > self.overbought { Signal::Sell }
            else if v < self.oversold { Signal::Buy }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// ROC (Rate of Change) Evaluator
pub struct ROCEvaluator {
    period_range: ParamRange,
}

impl ROCEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for ROCEvaluator {
    fn name(&self) -> &str { "ROC" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(12.0) as usize;
        let roc = ROC::new(period);
        let values = roc.calculate(&data.close);

        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v > 0.0 { Signal::Buy }
            else if v < 0.0 { Signal::Sell }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// Momentum Evaluator
pub struct MomentumEvaluator {
    period_range: ParamRange,
}

impl MomentumEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for MomentumEvaluator {
    fn name(&self) -> &str { "Momentum" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(10.0) as usize;
        let mom = Momentum::new(period);
        let values = mom.calculate(&data.close);

        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v > 0.0 { Signal::Buy }
            else if v < 0.0 { Signal::Sell }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// TRIX Evaluator
pub struct TRIXEvaluator {
    period_range: ParamRange,
}

impl TRIXEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for TRIXEvaluator {
    fn name(&self) -> &str { "TRIX" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(15.0) as usize;
        let trix = TRIX::new(period);
        let values = trix.calculate(&data.close);

        let signals: Vec<Signal> = values.windows(2).map(|w| {
            if w[1] > w[0] && w[1] > 0.0 { Signal::Buy }
            else if w[1] < w[0] && w[1] < 0.0 { Signal::Sell }
            else { Signal::Hold }
        }).collect();

        let mut final_signals = vec![Signal::Hold];
        final_signals.extend(signals);

        Ok(EvaluationResult { signals: final_signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// Ultimate Oscillator Evaluator
pub struct UltimateOscillatorEvaluator {
    short_range: ParamRange,
    medium_range: ParamRange,
    long_range: ParamRange,
}

impl UltimateOscillatorEvaluator {
    pub fn new(short_range: ParamRange, medium_range: ParamRange, long_range: ParamRange) -> Self {
        Self { short_range, medium_range, long_range }
    }
}

impl IndicatorEvaluator for UltimateOscillatorEvaluator {
    fn name(&self) -> &str { "UltimateOscillator" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let short = params.get("short").unwrap_or(7.0) as usize;
        let medium = params.get("medium").unwrap_or(14.0) as usize;
        let long = params.get("long").unwrap_or(28.0) as usize;

        let uo = UltimateOscillator::new(short, medium, long);
        let values = uo.calculate(&data.high, &data.low, &data.close);

        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v > 70.0 { Signal::Sell }
            else if v < 30.0 { Signal::Buy }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("short".to_string(), self.short_range.clone()),
            ("medium".to_string(), self.medium_range.clone()),
            ("long".to_string(), self.long_range.clone()),
        ]
    }
}

/// Chande Momentum Oscillator Evaluator
pub struct CMOEvaluator {
    period_range: ParamRange,
}

impl CMOEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for CMOEvaluator {
    fn name(&self) -> &str { "CMO" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(14.0) as usize;
        let cmo = ChandeMomentum::new(period);
        let values = cmo.calculate(&data.close);

        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v > 50.0 { Signal::Sell }
            else if v < -50.0 { Signal::Buy }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// Stochastic RSI Evaluator
pub struct StochRSIEvaluator {
    rsi_period_range: ParamRange,
    stoch_period_range: ParamRange,
}

impl StochRSIEvaluator {
    pub fn new(rsi_period_range: ParamRange, stoch_period_range: ParamRange) -> Self {
        Self { rsi_period_range, stoch_period_range }
    }
}

impl IndicatorEvaluator for StochRSIEvaluator {
    fn name(&self) -> &str { "StochRSI" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let rsi_period = params.get("rsi_period").unwrap_or(14.0) as usize;
        let stoch_period = params.get("stoch_period").unwrap_or(14.0) as usize;

        let stoch_rsi = StochasticRSI::new(rsi_period, stoch_period, 3, 3);
        let (k, _d) = stoch_rsi.calculate(&data.close);

        let signals: Vec<Signal> = k.iter().map(|&v| {
            if v > 80.0 { Signal::Sell }
            else if v < 20.0 { Signal::Buy }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: k, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("rsi_period".to_string(), self.rsi_period_range.clone()),
            ("stoch_period".to_string(), self.stoch_period_range.clone()),
        ]
    }
}

/// TSI (True Strength Index) Evaluator
pub struct TSIEvaluator {
    long_range: ParamRange,
    short_range: ParamRange,
}

impl TSIEvaluator {
    pub fn new(long_range: ParamRange, short_range: ParamRange) -> Self {
        Self { long_range, short_range }
    }
}

impl IndicatorEvaluator for TSIEvaluator {
    fn name(&self) -> &str { "TSI" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let long_period = params.get("long").unwrap_or(25.0) as usize;
        let short_period = params.get("short").unwrap_or(13.0) as usize;

        let tsi = TSI::new(long_period, short_period, 7);
        let (values, _signal) = tsi.calculate(&data.close);

        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v > 25.0 { Signal::Sell }
            else if v < -25.0 { Signal::Buy }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("long".to_string(), self.long_range.clone()),
            ("short".to_string(), self.short_range.clone()),
        ]
    }
}

// ============================================================================
// Trend Evaluators
// ============================================================================

/// ADX (Average Directional Index) Evaluator
pub struct ADXEvaluator {
    period_range: ParamRange,
}

impl ADXEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for ADXEvaluator {
    fn name(&self) -> &str { "ADX" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(14.0) as usize;
        let adx = ADX::new(period);
        let output = adx.calculate(&data.high, &data.low, &data.close);

        let signals: Vec<Signal> = output.adx.iter().enumerate().map(|(i, &adx_val)| {
            if adx_val > 25.0 {
                if output.plus_di[i] > output.minus_di[i] { Signal::Buy }
                else { Signal::Sell }
            } else {
                Signal::Hold
            }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: output.adx, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// SuperTrend Evaluator
pub struct SuperTrendEvaluator {
    period_range: ParamRange,
    multiplier_range: FloatParamRange,
}

impl SuperTrendEvaluator {
    pub fn new(period_range: ParamRange, multiplier_range: FloatParamRange) -> Self {
        Self { period_range, multiplier_range }
    }
}

impl IndicatorEvaluator for SuperTrendEvaluator {
    fn name(&self) -> &str { "SuperTrend" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(10.0) as usize;
        let multiplier = params.get("multiplier").unwrap_or(3.0);

        let st = SuperTrend::new(period, multiplier);
        let output = st.calculate(&data.high, &data.low, &data.close);

        let signals: Vec<Signal> = output.direction.iter().map(|&d| {
            if d > 0.0 { Signal::Buy }
            else if d < 0.0 { Signal::Sell }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: output.supertrend, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }

    fn float_parameter_space(&self) -> Vec<(String, FloatParamRange)> {
        vec![("multiplier".to_string(), self.multiplier_range.clone())]
    }
}

/// Parabolic SAR Evaluator
pub struct ParabolicSAREvaluator {
    af_start_range: FloatParamRange,
    af_max_range: FloatParamRange,
}

impl ParabolicSAREvaluator {
    pub fn new(af_start_range: FloatParamRange, af_max_range: FloatParamRange) -> Self {
        Self { af_start_range, af_max_range }
    }
}

impl IndicatorEvaluator for ParabolicSAREvaluator {
    fn name(&self) -> &str { "ParabolicSAR" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let af_start = params.get("af_start").unwrap_or(0.02);
        let af_max = params.get("af_max").unwrap_or(0.2);

        let psar = ParabolicSAR::new(af_start, 0.02, af_max);
        let output = psar.calculate(&data.high, &data.low);

        let signals: Vec<Signal> = output.trend.iter().map(|&t| {
            if t > 0.0 { Signal::Buy }
            else { Signal::Sell }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: output.sar, params: params.clone() })
    }

    fn float_parameter_space(&self) -> Vec<(String, FloatParamRange)> {
        vec![
            ("af_start".to_string(), self.af_start_range.clone()),
            ("af_max".to_string(), self.af_max_range.clone()),
        ]
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![]
    }
}

/// Aroon Evaluator
pub struct AroonEvaluator {
    period_range: ParamRange,
}

impl AroonEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for AroonEvaluator {
    fn name(&self) -> &str { "Aroon" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(25.0) as usize;
        let aroon = Aroon::new(period);
        let output = aroon.calculate(&data.high, &data.low);

        let signals: Vec<Signal> = output.oscillator.iter().map(|&v| {
            if v > 50.0 { Signal::Buy }
            else if v < -50.0 { Signal::Sell }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: output.oscillator, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// Coppock Curve Evaluator
pub struct CoppockEvaluator {
    wma_range: ParamRange,
    roc1_range: ParamRange,
    roc2_range: ParamRange,
}

impl CoppockEvaluator {
    pub fn new(wma_range: ParamRange, roc1_range: ParamRange, roc2_range: ParamRange) -> Self {
        Self { wma_range, roc1_range, roc2_range }
    }
}

impl IndicatorEvaluator for CoppockEvaluator {
    fn name(&self) -> &str { "Coppock" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let wma = params.get("wma").unwrap_or(10.0) as usize;
        let roc1 = params.get("roc1").unwrap_or(14.0) as usize;
        let roc2 = params.get("roc2").unwrap_or(11.0) as usize;

        let cop = CoppockCurve::new(wma, roc1, roc2);
        let values = cop.calculate(&data.close);

        let signals: Vec<Signal> = values.windows(2).map(|w| {
            if w[1] > 0.0 && w[0] <= 0.0 { Signal::Buy }
            else if w[1] < 0.0 && w[0] >= 0.0 { Signal::Sell }
            else { Signal::Hold }
        }).collect();

        let mut final_signals = vec![Signal::Hold];
        final_signals.extend(signals);

        Ok(EvaluationResult { signals: final_signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("wma".to_string(), self.wma_range.clone()),
            ("roc1".to_string(), self.roc1_range.clone()),
            ("roc2".to_string(), self.roc2_range.clone()),
        ]
    }
}

/// DPO (Detrended Price Oscillator) Evaluator
pub struct DPOEvaluator {
    period_range: ParamRange,
}

impl DPOEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for DPOEvaluator {
    fn name(&self) -> &str { "DPO" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(20.0) as usize;
        let dpo = DPO::new(period);
        let values = dpo.calculate(&data.close);

        let signals: Vec<Signal> = values.windows(2).map(|w| {
            if w[1] > 0.0 && w[0] <= 0.0 { Signal::Buy }
            else if w[1] < 0.0 && w[0] >= 0.0 { Signal::Sell }
            else { Signal::Hold }
        }).collect();

        let mut final_signals = vec![Signal::Hold];
        final_signals.extend(signals);

        Ok(EvaluationResult { signals: final_signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

// ============================================================================
// Volume Evaluators
// ============================================================================

/// OBV (On Balance Volume) Evaluator
pub struct OBVEvaluator {
    signal_period_range: ParamRange,
}

impl OBVEvaluator {
    pub fn new(signal_period_range: ParamRange) -> Self {
        Self { signal_period_range }
    }
}

impl IndicatorEvaluator for OBVEvaluator {
    fn name(&self) -> &str { "OBV" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let signal_period = params.get("signal_period").unwrap_or(20.0) as usize;
        let obv = OBV::new();
        let values = obv.calculate(&data.close, &data.volume);

        // Use EMA of OBV as signal line
        let ema = EMA::new(signal_period);
        let signal_line = ema.calculate(&values);

        let signals: Vec<Signal> = values.iter().zip(signal_line.iter()).map(|(&obv_val, &sig)| {
            if obv_val > sig { Signal::Buy }
            else if obv_val < sig { Signal::Sell }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("signal_period".to_string(), self.signal_period_range.clone())]
    }
}

/// MFI (Money Flow Index) Evaluator
pub struct MFIEvaluator {
    period_range: ParamRange,
}

impl MFIEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for MFIEvaluator {
    fn name(&self) -> &str { "MFI" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(14.0) as usize;
        let mfi = MFI::new(period);
        let values = mfi.calculate(&data.high, &data.low, &data.close, &data.volume);

        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v > 80.0 { Signal::Sell }
            else if v < 20.0 { Signal::Buy }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// CMF (Chaikin Money Flow) Evaluator
pub struct CMFEvaluator {
    period_range: ParamRange,
}

impl CMFEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for CMFEvaluator {
    fn name(&self) -> &str { "CMF" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(20.0) as usize;
        let cmf = CMF::new(period);
        let values = cmf.calculate(&data.high, &data.low, &data.close, &data.volume);

        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v > 0.05 { Signal::Buy }
            else if v < -0.05 { Signal::Sell }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// Force Index Evaluator
pub struct ForceIndexEvaluator {
    period_range: ParamRange,
}

impl ForceIndexEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for ForceIndexEvaluator {
    fn name(&self) -> &str { "ForceIndex" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(13.0) as usize;
        let fi = ForceIndex::new(period);
        let values = fi.calculate(&data.close, &data.volume);

        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v > 0.0 { Signal::Buy }
            else if v < 0.0 { Signal::Sell }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// VROC (Volume Rate of Change) Evaluator
pub struct VROCEvaluator {
    period_range: ParamRange,
}

impl VROCEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for VROCEvaluator {
    fn name(&self) -> &str { "VROC" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(14.0) as usize;
        let vroc = VROC::new(period);
        let values = vroc.calculate(&data.volume);

        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v > 50.0 { Signal::Buy }
            else if v < -50.0 { Signal::Sell }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

// ============================================================================
// Band/Channel Evaluators
// ============================================================================

/// Keltner Channels Evaluator
pub struct KeltnerEvaluator {
    ema_period_range: ParamRange,
    atr_period_range: ParamRange,
    multiplier_range: FloatParamRange,
}

impl KeltnerEvaluator {
    pub fn new(ema_period_range: ParamRange, atr_period_range: ParamRange, multiplier_range: FloatParamRange) -> Self {
        Self { ema_period_range, atr_period_range, multiplier_range }
    }
}

impl IndicatorEvaluator for KeltnerEvaluator {
    fn name(&self) -> &str { "Keltner" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let ema_period = params.get("ema_period").unwrap_or(20.0) as usize;
        let atr_period = params.get("atr_period").unwrap_or(10.0) as usize;
        let multiplier = params.get("multiplier").unwrap_or(2.0);

        let kc = KeltnerChannels::new(ema_period, atr_period, multiplier);
        let (middle, upper, lower) = kc.calculate(&data.high, &data.low, &data.close);

        let signals: Vec<Signal> = data.close.iter().enumerate().map(|(i, &c)| {
            if c < lower[i] { Signal::Buy }
            else if c > upper[i] { Signal::Sell }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: middle, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("ema_period".to_string(), self.ema_period_range.clone()),
            ("atr_period".to_string(), self.atr_period_range.clone()),
        ]
    }

    fn float_parameter_space(&self) -> Vec<(String, FloatParamRange)> {
        vec![("multiplier".to_string(), self.multiplier_range.clone())]
    }
}

/// Donchian Channels Evaluator
pub struct DonchianEvaluator {
    period_range: ParamRange,
}

impl DonchianEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for DonchianEvaluator {
    fn name(&self) -> &str { "Donchian" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(20.0) as usize;
        let dc = DonchianChannels::new(period);
        let (upper, middle, lower) = dc.calculate(&data.high, &data.low);

        let signals: Vec<Signal> = data.close.iter().enumerate().map(|(i, &c)| {
            if c >= upper[i] { Signal::Buy }  // Breakout up
            else if c <= lower[i] { Signal::Sell }  // Breakout down
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: middle, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

// ============================================================================
// Moving Average Evaluators (crossover strategies)
// ============================================================================

/// WMA Crossover Evaluator
pub struct WMAEvaluator {
    fast_range: ParamRange,
    slow_range: ParamRange,
}

impl WMAEvaluator {
    pub fn new(fast_range: ParamRange, slow_range: ParamRange) -> Self {
        Self { fast_range, slow_range }
    }
}

impl IndicatorEvaluator for WMAEvaluator {
    fn name(&self) -> &str { "WMA" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let fast = params.get("fast").unwrap_or(10.0) as usize;
        let slow = params.get("slow").unwrap_or(30.0) as usize;

        let fast_wma = WMA::new(fast);
        let slow_wma = WMA::new(slow);

        let fast_values = fast_wma.calculate(&data.close);
        let slow_values = slow_wma.calculate(&data.close);

        let signals: Vec<Signal> = fast_values.iter().zip(slow_values.iter())
            .map(|(&f, &s)| {
                if f > s { Signal::Buy }
                else if f < s { Signal::Sell }
                else { Signal::Hold }
            }).collect();

        Ok(EvaluationResult { signals, indicator_values: fast_values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("fast".to_string(), self.fast_range.clone()),
            ("slow".to_string(), self.slow_range.clone()),
        ]
    }
}

/// DEMA Crossover Evaluator
pub struct DEMAEvaluator {
    fast_range: ParamRange,
    slow_range: ParamRange,
}

impl DEMAEvaluator {
    pub fn new(fast_range: ParamRange, slow_range: ParamRange) -> Self {
        Self { fast_range, slow_range }
    }
}

impl IndicatorEvaluator for DEMAEvaluator {
    fn name(&self) -> &str { "DEMA" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let fast = params.get("fast").unwrap_or(12.0) as usize;
        let slow = params.get("slow").unwrap_or(26.0) as usize;

        let fast_dema = DEMA::new(fast);
        let slow_dema = DEMA::new(slow);

        let fast_values = fast_dema.calculate(&data.close);
        let slow_values = slow_dema.calculate(&data.close);

        let signals: Vec<Signal> = fast_values.iter().zip(slow_values.iter())
            .map(|(&f, &s)| {
                if f > s { Signal::Buy }
                else if f < s { Signal::Sell }
                else { Signal::Hold }
            }).collect();

        Ok(EvaluationResult { signals, indicator_values: fast_values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("fast".to_string(), self.fast_range.clone()),
            ("slow".to_string(), self.slow_range.clone()),
        ]
    }
}

/// TEMA Crossover Evaluator
pub struct TEMAEvaluator {
    fast_range: ParamRange,
    slow_range: ParamRange,
}

impl TEMAEvaluator {
    pub fn new(fast_range: ParamRange, slow_range: ParamRange) -> Self {
        Self { fast_range, slow_range }
    }
}

impl IndicatorEvaluator for TEMAEvaluator {
    fn name(&self) -> &str { "TEMA" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let fast = params.get("fast").unwrap_or(12.0) as usize;
        let slow = params.get("slow").unwrap_or(26.0) as usize;

        let fast_tema = TEMA::new(fast);
        let slow_tema = TEMA::new(slow);

        let fast_values = fast_tema.calculate(&data.close);
        let slow_values = slow_tema.calculate(&data.close);

        let signals: Vec<Signal> = fast_values.iter().zip(slow_values.iter())
            .map(|(&f, &s)| {
                if f > s { Signal::Buy }
                else if f < s { Signal::Sell }
                else { Signal::Hold }
            }).collect();

        Ok(EvaluationResult { signals, indicator_values: fast_values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("fast".to_string(), self.fast_range.clone()),
            ("slow".to_string(), self.slow_range.clone()),
        ]
    }
}

/// HMA (Hull Moving Average) Crossover Evaluator
pub struct HMAEvaluator {
    fast_range: ParamRange,
    slow_range: ParamRange,
}

impl HMAEvaluator {
    pub fn new(fast_range: ParamRange, slow_range: ParamRange) -> Self {
        Self { fast_range, slow_range }
    }
}

impl IndicatorEvaluator for HMAEvaluator {
    fn name(&self) -> &str { "HMA" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let fast = params.get("fast").unwrap_or(9.0) as usize;
        let slow = params.get("slow").unwrap_or(21.0) as usize;

        let fast_hma = HMA::new(fast);
        let slow_hma = HMA::new(slow);

        let fast_values = fast_hma.calculate(&data.close);
        let slow_values = slow_hma.calculate(&data.close);

        let signals: Vec<Signal> = fast_values.iter().zip(slow_values.iter())
            .map(|(&f, &s)| {
                if f > s { Signal::Buy }
                else if f < s { Signal::Sell }
                else { Signal::Hold }
            }).collect();

        Ok(EvaluationResult { signals, indicator_values: fast_values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("fast".to_string(), self.fast_range.clone()),
            ("slow".to_string(), self.slow_range.clone()),
        ]
    }
}

/// KAMA (Kaufman Adaptive Moving Average) Evaluator
pub struct KAMAEvaluator {
    period_range: ParamRange,
    fast_range: ParamRange,
    slow_range: ParamRange,
}

impl KAMAEvaluator {
    pub fn new(period_range: ParamRange, fast_range: ParamRange, slow_range: ParamRange) -> Self {
        Self { period_range, fast_range, slow_range }
    }
}

impl IndicatorEvaluator for KAMAEvaluator {
    fn name(&self) -> &str { "KAMA" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(10.0) as usize;
        let fast = params.get("fast").unwrap_or(2.0) as usize;
        let slow = params.get("slow").unwrap_or(30.0) as usize;

        let kama = KAMA::new(period, fast, slow);
        let values = kama.calculate(&data.close);

        // Price crossover KAMA
        let signals: Vec<Signal> = data.close.iter().zip(values.iter())
            .map(|(&c, &k)| {
                if c > k { Signal::Buy }
                else if c < k { Signal::Sell }
                else { Signal::Hold }
            }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("period".to_string(), self.period_range.clone()),
            ("fast".to_string(), self.fast_range.clone()),
            ("slow".to_string(), self.slow_range.clone()),
        ]
    }
}

// ============================================================================
// Volatility Evaluators
// ============================================================================

/// Historical Volatility Evaluator
pub struct HistVolEvaluator {
    period_range: ParamRange,
}

impl HistVolEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for HistVolEvaluator {
    fn name(&self) -> &str { "HistVol" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(20.0) as usize;
        let hv = HistoricalVolatility::new(period);
        let values = hv.calculate(&data.close);

        // Use volatility for regime detection (low vol = trend, high vol = reversal)
        let avg_vol: f64 = values.iter().sum::<f64>() / values.len() as f64;
        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v < avg_vol * 0.8 { Signal::Buy }  // Low vol, trend likely
            else if v > avg_vol * 1.2 { Signal::Hold }  // High vol, be cautious
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// Choppiness Index Evaluator
pub struct ChoppinessEvaluator {
    period_range: ParamRange,
}

impl ChoppinessEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for ChoppinessEvaluator {
    fn name(&self) -> &str { "Choppiness" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(14.0) as usize;
        let ci = ChoppinessIndex::new(period);
        let values = ci.calculate(&data.high, &data.low, &data.close);

        // Low choppiness = trending, high = ranging
        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v < 38.2 { Signal::Buy }  // Strong trend
            else if v > 61.8 { Signal::Hold }  // Choppy, avoid
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

// ============================================================================
// DSP Evaluators
// ============================================================================

/// Laguerre RSI Evaluator
pub struct LaguerreRSIEvaluator {
    gamma_range: FloatParamRange,
}

impl LaguerreRSIEvaluator {
    pub fn new(gamma_range: FloatParamRange) -> Self {
        Self { gamma_range }
    }
}

impl IndicatorEvaluator for LaguerreRSIEvaluator {
    fn name(&self) -> &str { "LaguerreRSI" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let gamma = params.get("gamma").unwrap_or(0.5);
        let lrsi = LaguerreRSI::new(gamma);
        let values = lrsi.calculate(&data.close);

        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v > 0.8 { Signal::Sell }
            else if v < 0.2 { Signal::Buy }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![]
    }

    fn float_parameter_space(&self) -> Vec<(String, FloatParamRange)> {
        vec![("gamma".to_string(), self.gamma_range.clone())]
    }
}

/// CG Oscillator Evaluator
pub struct CGOscillatorEvaluator {
    period_range: ParamRange,
}

impl CGOscillatorEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for CGOscillatorEvaluator {
    fn name(&self) -> &str { "CGOscillator" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(10.0) as usize;
        let cg = CGOscillator::new(period);
        let (values, _trigger) = cg.calculate(&data.close);

        let signals: Vec<Signal> = values.windows(2).map(|w| {
            if w[1] > w[0] { Signal::Buy }
            else if w[1] < w[0] { Signal::Sell }
            else { Signal::Hold }
        }).collect();

        let mut final_signals = vec![Signal::Hold];
        final_signals.extend(signals);

        Ok(EvaluationResult { signals: final_signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

// ============================================================================
// Composite Evaluators
// ============================================================================

/// TTM Squeeze Evaluator
pub struct TTMSqueezeEvaluator {
    bb_period_range: ParamRange,
    kc_period_range: ParamRange,
}

impl TTMSqueezeEvaluator {
    pub fn new(bb_period_range: ParamRange, kc_period_range: ParamRange) -> Self {
        Self { bb_period_range, kc_period_range }
    }
}

impl IndicatorEvaluator for TTMSqueezeEvaluator {
    fn name(&self) -> &str { "TTMSqueeze" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let bb_period = params.get("bb_period").unwrap_or(20.0) as usize;
        let kc_period = params.get("kc_period").unwrap_or(20.0) as usize;

        let config = TTMSqueezeConfig {
            bb_period,
            bb_std_dev: 2.0,
            kc_ema_period: kc_period,
            kc_atr_period: kc_period,
            kc_multiplier: 1.5,
            momentum_length: 12,
        };
        let ttm = TTMSqueeze::new(config);
        let output = ttm.calculate(&data.high, &data.low, &data.close);

        let signals: Vec<Signal> = output.momentum.iter().zip(output.squeeze_on.iter())
            .map(|(&mom, &squeeze)| {
                if !squeeze && mom > 0.0 { Signal::Buy }
                else if !squeeze && mom < 0.0 { Signal::Sell }
                else { Signal::Hold }
            }).collect();

        Ok(EvaluationResult { signals, indicator_values: output.momentum, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("bb_period".to_string(), self.bb_period_range.clone()),
            ("kc_period".to_string(), self.kc_period_range.clone()),
        ]
    }
}

/// Schaff Trend Cycle Evaluator
pub struct SchaffEvaluator {
    fast_range: ParamRange,
    slow_range: ParamRange,
    cycle_range: ParamRange,
}

impl SchaffEvaluator {
    pub fn new(fast_range: ParamRange, slow_range: ParamRange, cycle_range: ParamRange) -> Self {
        Self { fast_range, slow_range, cycle_range }
    }
}

impl IndicatorEvaluator for SchaffEvaluator {
    fn name(&self) -> &str { "Schaff" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let fast = params.get("fast").unwrap_or(23.0) as usize;
        let slow = params.get("slow").unwrap_or(50.0) as usize;
        let cycle = params.get("cycle").unwrap_or(10.0) as usize;

        let config = SchaffConfig { macd_fast: fast, macd_slow: slow, cycle_period: cycle, factor: 0.5 };
        let schaff = SchaffTrendCycle::new(config);
        let output = schaff.calculate(&data.close);

        let signals: Vec<Signal> = output.stc.iter().map(|&v| {
            if v > 75.0 { Signal::Sell }
            else if v < 25.0 { Signal::Buy }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: output.stc, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("fast".to_string(), self.fast_range.clone()),
            ("slow".to_string(), self.slow_range.clone()),
            ("cycle".to_string(), self.cycle_range.clone()),
        ]
    }
}

/// Elder Ray Evaluator
pub struct ElderRayEvaluator {
    period_range: ParamRange,
}

impl ElderRayEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for ElderRayEvaluator {
    fn name(&self) -> &str { "ElderRay" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(13.0) as usize;
        let er = ElderRay::new(period);
        let output = er.calculate(&data.high, &data.low, &data.close);

        let signals: Vec<Signal> = output.bull_power.iter().zip(output.bear_power.iter())
            .map(|(&b, &br)| {
                if b > 0.0 && br > 0.0 { Signal::Buy }  // Both positive
                else if b < 0.0 && br < 0.0 { Signal::Sell }  // Both negative
                else { Signal::Hold }
            }).collect();

        Ok(EvaluationResult { signals, indicator_values: output.bull_power, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

// ============================================================================
// Additional Simple Evaluators
// ============================================================================

/// Price vs SMA Evaluator (simple trend following)
pub struct PriceSMAEvaluator {
    period_range: ParamRange,
}

impl PriceSMAEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for PriceSMAEvaluator {
    fn name(&self) -> &str { "PriceSMA" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(50.0) as usize;
        let sma = SMA::new(period);
        let values = sma.calculate(&data.close);

        let signals: Vec<Signal> = data.close.iter().zip(values.iter())
            .map(|(&c, &s)| {
                if c > s { Signal::Buy }
                else if c < s { Signal::Sell }
                else { Signal::Hold }
            }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// Price vs EMA Evaluator
pub struct PriceEMAEvaluator {
    period_range: ParamRange,
}

impl PriceEMAEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for PriceEMAEvaluator {
    fn name(&self) -> &str { "PriceEMA" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(20.0) as usize;
        let ema = EMA::new(period);
        let values = ema.calculate(&data.close);

        let signals: Vec<Signal> = data.close.iter().zip(values.iter())
            .map(|(&c, &e)| {
                if c > e { Signal::Buy }
                else if c < e { Signal::Sell }
                else { Signal::Hold }
            }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// ZLEMA Crossover Evaluator
pub struct ZLEMAEvaluator {
    fast_range: ParamRange,
    slow_range: ParamRange,
}

impl ZLEMAEvaluator {
    pub fn new(fast_range: ParamRange, slow_range: ParamRange) -> Self {
        Self { fast_range, slow_range }
    }
}

impl IndicatorEvaluator for ZLEMAEvaluator {
    fn name(&self) -> &str { "ZLEMA" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let fast = params.get("fast").unwrap_or(12.0) as usize;
        let slow = params.get("slow").unwrap_or(26.0) as usize;

        let fast_zlema = ZLEMA::new(fast);
        let slow_zlema = ZLEMA::new(slow);

        let fast_values = fast_zlema.calculate(&data.close);
        let slow_values = slow_zlema.calculate(&data.close);

        let signals: Vec<Signal> = fast_values.iter().zip(slow_values.iter())
            .map(|(&f, &s)| {
                if f > s { Signal::Buy }
                else if f < s { Signal::Sell }
                else { Signal::Hold }
            }).collect();

        Ok(EvaluationResult { signals, indicator_values: fast_values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("fast".to_string(), self.fast_range.clone()),
            ("slow".to_string(), self.slow_range.clone()),
        ]
    }
}

/// Fisher Transform Evaluator
pub struct FisherTransformEvaluator {
    period_range: ParamRange,
}

impl FisherTransformEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for FisherTransformEvaluator {
    fn name(&self) -> &str { "FisherTransform" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(10.0) as usize;
        let ft = FisherTransform::new(period);
        let (values, trigger) = ft.calculate(&data.high, &data.low);

        let signals: Vec<Signal> = values.iter().zip(trigger.iter())
            .map(|(&v, &t)| {
                if v > t && v > 0.0 { Signal::Buy }
                else if v < t && v < 0.0 { Signal::Sell }
                else { Signal::Hold }
            }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// Awesome Oscillator Evaluator
pub struct AwesomeOscillatorEvaluator {
    fast_range: ParamRange,
    slow_range: ParamRange,
}

impl AwesomeOscillatorEvaluator {
    pub fn new(fast_range: ParamRange, slow_range: ParamRange) -> Self {
        Self { fast_range, slow_range }
    }
}

impl IndicatorEvaluator for AwesomeOscillatorEvaluator {
    fn name(&self) -> &str { "AwesomeOscillator" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let fast = params.get("fast").unwrap_or(5.0) as usize;
        let slow = params.get("slow").unwrap_or(34.0) as usize;

        let ao = AwesomeOscillator::with_periods(fast, slow);
        let values = ao.calculate(&data.high, &data.low);

        let signals: Vec<Signal> = values.windows(2).map(|w| {
            if w[1] > 0.0 && w[0] <= 0.0 { Signal::Buy }
            else if w[1] < 0.0 && w[0] >= 0.0 { Signal::Sell }
            else { Signal::Hold }
        }).collect();

        let mut final_signals = vec![Signal::Hold];
        final_signals.extend(signals);

        Ok(EvaluationResult { signals: final_signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("fast".to_string(), self.fast_range.clone()),
            ("slow".to_string(), self.slow_range.clone()),
        ]
    }
}

/// PPO (Percentage Price Oscillator) Evaluator
pub struct PPOEvaluator {
    fast_range: ParamRange,
    slow_range: ParamRange,
}

impl PPOEvaluator {
    pub fn new(fast_range: ParamRange, slow_range: ParamRange) -> Self {
        Self { fast_range, slow_range }
    }
}

impl IndicatorEvaluator for PPOEvaluator {
    fn name(&self) -> &str { "PPO" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let fast = params.get("fast").unwrap_or(12.0) as usize;
        let slow = params.get("slow").unwrap_or(26.0) as usize;

        let ppo = PPO::new(fast, slow, 9);
        let (values, signal, _hist) = ppo.calculate(&data.close);

        let signals: Vec<Signal> = values.iter().zip(signal.iter())
            .map(|(&v, &s)| {
                if v > s { Signal::Buy }
                else if v < s { Signal::Sell }
                else { Signal::Hold }
            }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![
            ("fast".to_string(), self.fast_range.clone()),
            ("slow".to_string(), self.slow_range.clone()),
        ]
    }
}

/// Vortex Indicator Evaluator
pub struct VortexEvaluator {
    period_range: ParamRange,
}

impl VortexEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for VortexEvaluator {
    fn name(&self) -> &str { "Vortex" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(14.0) as usize;
        let vi = VortexIndicator::new(period);
        let (plus_vi, minus_vi) = vi.calculate(&data.high, &data.low, &data.close);

        let signals: Vec<Signal> = plus_vi.iter().zip(minus_vi.iter())
            .map(|(&p, &m)| {
                if p > m { Signal::Buy }
                else if p < m { Signal::Sell }
                else { Signal::Hold }
            }).collect();

        Ok(EvaluationResult { signals, indicator_values: plus_vi, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}

/// KST (Know Sure Thing) Evaluator
pub struct KSTEvaluator {
    roc1_range: ParamRange,
}

impl KSTEvaluator {
    pub fn new(roc1_range: ParamRange) -> Self {
        Self { roc1_range }
    }
}

impl IndicatorEvaluator for KSTEvaluator {
    fn name(&self) -> &str { "KST" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let roc1 = params.get("roc1").unwrap_or(10.0) as usize;

        let kst = KST::with_params([roc1, roc1 * 2, roc1 * 3, roc1 * 4], [10, 10, 10, 15], 9);
        let (values, signal) = kst.calculate(&data.close);

        let signals: Vec<Signal> = values.iter().zip(signal.iter())
            .map(|(&v, &s)| {
                if v > s { Signal::Buy }
                else if v < s { Signal::Sell }
                else { Signal::Hold }
            }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("roc1".to_string(), self.roc1_range.clone())]
    }
}

/// DeMarker Evaluator
pub struct DeMarkerEvaluator {
    period_range: ParamRange,
}

impl DeMarkerEvaluator {
    pub fn new(period_range: ParamRange) -> Self {
        Self { period_range }
    }
}

impl IndicatorEvaluator for DeMarkerEvaluator {
    fn name(&self) -> &str { "DeMarker" }

    fn evaluate(&self, params: &IndicatorParams, data: &MarketData) -> Result<EvaluationResult> {
        let period = params.get("period").unwrap_or(14.0) as usize;
        let dm = DeMarker::new(period);
        let values = dm.calculate(&data.high, &data.low);

        let signals: Vec<Signal> = values.iter().map(|&v| {
            if v > 0.7 { Signal::Sell }
            else if v < 0.3 { Signal::Buy }
            else { Signal::Hold }
        }).collect();

        Ok(EvaluationResult { signals, indicator_values: values, params: params.clone() })
    }

    fn parameter_space(&self) -> Vec<(String, ParamRange)> {
        vec![("period".to_string(), self.period_range.clone())]
    }
}
