//! Application state management for the TUI.

use std::path::PathBuf;
use std::time::Instant;

/// Main application state.
pub struct App {
    /// Current active tab
    pub current_tab: Tab,
    /// Whether the app should quit
    pub should_quit: bool,
    /// Loaded time series data
    pub data: Option<TimeSeriesData>,
    /// Current input mode
    pub input_mode: InputMode,
    /// Status message with expiry
    pub status_message: Option<(String, Instant)>,
    /// Whether a background operation is running
    pub loading: bool,
    /// Prediction results
    pub predictions: Option<PredictionResult>,
    /// Anomaly detection results
    pub anomalies: Option<AnomalyResult>,
    /// Signal generation results
    pub signals: Option<SignalResult>,
    /// Pipeline state
    pub pipeline_state: PipelineState,
    /// Server status
    pub server_status: Option<ServerStatus>,
    /// Selected model for prediction
    pub selected_model: ModelType,
    /// Forecast steps
    pub forecast_steps: usize,
    /// Selected detector
    pub selected_detector: DetectorType,
    /// Detection threshold
    pub detection_threshold: f64,
    /// Selected signal strategy
    pub selected_strategy: StrategyType,
    /// Short window for signals
    pub short_window: usize,
    /// Long window for signals
    pub long_window: usize,
}

impl Default for App {
    fn default() -> Self {
        Self {
            current_tab: Tab::Data,
            should_quit: false,
            data: None,
            input_mode: InputMode::Normal,
            status_message: None,
            loading: false,
            predictions: None,
            anomalies: None,
            signals: None,
            pipeline_state: PipelineState::default(),
            server_status: None,
            selected_model: ModelType::Auto,
            forecast_steps: 30,
            selected_detector: DetectorType::ZScore,
            detection_threshold: 3.0,
            selected_strategy: StrategyType::SmaCrossover,
            short_window: 10,
            long_window: 30,
        }
    }
}

impl App {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a status message that will be displayed temporarily.
    pub fn set_status(&mut self, message: impl Into<String>) {
        self.status_message = Some((message.into(), Instant::now()));
    }

    /// Clear expired status messages (older than 5 seconds).
    pub fn clear_expired_status(&mut self) {
        if let Some((_, instant)) = &self.status_message {
            if instant.elapsed().as_secs() > 5 {
                self.status_message = None;
            }
        }
    }

    /// Move to next tab.
    pub fn next_tab(&mut self) {
        self.current_tab = self.current_tab.next();
    }

    /// Move to previous tab.
    pub fn previous_tab(&mut self) {
        self.current_tab = self.current_tab.previous();
    }

    /// Jump to a specific tab by number (1-6).
    pub fn goto_tab(&mut self, num: u8) {
        self.current_tab = Tab::from_num(num);
    }

    /// Check if data is loaded.
    pub fn has_data(&self) -> bool {
        self.data.is_some()
    }
}

/// Available tabs in the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tab {
    #[default]
    Data,
    Predict,
    Detect,
    Signal,
    Pipeline,
    Server,
}

impl Tab {
    pub fn next(self) -> Self {
        match self {
            Tab::Data => Tab::Predict,
            Tab::Predict => Tab::Detect,
            Tab::Detect => Tab::Signal,
            Tab::Signal => Tab::Pipeline,
            Tab::Pipeline => Tab::Server,
            Tab::Server => Tab::Data,
        }
    }

    pub fn previous(self) -> Self {
        match self {
            Tab::Data => Tab::Server,
            Tab::Predict => Tab::Data,
            Tab::Detect => Tab::Predict,
            Tab::Signal => Tab::Detect,
            Tab::Pipeline => Tab::Signal,
            Tab::Server => Tab::Pipeline,
        }
    }

    pub fn from_num(num: u8) -> Self {
        match num {
            1 => Tab::Data,
            2 => Tab::Predict,
            3 => Tab::Detect,
            4 => Tab::Signal,
            5 => Tab::Pipeline,
            6 => Tab::Server,
            _ => Tab::Data,
        }
    }

    pub fn index(self) -> usize {
        match self {
            Tab::Data => 0,
            Tab::Predict => 1,
            Tab::Detect => 2,
            Tab::Signal => 3,
            Tab::Pipeline => 4,
            Tab::Server => 5,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Tab::Data => "Data",
            Tab::Predict => "Predict",
            Tab::Detect => "Detect",
            Tab::Signal => "Signal",
            Tab::Pipeline => "Pipeline",
            Tab::Server => "Server",
        }
    }

    pub fn all() -> &'static [Tab] {
        &[
            Tab::Data,
            Tab::Predict,
            Tab::Detect,
            Tab::Signal,
            Tab::Pipeline,
            Tab::Server,
        ]
    }
}

/// Input mode for the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputMode {
    #[default]
    Normal,
    #[allow(dead_code)] // Will be used for text input
    Editing,
    FileDialog,
}

/// Time series data with metadata.
#[derive(Debug, Clone)]
pub struct TimeSeriesData {
    pub values: Vec<f64>,
    pub timestamps: Option<Vec<String>>,
    pub column_name: Option<String>,
    pub source_path: PathBuf,
    pub stats: DataStats,
}

#[allow(dead_code)] // Will be used when data loading is implemented
impl TimeSeriesData {
    pub fn new(values: Vec<f64>, source_path: PathBuf) -> Self {
        let stats = DataStats::calculate(&values);
        Self {
            values,
            timestamps: None,
            column_name: None,
            source_path,
            stats,
        }
    }

    pub fn with_timestamps(mut self, timestamps: Vec<String>) -> Self {
        self.timestamps = Some(timestamps);
        self
    }

    pub fn with_column_name(mut self, name: String) -> Self {
        self.column_name = Some(name);
        self
    }
}

/// Statistics for time series data.
#[derive(Debug, Clone, Default)]
pub struct DataStats {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub count: usize,
}

#[allow(dead_code)] // Will be used when data loading is implemented
impl DataStats {
    pub fn calculate(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let count = data.len();
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = data.iter().sum();
        let mean = sum / count as f64;
        let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
        let std = variance.sqrt();

        Self {
            min,
            max,
            mean,
            std,
            count,
        }
    }
}

/// Prediction model types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelType {
    #[default]
    Auto,
    Arima,
    Ses,
    Holt,
    HoltWinters,
    Knn,
}

impl ModelType {
    pub fn name(self) -> &'static str {
        match self {
            ModelType::Auto => "Auto",
            ModelType::Arima => "ARIMA",
            ModelType::Ses => "SES",
            ModelType::Holt => "Holt",
            ModelType::HoltWinters => "Holt-Winters",
            ModelType::Knn => "KNN",
        }
    }

    pub fn all() -> &'static [ModelType] {
        &[
            ModelType::Auto,
            ModelType::Arima,
            ModelType::Ses,
            ModelType::Holt,
            ModelType::HoltWinters,
            ModelType::Knn,
        ]
    }

    pub fn next(self) -> Self {
        match self {
            ModelType::Auto => ModelType::Arima,
            ModelType::Arima => ModelType::Ses,
            ModelType::Ses => ModelType::Holt,
            ModelType::Holt => ModelType::HoltWinters,
            ModelType::HoltWinters => ModelType::Knn,
            ModelType::Knn => ModelType::Auto,
        }
    }
}

/// Anomaly detector types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetectorType {
    #[default]
    ZScore,
    Iqr,
}

impl DetectorType {
    pub fn name(self) -> &'static str {
        match self {
            DetectorType::ZScore => "Z-Score",
            DetectorType::Iqr => "IQR",
        }
    }

    pub fn all() -> &'static [DetectorType] {
        &[DetectorType::ZScore, DetectorType::Iqr]
    }

    pub fn next(self) -> Self {
        match self {
            DetectorType::ZScore => DetectorType::Iqr,
            DetectorType::Iqr => DetectorType::ZScore,
        }
    }
}

/// Signal strategy types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrategyType {
    #[default]
    SmaCrossover,
    Momentum,
    MeanReversion,
}

impl StrategyType {
    pub fn name(self) -> &'static str {
        match self {
            StrategyType::SmaCrossover => "SMA Crossover",
            StrategyType::Momentum => "Momentum",
            StrategyType::MeanReversion => "Mean Reversion",
        }
    }

    pub fn all() -> &'static [StrategyType] {
        &[
            StrategyType::SmaCrossover,
            StrategyType::Momentum,
            StrategyType::MeanReversion,
        ]
    }

    pub fn next(self) -> Self {
        match self {
            StrategyType::SmaCrossover => StrategyType::Momentum,
            StrategyType::Momentum => StrategyType::MeanReversion,
            StrategyType::MeanReversion => StrategyType::SmaCrossover,
        }
    }
}

/// Prediction result.
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub model_name: String,
    pub original: Vec<f64>,
    pub forecast: Vec<f64>,
    pub lower_bound: Option<Vec<f64>>,
    pub upper_bound: Option<Vec<f64>>,
}

/// Anomaly detection result.
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    pub detector_name: String,
    pub is_anomaly: Vec<bool>,
    pub scores: Vec<f64>,
    #[allow(dead_code)] // Will be used for threshold display
    pub threshold: f64,
}

impl AnomalyResult {
    pub fn anomaly_count(&self) -> usize {
        self.is_anomaly.iter().filter(|&&x| x).count()
    }

    pub fn anomaly_indices(&self) -> Vec<usize> {
        self.is_anomaly
            .iter()
            .enumerate()
            .filter_map(|(i, &is_anom)| if is_anom { Some(i) } else { None })
            .collect()
    }
}

/// Signal generation result.
#[derive(Debug, Clone)]
pub struct SignalResult {
    pub strategy_name: String,
    pub signals: Vec<SignalType>,
    pub trades: Vec<TradeRecord>,
    pub total_return: f64,
    pub win_rate: f64,
}

/// Signal type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    Buy,
    Sell,
    #[allow(dead_code)] // Valid signal state
    Hold,
}

impl SignalType {
    pub fn symbol(self) -> &'static str {
        match self {
            SignalType::Buy => "▲",
            SignalType::Sell => "▼",
            SignalType::Hold => "-",
        }
    }
}

/// Trade record for signal history.
#[derive(Debug, Clone)]
pub struct TradeRecord {
    pub index: usize,
    pub signal: SignalType,
    pub price: f64,
    pub gain_loss: Option<f64>,
}

/// Pipeline state.
#[derive(Debug, Clone, Default)]
pub struct PipelineState {
    pub steps: Vec<PipelineStep>,
    #[allow(dead_code)] // Will be used for pipeline stats
    pub input_count: usize,
    pub output_count: usize,
}

/// Pipeline step.
#[derive(Debug, Clone)]
pub struct PipelineStep {
    pub name: String,
    pub params: String,
}

/// Server status.
#[derive(Debug, Clone)]
pub struct ServerStatus {
    pub running: bool,
    pub address: String,
    pub uptime_secs: u64,
    pub endpoints: Vec<EndpointStatus>,
    pub cpu_percent: f64,
    pub memory_percent: f64,
    pub connections: u32,
    pub requests_per_sec: f64,
}

/// Endpoint status.
#[derive(Debug, Clone)]
pub struct EndpointStatus {
    pub method: String,
    pub path: String,
    pub requests: u64,
    pub avg_latency_ms: f64,
    pub status: String,
}
