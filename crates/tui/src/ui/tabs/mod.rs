//! Tab-specific UI modules.

mod data;
mod detect;
mod pipeline;
mod predict;
mod server;
mod signal;

pub use data::draw_data_tab;
pub use detect::draw_detect_tab;
pub use pipeline::draw_pipeline_tab;
pub use predict::draw_predict_tab;
pub use server::draw_server_tab;
pub use signal::draw_signal_tab;
