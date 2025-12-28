//! Event handling for the TUI.

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use std::time::Duration;

use crate::app::{App, InputMode, Tab};

/// Handle keyboard events.
pub fn handle_key_event(app: &mut App, key: KeyEvent) {
    // Global shortcuts (work in all modes)
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.should_quit = true;
            return;
        }
        KeyCode::Char('q') if app.input_mode == InputMode::Normal => {
            app.should_quit = true;
            return;
        }
        _ => {}
    }

    match app.input_mode {
        InputMode::Normal => handle_normal_mode(app, key),
        InputMode::Editing => handle_editing_mode(app, key),
        InputMode::FileDialog => handle_file_dialog_mode(app, key),
    }
}

fn handle_normal_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        // Tab navigation
        KeyCode::Tab | KeyCode::Right | KeyCode::Char('l') => app.next_tab(),
        KeyCode::BackTab | KeyCode::Left | KeyCode::Char('h') => app.previous_tab(),
        KeyCode::Char('1') => app.goto_tab(1),
        KeyCode::Char('2') => app.goto_tab(2),
        KeyCode::Char('3') => app.goto_tab(3),
        KeyCode::Char('4') => app.goto_tab(4),
        KeyCode::Char('5') => app.goto_tab(5),
        KeyCode::Char('6') => app.goto_tab(6),

        // Tab-specific actions
        _ => match app.current_tab {
            Tab::Data => handle_data_tab_keys(app, key),
            Tab::Predict => handle_predict_tab_keys(app, key),
            Tab::Detect => handle_detect_tab_keys(app, key),
            Tab::Signal => handle_signal_tab_keys(app, key),
            Tab::Pipeline => handle_pipeline_tab_keys(app, key),
            Tab::Server => handle_server_tab_keys(app, key),
        },
    }
}

fn handle_data_tab_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Char('d') | KeyCode::Enter => {
            app.input_mode = InputMode::FileDialog;
            app.set_status("Enter file path to load...");
        }
        KeyCode::Char('r') => {
            if app.has_data() {
                app.set_status("Refreshing data...");
                // TODO: Reload data from source
            }
        }
        _ => {}
    }
}

fn handle_predict_tab_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Char('p') | KeyCode::Enter => {
            if app.has_data() {
                app.set_status("Running prediction...");
                app.loading = true;
                // TODO: Run prediction
            } else {
                app.set_status("No data loaded. Press 'd' to load data.");
            }
        }
        KeyCode::Char('m') => {
            app.selected_model = app.selected_model.next();
            app.set_status(format!("Model: {}", app.selected_model.name()));
        }
        KeyCode::Up => {
            if app.forecast_steps < 365 {
                app.forecast_steps += 1;
            }
        }
        KeyCode::Down => {
            if app.forecast_steps > 1 {
                app.forecast_steps -= 1;
            }
        }
        _ => {}
    }
}

fn handle_detect_tab_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Char('a') | KeyCode::Enter => {
            if app.has_data() {
                app.set_status("Detecting anomalies...");
                app.loading = true;
                // TODO: Run detection
            } else {
                app.set_status("No data loaded. Press 'd' to load data.");
            }
        }
        KeyCode::Char('m') => {
            app.selected_detector = app.selected_detector.next();
            app.set_status(format!("Detector: {}", app.selected_detector.name()));
        }
        KeyCode::Up => {
            app.detection_threshold += 0.1;
        }
        KeyCode::Down => {
            if app.detection_threshold > 0.1 {
                app.detection_threshold -= 0.1;
            }
        }
        _ => {}
    }
}

fn handle_signal_tab_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Char('s') | KeyCode::Enter => {
            if app.has_data() {
                app.set_status("Generating signals...");
                app.loading = true;
                // TODO: Generate signals
            } else {
                app.set_status("No data loaded. Press 'd' to load data.");
            }
        }
        KeyCode::Char('m') => {
            app.selected_strategy = app.selected_strategy.next();
            app.set_status(format!("Strategy: {}", app.selected_strategy.name()));
        }
        KeyCode::Up => {
            if app.short_window < app.long_window - 1 {
                app.short_window += 1;
            }
        }
        KeyCode::Down => {
            if app.short_window > 1 {
                app.short_window -= 1;
            }
        }
        _ => {}
    }
}

fn handle_pipeline_tab_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Char('r') | KeyCode::Enter => {
            if app.has_data() {
                app.set_status("Running pipeline...");
                // TODO: Run pipeline
            } else {
                app.set_status("No data loaded. Press 'd' to load data.");
            }
        }
        _ => {}
    }
}

fn handle_server_tab_keys(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Char('v') => {
            app.set_status("Starting server...");
            // TODO: Start/view server
        }
        KeyCode::Char('r') => {
            app.set_status("Refreshing server status...");
            // TODO: Refresh status
        }
        KeyCode::Char('s') => {
            app.set_status("Stopping server...");
            // TODO: Stop server
        }
        _ => {}
    }
}

fn handle_editing_mode(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Esc => {
            app.input_mode = InputMode::Normal;
        }
        KeyCode::Enter => {
            app.input_mode = InputMode::Normal;
            // TODO: Process input
        }
        _ => {}
    }
}

fn handle_file_dialog_mode(app: &mut App, key: KeyEvent) {
    if key.code == KeyCode::Esc {
        app.input_mode = InputMode::Normal;
        app.set_status("File loading cancelled.");
    }
}

/// Poll for events with a timeout.
pub fn poll_event(timeout: Duration) -> std::io::Result<Option<Event>> {
    if event::poll(timeout)? {
        Ok(Some(event::read()?))
    } else {
        Ok(None)
    }
}
