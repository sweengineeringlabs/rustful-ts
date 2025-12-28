//! Footer bar widget with keyboard shortcuts.

use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph},
};

use crate::app::{App, InputMode, Tab};

/// Draw the footer bar with context-sensitive help.
pub fn draw_footer(frame: &mut Frame, area: Rect, app: &App) {
    let help_text = match app.input_mode {
        InputMode::FileDialog => {
            "Enter file path | Esc: Cancel".to_string()
        }
        InputMode::Editing => {
            "Enter: Confirm | Esc: Cancel".to_string()
        }
        InputMode::Normal => {
            let tab_help = match app.current_tab {
                Tab::Data => "[d] Load data  [r] Refresh",
                Tab::Predict => "[p] Predict  [m] Model  [↑↓] Steps",
                Tab::Detect => "[a] Detect  [m] Method  [↑↓] Threshold",
                Tab::Signal => "[s] Signals  [m] Strategy  [↑↓] Window",
                Tab::Pipeline => "[r] Run pipeline",
                Tab::Server => "[v] View  [r] Refresh  [s] Stop",
            };
            format!(
                "{tab_help}  |  [1-6] Tab  [Tab/←→] Navigate  [q] Quit"
            )
        }
    };

    // Add status message if present
    let display_text = if let Some((status, _)) = &app.status_message {
        format!("{} | {}", status, help_text)
    } else {
        help_text
    };

    let footer = Paragraph::new(display_text)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::TOP));

    frame.render_widget(footer, area);
}
