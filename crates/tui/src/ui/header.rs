//! Header bar widget.

use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph},
};

use crate::app::App;

/// Draw the header bar with title.
pub fn draw_header(frame: &mut Frame, area: Rect, _app: &App) {
    let title = Paragraph::new("rustful-ts v0.2.0 - Time Series Analysis Suite")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center)
        .block(Block::default().borders(Borders::BOTTOM));

    frame.render_widget(title, area);
}
