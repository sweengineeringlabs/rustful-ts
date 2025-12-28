//! Main layout for the TUI.

use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Tabs},
};

use crate::app::{App, Tab};
use super::footer::draw_footer;
use super::header::draw_header;
use super::tabs::{
    draw_data_tab, draw_detect_tab, draw_pipeline_tab, draw_predict_tab, draw_server_tab,
    draw_signal_tab,
};

/// Draw the main UI layout.
pub fn draw_ui(frame: &mut Frame, app: &App) {
    let size = frame.area();

    // Create main layout: header, tabs, content, footer
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),  // Header
            Constraint::Length(3),  // Tab bar
            Constraint::Min(10),    // Content
            Constraint::Length(2),  // Footer
        ])
        .split(size);

    // Draw header
    draw_header(frame, chunks[0], app);

    // Draw tab bar
    draw_tab_bar(frame, chunks[1], app);

    // Draw content based on current tab
    let content_area = chunks[2];
    match app.current_tab {
        Tab::Data => draw_data_tab(frame, content_area, app),
        Tab::Predict => draw_predict_tab(frame, content_area, app),
        Tab::Detect => draw_detect_tab(frame, content_area, app),
        Tab::Signal => draw_signal_tab(frame, content_area, app),
        Tab::Pipeline => draw_pipeline_tab(frame, content_area, app),
        Tab::Server => draw_server_tab(frame, content_area, app),
    }

    // Draw footer
    draw_footer(frame, chunks[3], app);
}

/// Draw the tab bar.
fn draw_tab_bar(frame: &mut Frame, area: Rect, app: &App) {
    let titles: Vec<Line> = Tab::all()
        .iter()
        .enumerate()
        .map(|(i, tab)| {
            let style = if *tab == app.current_tab {
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            Line::from(format!(" {} {} ", i + 1, tab.name())).style(style)
        })
        .collect();

    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title(" Tabs "))
        .select(app.current_tab.index())
        .style(Style::default().fg(Color::White))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )
        .divider("|");

    frame.render_widget(tabs, area);
}
