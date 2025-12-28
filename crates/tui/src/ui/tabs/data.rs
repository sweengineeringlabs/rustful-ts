//! Data tab UI.

use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph},
};

use crate::app::App;
use crate::widgets::create_time_series_chart;

/// Draw the Data tab.
pub fn draw_data_tab(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Source info
            Constraint::Min(10),    // Chart
            Constraint::Length(3),  // Stats
        ])
        .split(area);

    // Draw source info
    draw_source_info(frame, chunks[0], app);

    // Draw chart
    draw_data_chart(frame, chunks[1], app);

    // Draw stats
    draw_stats(frame, chunks[2], app);
}

fn draw_source_info(frame: &mut Frame, area: Rect, app: &App) {
    let text = if let Some(data) = &app.data {
        let source = data.source_path.display();
        let points = data.values.len();
        let column = data.column_name.as_deref().unwrap_or("default");
        format!("Source: {source}    Points: {points}    Column: {column}")
    } else {
        "No data loaded. Press [d] to load a CSV or JSON file.".to_string()
    };

    let info = Paragraph::new(text)
        .style(Style::default().fg(Color::White))
        .block(Block::default().borders(Borders::ALL).title(" Data "));

    frame.render_widget(info, area);
}

fn draw_data_chart(frame: &mut Frame, area: Rect, app: &App) {
    if let Some(data) = &app.data {
        let chart = create_time_series_chart(&data.values, &data.stats, "Time Series", Color::Cyan);
        frame.render_widget(chart, area);
    } else {
        // Empty chart placeholder
        let placeholder = Block::default()
            .borders(Borders::ALL)
            .title(" Chart ")
            .style(Style::default().fg(Color::DarkGray));

        // Calculate inner area before rendering the block
        let inner = placeholder.inner(area);
        frame.render_widget(placeholder, area);

        // Center message
        let centered = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(45),
                Constraint::Length(1),
                Constraint::Percentage(45),
            ])
            .split(inner);
        let msg = Paragraph::new("Load data to see chart")
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray));
        frame.render_widget(msg, centered[1]);
    }
}

fn draw_stats(frame: &mut Frame, area: Rect, app: &App) {
    let text = if let Some(data) = &app.data {
        let s = &data.stats;
        format!(
            "Stats: min={:.2}  max={:.2}  mean={:.2}  std={:.2}  count={}",
            s.min, s.max, s.mean, s.std, s.count
        )
    } else {
        "Stats: --".to_string()
    };

    let stats = Paragraph::new(text)
        .style(Style::default().fg(Color::Green))
        .block(Block::default().borders(Borders::ALL).title(" Statistics "));

    frame.render_widget(stats, area);
}
