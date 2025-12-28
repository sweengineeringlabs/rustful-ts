//! Detect tab UI.

use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph, Row, Table},
};

use crate::app::{App, DetectorType};
use crate::widgets::create_anomaly_chart;

/// Draw the Detect tab.
pub fn draw_detect_tab(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Controls
            Constraint::Min(10),    // Chart
            Constraint::Length(10), // Anomaly list
        ])
        .split(area);

    // Draw controls
    draw_detect_controls(frame, chunks[0], app);

    // Draw chart
    draw_anomaly_chart_widget(frame, chunks[1], app);

    // Draw anomaly list
    draw_anomaly_list(frame, chunks[2], app);
}

fn draw_detect_controls(frame: &mut Frame, area: Rect, app: &App) {
    let detector_list: String = DetectorType::all()
        .iter()
        .map(|d| {
            if *d == app.selected_detector {
                format!("[{}]", d.name())
            } else {
                d.name().to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("  ");

    let text = format!(
        "Detector: {}    Threshold: {:.1}σ    [m] Change method  [↑↓] Adjust threshold  [a] Detect",
        detector_list, app.detection_threshold
    );

    let controls = Paragraph::new(text)
        .style(Style::default().fg(Color::White))
        .block(Block::default().borders(Borders::ALL).title(" Detect "));

    frame.render_widget(controls, area);
}

fn draw_anomaly_chart_widget(frame: &mut Frame, area: Rect, app: &App) {
    if let (Some(data), Some(anom)) = (&app.data, &app.anomalies) {
        let chart = create_anomaly_chart(&data.values, &anom.is_anomaly, &anom.detector_name);
        frame.render_widget(chart, area);
    } else if app.data.is_some() {
        let placeholder = Paragraph::new(format!(
            "Press [a] to detect anomalies with {} (threshold: {:.1}σ)",
            app.selected_detector.name(),
            app.detection_threshold
        ))
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Yellow))
        .block(Block::default().borders(Borders::ALL).title(" Anomaly Detection "));
        frame.render_widget(placeholder, area);
    } else {
        let placeholder = Paragraph::new("Load data first (press [d] on Data tab)")
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::ALL).title(" Anomaly Detection "));
        frame.render_widget(placeholder, area);
    }
}

fn draw_anomaly_list(frame: &mut Frame, area: Rect, app: &App) {
    if let Some(anom) = &app.anomalies {
        let indices = anom.anomaly_indices();
        let data = app.data.as_ref();

        let rows: Vec<Row> = indices
            .iter()
            .take(10) // Limit to first 10 for display
            .enumerate()
            .map(|(n, &i)| {
                let value = data.map(|d| d.values[i]).unwrap_or(0.0);
                let score = anom.scores.get(i).unwrap_or(&0.0);
                let timestamp = data
                    .and_then(|d| d.timestamps.as_ref())
                    .and_then(|ts| ts.get(i))
                    .cloned()
                    .unwrap_or_else(|| format!("Index {}", i));
                Row::new(vec![
                    format!("{}", n + 1),
                    timestamp,
                    format!("{:.2}", value),
                    format!("{:.2}σ", score),
                    "▲ Spike".to_string(),
                ])
            })
            .collect();

        let title = format!(
            " Anomalies Found: {} ({}) ",
            anom.anomaly_count(),
            anom.detector_name
        );

        let table = Table::new(
            rows,
            [
                Constraint::Length(4),
                Constraint::Length(15),
                Constraint::Length(12),
                Constraint::Length(10),
                Constraint::Length(10),
            ],
        )
        .header(
            Row::new(vec!["#", "Timestamp", "Value", "Score", "Type"])
                .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(Block::default().borders(Borders::ALL).title(title));

        frame.render_widget(table, area);
    } else {
        let placeholder = Paragraph::new("Run detection to see anomalies")
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::ALL).title(" Anomalies Found: 0 "));
        frame.render_widget(placeholder, area);
    }
}
