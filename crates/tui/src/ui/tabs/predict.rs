//! Predict tab UI.

use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph, Row, Table},
};

use crate::app::{App, ModelType};
use crate::widgets::create_forecast_chart;

/// Draw the Predict tab.
pub fn draw_predict_tab(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Controls
            Constraint::Min(10),    // Chart
            Constraint::Length(8),  // Summary table
        ])
        .split(area);

    // Draw controls
    draw_predict_controls(frame, chunks[0], app);

    // Draw chart
    draw_forecast_chart_widget(frame, chunks[1], app);

    // Draw summary
    draw_forecast_summary(frame, chunks[2], app);
}

fn draw_predict_controls(frame: &mut Frame, area: Rect, app: &App) {
    let model_list: String = ModelType::all()
        .iter()
        .map(|m| {
            if *m == app.selected_model {
                format!("[{}]", m.name())
            } else {
                m.name().to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("  ");

    let text = format!(
        "Model: {}    Steps: {}    Confidence: 95%    [m] Change model  [↑↓] Adjust steps  [p] Run",
        model_list, app.forecast_steps
    );

    let controls = Paragraph::new(text)
        .style(Style::default().fg(Color::White))
        .block(Block::default().borders(Borders::ALL).title(" Predict "));

    frame.render_widget(controls, area);
}

fn draw_forecast_chart_widget(frame: &mut Frame, area: Rect, app: &App) {
    if let Some(pred) = &app.predictions {
        let chart = create_forecast_chart(
            &pred.original,
            &pred.forecast,
            pred.lower_bound.as_deref(),
            pred.upper_bound.as_deref(),
            &pred.model_name,
        );
        frame.render_widget(chart, area);
    } else if app.data.is_some() {
        let placeholder = Paragraph::new(format!(
            "Press [p] to run {} forecast for {} steps",
            app.selected_model.name(),
            app.forecast_steps
        ))
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Yellow))
        .block(Block::default().borders(Borders::ALL).title(" Forecast "));
        frame.render_widget(placeholder, area);
    } else {
        let placeholder = Paragraph::new("Load data first (press [d] on Data tab)")
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::ALL).title(" Forecast "));
        frame.render_widget(placeholder, area);
    }
}

fn draw_forecast_summary(frame: &mut Frame, area: Rect, app: &App) {
    if let Some(pred) = &app.predictions {
        // Create summary table with forecast at key intervals
        let steps = pred.forecast.len();
        let intervals = [
            (steps / 4, format!("+{} days", steps / 4)),
            (steps / 2, format!("+{} days", steps / 2)),
            (3 * steps / 4, format!("+{} days", 3 * steps / 4)),
            (steps.saturating_sub(1), format!("+{} days", steps)),
        ];

        let rows: Vec<Row> = intervals
            .iter()
            .filter(|(i, _)| *i < pred.forecast.len())
            .map(|(i, label)| {
                let value = pred.forecast[*i];
                let bounds = if let (Some(lower), Some(upper)) =
                    (&pred.lower_bound, &pred.upper_bound)
                {
                    format!("±{:.1}", (upper[*i] - lower[*i]) / 2.0)
                } else {
                    "--".to_string()
                };
                Row::new(vec![label.clone(), format!("{:.2}", value), bounds])
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(12),
                Constraint::Length(12),
                Constraint::Length(10),
            ],
        )
        .header(
            Row::new(vec!["Interval", "Forecast", "CI"])
                .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Forecast Summary ({}) ", pred.model_name)),
        );

        frame.render_widget(table, area);
    } else {
        let placeholder = Paragraph::new("Run prediction to see summary")
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::ALL).title(" Forecast Summary "));
        frame.render_widget(placeholder, area);
    }
}
