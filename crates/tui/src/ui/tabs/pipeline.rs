//! Pipeline tab UI.

use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph},
};

use crate::app::App;

/// Draw the Pipeline tab.
pub fn draw_pipeline_tab(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Info
            Constraint::Min(5),     // Pipeline visualization
            Constraint::Length(5),  // Stats
        ])
        .split(area);

    // Draw info
    draw_pipeline_info(frame, chunks[0], app);

    // Draw pipeline chain
    draw_pipeline_chain(frame, chunks[1], app);

    // Draw stats
    draw_pipeline_stats(frame, chunks[2], app);
}

fn draw_pipeline_info(frame: &mut Frame, area: Rect, app: &App) {
    let text = if app.has_data() {
        let count = app.data.as_ref().map(|d| d.values.len()).unwrap_or(0);
        format!("Input data: {} points    [r] Run pipeline    [c] Configure steps", count)
    } else {
        "Load data first to configure pipeline".to_string()
    };

    let info = Paragraph::new(text)
        .style(Style::default().fg(Color::White))
        .block(Block::default().borders(Borders::ALL).title(" Pipeline "));

    frame.render_widget(info, area);
}

fn draw_pipeline_chain(frame: &mut Frame, area: Rect, app: &App) {
    // Default pipeline steps
    let steps = if app.pipeline_state.steps.is_empty() {
        vec![
            ("Raw", "input"),
            ("Normalize", "mean=0,std=1"),
            ("Difference", "d=1"),
            ("Scale", "0-1"),
            ("Processed", "output"),
        ]
    } else {
        app.pipeline_state
            .steps
            .iter()
            .map(|s| (s.name.as_str(), s.params.as_str()))
            .collect()
    };

    // Build ASCII pipeline visualization
    let mut chain = String::new();
    for (i, (name, _)) in steps.iter().enumerate() {
        if i > 0 {
            chain.push_str(" ──▶ ");
        }
        chain.push_str(&format!("[{}]", name));
    }
    chain.push('\n');

    // Add parameter line
    let mut params_line = String::new();
    for (i, (name, params)) in steps.iter().enumerate() {
        if i > 0 {
            params_line.push_str("     ");
        }
        let width = name.len() + 2;
        let param_display = if params.len() > width {
            &params[..width]
        } else {
            params
        };
        params_line.push_str(&format!("{:^width$}", param_display, width = width));
    }

    let display = format!(
        "\n    {}\n    {}",
        chain, params_line
    );

    let pipeline = Paragraph::new(display)
        .style(Style::default().fg(Color::Cyan))
        .alignment(Alignment::Left)
        .block(Block::default().borders(Borders::ALL).title(" Data Pipeline "));

    frame.render_widget(pipeline, area);
}

fn draw_pipeline_stats(frame: &mut Frame, area: Rect, app: &App) {
    let text = if app.has_data() {
        let input = app.data.as_ref().map(|d| d.values.len()).unwrap_or(0);
        let output = if app.pipeline_state.output_count > 0 {
            app.pipeline_state.output_count
        } else {
            input
        };
        format!(
            "Input: {} points    Output: {} points    Steps: {}",
            input,
            output,
            if app.pipeline_state.steps.is_empty() {
                3
            } else {
                app.pipeline_state.steps.len()
            }
        )
    } else {
        "No data loaded".to_string()
    };

    let stats = Paragraph::new(text)
        .style(Style::default().fg(Color::Green))
        .block(Block::default().borders(Borders::ALL).title(" Pipeline Stats "));

    frame.render_widget(stats, area);
}
