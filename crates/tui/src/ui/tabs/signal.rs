//! Signal tab UI.

use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph, Row, Table},
};

use crate::app::{App, SignalType, StrategyType};
use crate::widgets::create_signal_chart;

/// Draw the Signal tab.
pub fn draw_signal_tab(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Controls
            Constraint::Min(10),    // Chart
            Constraint::Length(10), // Trade history + performance
        ])
        .split(area);

    // Draw controls
    draw_signal_controls(frame, chunks[0], app);

    // Draw chart
    draw_signal_chart_widget(frame, chunks[1], app);

    // Draw trade history and performance
    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(chunks[2]);

    draw_trade_history(frame, bottom[0], app);
    draw_performance(frame, bottom[1], app);
}

fn draw_signal_controls(frame: &mut Frame, area: Rect, app: &App) {
    let strategy_list: String = StrategyType::all()
        .iter()
        .map(|s| {
            if *s == app.selected_strategy {
                format!("[{}]", s.name())
            } else {
                s.name().to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("  ");

    let text = format!(
        "Strategy: {}    Short: {}  Long: {}    [m] Change  [↑↓] Window  [s] Generate",
        strategy_list, app.short_window, app.long_window
    );

    let controls = Paragraph::new(text)
        .style(Style::default().fg(Color::White))
        .block(Block::default().borders(Borders::ALL).title(" Signal "));

    frame.render_widget(controls, area);
}

fn draw_signal_chart_widget(frame: &mut Frame, area: Rect, app: &App) {
    if let (Some(data), Some(sig)) = (&app.data, &app.signals) {
        let chart = create_signal_chart(&data.values, &sig.signals, &sig.strategy_name);
        frame.render_widget(chart, area);
    } else if app.data.is_some() {
        let placeholder = Paragraph::new(format!(
            "Press [s] to generate signals with {} (short: {}, long: {})",
            app.selected_strategy.name(),
            app.short_window,
            app.long_window
        ))
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Yellow))
        .block(Block::default().borders(Borders::ALL).title(" Price & Signals "));
        frame.render_widget(placeholder, area);
    } else {
        let placeholder = Paragraph::new("Load data first (press [d] on Data tab)")
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::ALL).title(" Price & Signals "));
        frame.render_widget(placeholder, area);
    }
}

fn draw_trade_history(frame: &mut Frame, area: Rect, app: &App) {
    if let Some(sig) = &app.signals {
        let rows: Vec<Row> = sig
            .trades
            .iter()
            .take(10)
            .map(|trade| {
                let signal_style = match trade.signal {
                    SignalType::Buy => Style::default().fg(Color::Green),
                    SignalType::Sell => Style::default().fg(Color::Red),
                    SignalType::Hold => Style::default().fg(Color::Gray),
                };
                let gain = trade
                    .gain_loss
                    .map(|g| format!("{:+.1}%", g * 100.0))
                    .unwrap_or_else(|| "--".to_string());
                Row::new(vec![
                    format!("{}", trade.index),
                    trade.signal.symbol().to_string(),
                    format!("{:.2}", trade.price),
                    gain,
                ])
                .style(signal_style)
            })
            .collect();

        let table = Table::new(
            rows,
            [
                Constraint::Length(8),
                Constraint::Length(8),
                Constraint::Length(12),
                Constraint::Length(10),
            ],
        )
        .header(
            Row::new(vec!["Index", "Signal", "Price", "Gain/Loss"])
                .style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .block(Block::default().borders(Borders::ALL).title(" Signal History "));

        frame.render_widget(table, area);
    } else {
        let placeholder = Paragraph::new("Generate signals to see history")
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::ALL).title(" Signal History "));
        frame.render_widget(placeholder, area);
    }
}

fn draw_performance(frame: &mut Frame, area: Rect, app: &App) {
    let text = if let Some(sig) = &app.signals {
        format!(
            "Total Return: {:+.1}%\nWin Rate: {:.0}%\nTrades: {}",
            sig.total_return * 100.0,
            sig.win_rate * 100.0,
            sig.trades.len()
        )
    } else {
        "Total Return: --\nWin Rate: --\nTrades: --".to_string()
    };

    let perf = Paragraph::new(text)
        .style(Style::default().fg(Color::Cyan))
        .block(Block::default().borders(Borders::ALL).title(" Performance "));

    frame.render_widget(perf, area);
}
