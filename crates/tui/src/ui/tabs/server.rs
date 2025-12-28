//! Server tab UI.

use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Gauge, Paragraph, Row, Table},
};

use crate::app::App;

/// Draw the Server tab.
pub fn draw_server_tab(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),  // Status
            Constraint::Min(10),    // Endpoints
            Constraint::Length(5),  // Resources
        ])
        .split(area);

    // Draw status
    draw_server_status(frame, chunks[0], app);

    // Draw endpoints
    draw_endpoints(frame, chunks[1], app);

    // Draw resources
    draw_resources(frame, chunks[2], app);
}

fn draw_server_status(frame: &mut Frame, area: Rect, app: &App) {
    let (status_text, status_style) = if let Some(server) = &app.server_status {
        if server.running {
            let uptime = format_uptime(server.uptime_secs);
            (
                format!(
                    "Status: ● RUNNING    Address: {}    Uptime: {}    Version: 0.2.0",
                    server.address, uptime
                ),
                Style::default().fg(Color::Green),
            )
        } else {
            (
                "Status: ○ STOPPED    [v] Start server".to_string(),
                Style::default().fg(Color::Red),
            )
        }
    } else {
        (
            "Status: ○ NOT CONNECTED    [v] Connect to server    [r] Refresh".to_string(),
            Style::default().fg(Color::Yellow),
        )
    };

    let status = Paragraph::new(status_text)
        .style(status_style)
        .block(Block::default().borders(Borders::ALL).title(" Server "));

    frame.render_widget(status, area);
}

fn draw_endpoints(frame: &mut Frame, area: Rect, app: &App) {
    let rows: Vec<Row> = if let Some(server) = &app.server_status {
        server
            .endpoints
            .iter()
            .map(|ep| {
                let status_style = if ep.status == "OK" {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default().fg(Color::Red)
                };
                Row::new(vec![
                    ep.method.clone(),
                    ep.path.clone(),
                    format!("{}", ep.requests),
                    format!("{:.1}ms", ep.avg_latency_ms),
                    format!("● {}", ep.status),
                ])
                .style(status_style)
            })
            .collect()
    } else {
        // Default endpoints display
        vec![
            Row::new(vec!["GET", "/health/live", "--", "--", "○ --"]),
            Row::new(vec!["GET", "/health/ready", "--", "--", "○ --"]),
            Row::new(vec!["POST", "/api/v1/forecast", "--", "--", "○ --"]),
            Row::new(vec!["POST", "/api/v1/detect", "--", "--", "○ --"]),
            Row::new(vec!["POST", "/api/v1/signal", "--", "--", "○ --"]),
        ]
        .into_iter()
        .map(|r| r.style(Style::default().fg(Color::DarkGray)))
        .collect()
    };

    let table = Table::new(
        rows,
        [
            Constraint::Length(8),
            Constraint::Length(20),
            Constraint::Length(12),
            Constraint::Length(12),
            Constraint::Length(10),
        ],
    )
    .header(
        Row::new(vec!["Method", "Path", "Requests", "Avg Latency", "Status"])
            .style(Style::default().add_modifier(Modifier::BOLD)),
    )
    .block(Block::default().borders(Borders::ALL).title(" Endpoints "));

    frame.render_widget(table, area);
}

fn draw_resources(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
            Constraint::Percentage(25),
        ])
        .split(area);

    let (cpu, mem, conn, rps) = if let Some(server) = &app.server_status {
        (
            server.cpu_percent,
            server.memory_percent,
            server.connections as f64,
            server.requests_per_sec,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    // CPU gauge
    let cpu_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" CPU "))
        .gauge_style(Style::default().fg(Color::Cyan))
        .percent((cpu as u16).min(100))
        .label(format!("{:.0}%", cpu));
    frame.render_widget(cpu_gauge, chunks[0]);

    // Memory gauge
    let mem_gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" Memory "))
        .gauge_style(Style::default().fg(Color::Green))
        .percent((mem as u16).min(100))
        .label(format!("{:.0}%", mem));
    frame.render_widget(mem_gauge, chunks[1]);

    // Connections
    let conn_text = Paragraph::new(format!("{:.0}", conn))
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Yellow))
        .block(Block::default().borders(Borders::ALL).title(" Connections "));
    frame.render_widget(conn_text, chunks[2]);

    // Requests per second
    let rps_text = Paragraph::new(format!("{:.0} req/s", rps))
        .alignment(Alignment::Center)
        .style(Style::default().fg(Color::Magenta))
        .block(Block::default().borders(Borders::ALL).title(" Throughput "));
    frame.render_widget(rps_text, chunks[3]);
}

fn format_uptime(secs: u64) -> String {
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;
    format!("{}h {}m {}s", hours, minutes, seconds)
}
