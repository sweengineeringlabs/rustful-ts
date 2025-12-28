//! Chart widgets for time series visualization.

use ratatui::{
    prelude::*,
    symbols::Marker,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType},
};

use crate::app::{DataStats, SignalType};

/// Create a basic time series chart.
pub fn create_time_series_chart<'a>(
    data: &'a [f64],
    stats: &DataStats,
    title: &'a str,
    color: Color,
) -> Chart<'a> {
    let points: Vec<(f64, f64)> = data
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v))
        .collect();

    // We need to own the data for the dataset
    let dataset = Dataset::default()
        .name(title)
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(color))
        .data(Box::leak(points.into_boxed_slice()));

    let x_max = (data.len() as f64).max(1.0);
    let y_min = stats.min - stats.std * 0.1;
    let y_max = stats.max + stats.std * 0.1;

    Chart::new(vec![dataset])
        .block(Block::default().borders(Borders::ALL).title(format!(" {} ", title)))
        .x_axis(
            Axis::default()
                .title("Time")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, x_max])
                .labels(vec![
                    Span::raw("0"),
                    Span::raw(format!("{}", data.len() / 2)),
                    Span::raw(format!("{}", data.len())),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("Value")
                .style(Style::default().fg(Color::Gray))
                .bounds([y_min, y_max])
                .labels(vec![
                    Span::raw(format!("{:.1}", y_min)),
                    Span::raw(format!("{:.1}", (y_min + y_max) / 2.0)),
                    Span::raw(format!("{:.1}", y_max)),
                ]),
        )
}

/// Create a forecast chart with original data and predictions.
pub fn create_forecast_chart<'a>(
    original: &'a [f64],
    forecast: &'a [f64],
    lower_bound: Option<&'a [f64]>,
    upper_bound: Option<&'a [f64]>,
    model_name: &'a str,
) -> Chart<'a> {
    let n = original.len();

    // Original data points
    let orig_points: Vec<(f64, f64)> = original
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v))
        .collect();

    // Forecast points (starting from end of original)
    let forecast_points: Vec<(f64, f64)> = forecast
        .iter()
        .enumerate()
        .map(|(i, &v)| ((n + i) as f64, v))
        .collect();

    let mut datasets = vec![
        Dataset::default()
            .name("Historical")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(Box::leak(orig_points.into_boxed_slice())),
        Dataset::default()
            .name("Forecast")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Yellow))
            .data(Box::leak(forecast_points.into_boxed_slice())),
    ];

    // Add confidence interval if available
    if let (Some(lower), Some(upper)) = (lower_bound, upper_bound) {
        let lower_points: Vec<(f64, f64)> = lower
            .iter()
            .enumerate()
            .map(|(i, &v)| ((n + i) as f64, v))
            .collect();
        let upper_points: Vec<(f64, f64)> = upper
            .iter()
            .enumerate()
            .map(|(i, &v)| ((n + i) as f64, v))
            .collect();

        datasets.push(
            Dataset::default()
                .name("Lower CI")
                .marker(Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::DarkGray))
                .data(Box::leak(lower_points.into_boxed_slice())),
        );
        datasets.push(
            Dataset::default()
                .name("Upper CI")
                .marker(Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::DarkGray))
                .data(Box::leak(upper_points.into_boxed_slice())),
        );
    }

    // Calculate bounds
    let all_values: Vec<f64> = original
        .iter()
        .chain(forecast.iter())
        .chain(lower_bound.into_iter().flatten())
        .chain(upper_bound.into_iter().flatten())
        .cloned()
        .collect();

    let y_min = all_values.iter().cloned().fold(f64::INFINITY, f64::min) * 0.95;
    let y_max = all_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1.05;
    let x_max = (n + forecast.len()) as f64;

    Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Forecast ({}) ", model_name)),
        )
        .x_axis(
            Axis::default()
                .title("Time")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, x_max])
                .labels(vec![
                    Span::raw("History"),
                    Span::styled("Now", Style::default().fg(Color::Yellow)),
                    Span::raw("Forecast"),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("Value")
                .style(Style::default().fg(Color::Gray))
                .bounds([y_min, y_max])
                .labels(vec![
                    Span::raw(format!("{:.1}", y_min)),
                    Span::raw(format!("{:.1}", (y_min + y_max) / 2.0)),
                    Span::raw(format!("{:.1}", y_max)),
                ]),
        )
}

/// Create an anomaly detection chart with markers.
pub fn create_anomaly_chart<'a>(
    data: &'a [f64],
    is_anomaly: &'a [bool],
    detector_name: &'a str,
) -> Chart<'a> {
    // Normal data points
    let normal_points: Vec<(f64, f64)> = data
        .iter()
        .enumerate()
        .filter(|(i, _)| !is_anomaly.get(*i).unwrap_or(&false))
        .map(|(i, &v)| (i as f64, v))
        .collect();

    // Anomaly points
    let anomaly_points: Vec<(f64, f64)> = data
        .iter()
        .enumerate()
        .filter(|(i, _)| *is_anomaly.get(*i).unwrap_or(&false))
        .map(|(i, &v)| (i as f64, v))
        .collect();

    let datasets = vec![
        Dataset::default()
            .name("Normal")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(Box::leak(normal_points.into_boxed_slice())),
        Dataset::default()
            .name("Anomaly")
            .marker(Marker::Block)
            .graph_type(GraphType::Scatter)
            .style(Style::default().fg(Color::Red))
            .data(Box::leak(anomaly_points.into_boxed_slice())),
    ];

    let y_min = data.iter().cloned().fold(f64::INFINITY, f64::min) * 0.95;
    let y_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1.05;
    let x_max = data.len() as f64;

    Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Anomaly Detection ({}) ", detector_name)),
        )
        .x_axis(
            Axis::default()
                .title("Time")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, x_max])
                .labels(vec![
                    Span::raw("0"),
                    Span::raw(format!("{}", data.len() / 2)),
                    Span::raw(format!("{}", data.len())),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("Value")
                .style(Style::default().fg(Color::Gray))
                .bounds([y_min, y_max])
                .labels(vec![
                    Span::raw(format!("{:.1}", y_min)),
                    Span::raw(format!("{:.1}", (y_min + y_max) / 2.0)),
                    Span::raw(format!("{:.1}", y_max)),
                ]),
        )
}

/// Create a signal chart with buy/sell markers.
pub fn create_signal_chart<'a>(
    data: &'a [f64],
    signals: &'a [SignalType],
    strategy_name: &'a str,
) -> Chart<'a> {
    // Price data
    let price_points: Vec<(f64, f64)> = data
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v))
        .collect();

    // Buy signals
    let buy_points: Vec<(f64, f64)> = signals
        .iter()
        .enumerate()
        .filter(|(_, s)| **s == SignalType::Buy)
        .filter_map(|(i, _)| data.get(i).map(|&v| (i as f64, v)))
        .collect();

    // Sell signals
    let sell_points: Vec<(f64, f64)> = signals
        .iter()
        .enumerate()
        .filter(|(_, s)| **s == SignalType::Sell)
        .filter_map(|(i, _)| data.get(i).map(|&v| (i as f64, v)))
        .collect();

    let datasets = vec![
        Dataset::default()
            .name("Price")
            .marker(Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(Box::leak(price_points.into_boxed_slice())),
        Dataset::default()
            .name("Buy")
            .marker(Marker::Block)
            .graph_type(GraphType::Scatter)
            .style(Style::default().fg(Color::Green))
            .data(Box::leak(buy_points.into_boxed_slice())),
        Dataset::default()
            .name("Sell")
            .marker(Marker::Block)
            .graph_type(GraphType::Scatter)
            .style(Style::default().fg(Color::Red))
            .data(Box::leak(sell_points.into_boxed_slice())),
    ];

    let y_min = data.iter().cloned().fold(f64::INFINITY, f64::min) * 0.95;
    let y_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1.05;
    let x_max = data.len() as f64;

    Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Price & Signals ({}) ", strategy_name)),
        )
        .x_axis(
            Axis::default()
                .title("Time")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, x_max])
                .labels(vec![
                    Span::raw("0"),
                    Span::raw(format!("{}", data.len() / 2)),
                    Span::raw(format!("{}", data.len())),
                ]),
        )
        .y_axis(
            Axis::default()
                .title("Price")
                .style(Style::default().fg(Color::Gray))
                .bounds([y_min, y_max])
                .labels(vec![
                    Span::raw(format!("{:.1}", y_min)),
                    Span::raw(format!("{:.1}", (y_min + y_max) / 2.0)),
                    Span::raw(format!("{:.1}", y_max)),
                ]),
        )
}
