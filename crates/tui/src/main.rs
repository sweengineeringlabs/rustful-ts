//! rustful-tui - Text User Interface for rustful-ts time series library.

mod app;
mod event;
mod services;
mod ui;
mod widgets;

use std::io;
use std::time::Duration;

use crossterm::{
    event::Event,
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;

use app::App;
use event::{handle_key_event, poll_event};
use ui::draw_ui;

fn main() -> anyhow::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = App::new();

    // Main loop
    let result = run_app(&mut terminal, &mut app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(err) = result {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }

    Ok(())
}

fn run_app<B: Backend>(terminal: &mut Terminal<B>, app: &mut App) -> anyhow::Result<()> {
    let tick_rate = Duration::from_millis(100);

    loop {
        // Draw UI
        terminal.draw(|frame| draw_ui(frame, app))?;

        // Clear expired status messages
        app.clear_expired_status();

        // Handle events
        if let Some(event) = poll_event(tick_rate)? {
            match event {
                Event::Key(key) => handle_key_event(app, key),
                Event::Resize(_, _) => {} // Terminal will redraw automatically
                _ => {}
            }
        }

        // Check if we should quit
        if app.should_quit {
            break;
        }
    }

    Ok(())
}
