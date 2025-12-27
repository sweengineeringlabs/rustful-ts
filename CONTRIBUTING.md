# Contributing to rustful-ts

Thank you for your interest in contributing!

## Getting Started

### Prerequisites

- Rust 1.70+
- Node.js 18+ or Bun
- wasm-pack (for WASM builds)

### Setup

```bash
# Clone the repository
git clone https://github.com/rustful-ts/rustful-ts.git
cd rustful-ts

# Build Rust crates
cargo build

# Build TypeScript
cd ts && npm install && npm run build
```

### Running Tests

```bash
# Rust tests
cargo test

# TypeScript tests
cd ts && npm test
```

## Development Workflow

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code refactoring

### Commit Messages

Follow conventional commits:

```
type(scope): description

feat(pipeline): add normalize step
fix(arima): correct coefficient calculation
docs(readme): update quick start example
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

### PR Requirements

- [ ] Tests pass (`cargo test` and `npm test`)
- [ ] Code follows existing style
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow conventions

## Code Style

### Rust

- Follow `rustfmt` defaults
- Run `cargo clippy` before committing
- Add doc comments for public APIs

### TypeScript

- Follow ESLint configuration
- Use TypeScript strict mode
- Add JSDoc comments for exports

## Project Structure

```
rustful-ts/
├── crates/              # Rust crates
│   ├── rustful-core/    # Core algorithms
│   ├── rustful-wasm/    # WASM bindings
│   └── ...
├── ts/                  # TypeScript package
│   └── src/
└── docs/                # Documentation
```

## Adding Features

### New Algorithm

1. Add Rust implementation in `crates/rustful-core/src/algorithms/`
2. Add WASM bindings in `crates/rustful-wasm/src/lib.rs`
3. Add TypeScript wrapper in `ts/src/core/`
4. Add tests and examples
5. Update documentation

### New Module

1. Create crate in `crates/`
2. Add to workspace in `Cargo.toml`
3. Create TypeScript bindings if needed
4. Add module overview in `crates/[name]/doc/overview.md`

## Questions?

- Check [SUPPORT.md](SUPPORT.md) for help
- Open a GitHub Discussion for questions
