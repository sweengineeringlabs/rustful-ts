# Developer Guide

**Audience**: Developers, Contributors

## WHAT: Development Hub

Central guide for developing and contributing to rustful-ts.

**Scope**:
- Development environment setup
- Build and test procedures
- Adding new algorithms
- Contributing guidelines

## WHY: Developer Onboarding

### Problems Addressed

1. **New Contributor Onboarding**
   - Current impact: Contributors need scattered information to get started
   - Consequence: Slower onboarding, inconsistent contributions

2. **Build Complexity**
   - Current impact: Multi-language project (Rust + TypeScript) requires multiple tools
   - Consequence: Build errors, environment issues

### Benefits

- **Single Reference**: One place for all development information
- **Consistent Process**: Standardized build and test procedures
- **Quality Contributions**: Clear guidelines for code quality

## HOW: Development Workflow

### Quick Start

```bash
# Clone
git clone https://github.com/rustful-ts/rustful-ts.git
cd rustful-ts

# Build Rust
cargo build

# Build TypeScript
cd ts && npm install && npm run build

# Run tests
cargo test
npm test
```

### Development Guides

| Guide | Description |
|-------|-------------|
| [Setup Guide](guide/setup-guide.md) | Environment setup |
| [Rust Optimization](guide/rust-optimization.md) | Performance optimization |

### Project Structure

```
rustful-ts/
├── crates/                  # Rust workspace
│   ├── algorithm/           # Core algorithms
│   ├── predictor/           # Predictor SEA module
│   ├── detector/            # Detector SEA module
│   ├── pipeline/            # Pipeline SEA module
│   ├── signal/              # Signal SEA module
│   ├── wasm/                # WASM bindings
│   └── ...                  # Other crates
├── ts/                      # TypeScript package
│   ├── src/                 # Source code
│   ├── tests/               # Tests
│   └── pkg/                 # WASM output
└── doc/                    # Documentation
```

### Build Commands

| Command | Description |
|---------|-------------|
| `cargo build` | Build all Rust crates |
| `cargo build -p rustful-core` | Build specific crate |
| `cargo test` | Run Rust tests |
| `cd ts && npm run build` | Build TypeScript |
| `cd ts && npm test` | Run TypeScript tests |
| `wasm-pack build --target bundler --out-dir ts/pkg` | Build WASM |

### Code Style

**Rust**:
```bash
cargo fmt          # Format code
cargo clippy       # Lint
```

**TypeScript**:
```bash
npm run lint       # ESLint
npm run build      # Type check
```

### Adding a New Algorithm

1. **Rust Implementation** (`crates/rustful-core/src/algorithms/`)
   - Implement `Predictor` trait
   - Add unit tests
   - Register in `mod.rs`

2. **WASM Bindings** (`crates/rustful-wasm/src/lib.rs`)
   - Create wrapper struct
   - Add `#[wasm_bindgen]` annotations

3. **TypeScript Wrapper** (`ts/src/core/`)
   - Implement `Predictor` interface
   - Add JSDoc documentation

4. **Tests and Documentation**
   - Add TypeScript tests
   - Update API reference

### Commit Messages

Use conventional commits:
```
feat: add MyAlgo algorithm
fix: correct ARIMA coefficient estimation
docs: update API reference
refactor: simplify internals
test: add edge case tests
```

### Pull Request Checklist

- [ ] `cargo fmt` - Code formatted
- [ ] `cargo clippy` - No warnings
- [ ] `cargo test` - Tests pass
- [ ] `npm run build` - TypeScript builds
- [ ] `npm test` - TypeScript tests pass
- [ ] Documentation updated

### Best Practices

**DO**:
- Run tests before committing
- Follow conventional commit messages
- Update documentation for public APIs
- Add tests for new features

**DON'T**:
- Commit without testing
- Skip code formatting
- Add dependencies without justification
- Break existing tests

## Summary

Development follows a standard fork-branch-PR workflow with Rust and TypeScript quality checks.

**Key Takeaways**:
1. Build both Rust and TypeScript before testing
2. Follow conventional commit format
3. Run all tests before submitting PR

---

**Related Documentation**:
- [Architecture](../3-design/architecture.md) - System design
- [CONTRIBUTING.md](../../CONTRIBUTING.md) - Contribution process
- [Setup Guide](guide/setup-guide.md) - Environment setup

**Last Updated**: 2025-12-28
**Version**: 0.2
