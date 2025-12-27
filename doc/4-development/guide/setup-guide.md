# Setup Guide

**Audience**: Developers, Contributors

## WHAT: Development Environment Setup

Complete guide for setting up a development environment for rustful-ts.

**Scope**:
- Required tools and versions
- Installation steps
- Verification procedures

## WHY: Consistent Development

### Problems Addressed

1. **Tool Version Mismatches**
   - Current impact: Different Rust/Node versions cause build failures
   - Consequence: Wasted time debugging environment issues

2. **Missing Dependencies**
   - Current impact: Incomplete setup leads to partial functionality
   - Consequence: Unable to build or test certain components

### Benefits

- **Reproducible Builds**: Same versions across all developers
- **Complete Functionality**: All tools available for full development
- **Quick Onboarding**: Clear steps from zero to productive

## HOW: Installation Steps

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | 1.70+ | Core compilation |
| Node.js | 18+ | TypeScript development |
| wasm-pack | Latest | WASM compilation |
| Git | Any | Version control |

### Step 1: Install Rust

```bash
# Install rustup (Rust toolchain manager)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
rustc --version
cargo --version
```

### Step 2: Install Node.js

```bash
# Using nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
nvm use 18

# Verify
node --version
npm --version
```

Or use Bun as alternative:
```bash
curl -fsSL https://bun.sh/install | bash
bun --version
```

### Step 3: Install wasm-pack

```bash
cargo install wasm-pack

# Verify
wasm-pack --version
```

### Step 4: Clone and Build

```bash
# Clone repository
git clone https://github.com/rustful-ts/rustful-ts.git
cd rustful-ts

# Build Rust crates
cargo build

# Build TypeScript
cd ts
npm install
npm run build
```

### Step 5: Verify Setup

```bash
# Run Rust tests
cargo test

# Run TypeScript tests
cd ts && npm test

# Build WASM (optional, for full development)
cd .. && wasm-pack build --target bundler --out-dir ts/pkg
```

### IDE Setup

**VS Code** (recommended):
```bash
# Install extensions
code --install-extension rust-lang.rust-analyzer
code --install-extension bradlc.vscode-tailwindcss
```

**Settings** (`.vscode/settings.json`):
```json
{
  "rust-analyzer.cargo.features": "all",
  "editor.formatOnSave": true
}
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| `cargo build` fails | Update Rust: `rustup update` |
| WASM build fails | Install wasm-pack: `cargo install wasm-pack` |
| TypeScript errors | Rebuild: `npm run build` |
| Test failures | Check Node version: `node --version` |

## Summary

Development requires Rust 1.70+, Node.js 18+, and wasm-pack for full WASM builds.

**Key Takeaways**:
1. Use rustup for Rust toolchain management
2. Node.js 18+ or Bun for TypeScript
3. wasm-pack only needed for WASM builds

---

**Related Documentation**:
- [Developer Guide](../developer-guide.md) - Development hub
- [Testing Guide](testing-guide.md) - Test organization

**Last Updated**: 2024-12-27
**Version**: 0.2
