# Prerequisites

System requirements and dependencies for rustful-ts.

## For Users (npm package)

| Requirement | Version | Notes |
|-------------|---------|-------|
| Node.js | 18+ | Required for runtime |
| npm/pnpm/yarn | latest | Package manager |

```bash
# Verify
node --version    # v18.0.0+
npm --version     # 9.0.0+
```

## For Developers (building from source)

### Required

| Tool | Version | Install Command | Purpose |
|------|---------|-----------------|---------|
| Rust | 1.75+ | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` | Compile Rust crates |
| wasm32 target | - | `rustup target add wasm32-unknown-unknown` | WASM compilation target |
| wasm-pack | 0.13+ | `cargo install wasm-pack` | Build WASM packages |
| wasm-opt | 117+ | `apt install binaryen` or `brew install binaryen` | WASM optimization |
| Node.js | 18+ | [nodejs.org](https://nodejs.org) | TypeScript runtime |

### Verification

```bash
# Rust toolchain
rustc --version                    # rustc 1.75.0+
rustup target list --installed     # includes wasm32-unknown-unknown
cargo --version                    # cargo 1.75.0+

# WASM tools
wasm-pack --version                # wasm-pack 0.13.0+
wasm-opt --version                 # binaryen version 117+

# Node.js
node --version                     # v18.0.0+
npm --version                      # 9.0.0+
```

## Per-Crate Requirements

| Crate | Additional Requirements | Notes |
|-------|------------------------|-------|
| rustful-core | None | Pure Rust, no external deps |
| rustful-wasm | wasm-pack, wasm32 target | WASM compilation |
| rustful-financial | None | Depends on rustful-core |
| rustful-anomaly | None | Depends on rustful-core |
| rustful-automl | None | Depends on rustful-core |
| rustful-forecast | None | Depends on rustful-core |
| rustful-server | None | Axum-based REST API |
| rustful-cli | None | Clap-based CLI |

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt update
sudo apt install build-essential pkg-config libssl-dev binaryen

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Add WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-pack
cargo install wasm-pack

# Install Node.js (via nvm recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20
```

### macOS

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install binaryen

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Add WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-pack
cargo install wasm-pack

# Install Node.js
brew install node
```

### Windows (WSL2 Recommended)

For Windows, we recommend using WSL2 with Ubuntu. Follow the Linux instructions after setting up WSL2.

```powershell
# Install WSL2
wsl --install -d Ubuntu

# Then follow Linux instructions inside WSL
```

## Troubleshooting

### Missing wasm32 target

```bash
error[E0463]: can't find crate for `core`
```

**Fix**: Install the WASM target:
```bash
rustup target add wasm32-unknown-unknown
```

### wasm-opt not found

```bash
Error: failed to execute `wasm-opt`
```

**Fix**: Install binaryen:
```bash
# Linux
sudo apt install binaryen

# macOS
brew install binaryen
```

### Permission denied during npm install

```bash
npm ERR! EACCES: permission denied
```

**Fix**: Use nvm to manage Node.js or fix npm permissions:
```bash
# Recommended: Use nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20
```
