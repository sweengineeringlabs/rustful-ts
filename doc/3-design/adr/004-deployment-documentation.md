# ADR 004: Deployment Documentation Structure

## Status

Accepted

## Context

rustful-ts requires multiple build tools and configurations depending on the deployment target (npm package, WASM, Rust crate). Users frequently encountered issues with:

**Issues identified**:
- Incorrect wasm-pack commands in documentation
- Missing prerequisites for WASM builds (wasm32 target, wasm-opt)
- No centralized installation guide
- Per-crate prerequisites scattered or missing

## Decision

Create a dedicated `doc/6-deployment/` section with:

1. **Centralized deployment docs**:
   - `overview.md` - Index of deployment documentation
   - `prerequisites.md` - All system requirements in one place
   - `installation.md` - npm and source installation guides
   - `wasm-build.md` - WASM targets, optimization, troubleshooting
   - `typescript-package.md` - Package structure and API usage

2. **Per-crate prerequisites**:
   - Add Prerequisites section to each crate's `doc/overview.md`
   - Include build commands specific to that crate
   - List dependencies (Cargo-managed and external)

3. **Correct build commands**:
   - Standardize on `wasm-pack build crates/rustful-wasm --target nodejs --out-dir ../../ts/pkg`
   - Document all three targets: nodejs, web, bundler

## Consequences

**Positive**:
- Single source of truth for installation
- Users can quickly identify missing prerequisites
- Reduced support issues from incorrect builds
- Each crate is self-documenting

**Negative**:
- Documentation must be kept in sync across locations
- More files to maintain

**Mitigations**:
- Cross-reference between central docs and crate docs
- Include build commands in CI to validate documentation
