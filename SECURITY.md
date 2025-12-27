# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| < 0.2   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please report it responsibly.

### How to Report

1. **Do NOT** open a public GitHub issue for security vulnerabilities
2. Email security concerns to the maintainers (see SUPPORT.md for contact)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Resolution Timeline**: Depends on severity
  - Critical: 24-48 hours
  - High: 7 days
  - Medium: 30 days
  - Low: 90 days

### Security Best Practices

When using rustful-ts:

1. **Input Validation**: Always validate time series data before processing
2. **Resource Limits**: Set appropriate limits for data size and computation time
3. **Dependency Updates**: Keep rustful-ts and its dependencies updated
4. **WASM Isolation**: The WASM module runs in a sandboxed environment

### Scope

This security policy covers:
- The Rust crates in `crates/`
- The TypeScript package in `ts/`
- The REST API in `crates/rustful-server/`
- The CLI in `crates/rustful-cli/`

### Recognition

We appreciate responsible disclosure and will acknowledge reporters in release notes (unless anonymity is requested).
