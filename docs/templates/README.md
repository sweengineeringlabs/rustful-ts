# Documentation Templates

Templates for rustful-ts documentation following SEA (Stratified Encapsulation Architecture) patterns.

## Available Templates

| Template | Use For |
|----------|---------|
| [crate-overview-template.md](crate-overview-template.md) | Module/crate documentation |
| [framework-doc-template.md](framework-doc-template.md) | Framework-wide documentation |

## WHAT-WHY-HOW Structure

All documentation follows this structure:

### WHAT: Clear Description
- Define what is being documented
- Scope and boundaries
- Key capabilities

### WHY: Problem and Motivation
- Problems being solved
- Impact if not addressed
- Benefits of the solution

### HOW: Implementation and Application
- Practical examples
- Usage patterns
- Best practices

## Documentation Rules

| Location | Format | Audience Section |
|----------|--------|------------------|
| Crate docs (`crates/*/doc/`) | WHAT-WHY-HOW | No |
| Framework docs (`docs/`) | Audience + WHAT-WHY-HOW | Yes |

## Creating Documentation

### Crate/Module Documentation

1. Copy `crate-overview-template.md`
2. Save to `crates/[name]/doc/overview.md`
3. Fill in WHAT-WHY-HOW sections
4. **Do NOT add Audience section**

### Framework Documentation

1. Copy `framework-doc-template.md`
2. Save to appropriate `docs/` subdirectory
3. **Define Audience** (required!)
4. Fill in WHAT-WHY-HOW sections
