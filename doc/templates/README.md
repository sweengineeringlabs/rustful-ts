# Documentation Templates

Templates for rustful-ts documentation following SEA (Stratified Encapsulation Architecture) patterns.

## Quick Reference

| Template | Use For |
|----------|---------|
| [crate-overview-template.md](crate-overview-template.md) | Module/crate documentation |
| [framework-doc-template.md](framework-doc-template.md) | Framework-wide documentation |

## How to Use

1. **Choose a template** based on what you're documenting
2. **Copy it** to your target location
3. **Replace placeholders** marked with `[BRACKETS]`
4. **Follow the customization guide** at the end of each template
5. **Remove the customization guide** before publishing

## Template Structure

Each template follows this structure:

```
# Title
**Audience**: [Who should read this]  (framework docs only)

## WHAT: [Description]
## WHY: [Motivation and benefits]
## HOW: [Implementation]
  - Examples
  - Best practices

## Summary (with Key Takeaways)
## Template Customization Guide
```

## Placeholder Convention

Templates use this placeholder format:
- `[Name]` - Replace with actual name
- `[Description]` - Your description
- `YYYY-MM-DD` - Actual dates
- `X.Y` - Version numbers

## Documentation Rules

| Location | Format | Audience Section |
|----------|--------|------------------|
| Crate docs (`crates/*/doc/`) | WHAT-WHY-HOW | No |
| Framework docs (`doc/`) | Audience + WHAT-WHY-HOW | Yes |

## Creating Documentation

### Crate/Module Documentation

1. Copy `crate-overview-template.md`
2. Save to `crates/[name]/doc/overview.md`
3. Fill in WHAT-WHY-HOW sections
4. **Do NOT add Audience section**
5. Remove Template Customization Guide

### Framework Documentation

1. Copy `framework-doc-template.md`
2. Save to appropriate `doc/` subdirectory
3. **Define Audience** (required!)
4. Fill in WHAT-WHY-HOW sections
5. Add Summary with Key Takeaways
6. Remove Template Customization Guide
