# Documentation Templates

Templates for rustful-ts documentation following the **Audience + WHAT-WHY-HOW** structure from the SEA Documentation Template Engine.

## Quick Reference

| Template | Use For | Target Location |
|----------|---------|-----------------|
| [crate-overview-template.md](crate-overview-template.md) | Module/crate documentation | `crates/*/doc/overview.md` |
| [framework-doc-template.md](framework-doc-template.md) | Framework-wide documentation | `doc/3-design/*.md`, `doc/4-development/guide/*.md` |

## How to Use

1. **Choose a template** based on what you're documenting
2. **Copy it** to your target location
3. **Replace placeholders** marked with `[BRACKETS]`
4. **Update dates** (`YYYY-MM-DD` → actual dates)

## Template Structure

### Crate/Module Overview Template

```markdown
# [Module/Crate Name] Overview

> **Scope**: High-level overview only. Implementation details belong in [Developer Guide](...).

## Audience
## WHAT
## WHY
## HOW
## Documentation

**Status**: [Stable / Beta / Alpha]
```

### Framework Documentation Template

```markdown
# [Document Title]

**Audience**: [Who should read this]

## WHAT: [Description]
## WHY: [Motivation and benefits]
## HOW: [Implementation]
  - Examples
  - Best practices (✅ DO / ❌ DON'T)
  - Decision Matrix
## Summary (with Key Takeaways)

**Last Updated**: YYYY-MM-DD
**Version**: X.Y
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
