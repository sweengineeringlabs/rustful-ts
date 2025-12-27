# [Crate Name] Overview

## WHAT: [Brief Description]

[1-2 sentence description of what this crate provides]

Key capabilities:
- **Capability 1** - Brief description
- **Capability 2** - Brief description
- **Capability 3** - Brief description

## WHY: [Problem Statement]

**Problems Solved**:
1. [Problem 1] - [Impact]
2. [Problem 2] - [Impact]
3. [Problem 3] - [Impact]

**When to Use**: [Describe scenarios where this crate should be used]

**When NOT to Use**: [Edge cases or alternative solutions]

## HOW: [Usage Guide]

### Basic Example

```rust
// Example usage
use crate_name::feature;

let result = feature::do_something()?;
```

### Feature 1

[Explanation of the feature]

```rust
// Example usage
```

### Feature 2

[Explanation of the feature]

```rust
// Example usage
```

## Relationship to Other Modules

| Module | Relationship |
|--------|--------------|
| [Module A] | [Dependency/Integration] |
| [Module B] | [Dependency/Integration] |

**Integration Points**:
- [Where this crate interfaces with others]
- [Data flows or dependencies]

## Examples and Tests

### Examples

**Location**: [`examples/`](../examples/)

- `basic.rs` - [Description]

### Tests

**Location**: [`tests/`](../tests/)

- `integration.rs` - [Description]

### Testing

```bash
cargo test -p [crate-name]
```

---

**Status**: [Stable/Beta/Planned]
**Roadmap**: See [backlog.md](../backlog.md) | [Framework Backlog](../../../doc/framework-backlog.md)

---

## Template Customization Guide

When using this template for your crate:

1. **Replace all placeholders**:
   - `[Crate Name]` - Your crate name (e.g., rustful-core)
   - `[Brief Description]` - One sentence describing the crate
   - `[Problem Statement]` - What problem this solves
   - `[Usage Guide]` - How to use the crate

2. **Fill in capabilities**:
   - List 3-5 key capabilities
   - Be specific about what each does
   - Use action verbs (Provides, Implements, Enables)

3. **Document relationships**:
   - List all crates this depends on
   - List all crates that depend on this
   - Describe integration points

4. **Add real examples**:
   - Replace placeholder code with working examples
   - Include imports and error handling
   - Show common use cases

5. **Link to actual files**:
   - Update examples/ path to real examples
   - Update tests/ path to real tests
   - Verify all links work

6. **Remove this section** after customization
