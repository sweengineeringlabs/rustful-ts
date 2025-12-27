# Architecture Decision Records

Index of architectural decisions for rustful-ts.

## ADR Format

Each ADR follows this structure:
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: What is the issue?
- **Decision**: What was decided?
- **Consequences**: What are the trade-offs?

## Decisions

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](001-workspace-structure.md) | Cargo Workspace Structure | Accepted | 2024-12-27 |
| [002](002-wasm-bindings.md) | WASM as Primary TypeScript Interface | Accepted | 2024-12-27 |
| [003](003-trait-based-design.md) | Trait-Based Algorithm Design | Accepted | 2024-12-27 |

## Adding New ADRs

1. Copy template from `docs/templates/adr-template.md`
2. Name: `NNN-short-title.md`
3. Fill in Context, Decision, Consequences
4. Update this index
