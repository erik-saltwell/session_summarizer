---
name: Explore Codebase
description: Navigate and understand codebase structure
---

## Explore Codebase

Use local code search and file inspection tools to explore and understand the codebase.

### Steps

1. Start with a global search to locate key modules and entry points.
2. Inspect package structure and imports to build a high-level map.
3. Follow call sites and references for functions/classes relevant to your task.
4. Identify tests that cover the target modules before making changes.
5. Read only the minimal set of files needed to answer the current question.

### Tips

- Start broad, then narrow down to the specific implementation area.
- Track related tests and interfaces while tracing behavior.
- Prefer targeted reads over scanning entire files.

## Token Efficiency Rules
- Start with the smallest useful search scope for the task.
- Escalate to broader scans only when targeted inspection is insufficient.
- Keep tool calls and output concise and task-focused.
