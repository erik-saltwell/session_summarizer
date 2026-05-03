## Coding Rules
- Always store dependencies in pyproject.toml so that "uv sync" sets up the environment as much as possible.
- When there is additional setup needed for the project beyond "uv sync" always put install instructions in readme.md
- Use type hints everywhere

## Additional Rules to consider
- src/session_summarizer/commands/AGENTS.md has additional rules and steps for creating new commands.
- src/session_summarizer/settings/AGENTS.md has additional rules for creating or editing settings.

## Agent skills

### Issue tracker

Issues live as markdown files under `.scratch/<feature-slug>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`), used as `Status:` markers in issue files. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
