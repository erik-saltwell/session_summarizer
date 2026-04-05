---
name: cmd-test
description: "Run a VSCode launch.json profile by name, automatically fix any Python exceptions, and verify optional success criteria. Use when asked to test or run a specific launch config."
argument-hint: "<profile-name> [and <success-criteria>]"
allowed-tools: Read, Bash, Edit, Write, Glob, Grep
---

## Arguments
$ARGUMENTS

## Available launch profiles
!`cd /home/eriksalt/proj/session_summarizer && python3 .claude/scripts/launch_to_cmd.py --list 2>&1`

## Your task

### Step 1 — Parse arguments
From **$ARGUMENTS**, extract:
- **Profile name**: the first word/phrase (before any "and", "make sure", or "ensure")
- **Success criteria**: everything after that keyword (may be empty)

### Step 2 — Build the command
Run the helper script to get the `uv run` command:
```bash
cd /home/eriksalt/proj/session_summarizer && python3 .claude/scripts/launch_to_cmd.py "<profile-name>"
```
If exit code is non-zero, report the error (profile not found or interactive picker) and stop.

### Step 3 — Run the command
Run the resulting `uv run ...` command from `/home/eriksalt/proj/session_summarizer`.
Capture stdout and stderr.

### Step 4 — Fix exceptions (iterate until success or give up)
If the output contains a Python traceback or error:
1. Read the traceback — identify the source file and line number
2. Read that source file
3. Understand the bug and fix it with Edit
4. Re-run the command

Repeat up to 5 times. If still failing after 5 attempts, report the remaining error and stop.

### Step 5 — Verify success criteria (if provided)
If success criteria were given (e.g., "settings.yaml has a new speakers entry"), verify them now:
- Read the relevant file(s)
- Check the condition
- If the condition is not met, investigate why and fix, then re-run

### Step 6 — Report
Summarize: whether the command succeeded, what (if anything) was fixed, and whether criteria were met.
