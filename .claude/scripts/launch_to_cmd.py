#!/usr/bin/env python3
"""Convert a VSCode launch.json profile to a uv run command line.

Usage:
  python launch_to_cmd.py <profile-name>   # find and print command
  python launch_to_cmd.py --list           # list all profiles
"""

import json
import sys
from pathlib import Path


def load_configs(launch_path: Path) -> list[dict]:
    # launch.json uses JSONC: // comments and trailing commas — strip both
    import re

    text = launch_path.read_text()
    text = re.sub(r"//[^\n]*", "", text)  # remove // line comments
    text = re.sub(r",(\s*[}\]])", r"\1", text)  # remove trailing commas
    return json.loads(text)["configurations"]


def find_profile(configs: list[dict], search: str) -> dict | None:
    # 1. Exact name match
    for c in configs:
        if c["name"] == search:
            return c
    # 2. Strip leading dash from both sides, compare lowercase
    s = search.lstrip("-").lower()
    for c in configs:
        if c["name"].lstrip("-").lower() == s:
            return c
    # 3. Match against args[0] (the CLI subcommand)
    for c in configs:
        args = c.get("args", [])
        if isinstance(args, list) and args and str(args[0]) == search:
            return c
    # 4. Substring match on stripped profile name
    for c in configs:
        if s in c["name"].lstrip("-").lower():
            return c
    return None


def config_to_cmd(config: dict) -> str:
    args = config.get("args", [])
    if isinstance(args, str):
        raise ValueError(f"Profile '{config['name']}' uses interactive arg picker — cannot translate.")
    module = config.get("module", "")
    args_str = " ".join(str(a) for a in args)
    if module == "pytest":
        return f"uv run pytest {args_str}"
    return f"uv run python -m {module} {args_str}"


script_dir = Path(__file__).parent
launch_path = script_dir.parent.parent / ".vscode" / "launch.json"
configs = load_configs(launch_path)

if len(sys.argv) > 1 and sys.argv[1] == "--list":
    for c in configs:
        args = c.get("args", [])
        first_arg = args[0] if isinstance(args, list) and args else "?"
        print(f"  {c['name']!s:30s}  →  {first_arg}")
    sys.exit(0)

if len(sys.argv) < 2:
    print("Usage: launch_to_cmd.py <profile-name>", file=sys.stderr)
    sys.exit(1)

profile = find_profile(configs, sys.argv[1])
if not profile:
    print(f"No profile matching '{sys.argv[1]}' found.\n\nAvailable profiles:", file=sys.stderr)
    for c in configs:
        print(f"  {c['name']}", file=sys.stderr)
    sys.exit(1)

try:
    print(config_to_cmd(profile))
except ValueError as e:
    print(str(e), file=sys.stderr)
    sys.exit(1)
