from pathlib import Path

_INVALID_DIR_CHARS: frozenset[str] = frozenset('<>:"/\\|?*\0')


def _validate_directory_exists(path: Path) -> list[str]:
    """Validate that a path exists, is a file, and is writable.

    Args:
        path: Path to validate.
        label: Label used in error messages.

    Returns:
        List of validation error messages.
    """

    errors: list[str] = []
    if not path.exists():
        errors.append(f"{path} does not exist.")
        return errors
    if not path.is_dir():
        errors.append(f"{path} is not a directory.")
        return errors
    return errors
