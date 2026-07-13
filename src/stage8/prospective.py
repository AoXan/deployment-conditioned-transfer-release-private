"""Hard boundaries for prospectively locked data."""

from pathlib import Path


FORBIDDEN_G2F_2024_TOKENS = ("observed_values", "observed-target", "observed_target")


def assert_unlabelled_g2f_2024_path(path: Path) -> None:
    normalized = str(path).lower().replace(" ", "_")
    if any(token in normalized for token in FORBIDDEN_G2F_2024_TOKENS):
        raise PermissionError("G2F 2024 observed-target firewall rejected the path")


def observed_target_presence(path: Path) -> dict[str, bool]:
    """Report directory-entry presence without opening protected label content."""
    return {
        "answer_file_present": path.is_file(),
        "content_read": False,
        "content_hash_computed": False,
    }
