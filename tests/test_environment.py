"""Phase 0 Environment and Structure Validation Tests."""
import os
import sys

def test_directory_structure():
    """Verify all required top-level and subdirectories exist."""
    required_dirs = [
        os.path.join("data", "raw"),
        os.path.join("data", "processed"),
        os.path.join("src", "agents"),
        os.path.join("src", "data"),
        os.path.join("src", "eval"),
        os.path.join("src", "models"),
        "notebooks",
        "results",
        "docs",
    ]
    for rel_path in required_dirs:
        assert os.path.isdir(rel_path), f"Required directory missing: {rel_path}"

def test_critical_files():
    """Verify required top-level configuration and tracking files exist."""
    required_files = [
        ".gitignore",
        "requirements.txt",
        "PROGRESS_LOG.md",
    ]
    for rel_path in required_files:
        assert os.path.isfile(rel_path), f"Required file missing: {rel_path}"
