"""
Security utilities for RAG Document Q&A.
Provides filename sanitization, file type validation, and rate limiting helpers.
"""

import os
import re
import logging
from pathlib import PurePosixPath
from typing import Optional

logger = logging.getLogger(__name__)

# ── File magic bytes for supported formats ──────────────────────

# Maps file extensions to their expected magic byte signatures.
# Each entry is (offset, bytes) — the signature must appear at the given offset.
FILE_SIGNATURES = {
    ".pdf": [(0, b"%PDF")],
    ".docx": [(0, b"PK\x03\x04")],  # ZIP-based format
    ".txt": [],   # No magic bytes; accept any content
    ".md": [],    # No magic bytes; accept any content
}

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal attacks.

    Strips directory components, removes dangerous characters,
    and ensures the result is a safe, flat filename.

    Args:
        filename: Raw filename (potentially from untrusted upload).

    Returns:
        A safe filename containing only alphanumerics, hyphens,
        underscores, and dots. Returns 'unnamed' if nothing valid remains.

    Examples:
        >>> sanitize_filename("../../etc/passwd")
        'passwd'
        >>> sanitize_filename("my report (2024).pdf")
        'my_report_2024.pdf'
        >>> sanitize_filename("../../../secret.txt")
        'secret.txt'
    """
    # Extract only the final filename component (strip all directory parts)
    name = PurePosixPath(filename).name

    # Also handle Windows-style backslash separators
    if "\\" in name:
        name = name.rsplit("\\", 1)[-1]

    # Remove any characters that aren't alphanumeric, hyphen, underscore, dot, or space
    name = re.sub(r"[^\w\s.\-]", "", name)

    # Replace whitespace with underscores
    name = re.sub(r"\s+", "_", name)

    # Remove leading dots (hidden files)
    name = name.lstrip(".")

    # Collapse multiple dots
    name = re.sub(r"\.{2,}", ".", name)

    if not name:
        return "unnamed"

    return name


def validate_file_magic(content: bytes, extension: str) -> bool:
    """
    Validate file content against expected magic bytes for the given extension.

    This prevents attacks where a malicious file is renamed to a supported
    extension (e.g., an executable renamed to .pdf).

    Args:
        content: Raw file bytes (at least the first 16 bytes).
        extension: Expected file extension (e.g., '.pdf').

    Returns:
        True if the file content matches expected signatures,
        or if the extension has no defined signatures (plain text).
        False if the content does not match.
    """
    ext = extension.lower()

    if ext not in FILE_SIGNATURES:
        logger.warning(f"No signature defined for extension: {ext}")
        return False

    signatures = FILE_SIGNATURES[ext]

    # Extensions with no defined signatures (txt, md) always pass
    if not signatures:
        return True

    for offset, magic in signatures:
        if len(content) < offset + len(magic):
            return False
        if content[offset:offset + len(magic)] != magic:
            return False

    return True


def validate_upload(
    filename: str,
    content: bytes,
    max_size: int,
) -> tuple[str, Optional[str]]:
    """
    Validate an uploaded file for security issues.

    Args:
        filename: Original filename from the upload.
        content: Raw file bytes.
        max_size: Maximum allowed file size in bytes.

    Returns:
        Tuple of (safe_filename, error_message).
        error_message is None if validation passes.
    """
    safe_name = sanitize_filename(filename)

    # Check extension
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return safe_name, (
            f"Unsupported file type: {ext}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # Check size
    if len(content) > max_size:
        return safe_name, (
            f"File exceeds maximum size of "
            f"{max_size // (1024 * 1024)} MB"
        )

    # Check magic bytes
    if not validate_file_magic(content, ext):
        return safe_name, (
            f"File content does not match expected format for {ext}. "
            f"The file may be corrupted or mislabeled."
        )

    return safe_name, None
