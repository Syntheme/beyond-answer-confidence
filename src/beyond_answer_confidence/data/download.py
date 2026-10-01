"""Integrity-checked downloads of pinned public files.

Every file fetched over plain URLs (not through the Hugging Face hub, which
pins revisions itself) has a hard-coded SHA-256. A download is verified in
memory before anything is written, then written to a temporary file in the
destination directory and moved into place with an atomic rename (:func:`os.replace`), so an
interrupted or tampered download never leaves a file that a later run would
trust. Existing local copies are verified on every load.
"""

import hashlib
import logging
import os
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

Fetcher = Callable[[str], bytes]
"""Downloads a URL and returns the body."""


class IntegrityError(ValueError):
    """A downloaded or local file does not have the expected SHA-256."""


def fetch(url: str) -> bytes:
    """Download a public file.

    Args:
        url: A fixed https URL.

    Returns:
        The body.

    Raises:
        ValueError: For a non-https URL.
    """
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https URL {url!r}")
    # A fixed https URL (checked above); no user input reaches it.
    with urllib.request.urlopen(url, timeout=120) as r:  # noqa: S310  # nosec B310
        body: bytes = r.read()
    return body


def sha256_hex(data: bytes) -> str:
    """Return the hex SHA-256 of some bytes.

    Args:
        data: The bytes.

    Returns:
        The digest.
    """
    return hashlib.sha256(data).hexdigest()


def pinned_file(
    url: str, dest: Path, sha256: str, fetcher: Fetcher | None = None
) -> Path:
    """Return a verified local copy of a pinned file, downloading it once.

    Args:
        url: Source URL (https).
        dest: Local path.
        sha256: Expected hex SHA-256 of the file.
        fetcher: Downloads a URL (default :func:`fetch`).

    Returns:
        ``dest``, whose contents have the expected digest.

    Raises:
        IntegrityError: If the local copy or the download has another digest
            (a failed download writes nothing).
    """
    if dest.exists():
        actual = sha256_hex(dest.read_bytes())
        if actual != sha256:
            raise IntegrityError(
                f"{dest} has SHA-256 {actual}, expected {sha256}; delete the "
                "file to download it again"
            )
        return dest
    logger.info("downloading %s", url)
    data = (fetcher or fetch)(url)
    actual = sha256_hex(data)
    if actual != sha256:
        raise IntegrityError(
            f"download of {url} has SHA-256 {actual}, expected {sha256}; "
            "nothing was written"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        Path(tmp).replace(dest)  # atomic rename (os.replace)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return dest
