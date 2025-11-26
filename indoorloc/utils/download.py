"""
Dataset Download Utilities for IndoorLoc

Provides functions for downloading and managing dataset files.
"""
import os
import hashlib
import zipfile
import io
from pathlib import Path
from typing import Optional, List

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def get_data_home() -> Path:
    """Get the default data cache directory.

    Priority:
    1. INDOORLOC_DATA environment variable
    2. ~/.cache/indoorloc/datasets

    Returns:
        Path to the data home directory.
    """
    data_home = os.environ.get('INDOORLOC_DATA')
    if data_home:
        return Path(data_home)
    return Path.home() / '.cache' / 'indoorloc' / 'datasets'


def check_integrity(filepath: Path, md5: Optional[str] = None) -> bool:
    """Check file integrity using MD5 hash.

    Args:
        filepath: Path to the file to check.
        md5: Expected MD5 hash (optional).

    Returns:
        True if file exists and hash matches (or no hash provided).
    """
    if not filepath.exists():
        return False

    if md5 is None:
        return True

    # Calculate file MD5
    hash_md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_md5.update(chunk)

    return hash_md5.hexdigest() == md5


def download_url(
    url: str,
    root: Path,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
) -> Path:
    """Download a file from URL.

    Args:
        url: URL to download from.
        root: Directory to save the file.
        filename: Filename to save as (defaults to URL basename).
        md5: Expected MD5 hash for verification.

    Returns:
        Path to the downloaded file.

    Raises:
        ImportError: If requests is not installed.
        RuntimeError: If download fails or hash mismatch.
    """
    if not HAS_REQUESTS:
        raise ImportError(
            "requests is required for downloading datasets.\n"
            "Install with: pip install requests"
        )

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.split('/')[-1]

    filepath = root / filename

    # Check if already downloaded
    if check_integrity(filepath, md5):
        return filepath

    print(f"Downloading {url}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get total size for progress
    total_size = int(response.headers.get('content-length', 0))

    # Download with progress
    downloaded = 0
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)

    print()  # New line after progress

    # Verify download
    if md5 is not None and not check_integrity(filepath, md5):
        filepath.unlink()  # Remove corrupted file
        raise RuntimeError(f"MD5 verification failed for {filepath}")

    return filepath


def download_and_extract_zip(
    url: str,
    root: Path,
    extract_files: Optional[List[str]] = None,
    md5: Optional[str] = None,
) -> None:
    """Download and extract a ZIP file.

    Args:
        url: URL to the ZIP file.
        root: Directory to extract to.
        extract_files: List of specific files to extract (None for all).
        md5: Expected MD5 hash of the ZIP file.

    Raises:
        ImportError: If requests is not installed.
        RuntimeError: If download or extraction fails.
    """
    if not HAS_REQUESTS:
        raise ImportError(
            "requests is required for downloading datasets.\n"
            "Install with: pip install requests"
        )

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from {url}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get total size for progress
    total_size = int(response.headers.get('content-length', 0))

    # Download to memory with progress
    downloaded = 0
    content = io.BytesIO()
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            content.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rDownloading: {percent:.1f}%", end='', flush=True)

    print()  # New line after progress

    # Verify MD5 if provided
    if md5 is not None:
        content.seek(0)
        hash_md5 = hashlib.md5()
        for chunk in iter(lambda: content.read(8192), b''):
            hash_md5.update(chunk)
        if hash_md5.hexdigest() != md5:
            raise RuntimeError("MD5 verification failed")
        content.seek(0)

    # Extract ZIP
    print("Extracting files...")
    content.seek(0)

    with zipfile.ZipFile(content) as zf:
        for member in zf.namelist():
            # Get just the filename (ignore directory structure in ZIP)
            filename = Path(member).name

            # Skip directories
            if not filename:
                continue

            # Filter specific files if requested
            if extract_files is not None:
                if filename not in extract_files:
                    continue

            target_path = root / filename

            with zf.open(member) as source, open(target_path, 'wb') as target:
                target.write(source.read())
            print(f"  Extracted: {filename}")

    print(f"Done! Files saved to {root}")


__all__ = [
    'get_data_home',
    'check_integrity',
    'download_url',
    'download_and_extract_zip',
]
