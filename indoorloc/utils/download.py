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


def download_from_zenodo(
    record_id: str,
    root: Path,
    filenames: Optional[List[str]] = None,
) -> List[Path]:
    """Download files from a Zenodo record.

    Args:
        record_id: Zenodo record ID (e.g., '1066041').
        root: Directory to save files.
        filenames: List of specific filenames to download (None for all).

    Returns:
        List of paths to downloaded files.

    Raises:
        ImportError: If requests is not installed.
        RuntimeError: If download fails.

    Example:
        >>> # Download specific files from Zenodo record
        >>> paths = download_from_zenodo(
        ...     record_id='1066041',
        ...     root=Path('data/dataset'),
        ...     filenames=['train.csv', 'test.csv']
        ... )
    """
    if not HAS_REQUESTS:
        raise ImportError(
            "requests is required for downloading from Zenodo.\n"
            "Install with: pip install requests"
        )

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # Query Zenodo API for record metadata
    api_url = f"https://zenodo.org/api/records/{record_id}"
    print(f"Fetching Zenodo record {record_id}...")

    response = requests.get(api_url)
    response.raise_for_status()

    record_data = response.json()

    # Get file list from record
    files = record_data.get('files', [])
    if not files:
        raise RuntimeError(f"No files found in Zenodo record {record_id}")

    # Filter files if specific filenames requested
    if filenames is not None:
        files = [f for f in files if f['key'] in filenames]
        if not files:
            raise RuntimeError(
                f"None of the requested files {filenames} found in record"
            )

    downloaded_paths = []

    # Download each file
    for file_info in files:
        filename = file_info['key']
        file_url = file_info['links']['self']
        file_size = file_info['size']
        file_checksum = file_info.get('checksum', '').split(':')[-1]  # md5:hash

        filepath = root / filename

        # Check if already downloaded
        if filepath.exists():
            print(f"File {filename} already exists, skipping...")
            downloaded_paths.append(filepath)
            continue

        print(f"Downloading {filename} ({file_size / 1024 / 1024:.1f} MB)...")

        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        # Download with progress
        downloaded = 0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    percent = (downloaded / file_size) * 100
                    print(f"\r  Progress: {percent:.1f}%", end='', flush=True)

        print()  # New line
        downloaded_paths.append(filepath)

    print(f"Done! Downloaded {len(downloaded_paths)} file(s) to {root}")
    return downloaded_paths


def download_from_uci(
    dataset_name: str,
    root: Path,
    filenames: List[str],
    base_url: Optional[str] = None,
) -> List[Path]:
    """Download files from UCI Machine Learning Repository.

    Args:
        dataset_name: UCI dataset name (e.g., 'indoor-localization').
        root: Directory to save files.
        filenames: List of filenames to download.
        base_url: Optional custom base URL (defaults to UCI ML database).

    Returns:
        List of paths to downloaded files.

    Raises:
        ImportError: If requests is not installed.
        RuntimeError: If download fails.

    Example:
        >>> # Download files from UCI repository
        >>> paths = download_from_uci(
        ...     dataset_name='indoor-localization',
        ...     root=Path('data/dataset'),
        ...     filenames=['data.csv', 'README.txt']
        ... )
    """
    if not HAS_REQUESTS:
        raise ImportError(
            "requests is required for downloading from UCI.\n"
            "Install with: pip install requests"
        )

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # Default UCI ML Repository base URL
    if base_url is None:
        base_url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/{dataset_name}"

    downloaded_paths = []

    # Download each file
    for filename in filenames:
        filepath = root / filename

        # Check if already downloaded
        if filepath.exists():
            print(f"File {filename} already exists, skipping...")
            downloaded_paths.append(filepath)
            continue

        # Construct full URL
        file_url = f"{base_url.rstrip('/')}/{filename}"

        print(f"Downloading {filename} from UCI...")

        try:
            response = requests.get(file_url, stream=True)
            response.raise_for_status()

            # Get file size if available
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
                            print(f"\r  Progress: {percent:.1f}%", end='', flush=True)

            print()  # New line
            downloaded_paths.append(filepath)

        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to download {filename}: {e}")
            continue

    if not downloaded_paths:
        raise RuntimeError(f"Failed to download any files from {base_url}")

    print(f"Done! Downloaded {len(downloaded_paths)} file(s) to {root}")
    return downloaded_paths


def download_from_figshare(
    article_id: str,
    root: Path,
    filenames: Optional[List[str]] = None,
) -> List[Path]:
    """Download files from a Figshare article.

    Args:
        article_id: Figshare article ID (e.g., '19596379').
        root: Directory to save files.
        filenames: List of specific filenames to download (None for all).

    Returns:
        List of paths to downloaded files.
    """
    if not HAS_REQUESTS:
        raise ImportError(
            "requests is required for downloading from Figshare.\n"
            "Install with: pip install requests"
        )

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # Query Figshare API for article metadata
    api_url = f"https://api.figshare.com/v2/articles/{article_id}"
    print(f"Fetching Figshare article {article_id}...")

    response = requests.get(api_url)
    response.raise_for_status()

    article_data = response.json()

    # Get file list from article
    files = article_data.get('files', [])
    if not files:
        raise RuntimeError(f"No files found in Figshare article {article_id}")

    # Filter files if specific filenames requested
    if filenames is not None:
        files = [f for f in files if f['name'] in filenames]

    downloaded_paths = []

    # Download each file
    for file_info in files:
        filename = file_info['name']
        file_url = file_info['download_url']
        file_size = file_info.get('size', 0)

        filepath = root / filename

        # Check if already downloaded
        if filepath.exists():
            print(f"File {filename} already exists, skipping...")
            downloaded_paths.append(filepath)
            continue

        size_mb = file_size / 1024 / 1024 if file_size else 0
        print(f"Downloading {filename} ({size_mb:.1f} MB)...")

        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        downloaded = 0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if file_size > 0:
                        percent = (downloaded / file_size) * 100
                        print(f"\r  Progress: {percent:.1f}%", end='', flush=True)

        print()
        downloaded_paths.append(filepath)

    print(f"Done! Downloaded {len(downloaded_paths)} file(s) to {root}")
    return downloaded_paths


def download_from_github(
    repo: str,
    root: Path,
    files: Optional[List[str]] = None,
    branch: str = 'main',
    release_tag: Optional[str] = None,
) -> List[Path]:
    """Download files from a GitHub repository.

    Args:
        repo: Repository in format 'owner/repo'.
        root: Directory to save files.
        files: List of file paths to download (relative to repo root).
        branch: Branch name (default 'main').
        release_tag: If specified, download release assets instead.

    Returns:
        List of paths to downloaded files.
    """
    if not HAS_REQUESTS:
        raise ImportError(
            "requests is required for downloading from GitHub.\n"
            "Install with: pip install requests"
        )

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    downloaded_paths = []

    if release_tag:
        # Download release assets
        api_url = f"https://api.github.com/repos/{repo}/releases/tags/{release_tag}"
        response = requests.get(api_url)
        response.raise_for_status()
        release_data = response.json()

        assets = release_data.get('assets', [])
        for asset in assets:
            if files and asset['name'] not in files:
                continue

            filepath = root / asset['name']
            if filepath.exists():
                print(f"File {asset['name']} already exists, skipping...")
                downloaded_paths.append(filepath)
                continue

            print(f"Downloading {asset['name']}...")
            download_url_path = asset['browser_download_url']
            response = requests.get(download_url_path, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            downloaded_paths.append(filepath)
    else:
        # Download raw files from branch
        if not files:
            raise ValueError("Must specify files to download from branch")

        for file_path in files:
            raw_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_path}"
            filename = Path(file_path).name
            filepath = root / filename

            if filepath.exists():
                print(f"File {filename} already exists, skipping...")
                downloaded_paths.append(filepath)
                continue

            print(f"Downloading {filename} from GitHub...")
            try:
                response = requests.get(raw_url, stream=True)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                downloaded_paths.append(filepath)
            except requests.exceptions.RequestException as e:
                print(f"Warning: Failed to download {filename}: {e}")

    print(f"Done! Downloaded {len(downloaded_paths)} file(s) to {root}")
    return downloaded_paths


__all__ = [
    'get_data_home',
    'check_integrity',
    'download_url',
    'download_and_extract_zip',
    'download_from_zenodo',
    'download_from_uci',
    'download_from_figshare',
    'download_from_github',
]
