"""Loader machinery shared by every dataset: file layout, downloads, checksums, metadata."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath

from ..core import SampleTable


def default_root() -> Path:
    """``$INDOORLOC_DATA``, else ``~/.cache/indoorloc/datasets`` as in 0.1 (an empty variable counts as unset)."""
    return Path(os.environ.get("INDOORLOC_DATA") or Path.home() / ".cache" / "indoorloc" / "datasets")


def sha256sum(path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


_USER_AGENT = "indoorloc (+https://github.com/qdtiger/indoorloc)"


def _permanent(err) -> bool:
    """An error no retry can fix: an HTTP 4xx other than timeout/too-many-requests, or a
    ``file://`` mirror that does not exist."""
    import urllib.error

    if isinstance(err, urllib.error.HTTPError):
        return 400 <= err.code < 500 and err.code not in (408, 425, 429)
    return isinstance(err, urllib.error.URLError) and isinstance(err.reason, FileNotFoundError)


def _fetch(mirrors, part: Path, hint: str, *, attempts: int = 3) -> tuple[str, str]:
    """Download the first mirror that answers into ``part``; returns ``(url used, sha256)``.

    Asks for gzip transfer encoding and inflates it; each mirror is tried ``attempts`` times
    (once for a permanent error such as HTTP 404).
    """
    import gzip
    import http.client
    import time
    import urllib.request
    import zlib

    mirrors = (mirrors,) if isinstance(mirrors, str) else tuple(mirrors)
    errors = []
    for url in mirrors:
        for attempt in range(attempts):
            digest = hashlib.sha256()
            try:
                request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip"})
                with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as fh:
                    headers = getattr(response, "headers", None) or {}
                    source = (gzip.GzipFile(fileobj=response)
                              if str(headers.get("Content-Encoding", "")).lower() == "gzip" else response)
                    for block in iter(lambda: source.read(1 << 20), b""):
                        digest.update(block)
                        fh.write(block)
                return url, digest.hexdigest()
            except (OSError, EOFError, zlib.error, http.client.HTTPException) as err:  # URLError is an OSError
                if attempt == attempts - 1 or _permanent(err):
                    errors.append(f"{url}: {err}")
                    break
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"could not download {'; '.join(errors)}. {hint}")


def _write(source, target: Path, expected: str | None) -> str | None:
    """Stream ``source`` (a binary file object) to ``target`` through ``target.part``: moved
    into place only if its sha256 is ``expected`` (or ``expected`` is None); returns the
    digest when it did not match (nothing written), else None."""
    import os

    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(part, "wb") as dst:
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
                dst.write(block)
        if expected and digest.hexdigest() != expected:
            return digest.hexdigest()
        os.replace(part, target)
        return None
    finally:
        part.unlink(missing_ok=True)


class Dataset:
    """A public dataset described by class attributes (as in torchvision and TorchGeo).

    Subclasses declare:

    ``name``           registry id, also the folder under ``default_root()``.
    ``urls``           mirrors of one archive or file, tried in order; or a dict
                       ``{relative file path: url or (mirrors...)}`` when files are fetched one by one.
    ``files``          split -> ``(relative path, sha256)`` or a tuple of such pairs when a split
                       spans several files. Paths are POSIX-style and relative to ``root``.
    ``split_aliases``  e.g. ``{"validation": "test"}``.
    ``meta``           dataset-level facts copied into every table (crs, units, license, doi, ...).

    and implement ``_parse(paths, split) -> SampleTable`` where ``paths`` is one ``Path``
    for a single-file split and a list of ``Path`` otherwise.

    ``load`` always returns physical units: no normalization, NaN for missing readings,
    float64 coordinates. Nothing is downloaded unless ``download=True`` (``load_dataset``
    passes True by default, as in 0.1). Every file is sha256-checked unless ``verify=False``.
    """

    name: str = ""
    urls: tuple[str, ...] | dict[str, str | tuple[str, ...]] = ()
    files: dict[str, tuple] = {}
    split_aliases: dict[str, str] = {}
    meta: dict = {}

    def __init__(self, root=None, *, download: bool = False, verify: bool = True):
        self.root = Path(root) if root is not None else default_root() / self.name
        self.download = download
        self.verify = verify

    # ----------------------------------------------------------------- layout
    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self.files)

    @property
    def default_splits(self) -> tuple[str, ...]:
        """What ``split=None`` loads: ``("train", "test")`` (the official split) when both
        exist, else ``("all",)``, else the first split. A dataset whose default depends on
        its options (e.g. a modality with a single split) overrides this."""
        splits = self.splits
        if {"train", "test"} <= set(splits):
            return ("train", "test")
        return ("all",) if "all" in splits else splits[:1]

    def _split(self, split: str | None) -> str:
        if split is None:
            split = self.default_splits[0] if self.default_splits else None
        split = self.split_aliases.get(split, split)
        if split not in self.files:
            raise ValueError(f"{self.name} has splits {self.splits} (aliases {self.split_aliases}), not {split!r}")
        return split

    def _entries(self, split: str) -> tuple[tuple[str, str | None], ...]:
        """The (relative path, sha256) pairs of a split, whichever way ``files`` spells them."""
        entry = self.files[self._split(split)]
        if len(entry) == 2 and isinstance(entry[0], str) and (entry[1] is None or isinstance(entry[1], str)):
            return (tuple(entry),)
        return tuple(tuple(e) for e in entry)

    # ----------------------------------------------------------------- checks
    def _downloadable(self, relpath: str) -> bool:
        """Whether ``urls`` names a source for ``relpath`` (a dict entry, or any archive mirror)."""
        if isinstance(self.urls, dict):
            return relpath in self.urls
        return bool(self.urls)

    def check(self, split: str) -> list[Path]:
        """Return the verified paths of a split's files, downloading them if allowed."""
        paths = []
        for relpath, expected in self._entries(split):
            path = self.root / PurePosixPath(relpath)
            if not path.is_file():
                if not self.download:
                    if not self._downloadable(relpath):  # download=True could not help: do not suggest it
                        raise FileNotFoundError(f"{path} not found, and {self.name or type(self).__name__} declares "
                                                "no download url for it: place the file there by hand, or pass "
                                                "root=... (the folder that holds it)")
                    raise FileNotFoundError(f"{path} not found; pass root=... or download=True")
                self._download(split)
                if not path.is_file():
                    raise FileNotFoundError(f"{path} not found after downloading {self.name}")
            if self.verify and expected and (actual := sha256sum(path)) != expected:
                raise ValueError(f"checksum mismatch for {path}: got {actual}, expected {expected} "
                                 "(delete the file to download it again, or pass verify=False)")
            paths.append(path)
        return paths

    def load(self, split: str | None = None) -> SampleTable:
        """One split as a SampleTable; ``None`` means the first of ``default_splits``
        (``"train"`` when the dataset has an official split, else ``"all"``)."""
        split = self._split(split)
        paths = self.check(split)
        split = self._split(split)
        table = self._parse(paths[0] if len(paths) == 1 else paths, split)
        hashes = {rel: sha for rel, sha in self._entries(split)}
        sha = next(iter(hashes.values())) if len(hashes) == 1 else hashes  # one file: the digest itself
        return table.replace(meta={**self.meta, **table.meta, "name": self.name, "split": split,
                                   "source_files": tuple(hashes), "sha256": sha})

    # --------------------------------------------------------------- download
    def _download(self, split: str | None = None) -> None:
        """Fetch the split's absent files into ``root`` (standard library only).

        ``urls`` as a dict ``{relative path: url or (mirrors...)}``: each absent file is fetched
        from its own url to its own path. ``urls`` as a tuple: mirrors of one archive (zip, tar,
        tar.gz) or of a single file. The download itself is kept as a wanted file when its
        sha256 is that file's (or, without a digest to compare, when the url names the file);
        otherwise each wanted file is looked up in the archive by its relative path or trailing
        path components, so an archive's top-level folder does not matter; when several members
        match, the shallowest one whose sha256 is right is taken. Every request is asked for
        gzip transfer encoding and tried three times per mirror (once on HTTP 4xx).

        A file is written under ``<name>.part``, sha256-checked (unless ``verify=False``) and
        only then moved into place: an interrupted, corrupted or ambiguous download leaves
        nothing under ``root`` and raises, so the next call downloads again.
        """
        import tarfile  # the download machinery loads only when something is downloaded
        import urllib.parse
        import zipfile

        splits = [split] if split else list(self.files)
        entries = {rel: sha for s in splits for rel, sha in self._entries(s)}
        absent = [rel for rel in entries if not (self.root / rel).is_file()]
        if not absent:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        hint = f"Behind a proxy that blocks the host, add it to no_proxy, or fetch the files yourself into {self.root}"
        wanted = {rel: entries[rel] if self.verify else None for rel in absent}  # None: nothing to compare
        mismatch = []

        if isinstance(self.urls, dict):
            for rel in absent:
                if rel not in self.urls:
                    raise RuntimeError(f"{self.name}: no url declared for {rel}; place the file under {self.root}")
            for rel in absent:
                part = self.root / (rel + ".part")
                part.parent.mkdir(parents=True, exist_ok=True)
                try:
                    _, digest = _fetch(self.urls[rel], part, hint)
                    if wanted[rel] and digest != wanted[rel]:
                        raise ValueError(f"checksum mismatch for {rel}: got {digest}, expected {wanted[rel]}; "
                                         f"{self.root / rel} was not written")
                    part.replace(self.root / rel)
                finally:
                    part.unlink(missing_ok=True)
            return

        if not self.urls:
            raise RuntimeError(f"{self.name}: no download urls declared; place the files under {self.root}")
        part = self.root / ".download.part"
        try:
            url, digest = _fetch(self.urls, part, hint)
            own = PurePosixPath(urllib.parse.urlparse(url).path).name
            remaining, named = [], {}
            for rel in absent:  # the wanted file may be the download itself (a zip read in place)
                if wanted[rel] == digest or (not wanted[rel] and PurePosixPath(rel).name == own):
                    with open(part, "rb") as src:
                        _write(src, self.root / rel, None)
                    continue
                if wanted[rel] and PurePosixPath(rel).name == own:  # corrupt, or an archive wrapping its namesake
                    named[rel] = f"{rel}: the download from {url} has sha256 {digest}, expected {wanted[rel]}"
                remaining.append(rel)

            def extract(members) -> None:
                """``members``: (name in the archive, opener) of every file in it."""
                for rel in remaining:
                    want = PurePosixPath(rel).parts
                    found = sorted((m for m in members if PurePosixPath(m[0]).parts[-len(want):] == want),
                                   key=lambda m: len(PurePosixPath(m[0]).parts))  # shallowest first, stable
                    if not found:
                        continue  # check() reports it as not found
                    names = [name for name, _ in found]
                    if not wanted[rel] and len(found) > 1 and len(PurePosixPath(names[0]).parts) == len(
                            PurePosixPath(names[1]).parts):
                        mismatch.append(f"{rel}: the archive holds {names} and no sha256 tells which one is meant")
                        continue
                    for _, opener in found if wanted[rel] else found[:1]:
                        with opener() as src:
                            if _write(src, self.root / rel, wanted[rel]) is None:
                                break
                    else:
                        mismatch.append(f"{rel}: no archive member matching it ({names}) has sha256 {wanted[rel]}")

            if remaining and zipfile.is_zipfile(part):
                with zipfile.ZipFile(part) as archive:
                    extract([(i.filename.replace("\\", "/"), lambda i=i: archive.open(i))
                             for i in archive.infolist() if not i.is_dir()])
            elif remaining and tarfile.is_tarfile(part):
                with tarfile.open(part) as archive:
                    extract([(i.name, lambda i=i: archive.extractfile(i)) for i in archive.getmembers() if i.isfile()])
            elif len(remaining) == 1 and remaining[0] not in named:  # a plain file behind a url not naming it
                with open(part, "rb") as src:
                    if (got := _write(src, self.root / remaining[0], wanted[remaining[0]])) is not None:
                        mismatch.append(f"{remaining[0]}: the download from {url} has sha256 {got}, "
                                        f"expected {wanted[remaining[0]]}")
        finally:
            part.unlink(missing_ok=True)
        reported = {m.partition(":")[0] for m in mismatch}
        mismatch += [msg for rel, msg in named.items() if rel not in reported and not (self.root / rel).is_file()]
        if mismatch:
            raise ValueError("checksum mismatch: " + "; ".join(mismatch) + " (those files were not written)")

    def _parse(self, paths, split: str) -> SampleTable:
        raise NotImplementedError
