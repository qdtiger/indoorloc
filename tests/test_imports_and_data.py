from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from conftest import BRIDGES, HEAVY, PROJECT, UJI_ROOT
from indoorloc.core import SampleTable
from indoorloc.datasets import Dataset, default_root, list_datasets, load_dataset

BLOCK = f"import sys\nfor m in {HEAVY!r}: sys.modules[m] = None\n"


def _python(code: str) -> None:
    subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT, check=True)


def test_every_module_imports_without_heavy_dependencies_and_loads_no_bridge():
    """Heavy packages blocked: every module but the declared bridges imports, and none of them
    loads a bridge (import-linter cannot see this: the bridge's own edge is in ignore_imports)."""
    _python(BLOCK + "import importlib, pkgutil, indoorloc as pkg\n"
            f"bridges = set({BRIDGES!r})\n"
            "names = [m.name for m in pkgutil.walk_packages(pkg.__path__, 'indoorloc.')]\n"
            "for name in names:\n"
            "    if name not in bridges:\n"
            "        importlib.import_module(name)\n"
            "assert not bridges & set(sys.modules), 'a library module imported a bridge'\n"
            "assert len(names) >= 20, names")


def test_top_level_import_is_lazy_and_l3_does_not_load_l4():
    _python("import sys, indoorloc as il\n"
            "assert 'numpy' not in sys.modules\n"
            "il.create_model\n"
            "assert 'indoorloc.evaluation' not in sys.modules  # L3 -> L4 only inside score/evaluate\n"
            "assert 'indoorloc.datasets' not in sys.modules")


def test_missing_optional_dependency_names_the_extra():
    _python(BLOCK + "import indoorloc as il\n"
            "t = il.SampleTable([[1.0]], [[0.0, 0.0]])\n"
            "try:\n    t.to_dataframe()\nexcept ImportError as e:\n    assert \"indoorloc[pandas]\" in str(e)\n"
            "else:\n    raise AssertionError('expected ImportError')")


@pytest.mark.skipif(not (UJI_ROOT / "validationData.csv").is_file(), reason="UJIIndoorLoc files not found")
def test_ujiindoorloc_validation_split_contract():
    t = load_dataset("ujiindoorloc", split="validation", root=UJI_ROOT, download=False)  # split alias
    assert t.X.shape == (1111, 520) and t.X.dtype == np.float32 and t.pos.dtype == np.float64
    assert np.isnan(t.X).any() and not np.any(t.X == 100) and np.nanmax(t.X) <= 0
    assert t.meta["crs"] == "EPSG:3857" and t.meta["split"] == "test" and len(t.meta["sha256"]) == 64
    assert set(np.unique(t.floor)) <= {0, 1, 2, 3, 4} and len(set(t.ids)) == 1111
    # no sentinels anywhere (rule 5.4): columns the file does not know are left out, not zero-filled
    assert t.meta["unknown_groups"] == ("user", "space", "relative_position") and sorted(t.groups) == ["device", "time"]
    assert "ujiindoorloc" in list_datasets()  # the 0.1 id is not listed (see test_legacy.py)


def test_checksum_mismatch_is_an_error(tmp_path):
    (tmp_path / "validationData.csv").write_text("WAP001\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_dataset("ujiindoorloc", split="test", root=tmp_path)


def test_torch_dataloader_yields_dict_batches_and_keeps_string_groups():
    torch = pytest.importorskip("torch")
    from indoorloc.datasets.torch_adapter import make_dataloader

    t = SampleTable(np.arange(20, dtype=np.float32).reshape(10, 2), np.zeros((10, 2)), floor=np.arange(10) % 3,
                    groups={"source": np.array(["sim"] * 5 + ["real"] * 5), "device": np.arange(10)})
    batches = list(make_dataloader(t, batch_size=4, shuffle=True, seed=0))
    first = batches[0]
    assert first["X"].dtype == torch.float32 and first["pos"].dtype == torch.float64
    assert first["floor"].dtype == torch.int64 and isinstance(first["groups.source"], np.ndarray)
    assert sorted(torch.cat([b["groups.device"] for b in batches]).tolist()) == list(range(10))
    again = next(iter(make_dataloader(t, batch_size=4, shuffle=True, seed=0)))
    assert torch.equal(first["X"], again["X"])  # seeded shuffling


class _Tiny(Dataset):
    name = "tiny"
    files = {"train": ("tiny.csv", hashlib.sha256(b"x,y\n1,2\n").hexdigest())}

    def _parse(self, path, split):
        raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
        return SampleTable(raw[:, :1], raw[:, 1:])


def test_download_tries_mirrors_unpacks_and_verifies(tmp_path):
    archive = tmp_path / "mirror" / "tiny.zip"
    archive.parent.mkdir()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("tiny-v1/tiny.csv", "x,y\n1,2\n")  # folders inside the archive are flattened

    class Mirrored(_Tiny):
        urls = ((tmp_path / "gone.zip").as_uri(), archive.as_uri())  # first mirror is down

    with pytest.raises(FileNotFoundError, match="download=True"):
        Mirrored(root=tmp_path / "data").load("train")  # the class never downloads on its own
    table = Mirrored(root=tmp_path / "data", download=True).load("train")
    assert table.X.tolist() == [[1.0]] and table.meta["sha256"] == _Tiny.files["train"][1]
    assert sorted(p.name for p in (tmp_path / "data").iterdir()) == ["tiny.csv"]

    class Tampered(Mirrored):
        files = {"train": ("tiny.csv", "0" * 64)}

    with pytest.raises(ValueError, match="checksum mismatch"):
        Tampered(root=tmp_path / "other", download=True).load("train")


def test_an_empty_data_variable_counts_as_unset(monkeypatch):
    monkeypatch.setenv("INDOORLOC_DATA", "")
    assert default_root() == Path.home() / ".cache" / "indoorloc" / "datasets"  # not the working directory
    monkeypatch.setenv("INDOORLOC_DATA", "/srv/data")
    assert default_root() == Path("/srv/data")
