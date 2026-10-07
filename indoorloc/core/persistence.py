"""Save/load any Estimator as ``config.json`` + ``arrays.npz``: no pickle, diffable, versioned.

config.json holds the class path, the constructor parameters, every other attribute set
by ``fit`` (arrays replaced by references into arrays.npz), the format and library
versions and free-form provenance (``info``: data sha256, split, git commit, ...).

Values are stored as JSON nodes:

    {"__estimator__": "mod:Cls", "params": ..., "state": ...}   an Estimator
    {"__object__": "mod:Cls", "data": {...}}   a data object (e.g. the ``apps.maps.FloorMap``
        parameter of a particle filter); ``data`` is its ``to_dict()``
    {"__array__": key}   a numeric, bool or str array stored in arrays.npz
    {"__tuple__": [...]}, {"__dict__": {...}}, {"__float__": "nan"}   tuples, str-keyed
        dicts and non-finite floats (lists, str, numbers, bool and None are plain JSON)

A data object is an instance of a class that opts in: it is not an Estimator, sets the class
attribute ``_save_via_dict = True`` and defines ``to_dict()`` (str keys; values of the kinds
above) and the classmethod ``from_dict(d)``, its exact inverse. The opt-in is explicit because
duck typing would also catch objects whose ``to_dict``/``from_dict`` do not round-trip (a
pandas DataFrame or an xarray Dataset loses its dtypes), and a model file must reload exactly.

Loading checks each class path against an allow-list BEFORE importing anything
(indoorloc itself, modules named by a registry, ``trusted_modules=``), checks that the class
it finds is defined in an allowed module too (a path cannot reach another package's class
through an attribute of an allowed module), then refuses anything that is not an Estimator
(for ``__estimator__``) or not an opted-in data class (for ``__object__``); the saved object
itself must be an Estimator; arrays are read with ``allow_pickle=False``.
Still, load only model files you trust: a trusted class decides what its state does.
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import numpy as np

from .._version import __version__
from .estimator import Estimator
from .registry import Registry

FORMAT_VERSION = 1


def _is_data_class(cls) -> bool:
    """A class that opted in to being saved as ``to_dict()`` and rebuilt by ``from_dict``."""
    return (isinstance(cls, type) and not issubclass(cls, Estimator) and getattr(cls, "_save_via_dict", False) is True
            and callable(getattr(cls, "to_dict", None)) and callable(getattr(cls, "from_dict", None)))


def _encode(value, arrays: dict, key: str):
    if isinstance(value, Estimator):
        cls = type(value)
        params = value.get_params(deep=False)
        state = value._get_state()
        return {"__estimator__": f"{cls.__module__}:{cls.__qualname__}",
                "params": {k: _encode(v, arrays, f"{key}.{k}") for k, v in params.items()},
                "state": {k: _encode(v, arrays, f"{key}.{k}") for k, v in state.items()}}
    if getattr(type(value), "_save_via_dict", False) is True:
        cls = type(value)
        if not _is_data_class(cls):
            raise TypeError(f"cannot save {key}: {cls.__qualname__} sets _save_via_dict but lacks to_dict() or "
                            "the classmethod from_dict()")
        data = value.to_dict()
        if not (isinstance(data, dict) and all(isinstance(k, str) for k in data)):
            raise TypeError(f"cannot save {key}: {cls.__qualname__}.to_dict() must return a dict with str keys")
        return {"__object__": f"{cls.__module__}:{cls.__qualname__}",
                "data": {k: _encode(v, arrays, f"{key}.{k}") for k, v in data.items()}}
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise TypeError(f"cannot save {key}: object arrays need pickle")
        name, i = key, 1
        while name in arrays:  # {"a.b": x, "a": {"b": y}} spell one path: never let y overwrite x
            name, i = f"{key}~{i}", i + 1
        arrays[name] = value
        return {"__array__": name}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": repr(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        items = [_encode(v, arrays, f"{key}[{i}]") for i, v in enumerate(value)]
        return items if isinstance(value, list) else {"__tuple__": items}
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {"__dict__": {k: _encode(v, arrays, f"{key}.{k}") for k, v in value.items()}}
    raise TypeError(f"cannot save {key}: unsupported type {type(value).__name__} (arrays, numbers, str, "
                    "list/tuple/dict, Estimators and data classes with _save_via_dict = True, to_dict() and "
                    "from_dict() are supported)")


def _import_class(path: str, trusted: tuple[str, ...], kind: str = "estimator") -> type:
    """``"pkg.mod:Cls"`` -> the class, importing ``pkg.mod`` only if the allow-list covers it.
    ``kind="estimator"`` accepts Estimator subclasses; ``kind="object"`` accepts opted-in data
    classes (``_save_via_dict = True``, ``to_dict``, ``from_dict``; not Estimators)."""
    module, _, qualname = path.partition(":")
    roots = (__name__.partition(".")[0], *trusted, *Registry.trusted_modules())

    def allowed(name) -> bool:
        return isinstance(name, str) and any(name == root or name.startswith(root + ".") for root in roots)

    if not allowed(module):
        raise ValueError(f"refusing to import {module!r} named in a model file: it is not part of "
                         "indoorloc, not a registered model and not in trusted_modules=")
    cls = importlib.import_module(module)
    for part in qualname.split("."):
        cls = getattr(cls, part)
    if not allowed(getattr(cls, "__module__", None)):  # e.g. "indoorloc.x:np.SomeClass" reaching numpy
        raise ValueError(f"refusing {path!r} from a model file: it resolves to a class of "
                         f"{getattr(cls, '__module__', None)!r}, which is not in the allow-list")
    if kind == "estimator":
        if not (isinstance(cls, type) and issubclass(cls, Estimator)):
            raise TypeError(f"{path} is not an indoorloc Estimator")
    elif not _is_data_class(cls):
        raise TypeError(f"{path} is not a data class (_save_via_dict = True, to_dict() and from_dict())")
    return cls


def _decode(node, arrays: dict, trusted: tuple[str, ...]):
    if isinstance(node, list):
        return [_decode(v, arrays, trusted) for v in node]
    if not isinstance(node, dict):
        return node
    if "__array__" in node:
        return arrays[node["__array__"]]
    if "__float__" in node:
        return float(node["__float__"])
    if "__tuple__" in node:
        return tuple(_decode(v, arrays, trusted) for v in node["__tuple__"])
    if "__dict__" in node:
        return {k: _decode(v, arrays, trusted) for k, v in node["__dict__"].items()}
    if "__object__" in node:
        if not isinstance(node.get("data"), dict):
            raise ValueError(f"malformed __object__ node for {node['__object__']!r} in a model file: no 'data' dict")
        cls = _import_class(node["__object__"], trusted, kind="object")
        obj = cls.from_dict({k: _decode(v, arrays, trusted) for k, v in node["data"].items()})
        if not isinstance(obj, cls):
            raise TypeError(f"{node['__object__']}.from_dict returned {type(obj).__name__}, not the class itself")
        return obj
    if "__estimator__" not in node:
        raise ValueError(f"unknown node {sorted(node)[:4]} in a model file (written by a newer indoorloc?)")
    cls = _import_class(node["__estimator__"], trusted)
    obj = cls(**{k: _decode(v, arrays, trusted) for k, v in node["params"].items()})
    obj._set_state({k: _decode(v, arrays, trusted) for k, v in node["state"].items()})
    return obj


def save(obj: Estimator, path, *, info: dict | None = None, compress: bool = True) -> Path:
    """Write ``config.json`` + ``arrays.npz`` into the directory ``path`` (created if needed).
    ``compress`` deflates the arrays (a k-NN radio map of dBm integers shrinks about 20x).

    Both files are written under temporary names first and then moved into place, the old
    ``config.json`` removed before the new ``arrays.npz`` arrives: a save that fails (a full
    disk, an interrupt) never leaves a readable model that pairs one save's config with
    another save's arrays; at worst ``load_model`` finds no ``config.json``."""
    import os

    if not isinstance(obj, Estimator):
        raise TypeError(f"save writes an indoorloc Estimator, not a {type(obj).__name__}")
    path = Path(path)
    arrays: dict = {}
    config = {"format": "indoorloc-estimator", "format_version": FORMAT_VERSION,
              "library_version": __version__, "numpy_version": np.__version__,
              "info": _encode(dict(info or {}), arrays, "info"), "object": _encode(obj, arrays, "obj")}
    text = json.dumps(config, indent=1) + "\n"
    path.mkdir(parents=True, exist_ok=True)  # only once everything encoded: a failed encoding leaves nothing
    parts = {name: path / f".{name}.part" for name in ("arrays.npz", "config.json")}
    try:
        with open(parts["arrays.npz"], "wb") as fh:  # a file object: numpy appends no ".npz"
            (np.savez_compressed if compress else np.savez)(fh, **arrays)
        parts["config.json"].write_text(text)
        (path / "config.json").unlink(missing_ok=True)
        os.replace(parts["arrays.npz"], path / "arrays.npz")
        os.replace(parts["config.json"], path / "config.json")
    finally:
        for part in parts.values():
            part.unlink(missing_ok=True)
    return path


def load_model(path, *, trusted_modules=()):
    """Inverse of ``save``; returns the fitted estimator (``info`` is on ``.saved_info_``).
    Classes outside indoorloc and the registries (estimators and data objects alike) load
    only if their module (or a parent package) is in ``trusted_modules``."""
    path = Path(path)
    trusted = (trusted_modules,) if isinstance(trusted_modules, str) else tuple(trusted_modules)
    config = json.loads((path / "config.json").read_text())
    if config.get("format") != "indoorloc-estimator" or config.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path} is not a format-{FORMAT_VERSION} indoorloc estimator")
    if not (isinstance(config.get("object"), dict) and "__estimator__" in config["object"]):
        raise ValueError(f"{path}/config.json does not hold an estimator")
    with np.load(path / "arrays.npz", allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    obj = _decode(config["object"], arrays, trusted)
    obj.saved_info_ = _decode(config["info"], arrays, trusted)
    return obj
