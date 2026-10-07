"""Name -> ``"module:attr"`` lookup that imports nothing until an entry is used.

Built-in entries are literal strings, so listing them never loads torch; a full
path such as ``"mypkg.models:MyLocalizer"`` works without registering anything.
"""
from __future__ import annotations

import importlib


class Registry:
    _modules: set[str] = set()  # every module some registry names: load_model may import these

    def __init__(self, kind: str, entries: dict | None = None):
        self.kind = kind
        self._entries: dict = {}
        for name, target in (entries or {}).items():
            self.register(name, target)

    def register(self, name: str, target=None, *, force: bool = False):
        """``register("x", "pkg.mod:Cls")``, ``register("x", Cls)`` or ``@register("x")``."""
        if not isinstance(name, str):
            raise TypeError(f"a {self.kind} name must be a string, got {type(name).__name__}")
        if target is None:
            return lambda obj: self.register(name, obj, force=force)
        if name.lower() in self._entries and not force:
            raise KeyError(f"{self.kind} {name!r} is already registered; pass force=True to replace it")
        self._entries[name.lower()] = target
        module = target.partition(":")[0] if isinstance(target, str) else getattr(target, "__module__", "")
        if module and (not isinstance(target, str) or ":" in target):  # an alias names no module
            Registry._modules.add(module)
        return target

    @classmethod
    def trusted_modules(cls) -> frozenset[str]:
        """Modules named by any registry entry (built in or registered by the user)."""
        return frozenset(cls._modules)

    def _is_alias(self, target) -> bool:
        return isinstance(target, str) and target.lower() in self._entries

    def names(self, aliases: bool = False) -> list[str]:
        return sorted(n for n, t in self._entries.items() if aliases or not self._is_alias(t))

    def __contains__(self, name: str) -> bool:
        return isinstance(name, str) and name.lower() in self._entries

    def get(self, name: str):
        """The entry called ``name`` (case-insensitive), a ``"module:attr"`` path, or a class as is."""
        if isinstance(name, type):  # create_model(MyLocalizer, k=3): the class itself, nothing to look up
            return name
        if not isinstance(name, str):
            raise TypeError(f"a {self.kind} is named by a string (registry name or 'module:attr') or given as a "
                            f"class, got {type(name).__name__}")
        target = self._entries.get(name.lower(), name)
        while self._is_alias(target):
            target = self._entries[target.lower()]
        if not isinstance(target, str):
            return target
        module, sep, attr = target.partition(":")
        if not sep:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {', '.join(self.names())}")
        return getattr(importlib.import_module(module), attr)
