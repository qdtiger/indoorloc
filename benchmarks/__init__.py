"""IndoorLoc's real-data benchmark matrix (not part of the installed package).

``python benchmarks/run.py`` runs every cell of ``matrix.py`` through the public command line
(``indoorloc benchmark``) and writes ``results/<dataset>.json``; ``python benchmarks/render.py``
turns them into ``docs/benchmarks.md`` and ``docs/benchmarks_zh.md``; ``python -m benchmarks.crosscheck``
recomputes the cross-checked cells outside the library. See ``README.md``.

The modules here only use the library's public API. ``protocols`` and ``baselines`` are
importable by path from the repository root, e.g.
``indoorloc benchmark --protocol benchmarks.protocols:LEAVE_ONE_USER_OUT``.
"""
