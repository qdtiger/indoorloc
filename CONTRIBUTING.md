# Contributing

Thanks for your interest in contributing to IndoorLoc!

## Development setup

- Follow the conda-based install in `README.md`.
- Install in editable mode:

```bash
pip install -e ".[full,dev]"
```

## Run checks locally

```bash
pytest -q
ruff check indoorloc/ --ignore=E501,F401
```

## Adding a dataset

1. Implement the dataset under `indoorloc/datasets/`.
2. Register it in the registry (see existing datasets for patterns).
3. Add/extend tests under `tests/`.
4. Document the dataset ID and download source.

## Reporting bugs

Please include:
- OS, Python version, and `torch` version
- Minimal reproducible snippet (or config)
- Full error log / traceback
