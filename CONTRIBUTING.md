# Contributing

Thanks for your interest in contributing to IndoorLoc! New datasets and new localization methods
are what the library needs most.

## Development setup

```bash
git clone https://github.com/qdtiger/indoorloc.git && cd indoorloc
pip install -e ".[full,dev]"     # Python 3.10+; numpy alone is enough for most of the library
```

## Before you write code

Read [docs/architecture/CONTRACTS.md](docs/architecture/CONTRACTS.md): the layer import rules, the
`SampleTable` / `Prediction` data layouts and units, and what every dataset, transform and method
must provide. [docs/guide/extending.md](docs/guide/extending.md) walks through adding a dataset
(`register_dataset`), a transform, a method (`register_model`) or a protocol (`register_protocol`).

A contribution needs:

- one class (and one registry entry) per dataset or method, with a docstring whose References
  section cites the original paper or data source;
- physical units, NaN for missing readings, float64 positions in the source's coordinate frame;
- a test against a known result (a closed-form case, a paper's toy example or a published number),
  not only a smoke test; tests run without network access, and real-data tests skip when the data
  is absent;
- numbers in docs or docstrings only if you ran them.

## Run the checks locally

```bash
python -m pytest -q                      # tests/ and benchmarks/tests/
lint-imports                             # the layer contracts in pyproject.toml
python docs/catalog.py --check           # the generated catalog tables match the registries
ruff check indoorloc examples benchmarks docs --select E9,F63,F7,F82
```

## Reporting bugs

Please include the OS, Python and numpy versions (plus torch / scikit-learn if involved), a minimal
reproducible snippet, and the full traceback.
