# AGENTS.md

WFDB-Python is a library for reading, writing and processing physiologic signals and annotations in the WFDB format. See [DEVELOPING.md](DEVELOPING.md) for the full contributor guide.

## Layout

- `src/wfdb/`: the package (`io`, `plot`, `processing`)
- `tests/`: pytest tests, which read sample files from `sample-data/` relative to the repository root, so run them from there
- `docs/`: Sphinx documentation, including the changelog in `docs/changes.rst`

## Commands

The project uses [uv](https://docs.astral.sh/uv/). `uv run` installs the `dev` dependency group automatically.

```sh
uv run pytest                     # all tests
uv run pytest -m "not network"    # only tests that work offline
uv run pytest -n auto             # in parallel
uv run ruff format .              # format
uv run ruff check .               # lint
uv run --only-group docs make -C docs html SPHINXOPTS="-W --keep-going"
```

## Conventions

- Format and lint with Ruff before committing. Lines (including docstrings and comments) are at most 88 columns.
- Docstrings use the numpy style. Start the docstring directly after the opening quotes with a summary that fits on one line, followed by a blank line and the details (if any).
- The docs build treats warnings as errors, so check it after changing a docstring.
- Tests are plain pytest functions or classes (no `unittest`). Tests that need PhysioNet are marked `network` and are skipped if PhysioNet cannot be reached. Where possible, a test has a `local` variant and a `physionet` variant through the `source` fixture.
- Add a changelog entry to `docs/changes.rst` for user-visible changes.
- The version is set in `pyproject.toml` only. Do not bump it as part of a feature or fix.
- Commits that only reformat code are listed in `.git-blame-ignore-revs`.
