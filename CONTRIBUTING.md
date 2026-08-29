# Contributing

Contributions are welcome. This is academic research software maintained on a
best-effort basis, so please open an issue before starting anything large — it
saves you from building something that duplicates work in progress.

**This repository is independent of Microsoft.** Contributions here are not
covered by, and do not require, the Microsoft Contributor License Agreement. To
contribute to the original project instead, see
[microsoft/InverseFoldDir](https://github.com/microsoft/InverseFoldDir).

## Reporting a problem

Open an issue at
<https://github.com/AlpTartici/inversefolddir/issues> and include:

- the exact command you ran,
- the complete error message,
- the output of `python check_install.py`.

That last one resolves most installation issues on its own.

For security vulnerabilities, do **not** open a public issue — see
[SECURITY.md](SECURITY.md).

## Making a change

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files
```

The hooks are deliberately few and all run from a clean checkout: whitespace,
syntax, large-file and secret scanning, and `bandit`. Nothing here needs a
local baseline file or a site-specific scanner.

Then:

```bash
python -m pytest tests/ -q
python check_install.py
```

A few things worth knowing before you send a pull request:

- **Model weights are not in git.** They are distributed through Hugging Face
  and downloaded by `scripts/download_checkpoints.py`. Do not commit `.pt`
  files; the `check-added-large-files` hook will stop you anyway.
- **Sampling behaviour is checkpoint-driven.** Architecture and flow parameters
  are read back from the checkpoint at load time, so changing a default in the
  code does not change how an existing checkpoint samples. See "Settings that do
  not travel with the checkpoint" in [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).
- **Failures should be loud.** A design run that produces nothing must exit
  non-zero, and anything that would silently substitute a wrong value —
  clamping an out-of-range lookup, using a GPU whose results cannot be trusted —
  should raise instead. Plausible-looking wrong output is the failure mode this
  project cares about most.

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
