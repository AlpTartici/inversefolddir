# Security

## Reporting a vulnerability

Please do **not** report security vulnerabilities through public GitHub issues.

Report them privately by email to **tartici@stanford.edu**, or through GitHub's
[private vulnerability reporting](https://github.com/AlpTartici/inversefolddir/security/advisories/new)
for this repository.

Please include as much of the following as you can:

- The type of issue and the affected file or component
- Steps to reproduce, including any input structures or configuration
- Whether the issue is reachable through the sampling, training, or evaluation
  entry points
- Any proof-of-concept code

This is academic research software maintained on a best-effort basis. You can
expect an initial acknowledgement within about two weeks.

## Scope

This project loads model checkpoints with `torch.load` and reads user-supplied
structure files. Treat both as trusted input: **only load checkpoints from
sources you trust.** PyTorch checkpoints can execute arbitrary code when
deserialized, which is a property of the format rather than a defect in this
project.

## Upstream

This repository is a derivative of
[microsoft/InverseFoldDir](https://github.com/microsoft/InverseFoldDir). Issues
affecting the original Microsoft-hosted code should be reported to Microsoft
through the process described in that repository. Microsoft does not maintain
or provide support for this repository.
