#!/usr/bin/env python3
# Copyright (c) 2026 Alp Tartici and contributors.
# Licensed under the MIT License.

"""
Verify that a CUDA device will actually compute, before anything depends on it.

A PyTorch build ships compiled kernels for a fixed list of GPU architectures.
Run it on a card whose architecture is not in that list and, depending on the
build, kernel launches can silently no-op: tensors come back as zeros instead of
raising. Nothing crashes. Sampling proceeds, the Dirichlet step gets a
degenerate concentration, and either the run dies far downstream with an
unrelated-looking error or -- worse -- it produces a sequence that means nothing.

This is not hypothetical, and it is not confined to one card. It bites whenever
the installed PyTorch predates the GPU: a torch built for sm_80/sm_86 on an
H100 (sm_90), or a torch built through sm_90 on a Blackwell card. Because users
install their own PyTorch wheel for their own driver, this repository cannot
pin the combination away.

So: check the architecture, then prove numerically that the device computes.
Both are cheap, and both run before the model is loaded.

Used by ``check_install.py`` and by the design entry points.
"""

import os

__all__ = ["describe_device", "check_cuda_device", "assert_cuda_device_usable",
           "DeviceCheckError"]

# Escape hatch for anyone who understands their setup better than this check
# does -- a debug build with PTX JIT, for instance. Deliberately awkward to set
# by accident.
_OVERRIDE_ENV = "IFD_SKIP_DEVICE_CHECK"


class DeviceCheckError(RuntimeError):
    """The selected CUDA device cannot be trusted to compute correctly."""


def describe_device(index=0):
    """Return a short human-readable description of a CUDA device."""
    import torch

    if not torch.cuda.is_available():
        return "CPU (no CUDA device visible)"
    try:
        name = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        return f"{name} (sm_{major}{minor}), torch {torch.__version__}"
    except Exception as exc:  # driver present but unusable
        return f"CUDA device {index}, unreadable properties: {exc}"


def check_cuda_device(index=0):
    """
    Test whether CUDA device ``index`` computes correctly.

    Returns ``(ok, message)``. ``ok`` is True when the device is safe to use, or
    when there is no CUDA device at all -- CPU-only is a supported, correct
    configuration, just a slower one.
    """
    import torch

    if os.environ.get(_OVERRIDE_ENV):
        return True, f"device check skipped ({_OVERRIDE_ENV} is set)"

    if not torch.cuda.is_available():
        return True, "no CUDA device; running on CPU"

    try:
        name = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
    except Exception as exc:
        return False, f"CUDA reports a device but its properties are unreadable: {exc}"

    arch = f"sm_{major}{minor}"
    try:
        arch_list = [a for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    except Exception:
        arch_list = []

    # An arch list without this card is the strongest available signal, but it is
    # not conclusive on its own: a build with forward-compatible PTX can JIT for a
    # newer card. Report it, then let the numeric test decide.
    arch_missing = bool(arch_list) and arch not in arch_list

    try:
        # The specific op that returned zeros in the field, plus a matmul, which
        # exercises a different kernel family (cuBLAS rather than elementwise).
        probe = 20.0 * torch.ones(21, device=f"cuda:{index}")
        product = (torch.randn(256, 256, device=f"cuda:{index}")
                   @ torch.randn(256, 256, device=f"cuda:{index}"))
        torch.cuda.synchronize(index)
    except Exception as exc:
        return False, (
            f"{name} ({arch}) raised while running a trivial CUDA op: {exc}\n"
            f"  This PyTorch ({torch.__version__}) was built for: "
            f"{' '.join(arch_list) or 'unknown'}"
        )

    elementwise_ok = bool(probe.min().item() > 0) and abs(probe.sum().item() - 420.0) < 1e-3
    matmul_ok = bool(torch.isfinite(product).all()) and product.abs().sum().item() > 0

    if elementwise_ok and matmul_ok:
        note = ""
        if arch_missing:
            note = (f"\n  Note: {arch} is not in this build's architecture list "
                    f"({' '.join(arch_list)}), so it is running through PTX JIT. "
                    "It computes correctly, but expect a slow first kernel launch.")
        return True, f"{name} ({arch}) computes correctly, torch {torch.__version__}{note}"

    failed = []
    if not elementwise_ok:
        failed.append(f"20*ones(21) summed to {probe.sum().item():g}, expected 420")
    if not matmul_ok:
        failed.append("a 256x256 matmul produced zeros or non-finite values")

    return False, (
        f"{name} ({arch}) returns wrong results for basic tensor operations:\n"
        + "".join(f"    - {f}\n" for f in failed)
        + f"  This PyTorch ({torch.__version__}) was built for: "
        f"{' '.join(arch_list) or 'unknown'}.\n"
        + (f"  {arch} is not in that list, which is the usual cause: kernel launches\n"
           "  silently do nothing instead of raising.\n" if arch_missing else "")
        + "  Fix: install a PyTorch build that supports this GPU\n"
          "  (https://pytorch.org/get-started/locally/), or run on CPU with\n"
          "  CUDA_VISIBLE_DEVICES=''."
    )


def assert_cuda_device_usable(index=0, verbose=True):
    """
    Raise :class:`DeviceCheckError` unless the CUDA device computes correctly.

    Call this before loading a model. Failing here costs a second; failing later
    costs a whole sampling run, and may not look like a hardware problem at all.
    """
    ok, message = check_cuda_device(index)
    if not ok:
        raise DeviceCheckError(
            "Refusing to run: this GPU does not compute correctly.\n\n"
            + message
            + f"\n\n  To bypass this check anyway, set {_OVERRIDE_ENV}=1."
        )
    if verbose:
        print(f"Device check: {message}")
    return message
