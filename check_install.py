#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Copyright (c) 2025-2026 Alp Tartici, Stanford University.
# Licensed under the MIT License.

"""
check_install.py

Verify an Inverse FoldDir installation and say plainly what, if anything, is
missing. Run this before opening an issue -- it usually identifies the problem.

    python check_install.py

Exits 0 if you can design sequences, 1 otherwise. Optional components are
reported but never cause a failure.
"""

import importlib
import platform
import sys
from pathlib import Path

REQUIRED = [
    ("torch", "PyTorch", "https://pytorch.org/get-started/locally/"),
    ("torch_geometric", "PyTorch Geometric", "pip install torch-geometric"),
    ("Bio", "Biopython", "pip install biopython"),
    ("numpy", "NumPy", "pip install numpy"),
    ("scipy", "SciPy", "pip install scipy"),
    ("pandas", "pandas", "pip install pandas"),
    ("yaml", "PyYAML", "pip install PyYAML"),
    ("tqdm", "tqdm", "pip install tqdm"),
    ("requests", "requests", "pip install requests"),
]

OPTIONAL = [
    ("torch_scatter", "torch-scatter", "faster graph ops; a built-in fallback is used without it"),
    ("torch_cluster", "torch-cluster", "only needed for some training utilities"),
    ("matplotlib", "matplotlib", "notebook plots; pip install -r requirements-notebooks.txt"),
    ("jupyter", "Jupyter", "notebooks/quickstart.ipynb; pip install -r requirements-notebooks.txt"),
]

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = DIM = RESET = ""


def version_of(module):
    for attribute in ("__version__", "version", "VERSION"):
        value = getattr(module, attribute, None)
        if isinstance(value, str):
            return value
    return "installed"


def heading(text):
    print(f"\n{text}\n" + "-" * len(text))


def main():
    print("=" * 62)
    print(" Inverse FoldDir installation check")
    print("=" * 62)
    print(f"Python  : {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"Platform: {platform.system()} {platform.machine()}")

    problems = []

    heading("Required packages")
    for module_name, label, fix in REQUIRED:
        try:
            module = importlib.import_module(module_name)
            print(f"  {GREEN}OK{RESET}      {label:<20} {version_of(module)}")
        except Exception as exc:
            print(f"  {RED}MISSING{RESET} {label:<20} {DIM}{exc}{RESET}")
            problems.append(f"{label}: {fix}")

    heading("Optional packages")
    for module_name, label, why in OPTIONAL:
        try:
            module = importlib.import_module(module_name)
            print(f"  {GREEN}OK{RESET}      {label:<20} {version_of(module)}")
        except Exception:
            print(f"  {YELLOW}absent{RESET}  {label:<20} {DIM}{why}{RESET}")

    # Hardware. Reporting the device name is not enough: a PyTorch build without
    # kernels for the installed GPU can return zeros instead of raising, so this
    # runs real arithmetic on the device and checks the answer.
    heading("Compute")
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import torch

        if torch.cuda.is_available():
            from device_check import check_cuda_device

            ok, message = check_cuda_device(0)
            if ok:
                print(f"  {GREEN}GPU{RESET}     {message}")
            else:
                print(f"  {RED}BROKEN{RESET}  {message}")
                problems.append(
                    "the GPU returns wrong results for basic tensor operations "
                    "(PyTorch build does not match this card)"
                )
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            print(f"  {GREEN}GPU{RESET}     Apple Silicon (MPS)")
        else:
            print(f"  {YELLOW}CPU only{RESET}  designs take roughly 5 minutes each")
    except Exception as exc:
        print(f"  {DIM}skipped ({exc}){RESET}")

    # Does the project itself import?
    heading("Project modules")
    for module_name, label in [
        ("gvp", "GVP layers"),
        ("models.dfm_model", "model definition"),
        ("training.soft_priors", "soft residue priors"),
        ("inversefolddir_tools", "results helpers"),
    ]:
        try:
            importlib.import_module(module_name)
            print(f"  {GREEN}OK{RESET}      {label}")
        except Exception as exc:
            print(f"  {RED}FAILED{RESET}  {label}: {exc}")
            problems.append(f"{label} did not import: {exc}")

    # Model weights
    heading("Model weights")
    checkpoints = sorted(Path("ckpts").glob("*.pt")) if Path("ckpts").exists() else []
    usable = [c for c in checkpoints if c.stat().st_size > 1_000_000]
    if usable:
        for path in usable:
            print(f"  {GREEN}OK{RESET}      {path} ({path.stat().st_size // 1_000_000} MB)")
    elif checkpoints:
        print(f"  {RED}PLACEHOLDER{RESET} files in ckpts/ are only a few hundred bytes.")
        print(f"  {DIM}These are Git LFS pointers, not real weights. Re-download them.{RESET}")
        problems.append("model weights are Git LFS placeholders, not real files")
    else:
        print(f"  {YELLOW}absent{RESET}  no .pt files in ckpts/")
        print(f"  {DIM}Download a checkpoint before designing:{RESET}")
        print(f"  {DIM}  python scripts/download_checkpoints.py{RESET}")

    # Loading a checkpoint is the next thing a user does, and it is where a
    # PyTorch-version or architecture mismatch actually shows up. Only runs when
    # real weights are present, and never turns an absent checkpoint into a
    # failure.
    if usable and not problems:
        heading("Model load")
        try:
            import io
            from contextlib import redirect_stdout

            import torch

            from training.sample_utils import load_model_distributed

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            class _Args:
                verbose = False
                t_max = None

            with redirect_stdout(io.StringIO()):
                model, _ = load_model_distributed(str(usable[0]), device, _Args())
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  {GREEN}OK{RESET}      {usable[0].name} loaded on {device.type} "
                  f"({n_params / 1e6:.1f}M parameters)")
        except Exception as exc:
            print(f"  {RED}FAILED{RESET}  {usable[0].name}: {exc}")
            problems.append(f"the checkpoint did not load: {exc}")

    # Verdict
    print("\n" + "=" * 62)
    if problems:
        print(f" {RED}Not ready.{RESET} Fix these:")
        for item in problems:
            print(f"   - {item}")
        print("\n See docs/INSTALL.md, or open an issue with this output:")
        print("   https://github.com/AlpTartici/inversefolddir/issues")
        print("=" * 62)
        return 1

    print(f" {GREEN}Ready to design sequences.{RESET}")
    print("\n Try it:")
    print("   python training/inpainting.py --pdb_input 3OGO \\")
    print("       --model ckpts/inverse_folddir_model.pt --mask-ratio 1.0 \\")
    print("       --output-dir output/test")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
