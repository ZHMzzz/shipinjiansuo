"""Unify wheel-bundled OpenMP inside this venv on macOS; keep originals."""
import os
import sys
import sysconfig
from pathlib import Path

if sys.platform != 'darwin':
    raise SystemExit('This compatibility step is macOS-only')
site = Path(sysconfig.get_paths()['purelib'])
target = site / 'torch/lib/libomp.dylib'
assert target.is_file(), target
for package in ['faiss', 'sklearn']:
    lib = site / package / '.dylibs/libomp.dylib'
    if lib.is_symlink() and lib.resolve() == target.resolve():
        continue
    if lib.exists():
        backup = lib.with_suffix('.dylib.original')
        if backup.exists():
            raise RuntimeError(f'Unexpected replacement of {lib}; inspect backup before changing it')
        lib.rename(backup)
        lib.symlink_to(os.path.relpath(target,lib.parent))
        print(f'{package}: using PyTorch OpenMP; original preserved at {backup}')
