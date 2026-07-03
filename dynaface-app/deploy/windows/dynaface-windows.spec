# -*- mode: python ; coding: utf-8 -*-
import os
import platform
import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_dynamic_libs,
    collect_data_files,
)

# version from env (optional)
version = os.getenv('version', '0.0.1')

# detect or override architecture
raw_arch = os.environ.get('arch') or platform.machine()
arch_l = raw_arch.lower()
if arch_l in ('amd64', 'x86_64'):
    target_arch = 'x86_64'
elif arch_l in ('arm64', 'aarch64'):
    target_arch = 'arm64'
else:
    raise ValueError(f"Unsupported architecture: {raw_arch}")

# --------- COLLECT OPENCV PROPERLY ---------
cv2_datas, cv2_binaries, cv2_hidden = collect_all("cv2")

# --------- COLLECT PyQt6 (DLLs/plugins/resources) ---------
pyqt_binaries = collect_dynamic_libs("PyQt6")
pyqt_datas = collect_data_files("PyQt6")

# --------- COLLECT ONNXRUNTIME DLLs into dedicated 'ort/' ---------
ort_bins_raw = collect_dynamic_libs("onnxruntime")
ort_bins = [(src, "ort") for (src, _dest) in ort_bins_raw]

# App-specific data
added_datas = [
    ("data/", "data"),
    ("dynaface_doc_icon.ico", "."),
]

# Extra hidden imports (add more if PyInstaller misses any)
extra_hiddenimports = [
    'scipy._lib.array_api_compat.numpy.fft'
]

# Final aggregates
datas = added_datas + cv2_datas + pyqt_datas
binaries = cv2_binaries + pyqt_binaries + ort_bins
hiddenimports = (cv2_hidden or []) + extra_hiddenimports

# ---- EXTRA: Force critical runtimes into ort/ ----
import sys

def _find_in_sitepackages(basenames):
    sp = Path(sys.executable).parent.parent / "Lib" / "site-packages"
    hits = []
    for root, _, files in os.walk(sp):
        for f in files:
            if f.lower() in [b.lower() for b in basenames]:
                hits.append(str(Path(root) / f))
    # Prefer copies under numpy.libs / onnxruntime / torch if multiple are found
    hits.sort(key=lambda p: (".libs" not in p, "onnxruntime" not in p, "torch" not in p))
    return hits[0] if hits else None

def _first_that_exists(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None

def _maybe(src, dest_subdir="ort"):
    return [(src, dest_subdir)] if src and os.path.isfile(src) else []

py_dll_dir = Path(sys.executable).parent              # venv ...\Scripts
base_py      = Path(sys.base_prefix)                  # base Python root
sys32_dir    = Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32"

extra_ort_bins = []

# MSVC CRTs commonly required by onnxruntime
for name in ("vcruntime140_1.dll", "vcruntime140.dll", "msvcp140.dll"):
    candidates = [
        str(py_dll_dir / name),
        str(base_py / name),
        str(sys32_dir / name),
        _find_in_sitepackages([name]),
    ]
    picked = _first_that_exists(candidates)
    if picked:
        print(f"[spec] bundling into ort/: {picked}")
        extra_ort_bins.append((picked, "ort"))

# OpenMP runtimes (keep both if present; harmless if unused)
for omp in ("libiomp5md.dll", "vcomp140.dll"):
    p = _find_in_sitepackages([omp])
    if p:
        print(f"[spec] bundling into ort/: {p}")
        extra_ort_bins.append((p, "ort"))

# Add them to binaries, all landing in ort/
binaries += extra_ort_bins

a = Analysis(
    ['dynaface_app.py'],
    pathex=[os.path.abspath('.')],
    binaries=binaries,
    datas=datas,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthook_paths.py'],  # make sure this file is beside the .spec at build time
    excludes=[],
    noarchive=False,
    optimize=0,
    hiddenimports=hiddenimports,
)

pyz = PYZ(a.pure)

# Splash (unchanged)
splash = Splash(
    'splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    text_size=12,
    minify_script=True,
    always_on_top=True,
)

splash_file = os.path.abspath('splash.png')
print(f"Splash image path: {splash_file}")

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    splash,
    splash.binaries,
    [],
    name='dynaface',
    icon='dynaface_icon.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
