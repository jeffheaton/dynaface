# -*- mode: python ; coding: utf-8 -*-
import os
import platform
import sys

block_cipher = None

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

# files to bundle alongside your scripts
added_files = [
    ("data/", "data"),               # your data directory
    # if you have a document‑icon for Windows, include it here:
    # ("dynaface_doc_icon.ico", "."),
]

a = Analysis(
    ["dynaface_app.py"],
    pathex=["."],
    binaries=[],
    datas=added_files,
    hiddenimports=['scipy._lib.array_api_compat.numpy.fft'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    # Windows‑specific settings:
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="dynaface.exe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                     # set to True for a console app
    disable_windowed_traceback=False,
    target_arch=target_arch,
    icon="dynaface_icon.ico",          # your Windows .ico file
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=f"Dynaface-win-{target_arch}",  # output folder name
)
