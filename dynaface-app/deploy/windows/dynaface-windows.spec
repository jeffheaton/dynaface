# -*- mode: python ; coding: utf-8 -*-
import os
import platform
from PyInstaller.utils.hooks import collect_all

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

# --------- COLLECT OPENCV PROPERLY ---------

cv2_datas, cv2_binaries, cv2_hidden = collect_all("cv2")

# Add any app-specific data
added_datas = [
    ("data/", "data")
]

# Add any extra hidden imports you know about
extra_hiddenimports = [
    'scipy._lib.array_api_compat.numpy.fft'
]

# Final lists
datas = added_datas + cv2_datas
binaries = cv2_binaries
hiddenimports = cv2_hidden + extra_hiddenimports

# --------- BUILD ---------

a = Analysis(
    ["dynaface_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    optimize=0,
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
    upx=False,  # UPX disabled for PyQt/OpenCV compatibility
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False once stable
    disable_windowed_traceback=False,
    target_arch=target_arch,
    icon="dynaface_icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=f"Dynaface-win-{target_arch}",
)