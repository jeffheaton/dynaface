# -*- mode: python ; coding: utf-8 -*-
import platform
from PyInstaller.utils.hooks import collect_all

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
    ("data/", "data"),
    ("dynaface_doc_icon.ico", ".")
]

# Add any extra hidden imports you know about
extra_hiddenimports = [
    'scipy._lib.array_api_compat.numpy.fft'
]

# Final lists
datas = added_datas + cv2_datas

a = Analysis(
    ['dynaface_app.py'],
    pathex=[os.path.abspath('.')],
    binaries=[],
    datas=datas,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
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