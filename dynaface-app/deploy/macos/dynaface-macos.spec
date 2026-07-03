# -*- mode: python ; coding: utf-8 -*-
import os
import platform
import sys

version = os.getenv('version')

# Accessing the environment variable
arch = os.environ.get('arch')

# Setting the target_arch based on the environment variable or detection logic
if not arch:
    arch = platform.machine()

if arch not in ['arm64', 'x86_64']:
    raise ValueError("Unsupported architecture: " + arch)

block_cipher = None

added_files = [
    ("data/", "data"),  # existing files you're adding
    ("dynaface_doc_icon.icns", "."),  # path to your icon file
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
    name="dynaface",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=arch,  # universal2
    codesign_identity=None,
    entitlements_file=None,
    icon="dynaface_icon.icns",
)


coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="app",
)

app = BUNDLE(
    coll,
    name=f'Dynaface-{arch}.app',
    icon="dynaface_icon.icns",
    bundle_identifier="com.heatonresearch.dynaface",
    info_plist={
        "NSPrincipalClass": "NSApplication",
        "NSAppleScriptEnabled": False,
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeExtensions": ["dyfc"],
                "CFBundleTypeName": "Dynaface Document",
                "CFBundleTypeRole": "None",
                "CFBundleTypeIconFile": "dynaface_doc_icon",
                "LSItemContentTypes": ["com.heatonresearch.dyfc"],
                "LSHandlerRank": "Owner",
            }
        ],
        "UTExportedTypeDeclarations": [
            {
                "UTTypeIdentifier": "com.heatonresearch.dyfc",
                "UTTypeConformsTo": ["public.data"],
                "UTTypeDescription": "Dynaface Document",
                "UTTypeTagSpecification": {"public.filename-extension": ["dyfc"]},
            }
        ],
        "LSBackgroundOnly": False,
        "NSRequiresAquaSystemAppearance": "No",
        "CFBundlePackageType": "APPL",
        "CFBundleSupportedPlatforms": ["MacOSX"],
        "CFBundleIdentifier": "com.heatonresearch.dynaface",
        "CFBundleVersion": version,
        "CFBundleShortVersionString": version,
        "UIRequiredDeviceCapabilities":[arch],
        "LSMinimumSystemVersion": "12.0",
        "LSApplicationCategoryType": "public.app-category.utilities",
        "ITSAppUsesNonExemptEncryption": False,
        "DTPlatformBuild": "13C90",
        "DTPlatformName": "macos",
    },
)
