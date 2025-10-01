# rthook_paths.py
import os, sys, ctypes, tempfile
from pathlib import Path

LOG = Path(tempfile.gettempdir()) / "dynaface_dll_diag.txt"


def log(msg: str):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def add_dir(p: Path):
    try:
        if p and p.is_dir():
            os.add_dll_directory(str(p))
            log(f"add_dll_directory: {p}")
            return True
    except Exception as e:
        log(f"add_dll_directory FAILED for {p}: {e}")
    return False


def preload(paths):
    for p in paths:
        try:
            ctypes.WinDLL(str(p))
            log(f"WinDLL OK: {p.name}")
        except Exception as e:
            log(f"WinDLL FAIL: {p.name} -> {e}")


MEI = Path(getattr(sys, "_MEIPASS", "")) if hasattr(sys, "_MEIPASS") else None
if MEI and MEI.exists():
    ort = MEI / "ort"
    qt_bin = MEI / "PyQt6" / "Qt6" / "bin"

    log(f"_MEIPASS={MEI}")
    add_dir(ort)  # ORT first
    add_dir(qt_bin)  # then Qt (ok if missing)

    # Log dir contents so we see what actually got bundled
    try:
        items = ", ".join(sorted([p.name for p in ort.glob("*")]))
        log(f"ort contents: {items}")
    except Exception:
        pass

    # Preload in a dependency-friendly order
    wants = [
        "vcruntime140_1.dll",
        "vcruntime140.dll",
        "msvcp140.dll",
        "libiomp5md.dll",
        "vcomp140.dll",
        "onnxruntime_providers_shared.dll",
        "onnxruntime.dll",
    ]
    present = [ort / w for w in wants if (ort / w).exists()]

    # Include any other DLLs in ort/ as a final sweep
    others = [p for p in ort.glob("*.dll") if p not in present]
    preload(present + sorted(others))

    log("rthook_paths.py done.")
