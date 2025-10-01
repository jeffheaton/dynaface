# rthook_diag.py  (temporary)
import os, sys, tempfile
from pathlib import Path

log = Path(tempfile.gettempdir()) / "dynaface_dll_diag.txt"
try:
    root = Path(getattr(sys, "_MEIPASS", ""))
    lines = []
    lines.append(f"_MEIPASS={root}\n")
    ort = root / "ort"
    lines.append(
        f"ort_exists={ort.exists()} contents={list(ort.iterdir()) if ort.exists() else 'N/A'}\n"
    )
    qt_bin = root / "PyQt6" / "Qt6" / "bin"
    lines.append(f"qt_bin_exists={qt_bin.exists()}\n")
    with open(log, "w", encoding="utf-8") as f:
        f.writelines(lines)
except Exception as e:
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"diag error: {e}\n")
