from PyInstaller.utils.hooks import collect_data_files

# Collect data files from the specified subdirectories
datas = collect_data_files(
    "dynaface", subdir="spiga/data/annotations", excludes=["__pyinstaller"]
)
datas += collect_data_files(
    "dynaface", subdir="spiga/models/weights", excludes=["__pyinstaller"]
)
datas += collect_data_files(
    "dynaface", subdir="spiga/data/models3D", excludes=["__pyinstaller"]
)
