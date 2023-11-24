from PyInstaller.utils.hooks import collect_data_files

# Collect data files from the specified subdirectories
datas = collect_data_files('facial_analysis', subdir='spiga/data/annotations', excludes=['__pyinstaller'])
datas += collect_data_files('facial_analysis', subdir='spiga/models/weights', excludes=['__pyinstaller'])
datas += collect_data_files('facial_analysis', subdir='spiga/data/models3D', excludes=['__pyinstaller'])


