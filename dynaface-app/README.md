pyinstaller -y -F --icon heaton_ca_icon.png --onefile --windowed heaton-ca.py

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

pip install PyQtWebEngine

pip install dynaface -f ./wheels

C:\Users\jeffh\AppData\Local\Programs\Python\Python312

.\venv\Scripts\activate.bat
.\venv\Scripts\Activate.ps1

Version:
build.sh
const_values.py
build.spec
heaton-ca-osx.spec

PyQt6==6.5.0
PyQt6-Qt6==6.5.0
PyQt6-sip==13.5.0
PyQt6-WebEngine==6.5.0
PyQt6-WebEngine-Qt6==6.5.0

pip install PyQt6-Charts

pyqtgraph-0.13.3
pip install PyQt6-Charts

Windows
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
