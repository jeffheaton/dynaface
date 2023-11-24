pyinstaller -y -F --icon heaton_ca_icon.png --onefile --windowed heaton-ca.py

python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install facial_analysis -f ./wheels

pip install -e .

C:\Users\jeffh\AppData\Local\Programs\Python\Python312

.\.venv\Scripts\activate.bat

Version:
    build.sh
    const_values.py
    build.spec
    heaton-ca-osx.spec
