cd ./dynaface-app
rm -rf ./venv || true
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ../dynaface-lib
pip install -e .
