cd ./dynaface-app
rm -rf ./venv || true
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ../dynaface-lib
pip install -e .
