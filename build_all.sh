cd ./dynaface-lib
python setup.py bdist_wheel
cp ./dist/*.whl ../dynaface-app/wheels/
cd ../dynaface-app
rm -rf ./venv || true
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install facial_analysis -f ./wheels
cd deploy/macos
rm -rf ./working || true
./build.sh