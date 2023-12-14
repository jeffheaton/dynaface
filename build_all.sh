#!/bin/bash

if [ -z "${app_certificate}" ]; then
    echo "Error: Environment variable app_certificate is not set."
    exit 1  # Exit with a non-zero value to indicate an error
fi

cd ./dynaface-lib
python setup.py bdist_wheel
mkdir -p ../dynaface-app/wheels/
cp ./dist/*.whl ../dynaface-app/wheels/
cd ../dynaface-app
cp $models/onet.pt ./data
cp $models/pnet.pt ./data
cp $models/rnet.pt ./data
cp $models/spiga_wflw.pt ./data
rm -rf ./venv || true
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install facial_analysis -f ./wheels
cd deploy/macos
rm -rf ./working || true
./build.sh