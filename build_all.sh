#!/bin/bash

if [ -z "${app_certificate}" ]; then
    echo "Error: Environment variable app_certificate is not set."
    exit 1  # Exit with a non-zero value to indicate an error
fi

cd ./dynaface-lib
cp $models/spiga_wflw.pt ./facial_analysis/spiga/models/weights/
python setup.py bdist_wheel
cp ./dist/*.whl ../dynaface-app/wheels/
cd ../dynaface-app
cp $models/onet.pt ./data
cp $models/pnet.pt ./data
cp $models/rnet.pt ./data
rm -rf ./venv || true
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install facial_analysis -f ./wheels
cd deploy/macos
rm -rf ./working || true
./build.sh