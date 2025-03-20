#!/bin/bash
set -e
set -o pipefail

trap 'echo "An error occurred. Exiting..."; exit 1;' ERR

if [ -z "${app_certificate}" ]; then
    echo "Error: Environment variable app_certificate is not set."
    exit 1  # Exit with a non-zero value to indicate an error
fi
# Baseline utilities to build package
rm -rf ./venv || true
python3.12 -m venv venv   
source venv/bin/activate   
pip install setuptools wheel
# Build it
cd ./dynaface-lib
python3.12 setup.py bdist_wheel
mkdir -p ../dynaface-app/wheels/
cp ./dist/*.whl ../dynaface-app/wheels/
cd ../dynaface-app
cp $models/onet.pt ./data
cp $models/pnet.pt ./data
cp $models/rnet.pt ./data
cp $models/spiga_wflw.pt ./data
rm -rf ./venv || true
python3.12 -m venv venv
source venv/bin/activate
pip3.12 install -r requirements.txt -f /Users/jeff/output/pytorch
pip3.12 install dynaface -f ./wheels -f /Users/jeff/output/PyTorch
cd deploy/macos
rm -rf ./working || true
./build.sh

echo "Build of library/app executed successfully."
exit 0
