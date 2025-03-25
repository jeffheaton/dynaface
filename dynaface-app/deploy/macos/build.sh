#!/bin/bash
set -e
set -o pipefail

trap 'echo "An error occurred. Exiting..."; exit 1;' ERR

if [ -z "${app_certificate}" ]; then
    echo "Error: Environment variable app_certificate is not set."
    exit 1  # Exit with a non-zero value to indicate an error
fi

if [ -z "${arch}" ]; then
    echo "Error: Environment variable arch is not set."
    exit 1  # Exit with a non-zero value to indicate an error
fi

# Constants
MODEL_BINARY_URL="https://github.com/jeffheaton/dynaface-models/releases/download/v1/dynaface_models.zip"

# Environment
cd ../..
rm -rf ./venv || true
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --upgrade https://data.heatonresearch.com/library/dynaface-0.1.11-py3-none-any.whl
cd deploy/macos

# Build it
rm -rf ./working
mkdir ./working

# Download and extract model binary
echo "** Downloading model binaries **"
TEMP_ZIP=$(mktemp)
curl -L "$MODEL_BINARY_URL" -o "$TEMP_ZIP"

echo "** Extracting model binaries to ./working/data **"
mkdir -p ./working/data
cp -r ../../data/. ./working/data
unzip -o "$TEMP_ZIP" -d ./working/data

echo "** Cleaning up temporary zip **"
rm "$TEMP_ZIP"

# Copy other files
cp ./entitlements.plist ./working
cp ./entitlements-nest.plist ./working
cp ./dynaface_icon.icns ./working
cp ./dynaface_doc_icon.icns ./working
cp ./dynaface-macos.spec ./working
cp ./build.sh ./working
cp ../../*.py ./working
cp -r ../../jth_ui ./working/jth_ui

cd ./working
echo "** Force PyTorch Upgrade **"

if [[ "$arch" == "x86_64" ]]; then
    echo "Detected MacOS Intel (x86_64)"

    pip install /Users/jeff/output/wheel/torch-2.4.0a0+git1cd4199-cp311-cp311-macosx_14_0_x86_64.whl
    pip install /Users/jeff/output/wheel/torchvision-0.9.0a0+761d09f-cp311-cp311-macosx_14_0_x86_64.whl
elif [[ "$arch" == "arm64" ]]; then
    echo "Detected MacOS ARM (arm64)"
    pip install /Users/jeff/output/wheel/torch-2.4.0a0+git1cd4199-cp311-cp311-macosx_14_0_arm64.whl
    pip install /Users/jeff/output/wheel/torchvision-0.9.0a0+761d09f-cp311-cp311-macosx_14_0_arm64.whl
else
    echo "Using existing PyTorch install"
fi

echo "** Pyinstaller **"
pyinstaller --clean --noconfirm --distpath dist --workpath build dynaface-macos.spec

echo "** Sign Deep **"
cp $provisionprofile dist/Dynaface-${arch}.app/Contents/embedded.provisionprofile
codesign --force --timestamp --deep --verbose --options runtime --sign "${app_certificate}" dist/Dynaface-${arch}.app

echo "** Sign nested **"
codesign --force --timestamp --verbose --options runtime --entitlements entitlements-nest.plist --sign "${app_certificate}" dist/Dynaface-${arch}.app/Contents/Frameworks/torch/bin/protoc
codesign --force --timestamp --verbose --options runtime --entitlements entitlements-nest.plist --sign "${app_certificate}" dist/Dynaface-${arch}.app/Contents/Frameworks/torch/bin/protoc-3.13.0.0
codesign --force --timestamp --verbose --options runtime --entitlements entitlements-nest.plist --sign "${app_certificate}" dist/Dynaface-${arch}.app/Contents/Frameworks/torch/bin/torch_shm_manager

echo "** Sign App **"
codesign --force --timestamp --verbose --options runtime --entitlements entitlements.plist --sign "${app_certificate}" dist/Dynaface-${arch}.app/Contents/MacOS/dynaface

echo "** Verify Sign **"
codesign --verify --verbose dist/Dynaface-${arch}.app

# Set permissions, sometimes the transport app will complain about this
echo "** Set Permissions **"
find dist/Dynaface-${arch}.app -type f -exec chmod a=u {} \;
find dist/Dynaface-${arch}.app -type d -exec chmod a=u {} \;

echo "** Package **"
productbuild --component dist/Dynaface-${arch}.app /Applications --sign "${installer_certificate}" --version "${version}" dist/Dynaface-${arch}.pkg

echo "Build of application executed successfully."
exit 0