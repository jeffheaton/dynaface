#!/bin/bash
if [ -z "${app_certificate}" ]; then
    echo "Error: Environment variable app_certificate is not set."
    exit 1  # Exit with a non-zero value to indicate an error
fi

if [ -z "${arch}" ]; then
    echo "Error: Environment variable arch is not set."
    exit 1  # Exit with a non-zero value to indicate an error
fi


# Environment
cd ../..
rm -rf ./venv || true
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd deploy/macos

# Build it
rm -rf ./working
mkdir ./working
cp ./entitlements.plist ./working
cp ./entitlements-nest.plist ./working
cp ./dynaface_icon.icns ./working
cp ./dynaface_doc_icon.icns ./working
cp ./dynaface-macos.spec ./working
cp ./build.sh ./working
cp ../../*.py ./working
cp -r ../../data ./working/data
cp -r ../../jth_ui ./working/jth_ui

cd ./working

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