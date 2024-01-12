#!/bin/bash
set -e
set -o pipefail

trap 'echo "An error occurred. Exiting..."; exit 1;' ERR

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
codesign --force --timestamp --deep --verbose --options runtime --sign "${app_certificate}" dist/Dynaface.app

echo "** Sign nested **"
codesign --force --timestamp --verbose --options runtime --entitlements entitlements-nest.plist --sign "${app_certificate}" dist/Dynaface.app/Contents/Frameworks/torch/bin/protoc
codesign --force --timestamp --verbose --options runtime --entitlements entitlements-nest.plist --sign "${app_certificate}" dist/Dynaface.app/Contents/Frameworks/torch/bin/protoc-3.13.0.0
codesign --force --timestamp --verbose --options runtime --entitlements entitlements-nest.plist --sign "${app_certificate}" dist/Dynaface.app/Contents/Frameworks/torch/bin/torch_shm_manager

echo "** Sign App **"
cp $provisionprofile dist/Dynaface.app/Contents/embedded.provisionprofile
codesign --force --timestamp --verbose --options runtime --entitlements entitlements.plist --sign "${app_certificate}" dist/Dynaface.app/Contents/MacOS/dynaface

echo "** Verify Sign **"
codesign --verify --verbose dist/Dynaface.app

echo "** Package **"
productbuild --component dist/Dynaface.app /Applications --sign "${installer_certificate}" --version "${version}" dist/Dynaface.pkg

echo "Build of application executed successfully."
exit 0