rm -rf ./working
mkdir ./working
cp ./entitlements.plist ./working
cp ./dynaface_icon.icns ./working
cp ./dynaface-macos.spec ./working
cp ./build.py ./working
cp ./build.sh ./working
cp ../../*.py ./working
cp -r ../../data ./working/data
cp -r ../../jth_ui ./working/jth_ui

cd ./working
python build.py \
    --app_name "Dynaface" \
    --version "$version" \
    --spec_file "dynaface-macos.spec" \
    --entitlements "entitlements.plist" \
    --provisioning_profile "$provisionprofile" \
    --app_certificate "$app_certificate" \
    --installer_certificate "$installer_certificate" \
    --output_dir "dist"
cd ..