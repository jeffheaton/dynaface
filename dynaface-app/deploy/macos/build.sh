#!/bin/bash
set -e
set -o pipefail

trap 'echo "An error occurred. Exiting..."; exit 1;' ERR

# Constants
MODEL_BINARY_URL="https://data.heatonresearch.com/dynaface/model/2/dynaface_models.zip"
DYNAFACE_VERSION="2.0.2"

# Apple Silicon only as of 2.0.0. Intel users are directed to the Unity build,
# which ships universal. The spec file reads `arch` from the environment.
export arch="arm64"

if [ -z "${app_certificate}" ]; then
    echo "Error: Environment variable app_certificate is not set."
    exit 1  # Exit with a non-zero value to indicate an error
fi

# ----------------------------
# Update version.py
# ----------------------------
echo "** Updating version.py **"
BUILD_DATE=$(date +"%Y-%m-%dT%H:%M:%S%z")
BUILD_NUM="${BUILD_NUMBER:-0}"
cat > ../../version.py <<EOF
VERSION = "${DYNAFACE_VERSION}"
BUILD_DATE = "${BUILD_DATE}"
BUILD = ${BUILD_NUM}
EOF
echo "** version.py updated: VERSION=${DYNAFACE_VERSION}, BUILD=${BUILD_NUM}, DATE=${BUILD_DATE} **"

# The spec file stamps CFBundleVersion / CFBundleShortVersionString from this.
export version="${DYNAFACE_VERSION}"

# Environment
cd ../..
rm -rf ./venv || true

# Local dev machines expose the interpreter as `python3.11`; CI sets PYTHON_EXE
# to the interpreter path from actions/setup-python.
PYTHON="${PYTHON_EXE:-python3.11}"
echo "** Using Python: ${PYTHON} **"
"$PYTHON" -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --upgrade "dynaface==${DYNAFACE_VERSION}"
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
cp ./dynaface_icon.icns ./working
cp ./dynaface_doc_icon.icns ./working
cp ./dynaface-macos.spec ./working
cp ./build.sh ./working
cp ../../*.py ./working
cp -r ../../jth_ui ./working/jth_ui

cd ./working

echo "** Pyinstaller **"
pyinstaller --clean --noconfirm --distpath dist --workpath build dynaface-macos.spec

APP="dist/Dynaface.app"

# Sign inside-out: every Mach-O inside the bundle, then the bundle itself.
# Apple discourages --deep for distribution and notarization rejects bundles
# whose nested code was not signed independently.
#
# Match on file type rather than extension: PyInstaller ad-hoc signs the bundled
# Qt and Python frameworks, whose binaries are named e.g. `QtCore` and `Python`,
# so an extension filter silently leaves them ad-hoc signed and notarization
# comes back Invalid.
echo "** Sign nested code **"
find "$APP" -type f -print0 |
    while IFS= read -r -d '' f; do
        if file -b "$f" | grep -q 'Mach-O'; then
            codesign --force --timestamp --options runtime \
                --sign "${app_certificate}" "$f"
        fi
    done

# Framework bundles must also be sealed as bundles, deepest first.
echo "** Sign frameworks **"
find "$APP" -depth -type d -name "*.framework" -print0 |
    while IFS= read -r -d '' fw; do
        codesign --force --timestamp --options runtime \
            --sign "${app_certificate}" "$fw"
    done

echo "** Sign App **"
codesign --force --timestamp --options runtime \
    --entitlements entitlements.plist \
    --sign "${app_certificate}" "$APP"

echo "** Verify Sign **"
codesign --verify --deep --strict --verbose=2 "$APP"

# ----------------------------
# Notarize
# ----------------------------
ZIP_NAME="dynaface-app-mac-${DYNAFACE_VERSION}.zip"

if [ -n "${AC_API_KEY_PATH}" ]; then
    echo "** Notarize **"
    # notarytool takes an archive, so the .app has to be zipped to submit it.
    # Use ditto, not zip: ditto preserves the symlinks, extended attributes and
    # signature structure that `zip` quietly mangles inside a bundle.
    ditto -c -k --keepParent "$APP" "dist/notarize.zip"

    # `notarytool submit --wait` exits 0 even when the verdict is Invalid; it
    # only fails on upload errors. Check the status explicitly, or a rejected
    # build sails on to stapler and dies with an unrelated Error 65.
    SUBMIT_JSON=$(xcrun notarytool submit "dist/notarize.zip" \
        --key "${AC_API_KEY_PATH}" \
        --key-id "${AC_API_KEY_ID}" \
        --issuer "${AC_API_ISSUER_ID}" \
        --wait --timeout 30m --output-format json)
    echo "$SUBMIT_JSON"
    rm -f "dist/notarize.zip"

    SUBMISSION_ID=$(echo "$SUBMIT_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
    STATUS=$(echo "$SUBMIT_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')

    if [ "$STATUS" != "Accepted" ]; then
        echo "** Notarization returned '${STATUS}' -- fetching the reason **"
        xcrun notarytool log "${SUBMISSION_ID}" \
            --key "${AC_API_KEY_PATH}" \
            --key-id "${AC_API_KEY_ID}" \
            --issuer "${AC_API_ISSUER_ID}" || true
        exit 1
    fi

    # The ticket staples to the .app and can never be stapled to a zip, so the
    # zip that users download has to be built AFTER this step. An unstapled app
    # fails Gatekeeper for anyone offline or behind a slow network.
    echo "** Staple **"
    xcrun stapler staple "$APP"
    xcrun stapler validate "$APP"

    echo "** Gatekeeper check **"
    spctl -a -vvv "$APP"
else
    echo "** WARNING: AC_API_KEY_PATH not set, skipping notarization. **"
    echo "** Gatekeeper will block this build on any machine but this one. **"
fi

echo "** Package ${ZIP_NAME} **"
ditto -c -k --keepParent "$APP" "dist/${ZIP_NAME}"

echo "Build of application executed successfully."
exit 0
