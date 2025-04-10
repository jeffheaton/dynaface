#!/bin/bash
# Example use: test_whl.sh 3.11 0.2.1
set -e  # Exit immediately if any command fails

# Check that two arguments (Python version and wheel version) are supplied.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <python_version> <wheel_version>" >&2
    exit 1
fi

PYTHON_VERSION=$1
WHEEL_VERSION=$2

# Upgrade pip for the specified Python interpreter.
pip${PYTHON_VERSION} install --upgrade pip

# Remove previous virtual environment if it exists.
rm -rf ./venv || true

# Create a new virtual environment using the specified Python version and activate it.
python${PYTHON_VERSION} -m venv venv
source venv/bin/activate

# Upgrade pip and setuptools inside the virtual environment.
pip install --upgrade pip setuptools

# Install the built wheel file. This installs your package with its metadata and dependencies.
pip install --upgrade "https://data.heatonresearch.com/library/dynaface-${WHEEL_VERSION}-py3-none-any.whl"

# Create a system temporary directory to host the cloned repository and the test files.
TMP_DIR=$(mktemp -d -t dynaface-tests-XXXXXX)
echo "Created temporary directory: $TMP_DIR"

# Set up a trap to clean up the temporary directory when the script exits.
cleanup() {
    echo "Cleaning up temporary directory: $TMP_DIR"
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# Clone the repository into a subdirectory within the temporary directory.
git clone https://github.com/jeffheaton/dynaface.git "$TMP_DIR/dynaface"

# Create a temporary directory for running tests inside the temporary directory.
mkdir "$TMP_DIR/temp_tests"

# Copy over the test files from the cloned repository.
cp -r "$TMP_DIR/dynaface/dynaface-lib/tests/"* "$TMP_DIR/temp_tests/"

# Also copy the tests_data folder if it exists.
if [ -d "$TMP_DIR/dynaface/dynaface-lib/tests_data" ]; then
    cp -r "$TMP_DIR/dynaface/dynaface-lib/tests_data" "$TMP_DIR/temp_tests/"
fi

# Change into the temporary tests directory and run the tests using the specified Python interpreter.
cd "$TMP_DIR/temp_tests"
python${PYTHON_VERSION} -m unittest discover -s .

# Deactivate the virtual environment when finished.
deactivate

