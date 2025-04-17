# PowerShell equivalent of the shell script

# Constants
$MODEL_BINARY_URL = "https://github.com/jeffheaton/dynaface-models/releases/download/v1/dynaface_models.zip"

# Move to project root
Set-Location ../..

# Remove existing virtual environment
if (Test-Path "./venv") {
    Remove-Item -Recurse -Force "./venv"
}

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
& "./venv/Scripts/Activate.ps1"

# Install dependencies
pip install -r requirements.txt
pip install --upgrade https://data.heatonresearch.com/library/dynaface-0.1.11-py3-none-any.whl

# Move to deploy/macos
Set-Location deploy/macos

# Prepare working directory
if (Test-Path "./working") {
    Remove-Item -Recurse -Force "./working"
}
New-Item -ItemType Directory -Path "./working" | Out-Null

# Download and extract model binary
Write-Output "** Downloading model binaries **"
$TEMP_ZIP = New-TemporaryFile
Invoke-WebRequest -Uri $MODEL_BINARY_URL -OutFile $TEMP_ZIP

Write-Output "** Extracting model binaries to ./working/data **"
New-Item -ItemType Directory -Path "./working/data" -Force | Out-Null
Copy-Item -Path "../../data/*" -Destination "./working/data" -Recurse -Force
Expand-Archive -Path $TEMP_ZIP -DestinationPath "./working/data" -Force

Write-Output "** Cleaning up temporary zip **"
Remove-Item -Force $TEMP_ZIP

# Copy needed files
cp ./dynaface_icon.icns ./working
cp ./dynaface-windows.spec ./working

# Run Pyinstaller

pyinstaller --clean --noconfirm --distpath dist --workpath build dynaface-windows.spec
