$ErrorActionPreference = "Stop"


# Constants
$MODEL_BINARY_URL = "https://github.com/jeffheaton/dynaface-models/releases/download/v1/dynaface_models.zip"

try {
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
    pip install --upgrade dynaface
    pip install --force-reinstall --upgrade onnxruntime==1.20.1

    # Move to deploy/windows
    Set-Location deploy/windows

    # Prepare working directory
    if (Test-Path "./working") {
        Remove-Item -Recurse -Force "./working"
    }
    New-Item -ItemType Directory -Path "./working" | Out-Null

    # Download and extract model binary
    Write-Output "** Downloading model binaries **"

    $TEMP_ZIP = [System.IO.Path]::GetTempFileName()
    $ZIP_FILE = "$TEMP_ZIP.zip"
    Rename-Item -Path $TEMP_ZIP -NewName $ZIP_FILE
    
    Invoke-WebRequest -Uri $MODEL_BINARY_URL -OutFile $ZIP_FILE
    Expand-Archive -Path $ZIP_FILE -DestinationPath "./working/data" -Force
    Remove-Item -Force $ZIP_FILE

    # Copy needed files
    Write-Output "** Copy files to working **"
    cp ../../data/* ./working/data
    cp ./dynaface_doc_icon.ico ./working
    cp ./dynaface_icon.ico ./working
    cp ./dynaface-windows.spec ./working
    cp ../../*.py ./working
    cp -r ../../jth_ui ./working/jth_ui
    
    # Run Pyinstaller
    Write-Output "**Run PyInstaller **"
    Set-Location ./working
    pyinstaller --clean --noconfirm --distpath dist --workpath build dynaface-windows.spec 
    
}
catch {
    Write-Error "An error occurred: $_"
    exit 1
}
