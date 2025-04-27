$ErrorActionPreference = "Stop"

# Save the initial directory
$InitialLocation = Get-Location

# Constants (scoped to the script)
$script:MODEL_BINARY_URL = "https://github.com/jeffheaton/dynaface-models/releases/download/v1/dynaface_models.zip"
$script:DYNAFACE_WHL = "https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/library/dynaface-0.2.2-py3-none-any.whl"

try {
    # Move to project root
    Set-Location ../..

    # Remove existing virtual environment
    if (Test-Path "./venv") {
        Remove-Item -Recurse -Force "./venv"
    }

    # Create virtual environment
    python3.11 -m venv venv
    $venvPython = Join-Path (Get-Location) "venv/Scripts/python.exe"
    $venvPip = Join-Path (Get-Location) "venv/Scripts/pip.exe"

    # Install dependencies with constraint
    & $venvPip install -r requirements.txt --constraint ./deploy/windows/constraints.txt

    # Debug output for DYNAFACE_WHL
    Write-Output "DYNAFACE_WHL: $script:DYNAFACE_WHL"

    # Install dynaface with constraint
    & $venvPip install --constraint ./deploy/windows/constraints.txt --upgrade $script:DYNAFACE_WHL

    # Move to deploy/windows
    Set-Location deploy/windows

    # Prepare working directory
    if (Test-Path "./working") {
        Remove-Item -Recurse -Force "./working"
    }
    New-Item -ItemType Directory -Path "./working" | Out-Null

    # Download and extract model binary
    Write-Output "** Downloading model binaries **"

    # Debug output for MODEL_BINARY_URL
    Write-Output "MODEL_BINARY_URL: $script:MODEL_BINARY_URL"

    $TEMP_ZIP = [System.IO.Path]::GetTempFileName()
    $ZIP_FILE = "$TEMP_ZIP.zip"
    Rename-Item -Path $TEMP_ZIP -NewName $ZIP_FILE
    
    Invoke-WebRequest -Uri $script:MODEL_BINARY_URL -OutFile $ZIP_FILE
    Expand-Archive -Path $ZIP_FILE -DestinationPath "./working/data" -Force
    Remove-Item -Force $ZIP_FILE

    # Copy needed files
    Write-Output "** Copy files to working **"
    cp ../../data/* ./working/data
    cp ./dynaface_doc_icon.ico ./working
    cp ./dynaface_icon.ico ./working
    cp ./splash.png ./working
    cp ./dynaface-windows.spec ./working
    cp ../../*.py ./working
    cp -r ../../jth_ui ./working/jth_ui
    
    # Run PyInstaller
    Write-Output "** Run PyInstaller **"
    Set-Location ./working
    & $venvPython -m PyInstaller --clean --noconfirm --distpath dist --workpath build dynaface-windows.spec 

}
catch {
    Write-Error "An error occurred: $_"
    exit 1
}
finally {
    # Always restore the initial directory
    Set-Location $InitialLocation
}
