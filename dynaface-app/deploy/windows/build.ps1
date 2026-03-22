$ErrorActionPreference = "Stop"

# Save the initial directory
$InitialLocation = Get-Location

# Constants
$script:MODEL_BINARY_URL = "https://github.com/jeffheaton/dynaface-models/releases/download/v1/dynaface_models.zip"
$script:DYNAFACE_WHL = "https://s3.us-east-1.amazonaws.com/data.heatonresearch.com/library/dynaface-0.3.0-py3-none-any.whl"

try {
    # ----------------------------
    # Update version.py
    # ----------------------------
    Write-Host "** Updating version.py **"
    $versionFile = "..\..\version.py"
    $currentVersion = "1.3.0"
    $buildDate = (Get-Date -Format "yyyy-MM-ddTHH:mm:sszzz")
    $buildNum = $env:BUILD_NUMBER
    if (-not $buildNum) { $buildNum = 0 }

    @"
VERSION = `"$currentVersion`"
BUILD_DATE = `"$buildDate`"
BUILD = $buildNum
"@ | Set-Content $versionFile -Encoding UTF8

    Write-Host "** version.py updated: VERSION=$currentVersion, BUILD=$buildNum, DATE=$buildDate **"

    # ----------------------------
    # Virtual Environment Setup
    # ----------------------------
    Set-Location ../..
    $appRoot = Get-Location

    if (Test-Path "./venv") {
        Remove-Item -Recurse -Force "./venv"
    }

    python3.11 -m venv venv
    $venvPython = Join-Path $appRoot "venv/Scripts/python.exe"
    $venvPip = Join-Path $appRoot "venv/Scripts/pip.exe"

    & $venvPip install -r requirements.txt
    & $venvPip install --upgrade $script:DYNAFACE_WHL

    # onnxruntime loads onnxruntime_providers_shared.dll at import time.
    # PyInstaller's isolated analysis subprocess doesn't have this DLL in PATH,
    # causing rembg/bg.py to call sys.exit(1) and crash the build.
    # Adding the capi directory to PATH fixes DLL resolution during analysis.
    $ortCapi = Join-Path $appRoot "venv\Lib\site-packages\onnxruntime\capi"
    if (Test-Path $ortCapi) {
        $env:PATH = "$ortCapi;$env:PATH"
        Write-Host "** Added onnxruntime capi to PATH for PyInstaller analysis **"
    }

    Set-Location deploy/windows

    # ----------------------------
    # Working Directory Prep
    # ----------------------------
    if (Test-Path "./working") {
        Remove-Item -Recurse -Force "./working"
    }
    New-Item -ItemType Directory -Path "./working" | Out-Null

    $workingDataDir = "./working/data"
    if (-not (Test-Path $workingDataDir)) {
        New-Item -ItemType Directory -Path $workingDataDir -Force | Out-Null
    }

    # ----------------------------
    # Download & Extract Models
    # ----------------------------
    Write-Host "** Downloading model binaries **"
    $TEMP_ZIP = [System.IO.Path]::GetTempFileName()
    $ZIP_FILE = "$TEMP_ZIP.zip"
    Rename-Item -Path $TEMP_ZIP -NewName $ZIP_FILE

    # Replaced Invoke-WebRequest with native curl.exe for speed/reliability
    & curl.exe --fail --location --retry 5 --retry-delay 2 --connect-timeout 30 --max-time 600 -o $ZIP_FILE $script:MODEL_BINARY_URL

    Expand-Archive -Path $ZIP_FILE -DestinationPath $workingDataDir -Force
    Remove-Item -Force $ZIP_FILE

    # ----------------------------
    # Copy Build Files
    # ----------------------------
    Write-Host "** Copying project files to working directory **"
    Copy-Item ../../data/* -Destination $workingDataDir -Recurse -Force
    Copy-Item ./dynaface_doc_icon.ico -Destination ./working
    Copy-Item ./dynaface_icon.ico -Destination ./working
    Copy-Item ./splash.png -Destination ./working
    Copy-Item ./dynaface-windows.spec -Destination ./working
    Copy-Item ../../*.py -Destination ./working
    Copy-Item ../../jth_ui -Destination ./working/jth_ui -Recurse -Force
    Copy-Item ./rthook_paths.py -Destination ./working
    Copy-Item ./rthook_diag.py -Destination ./working

    # ----------------------------
    # Run PyInstaller
    # ----------------------------
    Write-Host "** Running PyInstaller **"
    Set-Location ./working
    & $venvPython -m PyInstaller --clean --noconfirm --distpath dist --workpath build dynaface-windows.spec
}
catch {
    Write-Error "An error occurred: $_"
    exit 1
}
finally {
    Set-Location $InitialLocation
}
