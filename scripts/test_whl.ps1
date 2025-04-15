# Usage: .\test_whl.ps1 3.11 0.2.1

param (
    [Parameter(Mandatory = $true)][string]$python_version,
    [Parameter(Mandatory = $true)][string]$wheel_version
)

$ErrorActionPreference = "Stop"

# Paths
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$venvPath = Join-Path $scriptPath "venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

# Remove existing virtual environment
if (Test-Path $venvPath) {
    Remove-Item -Recurse -Force $venvPath
}

# Create virtual environment using the specified version
& "python$python_version" -m venv $venvPath

# Ensure the Python executable exists
if (!(Test-Path $venvPython)) {
    throw "Virtual environment Python not found at $venvPython"
}

# Upgrade pip and setuptools inside the virtual environment
& $venvPython -m pip install --upgrade pip setuptools

# Install the specified wheel
$wheelUrl = "https://data.heatonresearch.com/library/dynaface-$wheel_version-py3-none-any.whl"
& $venvPython -m pip install --upgrade $wheelUrl

# Create temp test directory
$tmp_dir = New-TemporaryFile
Remove-Item $tmp_dir
$tmp_dir = New-Item -ItemType Directory -Path ([System.IO.Path]::Combine($env:TEMP, "dynaface-tests-" + [System.Guid]::NewGuid().ToString("N")))
Write-Host "Created temporary directory: $($tmp_dir.FullName)"

# Register cleanup
$cleanup = {
    Write-Host "Cleaning up temporary directory: $($using:tmp_dir.FullName)"
    Remove-Item -Recurse -Force $using:tmp_dir
}
Register-EngineEvent PowerShell.Exiting -Action $cleanup | Out-Null

# Clone the repo
git clone https://github.com/jeffheaton/dynaface.git "$($tmp_dir.FullName)\dynaface"

# Copy test files
$temp_tests = "$($tmp_dir.FullName)\temp_tests"
New-Item -ItemType Directory -Path $temp_tests | Out-Null
Copy-Item -Recurse "$($tmp_dir.FullName)\dynaface\dynaface-lib\tests\*" $temp_tests

# Optionally copy test_data
$tests_data = "$($tmp_dir.FullName)\dynaface\dynaface-lib\tests_data"
if (Test-Path $tests_data) {
    Copy-Item -Recurse $tests_data "$temp_tests\tests_data"
}

# Run tests
Push-Location $temp_tests
& $venvPython -m unittest discover -s .
Pop-Location
