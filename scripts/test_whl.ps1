# Example use: .\test_whl.ps1 3.11 0.2.1
param (
    [Parameter(Mandatory = $true)][string]$python_version,
    [Parameter(Mandatory = $true)][string]$wheel_version
)

$ErrorActionPreference = "Stop"

# Upgrade pip for the specified Python interpreter
& "pip$python_version" install --upgrade pip

# Remove previous virtual environment if it exists
if (Test-Path "./venv") {
    Remove-Item -Recurse -Force "./venv"
}

# Create a new virtual environment using the specified Python version
& "python$python_version" -m venv venv

# Activate the virtual environment (Windows-specific)
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip and setuptools inside the virtual environment
pip install --upgrade pip setuptools

# Install the built wheel file
pip install --upgrade "https://data.heatonresearch.com/library/dynaface-$wheel_version-py3-none-any.whl"

# Create a temporary directory
$tmp_dir = New-TemporaryFile
Remove-Item $tmp_dir
$tmp_dir = New-Item -ItemType Directory -Path ([System.IO.Path]::Combine($env:TEMP, "dynaface-tests-" + [System.Guid]::NewGuid().ToString("N")))

Write-Host "Created temporary directory: $($tmp_dir.FullName)"

# Define cleanup logic
$cleanup = {
    Write-Host "Cleaning up temporary directory: $($tmp_dir.FullName)"
    Remove-Item -Recurse -Force $tmp_dir
}
Register-EngineEvent PowerShell.Exiting -Action $cleanup | Out-Null

# Clone the repository
git clone https://github.com/jeffheaton/dynaface.git "$($tmp_dir.FullName)\dynaface"

# Copy test files
$temp_tests = "$($tmp_dir.FullName)\temp_tests"
New-Item -ItemType Directory -Path $temp_tests | Out-Null
Copy-Item -Recurse "$($tmp_dir.FullName)\dynaface\dynaface-lib\tests\*" $temp_tests

# Copy test_data folder if it exists
$tests_data = "$($tmp_dir.FullName)\dynaface\dynaface-lib\tests_data"
if (Test-Path $tests_data) {
    Copy-Item -Recurse $tests_data "$temp_tests\tests_data"
}

# Change to test directory and run tests
Push-Location $temp_tests
& "python$python_version" -m unittest discover -s .
Pop-Location

# Deactivate the virtual environment
& ".\venv\Scripts\Deactivate.ps1"
