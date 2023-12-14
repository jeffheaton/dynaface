# Change directory to dynaface-app
Set-Location -Path ".\dynaface-app"

# Remove venv folder if it exists (ignore errors if it doesn't)
Remove-Item -Path ".\venv" -Recurse -Force -ErrorAction SilentlyContinue

# Create a virtual environment using Python 3.11
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install Python dependencies from requirements.txt
pip install -r .\requirements.txt

# Change directory to dynaface-lib
Set-Location -Path "..\dynaface-lib"

# Install the dynaface-lib package in editable mode
pip install -e .


