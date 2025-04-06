cd ./dynaface-app
rm -rf ./venv || true
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download and unzip dynaface_models.zip
mkdir -p ./data
curl -L -o ./data/dynaface_models.zip https://github.com/jeffheaton/dynaface-models/releases/download/v1/dynaface_models.zip
unzip -o ./data/dynaface_models.zip -d ./data/
rm ./data/dynaface_models.zip

cd ../dynaface-lib
pip install -e .[dev]
