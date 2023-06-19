# facial-analysis
pip install --upgrade pip

python3 -m venv venv
source venv/bin/activate
#python setup.py install
pip install -e .

python -m unittest discover -s tests

git clone https://github.com/andresprados/SPIGA.git
cd SPIGA
pip install -e .
cd ..
rm -rf ./SPIGA

https://www.geeksforgeeks.org/using-jupyter-notebook-in-virtual-environment/

ipython kernel install --user --name=venv

python setup.py sdist bdist_wheel

cp -R /content/drive/MyDrive/facial_analysis/ /content
