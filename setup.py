from setuptools import setup, find_packages

setup(
    name='facial_analysis',
    version='0.1.0',
    url='your-repo-url-if-any',
    author='Author Name',
    author_email='author@gmail.com',
    description='Description of my package',
    packages=find_packages(),    
    install_requires=["dlib>=19.24.1","opencv-python>=4.7.0","Pillow>=8.4.0","matplotlib>=3.7.1","scikit-learn>=1.2.2","plotly>=5.9.0","kaleido>=0.2.1"], 
)
