from setuptools import setup, find_packages
setup(
    name='irisclassifier',
    version='0.1.0',
    description='A Python library for Iris flower classification',
    author='Author Name',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'seaborn',
        'matplotlib',
        'scikit-learn',
        'tensorflow-macos',
        'pytest'
    ],
)