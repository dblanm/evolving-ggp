from setuptools import setup

setup(
    name="e-ggp",
    version='0.1',
    description='evolving-Graph Gaussian process module',
    author='David Blanco Mulero',
    packages=['e-ggp'],
    url="",
    license="Clear BSD License",
    install_requires = ['gpytorch==1.3.0', 'torch==1.5.1', 'matplotlib==3.1.2',
    'seaborn==0.11.0', 'numpy==1.17.4', 'pandas==1.0.4', 'scipy==1.3.3', 'dill==0.3.2',
    'ipython==7.19.0', 'python_igraph==0.8.3', 'scikit_learn==0.24.1']
)
