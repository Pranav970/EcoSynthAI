from setuptools import setup, find_packages

setup(
    name='ecosynthai',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
        'transformers',
        'scrapy',
        'geopandas'
    ],
    author='EcoSynthAI Team',
    description='Generative AI for Synthetic Ecosystem Design'
)