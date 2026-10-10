from setuptools import setup, find_packages

setup(
    name="complex-moe",
    version="0.1.0",
    author="Yang Peng",
    description="Complex-valued Mixture-of-Experts Language Model for Chinese",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "sentencepiece>=0.1.99",
    ],
)
