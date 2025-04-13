from setuptools import setup, find_packages

setup(
    name="optim",
    version="0.1.0",
    author="David Ulloa",
    author_email="dulloa6310@gmail.com",
    description="A simple package for optimization algorithms",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/davidulloa6310/optimizers",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
        "optax",
        "jax",
        "flax",
        "tensorflow-datasets",
        "tensorflow",
        "tqdm",
        "scikit-learn",
    ],
)
