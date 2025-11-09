"""
Setup script for the Bioinformatics MSA project.
"""

from setuptools import setup, find_packages

setup(
    name="bioinformatics-msa",
    version="0.1.0",
    description="Optimal Multiple Sequence Alignment with Heuristic Pruning",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "biopython>=1.81",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pytest>=7.4.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)

