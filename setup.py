from setuptools import setup, find_packages

setup(
    name="gnnatpg",
    version="0.1.0",
    description="GNN & RL-Based ATPG for Open-Source RISC-V Processor Cores",
    author="Somesh TG, Shreya Ranjitha, Mridula B",
    author_email="",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.2.0",
        "tensorboard>=2.13.0",
    ],
    entry_points={
        "console_scripts": [
            "gnnatpg=gnnatpg.scripts.main_pipeline:main",
        ],
    },
)
