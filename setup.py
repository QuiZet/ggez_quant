"""Setup script for ggez_quant"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="ggez_quant",
    version="0.1.0",
    author="Simon",
    description="A lightweight library for quantizing LLMs to GGUF format",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ggez_quant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "huggingface_hub>=0.19.0",
        "transformers>=4.35.0",
        "numpy>=1.23.0,<2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ggez_quant=core.cli:main",
        ],
    },
    package_dir={"core": "core"},
)
