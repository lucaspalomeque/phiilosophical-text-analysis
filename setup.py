"""
Setup configuration for philosophical-text-analysis package.
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read README.md for long description
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')

# Read version from __init__.py
def get_version():
    """Get version from src/philosophical_analysis/__init__.py"""
    version_file = this_directory / "src" / "philosophical_analysis" / "__init__.py"
    if version_file.exists():
        with open(version_file, 'r') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"').strip("'")
    return "0.1.0"

# Development dependencies
dev_requirements = [
    "pytest>=6.2.0",
    "pytest-cov>=3.0.0",
    "black>=22.0.0",
    "isort>=5.10.0",
    "flake8>=4.0.0",
    "mypy>=0.910",
    "pre-commit>=2.15.0",
]

# Documentation dependencies
docs_requirements = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "myst-parser>=0.18.0",
]

# Visualization dependencies
viz_requirements = [
    "plotly>=5.0.0",
    "wordcloud>=1.8.0",
    "ipywidgets>=7.6.0",
]

# Notebook dependencies
notebook_requirements = [
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
    "nbconvert>=6.0.0",
]

setup(
    name="philosophical-text-analysis",
    version=get_version(),
    author="Your Name",  # TODO: Change this
    author_email="your.email@example.com",  # TODO: Change this
    description="Automated analysis of philosophical texts using psycholinguistic techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/philosophical-text-analysis",  # TODO: Change this
    project_urls={
        "Bug Reports": "https://github.com/yourusername/philosophical-text-analysis/issues",
        "Source": "https://github.com/yourusername/philosophical-text-analysis",
        "Documentation": "https://philosophical-text-analysis.readthedocs.io/",
    },
    
    # Package configuration
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Include non-Python files
    include_package_data=True,
    package_data={
        "philosophical_analysis": [
            "config/*.yaml",
            "data/reference/*.json",
        ],
    },
    
    # Dependencies
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.7",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "requests>=2.26.0",
        "beautifulsoup4>=4.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
        "click>=8.0.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": dev_requirements,
        "docs": docs_requirements,
        "viz": viz_requirements,
        "notebook": notebook_requirements,
        "all": dev_requirements + docs_requirements + viz_requirements + notebook_requirements,
    },
    
    # Command line interface
    entry_points={
        "console_scripts": [
            "philo-analyze=philosophical_analysis.cli:main",
        ],
    },
    
    # Metadata
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.8",
    zip_safe=False,
    
    # Keywords for search
    keywords="philosophy text-analysis nlp psycholinguistics semantic-analysis machine-learning",
    
    # License
    license="MIT",
    platforms=["any"],
)