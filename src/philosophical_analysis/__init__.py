"""
Philosophical Text Analysis Package.

A Python package for analyzing philosophical texts using psycholinguistic techniques
based on the paper "Automated analysis of free speech predicts psychosis onset in high-risk youths".
"""

__version__ = "0.1.0"
__author__ = "Zeche"  
__email__ = "lucas@electricsheeps.co"  
__license__ = "MIT"
__copyright__ = "Copyright 2024, Philosophical Text Analysis Project"

# Main imports
from .core.analyzer import PhilosophicalAnalyzer
from .core.integrated_analyzer import IntegratedPhilosophicalAnalyzer
from .visualization import VisualizationGenerator

# Version info tuple
__version_info__ = tuple(map(int, __version__.split('.')))

# All public exports
__all__ = [
    "PhilosophicalAnalyzer",
    "IntegratedPhilosophicalAnalyzer",
    "VisualizationGenerator",
    "__version__",
    "__author__",
    "__email__",
]

# Package metadata
def get_package_info():
    """Get package information."""
    return {
        "name": "philosophical-text-analysis",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "license": __license__,
        "description": "Automated analysis of philosophical texts using psycholinguistic techniques",
    }