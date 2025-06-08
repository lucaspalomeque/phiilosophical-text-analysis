"""
Visualization module for Philosophical Text Analysis.

This module provides tools for generating interactive visualizations
of philosophical text analysis results.
"""

from .generator import VisualizationGenerator
from .semantic_network import SemanticNetworkGenerator

__all__ = [
    'VisualizationGenerator',
    'SemanticNetworkGenerator'
]

# Version info
__version__ = '1.0.0'

# Module metadata
__author__ = 'Philosophical Text Analysis Project'
__description__ = 'Interactive visualizations for philosophical text analysis'