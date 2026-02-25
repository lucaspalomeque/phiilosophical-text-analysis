"""
Tests for the visualization generator components.

This module tests the VisualizationGenerator class and its methods for 
creating dashboard data, temporal visualizations, and semantic networks.
"""

import pytest
import tempfile
import json
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from philosophical_analysis.visualization.generator import VisualizationGenerator
from philosophical_analysis.visualization.semantic_network import SemanticNetworkGenerator


class TestVisualizationGenerator:
    """Test the visualization generator component."""
    
    def setup_method(self):
        """Set up the visualization generator for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.viz_gen = VisualizationGenerator(output_dir=str(self.output_dir))
    
    def teardown_method(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_initialization(self):
        """Test that the visualization generator initializes correctly."""
        assert self.viz_gen.output_dir == self.output_dir
        assert 'primary' in self.viz_gen.color_scheme
        assert 'secondary' in self.viz_gen.color_scheme
    
    def test_dashboard_data_generation(self):
        """Test generating dashboard data from analysis results."""
        # Create mock analysis results
        data = {
            'text_id': ['kant_critique', 'nietzsche_zarathustra', 'hume_treatise'],
            'first_order_coherence': [0.75, 0.68, 0.62],
            'second_order_coherence': [0.85, 0.72, 0.66],
            'target_determiners_freq': [0.012, 0.008, 0.010],
            'max_phrase_length': [22, 18, 15],
            'avg_sentence_length': [25, 18, 20],
            'temporal_coherence': [0.70, 0.65, 0.60],
        }
        results = pd.DataFrame(data)
        
        # Generate dashboard data
        dashboard = self.viz_gen.generate_dashboard_data(results)
        
        # Verify structure
        assert 'philosophers' in dashboard
        assert 'stats' in dashboard
        assert 'timestamp' in dashboard
        
        # Verify philosopher data
        assert 'KANT' in dashboard['philosophers']
        assert 'NIETZSCHE' in dashboard['philosophers']
        assert 'HUME' in dashboard['philosophers']
        
        # Check specific metrics
        assert dashboard['philosophers']['KANT']['first_order_coherence'] == 0.75
        assert dashboard['philosophers']['NIETZSCHE']['determiner_frequency'] == 0.008
    
    def test_temporal_data_generation(self):
        """Test generating temporal coherence visualization data."""
        # Create mock analysis results with coherence timeline
        data = {
            'text_id': ['kant_critique', 'nietzsche_zarathustra'],
            'coherence_timeline': [[0.5, 0.6, 0.7], [0.4, 0.5, 0.6]],
        }
        results = pd.DataFrame(data)
        
        # Mock the _generate_coherence_timeline method to return fixed data
        with patch.object(
            VisualizationGenerator, 
            '_generate_coherence_timeline', 
            return_value=[0.5, 0.6, 0.7]
        ):
            temporal = self.viz_gen.generate_temporal_data(results)
        
        # Verify structure
        assert 'KANT' in temporal
        assert 'NIETZSCHE' in temporal
        
        # Check coherence timeline
        assert 'coherence_timeline' in temporal['KANT']
        assert isinstance(temporal['KANT']['coherence_timeline'], list)
        assert len(temporal['KANT']['coherence_timeline']) > 0
    
    def test_saving_json_files(self):
        """Test saving visualization data to JSON files."""
        # Create mock visualization data
        viz_data = {
            'dashboard': {
                'philosophers': {
                    'KANT': {'coherence': 0.75},
                    'NIETZSCHE': {'coherence': 0.68}
                },
                'stats': {'avg_coherence': 0.715}
            },
            'temporal': {
                'KANT': {'coherence_timeline': [0.5, 0.6, 0.7]},
                'NIETZSCHE': {'coherence_timeline': [0.4, 0.5, 0.6]}
            }
        }
        
        # Call the save method
        self.viz_gen.save_json_data(viz_data)
        
        # Check that files were created
        dashboard_file = self.output_dir / "dashboard_data.json"
        temporal_file = self.output_dir / "temporal_data.json"
        
        assert dashboard_file.exists()
        assert temporal_file.exists()
        
        # Verify file contents
        with open(dashboard_file, 'r') as f:
            dashboard_content = json.load(f)
            assert 'philosophers' in dashboard_content
            assert 'KANT' in dashboard_content['philosophers']
        
        with open(temporal_file, 'r') as f:
            temporal_content = json.load(f)
            assert 'KANT' in temporal_content
            assert 'coherence_timeline' in temporal_content['KANT']
    
    def test_error_handling(self):
        """Test graceful error handling in the visualization generator."""
        # Create a minimal DataFrame that will cause specific errors
        data = {
            'text_id': ['problematic_text'],
            # Missing important columns will cause errors in specific methods
        }
        results = pd.DataFrame(data)
        
        # Test that generate_all_visualizations handles errors gracefully
        viz_data = self.viz_gen.generate_all_visualizations(
            analysis_results=results,
            texts=None
        )
        
        # Even with errors, it should return some data structure
        assert 'dashboard' in viz_data
        assert 'temporal' in viz_data
        
        # Placeholder data should be generated when errors occur
        assert viz_data['dashboard'] is not None


class TestSemanticNetworkGenerator:
    """Test the semantic network generator component."""
    
    def setup_method(self):
        """Set up the semantic network generator for testing."""
        self.network_gen = SemanticNetworkGenerator()
    
    def test_network_generation(self):
        """Test generating a semantic network from philosophical texts."""
        # Create sample texts
        texts = {
            'kant_test': "The categorical imperative is a central philosophical concept in Kant's moral philosophy.",
            'nietzsche_test': "Nietzsche discusses the will to power and eternal recurrence as fundamental concepts."
        }
        
        # Generate network
        network = self.network_gen.generate_network(texts)
        
        # Verify structure
        assert 'nodes' in network
        assert 'links' in network
        assert 'metadata' in network
        
        # Check nodes
        assert len(network['nodes']) > 0
        assert 'id' in network['nodes'][0]
        assert 'label' in network['nodes'][0]
        
        # Check links
        if len(network['links']) > 0:
            assert 'source' in network['links'][0]
            assert 'target' in network['links'][0]
            assert 'strength' in network['links'][0]
