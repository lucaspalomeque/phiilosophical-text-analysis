"""
Tests for the visualization generator module.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import json

from philosophical_analysis.visualization import VisualizationGenerator


class TestVisualizationGenerator:
    """Test suite for visualization generator."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample analysis results."""
        return pd.DataFrame([
            {
                'text_id': 'kant_critique',
                'sentence_count': 100,
                'first_order_coherence': 0.75,
                'second_order_coherence': 0.85,
                'target_determiners_freq': 0.015,
                'max_phrase_length': 25,
                'avg_sentence_length': 22.5,
            },
            {
                'text_id': 'nietzsche_zarathustra',
                'sentence_count': 150,
                'first_order_coherence': 0.65,
                'second_order_coherence': 0.90,
                'target_determiners_freq': 0.008,
                'max_phrase_length': 18,
                'avg_sentence_length': 18.3,
            }
        ])
    
    @pytest.fixture
    def sample_texts(self):
        """Create sample texts."""
        return {
            'kant_critique': "The categorical imperative is the central philosophical concept.",
            'nietzsche_zarathustra': "Thus spoke Zarathustra about the eternal recurrence."
        }
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_generator_initialization(self, temp_output_dir):
        """Test generator initialization."""
        generator = VisualizationGenerator(str(temp_output_dir))
        
        assert generator.output_dir == temp_output_dir
        assert generator.color_scheme['primary'] == '#00FF00'
    
    def test_dashboard_data_generation(self, sample_results, temp_output_dir):
        """Test dashboard data generation."""
        generator = VisualizationGenerator(str(temp_output_dir))
        dashboard_data = generator.generate_dashboard_data(sample_results)
        
        assert 'philosophers' in dashboard_data
        assert 'stats' in dashboard_data
        assert 'timestamp' in dashboard_data
        
        # Check philosopher data
        assert 'KANT' in dashboard_data['philosophers']
        assert 'NIETZSCHE' in dashboard_data['philosophers']
        
        # Check metrics
        kant_data = dashboard_data['philosophers']['KANT']
        assert kant_data['first_order_coherence'] == 0.75
        assert kant_data['max_phrase_length'] == 25
    
    def test_temporal_data_generation(self, sample_results, temp_output_dir):
        """Test temporal coherence data generation."""
        generator = VisualizationGenerator(str(temp_output_dir))
        temporal_data = generator.generate_temporal_data(sample_results)
        
        assert 'KANT' in temporal_data
        assert 'NIETZSCHE' in temporal_data
        
        # Check temporal metrics
        kant_temporal = temporal_data['KANT']
        assert 'coherence_timeline' in kant_temporal
        assert 'avg_coherence' in kant_temporal
        assert 'volatility' in kant_temporal
        assert len(kant_temporal['coherence_timeline']) > 0
    
    def test_semantic_network_generation(self, sample_texts, sample_results, temp_output_dir):
        """Test semantic network generation."""
        generator = VisualizationGenerator(str(temp_output_dir))
        network_data = generator.generate_semantic_network(sample_texts, sample_results)
        
        assert 'nodes' in network_data
        assert 'links' in network_data
        assert 'metadata' in network_data
        
        # Should have some nodes
        assert len(network_data['nodes']) > 0
        
        # Check node structure
        if network_data['nodes']:
            node = network_data['nodes'][0]
            assert 'id' in node
            assert 'label' in node
            assert 'category' in node
    
    def test_json_data_saving(self, sample_results, temp_output_dir):
        """Test JSON data file generation."""
        generator = VisualizationGenerator(str(temp_output_dir))
        
        # Generate some data
        viz_data = {
            'dashboard': generator.generate_dashboard_data(sample_results),
            'temporal': generator.generate_temporal_data(sample_results)
        }
        
        # Save JSON data
        generator.save_json_data(viz_data)
        
        # Check files were created
        dashboard_json = temp_output_dir / 'dashboard_data.json'
        temporal_json = temp_output_dir / 'temporal_data.json'
        combined_json = temp_output_dir / 'all_visualizations_data.json'
        
        assert dashboard_json.exists()
        assert temporal_json.exists()
        assert combined_json.exists()
        
        # Verify JSON content
        with open(dashboard_json, 'r') as f:
            loaded_data = json.load(f)
            assert 'philosophers' in loaded_data
    
    def test_full_generation_pipeline(self, sample_results, sample_texts, temp_output_dir):
        """Test complete visualization generation pipeline."""
        generator = VisualizationGenerator(str(temp_output_dir))
        
        # Generate all visualizations (without HTML update)
        viz_data = generator.generate_all_visualizations(
            sample_results,
            sample_texts,
            save_html=False
        )
        
        # Check all components generated
        assert 'dashboard' in viz_data
        assert 'temporal' in viz_data
        assert 'network' in viz_data
        
        # Verify JSON files created
        json_files = list(temp_output_dir.glob('*.json'))
        assert len(json_files) > 0
    
    def test_error_handling(self, temp_output_dir):
        """Test error handling with invalid data."""
        generator = VisualizationGenerator(str(temp_output_dir))
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        dashboard_data = generator.generate_dashboard_data(empty_df)
        
        # Should still return valid structure
        assert 'philosophers' in dashboard_data
        assert 'stats' in dashboard_data
    
    @pytest.mark.parametrize("philosopher,expected_color", [
        ("nietzsche_test", "#00FF00"),  # Primary
        ("kant_test", "#FFFFFF"),        # Secondary
        ("hume_test", "#888888"),        # Tertiary
    ])
    def test_color_assignment(self, philosopher, expected_color, temp_output_dir):
        """Test color assignment for different philosophers."""
        generator = VisualizationGenerator(str(temp_output_dir))
        
        results = pd.DataFrame([{
            'text_id': philosopher,
            'first_order_coherence': 0.5,
            'max_phrase_length': 10,
            'avg_sentence_length': 15,
        }])
        
        dashboard_data = generator.generate_dashboard_data(results)
        phil_name = philosopher.split('_')[0].upper()
        
        if phil_name in dashboard_data['philosophers']:
            assert dashboard_data['philosophers'][phil_name]['color'] == expected_color


def test_visualization_module_imports():
    """Test that visualization module imports correctly."""
    try:
        from philosophical_analysis.visualization import VisualizationGenerator
        assert VisualizationGenerator is not None
    except ImportError as e:
        pytest.fail(f"Failed to import visualization module: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])