"""
End-to-end integration tests for the full philosophical analysis pipeline.

This test suite verifies the complete flow from text input through analysis to visualization output.
"""

import pytest
import tempfile
from pathlib import Path
import json
import pandas as pd
import numpy as np

from philosophical_analysis.core.integrated_analyzer import IntegratedPhilosophicalAnalyzer
from philosophical_analysis.visualization.generator import VisualizationGenerator


@pytest.mark.integration
@pytest.mark.usefixtures("mock_nltk_data")
class TestFullAnalysisPipeline:
    """Test the complete analysis pipeline from text input to visualization."""
    
    def setup_method(self):
        """Initialize components for the pipeline test."""
        self.analyzer = IntegratedPhilosophicalAnalyzer()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
        self.viz_generator = VisualizationGenerator(output_dir=str(self.output_dir))
    
    def teardown_method(self):
        """Clean up temporary directory after test."""
        self.temp_dir.cleanup()
    
    def test_full_pipeline_with_sample_texts(self, sample_philosophical_texts):
        """Test the full pipeline from text input to visualization output."""
        # 1. Set up test data
        texts = {
            'kant': sample_philosophical_texts['kant_style'],
            'nietzsche': sample_philosophical_texts['nietzsche_style'],
            'hume': sample_philosophical_texts['hume_style'],
            'incoherent': sample_philosophical_texts['incoherent']
        }
        labels = {'kant': 1, 'nietzsche': 1, 'hume': 1, 'incoherent': 0}
        
        # 2. Run the analysis pipeline
        # 2.1 Fit the analyzer on training texts
        self.analyzer.fit(texts, labels)
        
        # 2.2 Analyze all texts
        results = self.analyzer.analyze_multiple_texts(texts)
        
        # 3. Verify analysis results structure
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(texts)
        assert 'text_id' in results.columns
        assert 'first_order_coherence' in results.columns
        assert 'target_determiners_freq' in results.columns
        
        # 4. Generate visualization data
        viz_data = self.viz_generator.generate_all_visualizations(
            analysis_results=results,
            texts=texts,
            save_html=True
        )
        
        # 5. Check dashboard data structure
        assert 'dashboard' in viz_data
        dashboard = viz_data['dashboard']
        assert 'philosophers' in dashboard
        assert 'stats' in dashboard
        
        # 6. Check temporal data structure
        assert 'temporal' in viz_data
        temporal = viz_data['temporal']
        for philosopher in texts.keys():
            philosopher_name = philosopher.upper()
            if philosopher_name in temporal:
                assert 'coherence_timeline' in temporal[philosopher_name]
                assert isinstance(temporal[philosopher_name]['coherence_timeline'], list)
        
        # 7. Check network data if available
        if 'network' in viz_data and viz_data['network']:
            assert 'nodes' in viz_data['network']
            assert 'links' in viz_data['network']
            
        # 8. Verify JSON output files were created
        dashboard_json = self.output_dir / "dashboard_data.json"
        temporal_json = self.output_dir / "temporal_data.json"
        
        assert dashboard_json.exists()
        assert temporal_json.exists()
        
        # 9. Verify JSON file content is valid
        with open(dashboard_json, 'r') as f:
            dashboard_content = json.load(f)
            assert 'philosophers' in dashboard_content
            
        with open(temporal_json, 'r') as f:
            temporal_content = json.load(f)
            assert isinstance(temporal_content, dict)
    
    def test_pipeline_with_edge_cases(self, sample_philosophical_texts):
        """Test the pipeline with edge cases like very short texts."""
        # 1. Set up test data with edge cases
        texts = {
            'normal': sample_philosophical_texts['kant_style'],
            'very_short': "This is an extremely short text that may not have enough content for proper analysis.",
            'repetitive': "The same words repeated. The same words repeated. The same words repeated." * 10,
            'numbers_heavy': "The year was 1781 when Kant published his work. In 1748, Hume published his inquiry. By 1889, Nietzsche had completed most of his work."
        }
        
        # 2. Run the analysis pipeline - we don't need labels for this test
        try:
            self.analyzer.fit(texts)
            results = self.analyzer.analyze_multiple_texts(texts)
            
            # 3. Generate visualization with edge cases
            viz_data = self.viz_generator.generate_all_visualizations(
                analysis_results=results,
                texts=texts,
                save_html=True
            )
            
            # 4. Verify results were generated for all texts
            assert len(results) == len(texts)
            
            # 5. Check that visualization handles edge cases
            assert 'dashboard' in viz_data
            for text_id in texts.keys():
                philosopher = self.viz_generator._extract_philosopher_name(text_id)
                assert philosopher.upper() in viz_data['dashboard']['philosophers']
                
        except Exception as e:
            pytest.fail(f"Pipeline failed with edge case data: {e}")
