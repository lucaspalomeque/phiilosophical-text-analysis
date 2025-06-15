import pytest
import numpy as np
from philosophical_analysis.core.integrated_analyzer import IntegratedPhilosophicalAnalyzer

@pytest.mark.usefixtures("mock_nltk_data")
class TestIntegratedPhilosophicalAnalyzer:
    """Integration tests for the IntegratedPhilosophicalAnalyzer."""

    def setup_method(self):
        """Initialize the integrated analyzer for each test."""
        self.analyzer = IntegratedPhilosophicalAnalyzer()

    def test_initialization_and_component_loading(self):
        """Test that the main analyzer and its components initialize correctly."""
        assert self.analyzer.pos_analyzer is not None
        assert self.analyzer.coherence_analyzer is not None
        assert self.analyzer.classifier is not None

    def test_fit_pipeline(self, sample_philosophical_texts):
        """Test that the full fitting pipeline runs without errors."""
        texts = {
            'kant': sample_philosophical_texts['kant_style'],
            'incoherent': sample_philosophical_texts['incoherent']
        }
        labels = {'kant': 1, 'incoherent': 0}
        self.analyzer.fit(texts, labels)
        assert self.analyzer.coherence_analyzer.is_fitted
        assert self.analyzer.classifier.is_fitted

    def test_analyze_single_text_end_to_end(self, sample_philosophical_texts):
        """Test the full analysis of a single text from end to end."""
        texts_for_fit = {
            'kant': sample_philosophical_texts['kant_style'],
            'incoherent': sample_philosophical_texts['incoherent']
        }
        labels = {'kant': 1, 'incoherent': 0}
        self.analyzer.fit(texts_for_fit, labels)

        text_to_analyze = sample_philosophical_texts['nietzsche_style']
        result = self.analyzer.analyze_text(text_to_analyze, 'nietzsche_test')

        assert 'text_id' in result
        # Check for keys from all components
        assert 'target_determiners_freq' in result  # From POS analyzer
        assert 'first_order_coherence' in result    # From Coherence analyzer
        assert 'predicted_label' in result          # From Classifier
        assert 'classification_confidence' in result

    def test_cross_validation_integrated(self, sample_philosophical_texts):
        """Test that the integrated cross-validation runs."""
        texts = {
            'kant': sample_philosophical_texts['kant_style'],
            'hume': sample_philosophical_texts['hume_style'],
            'incoherent': sample_philosophical_texts['incoherent']
        }
        labels = {'kant': 1, 'hume': 1, 'incoherent': 0}
        
        # This test mainly ensures the process completes without crashing
        results = self.analyzer.cross_validate(texts, labels)
        
        assert 'accuracy' in results
        assert 'f1_score' in results
        assert results['accuracy'] >= 0.0  # Just check that it exists and is a float
