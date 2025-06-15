import pytest
import numpy as np
from philosophical_analysis.core.enhanced_coherence import EnhancedCoherenceAnalyzer

@pytest.mark.usefixtures("mock_nltk_data")
class TestEnhancedCoherenceAnalyzer:
    """Unit tests for the EnhancedCoherenceAnalyzer."""

    def setup_method(self):
        """Initialize the analyzer for each test."""
        self.analyzer = EnhancedCoherenceAnalyzer(n_components=2, window_size=2)

    def test_initialization(self):
        """Test correct initialization of the analyzer."""
        assert self.analyzer.n_components == 2
        assert self.analyzer.window_size == 2
        assert not self.analyzer.is_fitted

    def test_fit(self, sample_philosophical_texts):
        """Test that the fit method runs and marks the analyzer as fitted."""
        # Use all texts to ensure enough sentences for LSA
        self.analyzer.fit(sample_philosophical_texts)
        assert self.analyzer.is_fitted
        assert self.analyzer.vectorizer is not None
        assert self.analyzer.lsa_model is not None

    def test_comprehensive_analysis_after_fitting(self, sample_philosophical_texts):
        """Test the full analysis pipeline on a text after fitting."""
        texts_to_fit = {"kant": sample_philosophical_texts["kant_style"], "hume": sample_philosophical_texts["hume_style"]}
        self.analyzer.fit(texts_to_fit)
        
        analysis_text = sample_philosophical_texts["nietzsche_style"]
        result = self.analyzer.comprehensive_analysis(analysis_text, "nietzsche_test")

        assert result['text_id'] == "nietzsche_test"
        assert 'first_order_coherence' in result
        assert 'second_order_coherence' in result
        assert 'temporal_coherence' in result
        assert 'p_value' in result
        assert result['sentence_count'] > 0
        assert isinstance(result['first_order_coherence'], (float, np.floating))

    def test_analysis_on_insufficient_sentences(self):
        """Test that fitting with too few sentences raises a ValueError."""
        text = "This is a single sentence. This is another one."
        with pytest.raises(ValueError, match="Insufficient sentences for LSA fitting"):
            self.analyzer.fit({"short": text})

    def test_analysis_before_fitting(self):
        """Test that attempting analysis before fitting raises an exception."""
        with pytest.raises(ValueError, match="Analyzer must be fitted before analysis"):
            self.analyzer.comprehensive_analysis("some text", "unfitted_test")
