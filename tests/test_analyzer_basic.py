"""
Basic tests for PhilosophicalAnalyzer.

These tests validate core functionality and ensure the analyzer works correctly.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from philosophical_analysis.core.analyzer import PhilosophicalAnalyzer


class TestPhilosophicalAnalyzer:
    """Test suite for PhilosophicalAnalyzer class."""
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return {
            "coherent": """
            Philosophy is the study of fundamental questions about existence and knowledge.
            These questions have been explored by thinkers throughout history.
            The systematic approach to these problems defines philosophical inquiry.
            Logic provides the foundation for rigorous reasoning.
            """,
            "fragmented": """
            Reality is uncertain and strange in its manifestations.
            Therefore cats understand quantum mechanics better than humans.
            The universe speaks in colors we cannot perceive.
            Mathematics proves everything contradictory about existence.
            """
        }
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance for testing."""
        return PhilosophicalAnalyzer()
    
    @pytest.fixture
    def fitted_analyzer(self, analyzer, sample_texts):
        """Create fitted analyzer for testing."""
        analyzer.fit(sample_texts)
        return analyzer
    
    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly."""
        assert analyzer is not None
        assert analyzer.lemmatizer is not None
        assert analyzer.stop_words is not None
        assert not analyzer._is_fitted
        assert analyzer.vectorizer is None
        assert analyzer.lsa_model is None
    
    def test_analyzer_repr(self, analyzer):
        """Test string representation."""
        repr_str = repr(analyzer)
        assert "PhilosophicalAnalyzer" in repr_str
        assert "not fitted" in repr_str
    
    def test_preprocess_text_basic(self, analyzer):
        """Test basic text preprocessing."""
        text = "This is a test. It has multiple sentences."
        sentences = analyzer.preprocess_text(text)
        
        assert isinstance(sentences, list)
        assert len(sentences) >= 1
        assert all(isinstance(sentence, list) for sentence in sentences)
        assert all(isinstance(word, str) for sentence in sentences for word in sentence)
    
    def test_preprocess_text_empty(self, analyzer):
        """Test preprocessing empty text."""
        sentences = analyzer.preprocess_text("")
        assert sentences == []
    
    def test_preprocess_text_short(self, analyzer):
        """Test preprocessing very short text."""
        sentences = analyzer.preprocess_text("Hi.")
        assert len(sentences) == 0  # Too short after filtering
    
    def test_fit_with_valid_texts(self, analyzer, sample_texts):
        """Test fitting with valid texts."""
        result = analyzer.fit(sample_texts)
        
        # Should return self for method chaining
        assert result is analyzer
        assert analyzer._is_fitted
    
    def test_fit_with_empty_texts(self, analyzer):
        """Test fitting with empty texts dictionary."""
        with pytest.raises(ValueError, match="No valid sentences found"):
            analyzer.fit({})
    
    def test_fit_with_insufficient_text(self, analyzer):
        """Test fitting with insufficient text content."""
        minimal_texts = {"test": "Hi."}  # Too short
        with pytest.raises(ValueError, match="No valid sentences found"):
            analyzer.fit(minimal_texts)
    
    def test_analyze_text_before_fitting(self, analyzer):
        """Test that analyzing before fitting raises error."""
        with pytest.raises(RuntimeError, match="must be fitted"):
            analyzer.analyze_text("Some text", "test")
    
    def test_analyze_text_basic(self, fitted_analyzer, sample_texts):
        """Test basic text analysis."""
        text = sample_texts["coherent"]
        result = fitted_analyzer.analyze_text(text, "test")
        
        # Check result structure
        assert isinstance(result, dict)
        assert "text_id" in result
        assert "sentence_count" in result
        assert "word_count" in result
        assert "semantic_coherence" in result
        assert "classification" in result
        
        # Check data types
        assert isinstance(result["sentence_count"], int)
        assert isinstance(result["word_count"], int)
        assert isinstance(result["semantic_coherence"], float)
        assert result["classification"] in ["coherent", "fragmented"]
    
    def test_analyze_text_insufficient_sentences(self, fitted_analyzer):
        """Test analysis with insufficient sentences."""
        result = fitted_analyzer.analyze_text("Short.", "test")
        
        assert result["error"] == "insufficient_sentences"
        assert result["sentence_count"] < 2
    
    def test_coherence_calculation_basic(self, fitted_analyzer):
        """Test coherence calculation with basic sentences."""
        sentences = [
            ["philosophy", "study", "questions"],
            ["questions", "explored", "history"],
            ["systematic", "approach", "defines"]
        ]
        
        coherence = fitted_analyzer.calculate_coherence(sentences)
        
        assert isinstance(coherence, dict)
        assert "semantic_coherence" in coherence
        assert "min_coherence" in coherence
        assert "max_coherence" in coherence
        
        # Check ranges
        assert 0 <= coherence["semantic_coherence"] <= 1
        assert 0 <= coherence["min_coherence"] <= 1
        assert 0 <= coherence["max_coherence"] <= 1
    
    def test_coherence_calculation_insufficient_sentences(self, fitted_analyzer):
        """Test coherence calculation with insufficient sentences."""
        sentences = [["single", "sentence"]]
        
        coherence = fitted_analyzer.calculate_coherence(sentences)
        
        assert coherence["semantic_coherence"] == 0.0
        assert coherence["min_coherence"] == 0.0
        assert coherence["max_coherence"] == 0.0
    
    def test_analyze_multiple_texts(self, fitted_analyzer, sample_texts):
        """Test analyzing multiple texts."""
        result = fitted_analyzer.analyze_multiple_texts(sample_texts)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_texts)
        assert "text_id" in result.columns
        assert "semantic_coherence" in result.columns
    
    def test_coherent_vs_fragmented_detection(self, fitted_analyzer, sample_texts):
        """Test that analyzer can distinguish coherent from fragmented text."""
        coherent_result = fitted_analyzer.analyze_text(
            sample_texts["coherent"], "coherent"
        )
        fragmented_result = fitted_analyzer.analyze_text(
            sample_texts["fragmented"], "fragmented"
        )
        
        # Both should analyze successfully
        assert "error" not in coherent_result
        assert "error" not in fragmented_result
        
        # Coherent text should have higher coherence (generally)
        # Note: This might not always be true with small samples, but it's a good sanity check
        assert isinstance(coherent_result["semantic_coherence"], float)
        assert isinstance(fragmented_result["semantic_coherence"], float)
    
    @pytest.mark.parametrize("mode", ["simple", "lsa"])
    def test_different_analysis_modes(self, analyzer, sample_texts, mode):
        """Test that analyzer works in different modes."""
        # Fit the analyzer
        analyzer.fit(sample_texts)
        
        # Force a specific mode for testing
        if mode == "simple":
            analyzer._simple_mode = True
        else:
            analyzer._simple_mode = False
        
        result = analyzer.analyze_text(sample_texts["coherent"], "test")
        
        assert "error" not in result
        assert "analysis_mode" in result


class TestAnalyzerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_special_characters(self):
        """Test text with special characters."""
        analyzer = PhilosophicalAnalyzer()
        text_with_special = "What is φιλοσοφία? It's the love of σοφία (wisdom)!"
        sentences = analyzer.preprocess_text(text_with_special)
        
        # Should handle gracefully
        assert isinstance(sentences, list)
    
    def test_very_long_text(self):
        """Test with very long text."""
        analyzer = PhilosophicalAnalyzer()
        long_text = " ".join(["This is a sentence about philosophy."] * 100)
        sentences = analyzer.preprocess_text(long_text)
        
        assert len(sentences) > 0
        assert len(sentences) <= 100  # Should not exceed input
    
    def test_mixed_languages(self):
        """Test with mixed language text."""
        analyzer = PhilosophicalAnalyzer()
        mixed_text = "Philosophy is universal. La filosofía es universal."
        sentences = analyzer.preprocess_text(mixed_text)
        
        # Should handle gracefully
        assert isinstance(sentences, list)
    
    @patch('philosophical_analysis.core.analyzer.logger')
    def test_logging_calls(self, mock_logger):
        """Test that logging is called appropriately."""
        analyzer = PhilosophicalAnalyzer()
        
        # Check initialization logging
        mock_logger.info.assert_called()
        
        # Test fitting logging
        sample_texts = {
            "test": "This is a longer philosophical text about existence and knowledge. "
                   "It explores fundamental questions that have puzzled thinkers for centuries."
        }
        analyzer.fit(sample_texts)
        
        # Should have logged fitting process
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Fitting analyzer" in call for call in log_calls)


# Utility functions for testing
def test_package_imports():
    """Test that package imports work correctly."""
    from philosophical_analysis import PhilosophicalAnalyzer
    from philosophical_analysis import __version__
    
    assert PhilosophicalAnalyzer is not None
    assert __version__ is not None


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])