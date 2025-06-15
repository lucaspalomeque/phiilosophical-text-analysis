import pytest
from philosophical_analysis.core.pos_analyzer import AdvancedPOSAnalyzer

@pytest.mark.usefixtures("mock_nltk_data")
class TestAdvancedPOSAnalyzer:
    """Unit tests for the AdvancedPOSAnalyzer class."""

    def setup_method(self):
        """Initialize the analyzer before each test."""
        self.analyzer = AdvancedPOSAnalyzer()

    def test_initialization(self):
        """Test that the analyzer initializes with the correct target determiners."""
        expected_determiners = {'that', 'what', 'whatever', 'which', 'whichever'}
        assert self.analyzer.target_determiners == expected_determiners

    def test_full_pos_analysis_simple_sentence(self):
        """Test analysis on a simple sentence with target determiners."""
        text = "This is a test. That is what we want."
        result = self.analyzer.full_pos_analysis(text, "simple_test")

        assert result['text_id'] == "simple_test"
        assert result['sentence_count'] == 2
        assert result['total_words'] == 9
        assert result['target_determiners_count'] == 2  # 'that', 'what'
        assert pytest.approx(result['target_determiners_freq']) == 2 / 9
        assert result['max_phrase_length'] > 0

    def test_full_pos_analysis_no_target_determiners(self):
        """Test analysis on a text without any target determiners."""
        text = "This is a sentence without the specific words."
        result = self.analyzer.full_pos_analysis(text, "no_determiners_test")

        assert result['target_determiners_count'] == 0
        assert result['target_determiners_freq'] == 0.0

    def test_full_pos_analysis_empty_text(self):
        """Test that the analyzer handles empty strings gracefully."""
        text = ""
        result = self.analyzer.full_pos_analysis(text, "empty_test")

        assert result['sentence_count'] == 0
        assert result['total_words'] == 0
        assert result['target_determiners_count'] == 0
        assert result['target_determiners_freq'] == 0.0
        assert result['max_phrase_length'] == 0

    def test_full_pos_analysis_with_fixture(self, sample_philosophical_texts):
        """Test analysis using a fixture with philosophical text."""
        text = sample_philosophical_texts["kant_style"]
        result = self.analyzer.full_pos_analysis(text, "kant_fixture_test")

        assert result['text_id'] == "kant_fixture_test"
        assert result['sentence_count'] == 8
        assert result['total_words'] > 20  # Check for substantial text processing
        assert 'target_determiners_freq' in result
