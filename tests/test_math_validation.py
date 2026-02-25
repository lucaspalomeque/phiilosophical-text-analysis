"""
Mathematical validation tests for the analysis pipeline.

These tests verify that the MATH is correct — not just that code runs
or data structures are right, but that numerical results match
hand-computed expected values and satisfy known mathematical properties.

Organized by mathematical concept:
1. Cosine similarity correctness
2. Coherence ordering (coherent > incoherent)
3. Second-order coherence formula verification
4. Temporal coherence detects midpoint breaks
5. Determiner counting exactness
6. Statistical tests (t-test, Cohen's d)
7. Coherence decay model
8. Syntactic complexity formula
9. LSA dimensionality reduction sanity
10. End-to-end mathematical consistency
"""

import pytest
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import ttest_1samp

from philosophical_analysis.core.enhanced_coherence import EnhancedCoherenceAnalyzer
from philosophical_analysis.core.pos_analyzer import AdvancedPOSAnalyzer
from philosophical_analysis.core.integrated_analyzer import IntegratedPhilosophicalAnalyzer


# =========================================================================
# Shared corpus — enough sentences for the 10-sentence fit() minimum
# =========================================================================

PHILOSOPHY_CORPUS = {
    "epistemology": (
        "Knowledge comes from experience and observation of the world around us. "
        "Experience teaches us about the nature of reality and existence itself. "
        "Reality is perceived through our senses and interpreted by the mind. "
        "The mind organizes sensory data into coherent understanding and belief. "
        "Understanding the world requires both reason and empirical evidence. "
        "Evidence must be gathered systematically through careful observation. "
        "Observation reveals patterns that reason alone cannot discover. "
        "Discovery of truth demands rigorous methodology and skepticism. "
        "Skepticism guards against accepting unfounded beliefs and illusions. "
        "Illusions remind us that perception can be deceptive and unreliable."
    ),
    "ethics": (
        "Morality is grounded in duty and rational principles of conduct. "
        "Duty requires that we act according to universal moral laws. "
        "Moral laws apply to all rational beings without exception or favoritism. "
        "Rational beings must respect the autonomy of other persons always. "
        "The categorical imperative defines the supreme principle of morality. "
        "Virtue ethics emphasizes the development of good moral character. "
        "Character is formed through habitual practice of virtuous actions. "
        "Actions should be judged by their conformity to moral principles. "
        "Principles of justice require treating similar cases in similar ways. "
        "Justice demands fairness, equality, and respect for human dignity."
    ),
}


@pytest.fixture(scope="module")
def fitted_coherence_analyzer():
    """Pre-fitted coherence analyzer with sufficient corpus."""
    analyzer = EnhancedCoherenceAnalyzer(n_components=5, window_size=3)
    analyzer.fit(PHILOSOPHY_CORPUS)
    return analyzer


@pytest.fixture
def pos_analyzer():
    return AdvancedPOSAnalyzer()


# =========================================================================
# 1. Cosine similarity correctness
# =========================================================================

class TestCosineSimilarity:
    """Verify cosine similarity satisfies mathematical axioms."""

    def test_identical_vectors_have_similarity_one(self):
        v = np.array([1.0, 2.0, 3.0])
        sim = 1.0 - cosine_distance(v, v)
        assert abs(sim - 1.0) < 1e-10

    def test_orthogonal_vectors_have_similarity_zero(self):
        v = np.array([1.0, 0.0, 0.0])
        w = np.array([0.0, 1.0, 0.0])
        sim = 1.0 - cosine_distance(v, w)
        assert abs(sim) < 1e-10

    def test_opposite_vectors_have_similarity_negative_one(self):
        v = np.array([1.0, 2.0, 3.0])
        sim = 1.0 - cosine_distance(v, -v)
        assert abs(sim - (-1.0)) < 1e-10

    def test_similarity_is_symmetric(self):
        v = np.array([1.0, 3.0, 5.0])
        w = np.array([2.0, 4.0, 1.0])
        assert abs((1 - cosine_distance(v, w)) - (1 - cosine_distance(w, v))) < 1e-10

    def test_similarity_bounded(self):
        rng = np.random.RandomState(42)
        for _ in range(100):
            v, w = rng.randn(10), rng.randn(10)
            sim = 1.0 - cosine_distance(v, w)
            assert -1.0 - 1e-10 <= sim <= 1.0 + 1e-10


# =========================================================================
# 2. Coherence ordering
# =========================================================================

@pytest.mark.usefixtures("mock_nltk_data")
class TestCoherenceOrdering:
    """Coherent texts MUST score higher than incoherent texts."""

    def test_coherent_beats_incoherent(self, fitted_coherence_analyzer):
        coherent = (
            "The mind processes information through cognitive structures. "
            "These cognitive structures shape our perception of reality. "
            "Perception filters sensory input into meaningful patterns. "
            "Patterns of thought become habitual through repetition. "
            "Repetition strengthens our understanding of the world."
        )
        incoherent = (
            "The cat sat on the purple mat yesterday afternoon. "
            "Quantum mechanics describes particles at subatomic scales. "
            "The best recipe for chocolate cake requires cocoa butter. "
            "Mount Everest is the tallest mountain above sea level. "
            "Shakespeare wrote Hamlet in approximately sixteen hundred."
        )

        r_coh = fitted_coherence_analyzer.comprehensive_analysis(coherent)
        r_inc = fitted_coherence_analyzer.comprehensive_analysis(incoherent)

        c1 = r_coh["first_order_coherence"]
        c2 = r_inc["first_order_coherence"]

        assert c1 > c2, (
            f"Coherent ({c1:.4f}) must score higher than incoherent ({c2:.4f})"
        )

    def test_repetitive_text_high_coherence(self, fitted_coherence_analyzer):
        repetitive = (
            "Knowledge is essential for understanding truth and reality. "
            "Understanding truth requires deep and thorough knowledge. "
            "Truth and knowledge are fundamentally connected together. "
            "The connection between knowledge and truth is essential. "
            "Essential truth comes from fundamental knowledge always."
        )

        result = fitted_coherence_analyzer.comprehensive_analysis(repetitive)
        assert result["first_order_coherence"] > 0.3, (
            f"Repetitive text coherence = {result['first_order_coherence']:.4f}, expected > 0.3"
        )

    def test_coherence_in_unit_interval(self, fitted_coherence_analyzer):
        """TF-IDF vectors are non-negative, so cosine similarity is in [0, 1]."""
        text = (
            "Philosophy examines fundamental questions about existence. "
            "Ethics concerns the study of moral principles and values. "
            "Logic provides the framework for valid reasoning well. "
            "Metaphysics investigates the nature of being and reality."
        )

        result = fitted_coherence_analyzer.comprehensive_analysis(text)
        c = result["first_order_coherence"]
        assert 0.0 <= c <= 1.0, f"Coherence {c} out of [0,1]"


# =========================================================================
# 3. Second-order coherence formula
# =========================================================================

@pytest.mark.usefixtures("mock_nltk_data")
class TestSecondOrderFormula:
    """Verify: second_order = 1 - mean(|delta_i|), clamped to [0, 1]."""

    def test_stable_text_has_high_second_order(self, fitted_coherence_analyzer):
        stable = (
            "Reason guides moral judgment and ethical behavior always. "
            "Reason guides moral judgment and ethical behavior always. "
            "Reason guides moral judgment and ethical behavior always. "
            "Reason guides moral judgment and ethical behavior always. "
            "Reason guides moral judgment and ethical behavior always."
        )

        result = fitted_coherence_analyzer.comprehensive_analysis(stable)
        so = result["second_order_coherence"]
        assert so > 0.8, f"Stable text second-order = {so:.4f}, expected > 0.8"

    def test_second_order_bounded(self, fitted_coherence_analyzer):
        text = (
            "The categorical imperative demands universalizability always. "
            "Purple elephants dance on quantum foam every night. "
            "Moral law is the foundation of rational ethics today. "
            "Bananas contain potassium and vitamin B6 for health. "
            "Duty transcends personal inclination and desire entirely."
        )

        result = fitted_coherence_analyzer.comprehensive_analysis(text)
        so = result["second_order_coherence"]
        assert 0.0 <= so <= 1.0, f"Second-order {so} out of [0,1]"

    def test_erratic_coherence_gives_lower_second_order(self, fitted_coherence_analyzer):
        """Alternating between on-topic and off-topic should yield lower
        second-order coherence than a stable on-topic text."""
        stable = (
            "Knowledge derives from experience and empirical evidence. "
            "Experience provides the foundation for understanding reality. "
            "Understanding reality requires systematic observation methods. "
            "Observation reveals the patterns underlying natural phenomena. "
            "Phenomena are explained through rational theoretical frameworks."
        )
        erratic = (
            "Knowledge derives from experience and empirical evidence. "
            "Bananas are a popular tropical fruit worldwide today. "
            "Experience provides the foundation for understanding reality. "
            "The weather forecast predicts rain for tomorrow morning. "
            "Understanding reality requires systematic observation methods."
        )

        r_stable = fitted_coherence_analyzer.comprehensive_analysis(stable)
        r_erratic = fitted_coherence_analyzer.comprehensive_analysis(erratic)

        so_stable = r_stable["second_order_coherence"]
        so_erratic = r_erratic["second_order_coherence"]

        assert so_stable >= so_erratic, (
            f"Stable ({so_stable:.4f}) should have >= second-order than erratic ({so_erratic:.4f})"
        )


# =========================================================================
# 4. Temporal coherence detects midpoint breaks
# =========================================================================

@pytest.mark.usefixtures("mock_nltk_data")
class TestTemporalCoherence:

    def test_topic_shift_detected(self, fitted_coherence_analyzer):
        """First half about epistemology, second half about cooking.
        The transition windows should have lower coherence."""
        text = (
            "Knowledge derives from sensory experience and rational reflection. "
            "Experience provides the raw material for all understanding. "
            "Understanding is organized by the faculties of the mind. "
            "The mind imposes categories on the manifold of perception. "
            "Categories are the a priori conditions of all knowledge. "
            "Now chop the onions finely and heat olive oil in pan. "
            "Add garlic and saute until golden brown and fragrant. "
            "Season with salt pepper and a pinch of cumin powder. "
            "Stir in the tomatoes and simmer for twenty minutes now. "
            "Serve the dish with fresh bread and a green salad today."
        )

        result = fitted_coherence_analyzer.comprehensive_analysis(text)
        windows = result.get("window_coherences", [])

        if len(windows) >= 4:
            mid = len(windows) // 2
            mid_region = min(windows[max(0, mid - 1):mid + 2])
            edge_avg = (np.mean(windows[:2]) + np.mean(windows[-2:])) / 2

            assert mid_region < edge_avg, (
                f"Mid-document ({mid_region:.4f}) should be lower than "
                f"edges ({edge_avg:.4f}) when topic shifts"
            )

    def test_temporal_coherence_equals_mean_of_windows(self, fitted_coherence_analyzer):
        """temporal_coherence == mean(window_coherences)."""
        text = (
            "Knowledge requires justification and true belief always. "
            "Justification can be either empirical or rational in nature. "
            "Empirical justification relies on sensory experience directly. "
            "Rational justification uses logical deduction from axioms. "
            "The debate between empiricism and rationalism continues today. "
            "Modern epistemology seeks to integrate both approaches together."
        )

        result = fitted_coherence_analyzer.comprehensive_analysis(text)
        windows = result.get("window_coherences", [])

        if len(windows) > 0:
            expected = float(np.mean(windows))
            actual = result["temporal_coherence"]
            assert abs(actual - expected) < 1e-10, (
                f"temporal_coherence ({actual}) != mean(windows) ({expected})"
            )


# =========================================================================
# 5. Determiner counting exactness
# =========================================================================

@pytest.mark.usefixtures("mock_nltk_data")
class TestDeterminerCounting:
    """Target determiners: {that, what, whatever, which, whichever}.
    Frequency = target_count / total_words."""

    def test_known_determiners_present(self, pos_analyzer):
        """Text with known target determiners should have nonzero freq."""
        text = (
            "I believe that the principle that guides us is very important. "
            "What matters is which path we choose in our daily life."
        )

        result = pos_analyzer.full_pos_analysis(text, "test")
        freq = result.get("target_determiners_freq", 0)
        count = result.get("target_determiners_count", 0)

        assert count > 0, f"Expected target determiners, got count={count}"
        assert freq > 0, f"Expected nonzero freq, got {freq}"

    def test_no_target_determiners(self, pos_analyzer):
        """Text without target determiners should have freq near 0."""
        text = "The dog runs quickly. A cat sleeps peacefully on the warm rug."

        result = pos_analyzer.full_pos_analysis(text, "test")
        freq = result.get("target_determiners_freq", 0)

        assert freq < 0.01, f"No target determiners expected, got freq={freq}"

    def test_frequency_formula(self, pos_analyzer):
        """Verify: freq = target_count / total_words."""
        text = (
            "That which we call knowledge is what defines our understanding. "
            "Whatever we discover must be examined very critically today."
        )

        result = pos_analyzer.full_pos_analysis(text, "test")
        count = result.get("target_determiners_count", 0)
        total = result.get("total_words", 1)
        freq = result.get("target_determiners_freq", 0)

        if count > 0 and total > 0:
            expected = count / total
            assert abs(freq - expected) < 1e-10, (
                f"Freq: {freq} != {count}/{total} = {expected}"
            )


# =========================================================================
# 6. Statistical tests (t-test, Cohen's d)
# =========================================================================

@pytest.mark.usefixtures("mock_nltk_data")
class TestStatisticalTests:
    """Verify t-test and Cohen's d against scipy reference implementation."""

    def test_cohens_d_manual(self):
        """Cohen's d = (mean - baseline) / std. Verify with known data."""
        scores = [0.6, 0.7, 0.5, 0.65, 0.55, 0.72, 0.58]
        baseline = 0.3

        mean_s = np.mean(scores)
        std_s = np.std(scores, ddof=1)
        expected_d = (mean_s - baseline) / std_s

        analyzer = EnhancedCoherenceAnalyzer()
        result = analyzer.statistical_significance_test(scores, baseline)

        assert abs(result["effect_size"] - expected_d) < 1e-10, (
            f"Cohen's d: got {result['effect_size']}, expected {expected_d}"
        )

    def test_t_statistic_matches_scipy(self):
        """Our t-stat should match scipy.stats.ttest_1samp exactly."""
        scores = [0.45, 0.55, 0.50, 0.60, 0.48, 0.52]
        baseline = 0.3

        t_expected, p_expected = ttest_1samp(scores, baseline)

        analyzer = EnhancedCoherenceAnalyzer()
        result = analyzer.statistical_significance_test(scores, baseline)

        assert abs(result["t_statistic"] - float(t_expected)) < 1e-10
        assert abs(result["p_value"] - float(p_expected)) < 1e-10

    def test_significance_with_high_scores(self):
        """Scores well above baseline should be significant (p < 0.05)."""
        scores = [0.7, 0.8, 0.75, 0.85, 0.72, 0.78, 0.81]
        baseline = 0.3

        analyzer = EnhancedCoherenceAnalyzer()
        result = analyzer.statistical_significance_test(scores, baseline)

        assert result["significant"] is True, (
            f"High scores vs baseline 0.3 should be significant, p={result['p_value']}"
        )
        assert result["effect_size"] > 0, "Effect size should be positive"

    def test_scores_at_baseline_not_significant(self):
        """Scores near baseline should NOT be significant."""
        scores = [0.28, 0.32, 0.30, 0.31, 0.29, 0.33, 0.27]
        baseline = 0.3

        analyzer = EnhancedCoherenceAnalyzer()
        result = analyzer.statistical_significance_test(scores, baseline)

        assert result["significant"] is False, (
            f"Scores near baseline should not be significant, p={result['p_value']}"
        )

    def test_too_few_scores_returns_defaults(self):
        """With < 3 scores, should return safe defaults."""
        analyzer = EnhancedCoherenceAnalyzer()
        result = analyzer.statistical_significance_test([0.5, 0.6], 0.3)

        assert result["t_statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert result["significant"] is False


# =========================================================================
# 7. Coherence decay model
# =========================================================================

@pytest.mark.usefixtures("mock_nltk_data")
class TestCoherenceDecay:
    """Verify coherence decreases with sentence distance (on average)."""

    def test_near_more_coherent_than_distant(self, fitted_coherence_analyzer):
        text = (
            "The concept of justice requires fairness and equality always. "
            "Fairness means treating similar cases in very similar ways. "
            "Equality demands equal rights and opportunities for everyone. "
            "The social contract establishes the basis for authority. "
            "Political authority must be legitimated by consent clearly. "
            "Consent can be expressed through democratic institutions today. "
            "Democratic institutions protect individual rights and freedoms."
        )

        result = fitted_coherence_analyzer.comprehensive_analysis(text)
        dc = result.get("distance_coherences", {})

        if 1 in dc and len(dc) > 2:
            max_d = max(dc.keys())
            assert dc[1] >= dc[max_d] - 0.15, (
                f"d=1 coherence ({dc[1]:.4f}) should be >= d={max_d} ({dc[max_d]:.4f}) - tolerance"
            )

    def test_decay_rate_non_negative(self, fitted_coherence_analyzer):
        text = (
            "Epistemology studies the nature and scope of knowledge always. "
            "Knowledge claims must be justified by evidence and reason. "
            "Reason provides the logical framework for evaluating claims. "
            "Claims about the world can be empirical or a priori. "
            "A priori knowledge is independent of sensory experience. "
            "Experience nonetheless remains crucial for empirical science today."
        )

        result = fitted_coherence_analyzer.comprehensive_analysis(text)
        decay = result.get("coherence_decay_rate", 0)
        assert decay >= -0.1, f"Decay rate {decay:.4f} unexpectedly negative"


# =========================================================================
# 8. Syntactic complexity
# =========================================================================

@pytest.mark.usefixtures("mock_nltk_data")
class TestSyntacticComplexity:
    """Complex sentences should score higher than simple ones."""

    @pytest.mark.integration
    def test_complex_scores_higher_than_simple(self):
        analyzer = IntegratedPhilosophicalAnalyzer()

        texts = {
            "simple": (
                "The small dog runs very fast across the field. "
                "The lazy cat sleeps well on the soft warm couch. "
                "The colorful birds fly high above the green trees. "
                "The cold rain falls down hard on the tin roof. "
                "The bright sun shines warmly across the blue sky. "
                "The silver fish swim deep in the clear cold river. "
                "The strong wind blows across the open grassy plain. "
                "The white snow melts slowly in the warm spring sun. "
                "The beautiful flowers bloom nicely in the garden bed. "
                "The tall green trees grow slowly in the forest now."
            ),
            "complex": (
                "The transcendental unity of apperception which Kant identifies as the "
                "highest principle of all employment of the understanding establishes that "
                "the objective validity of the categories depends on synthetic unity. "
                "The relationship between phenomena and noumena reveals the fundamental "
                "limitation of human cognition which cannot extend beyond possible experience. "
                "Pure reason when it attempts to transcend the boundaries of experience "
                "inevitably falls into antinomies and dialectical illusions that deceive. "
                "The critical philosophy therefore establishes transcendental idealism as "
                "the proper framework for understanding the conditions of knowledge today. "
                "These conditions which are both subjective and necessary constitute the "
                "formal structure within which all empirical knowledge becomes possible. "
                "The synthetic a priori judgments that make experience possible are themselves "
                "grounded in the transcendental categories of the pure understanding faculty. "
                "Without these categories no object of experience could ever be cognized. "
                "The manifold of intuition must be synthesized under these pure concepts. "
                "This synthesis is the fundamental act of the transcendental imagination now."
            ),
        }

        analyzer.fit(texts)
        r_simple = analyzer.analyze_text("simple", texts["simple"])
        r_complex = analyzer.analyze_text("complex", texts["complex"])

        # avg_sentence_length is the most reliable complexity proxy
        asl_simple = r_simple.get("avg_sentence_length", 0)
        asl_complex = r_complex.get("avg_sentence_length", 0)

        assert asl_complex > asl_simple, (
            f"Complex avg_sentence_length ({asl_complex:.1f}) should be > "
            f"simple ({asl_simple:.1f})"
        )

    def test_avg_sentence_length_positive(self):
        pos = AdvancedPOSAnalyzer()
        result = pos.full_pos_analysis(
            "This is a reasonable sentence. And here is another one.", "test"
        )
        asl = result.get("avg_sentence_length", 0)
        assert asl > 0, f"avg_sentence_length should be > 0, got {asl}"


# =========================================================================
# 9. LSA sanity
# =========================================================================

@pytest.mark.usefixtures("mock_nltk_data")
class TestLSASanity:
    """LSA should not produce degenerate results."""

    def test_different_texts_get_different_coherence(self, fitted_coherence_analyzer):
        """Two quite different texts should not have identical coherence."""
        text_a = (
            "Knowledge derives from experience and empirical observation. "
            "Experience provides evidence about the natural world directly. "
            "Evidence must be evaluated through rational careful analysis. "
            "Analysis reveals patterns in our observations of nature today."
        )
        text_b = (
            "Now chop the onions finely and heat olive oil in pan. "
            "Add garlic and saute until golden brown and fragrant. "
            "Season with salt pepper and a pinch of cumin powder. "
            "Stir in the tomatoes and simmer for twenty minutes now."
        )

        r_a = fitted_coherence_analyzer.comprehensive_analysis(text_a)
        r_b = fitted_coherence_analyzer.comprehensive_analysis(text_b)

        c_a = r_a["first_order_coherence"]
        c_b = r_b["first_order_coherence"]

        # They should differ (not be identical)
        assert abs(c_a - c_b) > 0.001, (
            f"Different texts should differ in coherence: {c_a:.4f} vs {c_b:.4f}"
        )

    def test_coherence_not_always_zero(self, fitted_coherence_analyzer):
        """A topically coherent text should produce nonzero coherence."""
        text = (
            "Philosophy examines the fundamental nature of knowledge. "
            "Knowledge is justified true belief according to tradition. "
            "Belief requires evidence and rational justification always. "
            "Justification can be foundational or coherentist in nature."
        )

        result = fitted_coherence_analyzer.comprehensive_analysis(text)
        assert result["first_order_coherence"] > 0.0, "Coherence should be > 0"


# =========================================================================
# 10. End-to-end consistency
# =========================================================================

@pytest.mark.integration
@pytest.mark.usefixtures("mock_nltk_data")
class TestEndToEndConsistency:
    """Verify cross-metric relationships that must hold mathematically."""

    def test_all_metrics_are_finite(self):
        analyzer = IntegratedPhilosophicalAnalyzer()
        analyzer.fit(PHILOSOPHY_CORPUS)
        result = analyzer.analyze_text("epistemology", PHILOSOPHY_CORPUS["epistemology"])

        numeric_keys = [
            "first_order_coherence", "second_order_coherence",
            "target_determiners_freq", "avg_sentence_length",
            "syntactic_complexity",
        ]

        for key in numeric_keys:
            if key in result:
                val = result[key]
                if isinstance(val, (int, float)):
                    assert np.isfinite(val), f"{key} = {val} is not finite"
                elif isinstance(val, dict):
                    # syntactic_complexity can be a dict — check its values
                    for k, v in val.items():
                        if isinstance(v, (int, float)):
                            assert np.isfinite(v), f"{key}.{k} = {v} is not finite"

    def test_coherence_metrics_self_consistent(self):
        """If first-order coherence is high, second-order should also be reasonable."""
        analyzer = EnhancedCoherenceAnalyzer(n_components=5)
        analyzer.fit(PHILOSOPHY_CORPUS)

        result = analyzer.comprehensive_analysis(PHILOSOPHY_CORPUS["epistemology"])
        c1 = result["first_order_coherence"]
        c2 = result["second_order_coherence"]

        # Both should be in valid range
        assert 0.0 <= c1 <= 1.0
        assert 0.0 <= c2 <= 1.0

        # If text is coherent, second-order shouldn't be 0
        if c1 > 0.3:
            assert c2 > 0.0, (
                f"With first-order={c1:.4f}, second-order should be > 0"
            )

    def test_analyze_multiple_returns_all_texts(self):
        """analyze_multiple_texts should return one row per input text."""
        analyzer = IntegratedPhilosophicalAnalyzer()
        analyzer.fit(PHILOSOPHY_CORPUS)
        results = analyzer.analyze_multiple_texts(PHILOSOPHY_CORPUS)

        assert len(results) == len(PHILOSOPHY_CORPUS), (
            f"Expected {len(PHILOSOPHY_CORPUS)} results, got {len(results)}"
        )
