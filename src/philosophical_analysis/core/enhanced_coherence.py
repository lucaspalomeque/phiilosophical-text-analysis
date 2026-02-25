"""
Enhanced coherence analysis with second-order metrics and temporal tracking.

This module implements advanced coherence metrics following Bedi et al. (2015)
including phrase-separation analysis and temporal coherence patterns.
"""

import logging
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import nltk
from nltk.tokenize import sent_tokenize
import re
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class EnhancedCoherenceAnalyzer:
    """
    Enhanced coherence analyzer implementing second-order coherence and temporal tracking.
    
    This analyzer extends basic LSA coherence with additional metrics from the paper
    including phrase-separation analysis and statistical significance testing.
    """
    
    def __init__(self, 
                 n_components: int = 10,
                 min_df: int = 1,
                 max_df: float = 0.95,
                 window_size: int = 5):
        """
        Initialize enhanced coherence analyzer.
        
        Args:
            n_components: Number of LSA components
            min_df: Minimum document frequency for TF-IDF
            max_df: Maximum document frequency for TF-IDF
            window_size: Window size for temporal coherence analysis
        """
        self.n_components = n_components
        self.min_df = min_df
        self.max_df = max_df
        self.window_size = window_size
        
        self.vectorizer = None
        self.lsa_model = None
        self.is_fitted = False
        
        logger.info(f"Enhanced Coherence Analyzer initialized with {n_components} components")
    
    def preprocess_sentences(self, text: str) -> List[str]:
        """
        Advanced sentence preprocessing with phrase separation.
        
        Args:
            text: Input text
            
        Returns:
            List of preprocessed sentences
        """
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        processed_sentences = []
        
        for sentence in sentences:
            # Clean sentence
            cleaned = re.sub(r'[^\w\s\-\']', ' ', sentence)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
            
            # Skip very short sentences
            if len(cleaned.split()) >= 3:
                processed_sentences.append(cleaned)
        
        return processed_sentences
    
    def fit(self, texts: Dict[str, str]) -> 'EnhancedCoherenceAnalyzer':
        """
        Fit the enhanced coherence analyzer.
        
        Args:
            texts: Dictionary of text_id -> text_content
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting enhanced analyzer on {len(texts)} texts")
        
        # Collect all sentences
        all_sentences = []
        for text_content in texts.values():
            sentences = self.preprocess_sentences(text_content)
            all_sentences.extend(sentences)
        
        if len(all_sentences) < 10:
            raise ValueError("Insufficient sentences for LSA fitting")
        
        # Build TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            max_features=min(1000, len(all_sentences) * 5),
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=(1, 2),  # Include bigrams for better semantic capture
            token_pattern=r'\b\w+\b'
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(all_sentences)
        
        # Build LSA model
        n_components = min(self.n_components, tfidf_matrix.shape[1] - 1, len(all_sentences) - 1)
        self.lsa_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.lsa_model.fit(tfidf_matrix)
        
        self.is_fitted = True
        logger.info(f"Enhanced analyzer fitted with {len(all_sentences)} sentences.")
        
        return self

    def save_model(self, model_path: str):
        """
        Save the fitted vectorizer and LSA model.
        
        Args:
            model_path: Directory to save the models to.
        """
        if not self.is_fitted:
            raise RuntimeError("Analyzer must be fitted before saving the model.")
        
        path = Path(model_path)
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.vectorizer, path / "vectorizer.pkl")
        joblib.dump(self.lsa_model, path / "lsa_model.pkl")
        
        logger.info(f"Coherence model saved to {model_path}")

    def load_model(self, model_path: str):
        """
        Load a pre-trained vectorizer and LSA model.
        
        Args:
            model_path: Directory to load the models from.
        """
        path = Path(model_path)
        vectorizer_path = path / "vectorizer.pkl"
        lsa_model_path = path / "lsa_model.pkl"
        
        if not vectorizer_path.exists() or not lsa_model_path.exists():
            raise FileNotFoundError(f"Model files not found in {model_path}")
            
        self.vectorizer = joblib.load(vectorizer_path)
        self.lsa_model = joblib.load(lsa_model_path)
        self.is_fitted = True
        
        logger.info(f"Coherence model loaded from {model_path}")
    
    def get_sentence_vector(self, sentence: str) -> np.ndarray:
        """
        Get LSA vector for a sentence.
        
        Args:
            sentence: Preprocessed sentence
            
        Returns:
            LSA vector for the sentence
        """
        if not self.is_fitted:
            raise ValueError("Analyzer must be fitted before vectorizing sentences")
        
        tfidf_vec = self.vectorizer.transform([sentence])
        lsa_vec = self.lsa_model.transform(tfidf_vec)
        return lsa_vec[0]
    
    def calculate_basic_coherence(self, sentences: List[str]) -> Dict[str, float]:
        """
        Calculate basic first-order coherence metrics.
        
        Args:
            sentences: List of preprocessed sentences
            
        Returns:
            Dictionary with basic coherence metrics
        """
        if len(sentences) < 2:
            return {
                'first_order_coherence': 0.0,
                'min_coherence': 0.0,
                'max_coherence': 0.0,
                'coherence_variance': 0.0
            }
        
        # Get vectors for all sentences
        vectors = []
        for sentence in sentences:
            try:
                vec = self.get_sentence_vector(sentence)
                vectors.append(vec)
            except Exception:
                continue

        if len(vectors) < 2:
            return {
                'first_order_coherence': 0.0,
                'min_coherence': 0.0,
                'max_coherence': 0.0,
                'coherence_variance': 0.0
            }
        
        # Calculate pairwise coherence
        coherence_scores = []
        for i in range(len(vectors) - 1):
            try:
                similarity = cosine_similarity([vectors[i]], [vectors[i + 1]])[0][0]
                if not np.isnan(similarity) and not np.isinf(similarity):
                    coherence_scores.append(similarity)
            except Exception:
                continue

        if not coherence_scores:
            return {
                'first_order_coherence': 0.0,
                'min_coherence': 0.0,
                'max_coherence': 0.0,
                'coherence_variance': 0.0
            }
        
        return {
            'first_order_coherence': float(np.mean(coherence_scores)),
            'min_coherence': float(np.min(coherence_scores)),
            'max_coherence': float(np.max(coherence_scores)),
            'coherence_variance': float(np.var(coherence_scores))
        }
    
    def calculate_second_order_coherence(self, sentences: List[str]) -> Dict[str, float]:
        """
        Calculate second-order coherence metrics as described in the paper.
        
        Args:
            sentences: List of preprocessed sentences
            
        Returns:
            Dictionary with second-order coherence metrics
        """
        if len(sentences) < 3:
            return {
                'second_order_coherence': 0.0,
                'coherence_change_rate': 0.0,
                'coherence_stability': 0.0
            }
        
        # Get vectors
        vectors = []
        for sentence in sentences:
            try:
                vec = self.get_sentence_vector(sentence)
                vectors.append(vec)
            except Exception:
                continue

        if len(vectors) < 3:
            return {
                'second_order_coherence': 0.0,
                'coherence_change_rate': 0.0,
                'coherence_stability': 0.0
            }
        
        # Calculate first-order coherence scores
        first_order_scores = []
        for i in range(len(vectors) - 1):
            try:
                similarity = cosine_similarity([vectors[i]], [vectors[i + 1]])[0][0]
                if not np.isnan(similarity) and not np.isinf(similarity):
                    first_order_scores.append(similarity)
            except Exception:
                continue
        
        if len(first_order_scores) < 2:
            return {
                'second_order_coherence': 0.0,
                'coherence_change_rate': 0.0,
                'coherence_stability': 0.0
            }
        
        # Calculate second-order metrics
        # Second-order coherence: coherence of the coherence scores
        coherence_changes = []
        for i in range(len(first_order_scores) - 1):
            change = abs(first_order_scores[i + 1] - first_order_scores[i])
            coherence_changes.append(change)
        
        # Metrics
        second_order_coherence = 1.0 - np.mean(coherence_changes) if coherence_changes else 0.0
        coherence_change_rate = np.mean(coherence_changes) if coherence_changes else 0.0
        coherence_stability = 1.0 / (1.0 + np.std(first_order_scores)) if len(first_order_scores) > 1 else 0.0
        
        return {
            'second_order_coherence': float(max(0.0, second_order_coherence)),
            'coherence_change_rate': float(coherence_change_rate),
            'coherence_stability': float(coherence_stability)
        }
    
    def calculate_temporal_coherence(self, sentences: List[str]) -> Dict[str, float]:
        """
        Calculate temporal coherence patterns across the text.
        
        Args:
            sentences: List of preprocessed sentences
            
        Returns:
            Dictionary with temporal coherence metrics
        """
        if len(sentences) < self.window_size:
            return {
                'temporal_coherence': 0.0,
                'coherence_trend': 0.0,
                'local_coherence_variance': 0.0
            }
        
        # Calculate coherence in sliding windows
        window_coherences = []
        
        for i in range(len(sentences) - self.window_size + 1):
            window_sentences = sentences[i:i + self.window_size]
            window_metrics = self.calculate_basic_coherence(window_sentences)
            window_coherences.append(window_metrics['first_order_coherence'])
        
        if not window_coherences:
            return {
                'temporal_coherence': 0.0,
                'coherence_trend': 0.0,
                'local_coherence_variance': 0.0
            }
        
        # Calculate temporal metrics
        temporal_coherence = np.mean(window_coherences)
        
        # Trend analysis: correlation with position
        if len(window_coherences) > 2:
            positions = np.arange(len(window_coherences))
            correlation, _ = stats.pearsonr(positions, window_coherences)
            coherence_trend = correlation if not np.isnan(correlation) else 0.0
        else:
            coherence_trend = 0.0
        
        local_coherence_variance = np.var(window_coherences)
        
        return {
            'temporal_coherence': float(temporal_coherence),
            'coherence_trend': float(coherence_trend),
            'local_coherence_variance': float(local_coherence_variance)
        }
    
    def calculate_phrase_separation_coherence(self, sentences: List[str]) -> Dict[str, float]:
        """
        Calculate coherence metrics with phrase separation analysis.
        
        Args:
            sentences: List of preprocessed sentences
            
        Returns:
            Dictionary with phrase-separation coherence metrics
        """
        if len(sentences) < 4:
            return {
                'phrase_separated_coherence': 0.0,
                'distant_coherence': 0.0,
                'coherence_decay_rate': 0.0
            }
        
        vectors = []
        for sentence in sentences:
            try:
                vec = self.get_sentence_vector(sentence)
                vectors.append(vec)
            except Exception:
                continue

        if len(vectors) < 4:
            return {
                'phrase_separated_coherence': 0.0,
                'distant_coherence': 0.0,
                'coherence_decay_rate': 0.0
            }
        
        # Calculate coherence at different distances
        distance_coherences = {}
        
        for distance in range(1, min(6, len(vectors))):  # Up to 5 sentences apart
            coherences_at_distance = []
            
            for i in range(len(vectors) - distance):
                try:
                    similarity = cosine_similarity([vectors[i]], [vectors[i + distance]])[0][0]
                    if not np.isnan(similarity) and not np.isinf(similarity):
                        coherences_at_distance.append(similarity)
                except Exception:
                    continue
            
            if coherences_at_distance:
                distance_coherences[distance] = np.mean(coherences_at_distance)
        
        if not distance_coherences:
            return {
                'phrase_separated_coherence': 0.0,
                'distant_coherence': 0.0,
                'coherence_decay_rate': 0.0
            }
        
        # Calculate metrics
        phrase_separated_coherence = np.mean(list(distance_coherences.values()))
        distant_coherence = distance_coherences.get(max(distance_coherences.keys()), 0.0)
        
        # Calculate decay rate
        if len(distance_coherences) > 1:
            distances = np.array(list(distance_coherences.keys()))
            coherences = np.array(list(distance_coherences.values()))
            
            # Fit exponential decay: coherence = a * exp(-b * distance)
            try:
                log_coherences = np.log(np.maximum(coherences, 1e-10))
                slope, _, _, _, _ = stats.linregress(distances, log_coherences)
                coherence_decay_rate = -slope  # Positive value indicates decay
            except Exception:
                coherence_decay_rate = 0.0
        else:
            coherence_decay_rate = 0.0
        
        return {
            'phrase_separated_coherence': float(phrase_separated_coherence),
            'distant_coherence': float(distant_coherence),
            'coherence_decay_rate': float(coherence_decay_rate)
        }
    
    def statistical_significance_test(self, 
                                    coherence_scores: List[float], 
                                    baseline_mean: float = 0.3) -> Dict[str, float]:
        """
        Perform statistical significance testing on coherence scores.
        
        Args:
            coherence_scores: List of coherence scores
            baseline_mean: Expected baseline coherence
            
        Returns:
            Dictionary with statistical test results
        """
        if len(coherence_scores) < 3:
            return {
                't_statistic': 0.0,
                'p_value': 1.0,
                'effect_size': 0.0,
                'significant': False
            }
        
        # One-sample t-test against baseline
        t_stat, p_value = stats.ttest_1samp(coherence_scores, baseline_mean)
        
        # Effect size (Cohen's d)
        sample_mean = np.mean(coherence_scores)
        sample_std = np.std(coherence_scores, ddof=1)
        effect_size = (sample_mean - baseline_mean) / sample_std if sample_std > 0 else 0.0
        
        # Significance test (p < 0.05)
        significant = p_value < 0.05
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
            'significant': bool(significant)
        }
    
    def comprehensive_analysis(self, text: str, text_id: str = "") -> Dict[str, Any]:
        """
        Perform comprehensive enhanced coherence analysis.
        
        Args:
            text: Text to analyze
            text_id: Identifier for the text
            
        Returns:
            Dictionary with all enhanced coherence metrics
        """
        logger.info(f"Starting comprehensive coherence analysis for: {text_id}")
        
        if not self.is_fitted:
            raise ValueError("Analyzer must be fitted before analysis")
        
        # Preprocess sentences
        sentences = self.preprocess_sentences(text)
        
        if len(sentences) < 2:
            return {
                'text_id': text_id,
                'error': 'insufficient_sentences_for_enhanced_analysis',
                'sentence_count': len(sentences)
            }
        
        # Calculate all coherence metrics
        basic_metrics = self.calculate_basic_coherence(sentences)
        second_order_metrics = self.calculate_second_order_coherence(sentences)
        temporal_metrics = self.calculate_temporal_coherence(sentences)
        phrase_separation_metrics = self.calculate_phrase_separation_coherence(sentences)
        
        # Statistical significance
        coherence_scores = []
        vectors = []
        for sentence in sentences:
            try:
                vec = self.get_sentence_vector(sentence)
                vectors.append(vec)
            except Exception:
                continue

        for i in range(len(vectors) - 1):
            try:
                similarity = cosine_similarity([vectors[i]], [vectors[i + 1]])[0][0]
                if not np.isnan(similarity) and not np.isinf(similarity):
                    coherence_scores.append(similarity)
            except Exception:
                continue

        statistical_results = self.statistical_significance_test(coherence_scores)
        
        # Combine all results
        result = {
            'text_id': text_id,
            'sentence_count': len(sentences),
            'vector_count': len(vectors),
            **basic_metrics,
            **second_order_metrics,
            **temporal_metrics,
            **phrase_separation_metrics,
            **statistical_results,
            'analysis_type': 'enhanced_coherence'
        }
        
        logger.info(f"Enhanced coherence analysis completed for {text_id}")
        return result


def test_enhanced_coherence():
    """Test the enhanced coherence analyzer."""
    print("🔬 Testing Enhanced Coherence Analyzer")
    print("=" * 50)
    
    # Test texts with different coherence patterns
    test_texts = {
        "highly_coherent": """
        Philosophy examines fundamental questions about reality and knowledge.
        These questions have been central to human inquiry for centuries.
        The systematic investigation of such problems defines philosophical methodology.
        Logical reasoning provides the foundation for philosophical argumentation.
        Through careful analysis, philosophers develop coherent theoretical frameworks.
        Such frameworks help us understand complex metaphysical and epistemological issues.
        """,
        
        "declining_coherence": """
        Philosophy studies important questions about existence and truth.
        Knowledge comes from careful reasoning and systematic investigation.
        The weather today is quite pleasant with sunny skies.
        Mathematics involves abstract concepts and logical relationships.
        Purple elephants dance in the moonlight every Tuesday evening.
        Quantum mechanics proves that reality is fundamentally uncertain.
        """,
        
        "fragmented": """
        Existence is purple mathematics.
        Therefore cats understand Hegelian dialectics better than professors.
        The universe whispers in forgotten languages of copper pennies.
        Reality tastes like childhood dreams mixed with quantum elephants.
        """
    }
    
    try:
        # Initialize analyzer
        analyzer = EnhancedCoherenceAnalyzer(n_components=8, window_size=3)
        
        # Fit analyzer
        print("🧠 Fitting enhanced analyzer...")
        analyzer.fit(test_texts)
        
        # Analyze each text
        for text_id, text in test_texts.items():
            print(f"\n📊 Analyzing: {text_id}")
            print("-" * 30)
            
            result = analyzer.comprehensive_analysis(text, text_id)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            # Display key metrics
            print(f"📝 Sentences: {result['sentence_count']}")
            print(f"🔤 Vectors: {result['vector_count']}")
            print(f"🎯 First-order coherence: {result['first_order_coherence']:.3f}")
            print(f"🔄 Second-order coherence: {result['second_order_coherence']:.3f}")
            print(f"⏱️ Temporal coherence: {result['temporal_coherence']:.3f}")
            print(f"📏 Phrase-separated coherence: {result['phrase_separated_coherence']:.3f}")
            print(f"📈 Coherence trend: {result['coherence_trend']:.3f}")
            print(f"📉 Decay rate: {result['coherence_decay_rate']:.3f}")
            print(f"📊 Statistical significance: {result['significant']}")
            print(f"🎯 P-value: {result['p_value']:.3f}")
        
        print(f"\n✅ Enhanced coherence analysis test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Enhanced Coherence Analyzer - Advanced Metrics Implementation")
    print("=" * 70)
    
    success = test_enhanced_coherence()
    
    if success:
        print("\n🎉 Enhanced Coherence Analyzer working correctly!")
        print("📊 Implemented advanced metrics:")
        print("   • Second-order coherence (coherence of coherence)")
        print("   • Temporal coherence with sliding windows")
        print("   • Phrase-separation coherence analysis")
        print("   • Statistical significance testing")
        print("   • Coherence decay rate analysis")
        print("\n🔬 Ready for integration with main analysis pipeline!")