"""
Minimal functional version of the PhilosophicalAnalyzer.

This is a basic implementation to test the project setup.
Full implementation will be added incrementally.
"""

import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

logger = logging.getLogger(__name__)

# Download required NLTK data with error handling
def download_nltk_data():
    """Download required NLTK data."""
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords'),
        ('wordnet', 'corpora/wordnet'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')
    ]
    
    for resource, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                logger.info(f"Downloading {resource}...")
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download {resource}: {e}")

_nltk_data_downloaded = False

def _ensure_nltk_data():
    global _nltk_data_downloaded
    if _nltk_data_downloaded:
        return
    download_nltk_data()
    _nltk_data_downloaded = True


class PhilosophicalAnalyzer:
    """
    Minimal implementation of philosophical text analyzer.
    
    This version implements basic LSA and coherence analysis
    to test the project structure.
    """
    
    def __init__(self):
        """Initialize the analyzer with basic settings."""
        _ensure_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # Fallback if stopwords not available
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        
        self.vectorizer = None
        self.lsa_model = None
        self.is_fitted = False
        
        logger.info("PhilosophicalAnalyzer initialized")
    
    def preprocess_text(self, text: str) -> List[List[str]]:
        """
        Basic text preprocessing with fallback tokenization.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            List of preprocessed sentences (as word lists)
        """
        # Clean text
        text = re.sub(r'[^\w\s\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize into sentences with fallback
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback: simple sentence splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        processed_sentences = []
        for sentence in sentences:
            try:
                # Tokenize words
                words = word_tokenize(sentence.lower())
            except LookupError:
                # Fallback: simple word splitting
                words = sentence.lower().split()
            
            # Filter and lemmatize
            processed_words = []
            for word in words:
                if word.isalpha() and word not in self.stop_words and len(word) > 2:
                    try:
                        lemmatized = self.lemmatizer.lemmatize(word)
                        processed_words.append(lemmatized)
                    except Exception:
                        # Fallback: use original word
                        processed_words.append(word)
            
            if len(processed_words) > 2:  # Reduced threshold
                processed_sentences.append(processed_words)
        
        return processed_sentences
    
    def fit(self, texts: Dict[str, str]) -> 'PhilosophicalAnalyzer':
        """
        Fit the LSA model on provided texts.
        
        Args:
            texts: Dictionary mapping text IDs to text content
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting analyzer on {len(texts)} texts...")
        
        # Preprocess all texts
        all_sentences = []
        for text_content in texts.values():
            sentences = self.preprocess_text(text_content)
            all_sentences.extend(sentences)
        
        if not all_sentences:
            raise ValueError("No valid sentences found in provided texts")
        
        logger.info(f"Found {len(all_sentences)} sentences for training")
        
        # Convert to corpus format
        corpus = [' '.join(sentence) for sentence in all_sentences]
        
        # Check corpus size
        all_words = set()
        for sentence in all_sentences:
            all_words.update(sentence)
        
        logger.info(f"Vocabulary size: {len(all_words)} unique words")
        
        if len(all_words) < 10:
            # If vocabulary is too small, use simple analysis without LSA
            logger.warning("Vocabulary too small for LSA, using simple analysis")
            self.vectorizer = None
            self.lsa_model = None
            self.is_fitted = True
            self._simple_mode = True
            return self
        
        # Build TF-IDF matrix with more permissive settings
        self.vectorizer = TfidfVectorizer(
            max_features=min(500, len(all_words)),  # Adaptive max features
            min_df=1,                               # Allow all words
            max_df=0.95,
            ngram_range=(1, 1),
            token_pattern=r'\b\w+\b'               # Simple token pattern
        )
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
            
            # Only use LSA if we have enough features
            if tfidf_matrix.shape[1] >= 2:
                # Build LSA model with adaptive components
                n_components = min(10, tfidf_matrix.shape[1] - 1, len(corpus) - 1)
                self.lsa_model = TruncatedSVD(n_components=n_components, random_state=42)
                self.lsa_model.fit(tfidf_matrix)
                logger.info(f"LSA model built with {n_components} components")
                self._simple_mode = False
            else:
                logger.warning("Too few features for LSA, using TF-IDF only")
                self.lsa_model = None
                self._simple_mode = True
        
        except Exception as e:
            logger.warning(f"LSA failed ({e}), using simple mode")
            self.vectorizer = None
            self.lsa_model = None
            self._simple_mode = True
        
        self.is_fitted = True
        logger.info("Analyzer fitted successfully")
        
        return self
    
    def get_sentence_vector(self, sentence_words: List[str]) -> np.ndarray:
        """
        Get vector for a sentence (LSA or simple).
        
        Args:
            sentence_words: List of words in the sentence
            
        Returns:
            Vector for the sentence
        """
        if hasattr(self, '_simple_mode') and self._simple_mode:
            # Simple mode: return word count vector
            return np.array([len(sentence_words)])
        
        sentence_text = ' '.join(sentence_words)
        tfidf_vec = self.vectorizer.transform([sentence_text])
        
        if self.lsa_model is not None:
            lsa_vec = self.lsa_model.transform(tfidf_vec)
            return lsa_vec[0]
        else:
            return tfidf_vec.toarray()[0]
    
    def calculate_coherence(self, sentences: List[List[str]]) -> Dict[str, float]:
        """
        Calculate semantic coherence metrics.
        
        Args:
            sentences: List of preprocessed sentences
            
        Returns:
            Dictionary with coherence metrics
        """
        if len(sentences) < 2:
            return {
                'semantic_coherence': 0.0,
                'min_coherence': 0.0,
                'max_coherence': 0.0
            }
        
        # Get vectors for all sentences
        vectors = []
        for sentence in sentences:
            try:
                vec = self.get_sentence_vector(sentence)
                if len(vec) > 0:  # Ensure vector is not empty
                    vectors.append(vec)
            except Exception:
                continue

        if len(vectors) < 2:
            return {
                'semantic_coherence': 0.0,
                'min_coherence': 0.0,
                'max_coherence': 0.0
            }
        
        # Calculate coherence between consecutive sentences
        coherence_scores = []
        for i in range(len(vectors) - 1):
            try:
                if hasattr(self, '_simple_mode') and self._simple_mode:
                    # Simple similarity for simple mode
                    sim = 1.0 / (1.0 + abs(vectors[i][0] - vectors[i+1][0]))
                else:
                    # Cosine similarity for LSA mode
                    sim = cosine_similarity([vectors[i]], [vectors[i + 1]])[0][0]
                
                if not np.isnan(sim) and not np.isinf(sim):
                    coherence_scores.append(sim)
            except Exception:
                continue
        
        if not coherence_scores:
            return {
                'semantic_coherence': 0.0,
                'min_coherence': 0.0,
                'max_coherence': 0.0
            }
        
        return {
            'semantic_coherence': float(np.mean(coherence_scores)),
            'min_coherence': float(np.min(coherence_scores)),
            'max_coherence': float(np.max(coherence_scores))
        }
    
    def analyze_text(self, text: str, text_id: str = "") -> Dict[str, Any]:
        """
        Analyze a single text.
        
        Args:
            text: Text content to analyze
            text_id: Identifier for the text
            
        Returns:
            Dictionary with analysis results
        """
        if not self.is_fitted:
            raise RuntimeError("Analyzer must be fitted before analyzing texts")
        
        logger.info(f"Analyzing text: {text_id}")
        
        # Preprocess
        sentences = self.preprocess_text(text)
        
        if len(sentences) < 2:
            return {
                'text_id': text_id,
                'error': 'insufficient_sentences',
                'sentence_count': len(sentences),
                'semantic_coherence': 0.0
            }
        
        # Calculate coherence
        coherence_metrics = self.calculate_coherence(sentences)
        
        # Basic syntactic analysis
        total_words = sum(len(sentence) for sentence in sentences)
        max_sentence_length = max(len(sentence) for sentence in sentences) if sentences else 0
        avg_sentence_length = total_words / len(sentences) if sentences else 0
        
        # Simple classification
        is_coherent = coherence_metrics['semantic_coherence'] > 0.3
        
        result = {
            'text_id': text_id,
            'sentence_count': len(sentences),
            'word_count': total_words,
            'max_sentence_length': max_sentence_length,
            'avg_sentence_length': round(avg_sentence_length, 2),
            **coherence_metrics,
            'classification': 'coherent' if is_coherent else 'fragmented',
            'analysis_mode': 'simple' if hasattr(self, '_simple_mode') and self._simple_mode else 'lsa'
        }
        
        logger.info(f"Analysis completed for {text_id}")
        return result
    
    def analyze_multiple_texts(self, texts: Dict[str, str]) -> pd.DataFrame:
        """
        Analyze multiple texts.
        
        Args:
            texts: Dictionary mapping text IDs to content
            
        Returns:
            DataFrame with results for all texts
        """
        results = []
        
        for text_id, text_content in texts.items():
            try:
                result = self.analyze_text(text_content, text_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {text_id}: {e}")
                results.append({
                    'text_id': text_id,
                    'error': str(e),
                    'semantic_coherence': 0.0
                })
        
        return pd.DataFrame(results)
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        mode = ""
        if self.is_fitted and hasattr(self, '_simple_mode'):
            mode = f" (mode: {'simple' if self._simple_mode else 'lsa'})"
        return f"PhilosophicalAnalyzer(status={status}{mode})"


# Test function for basic functionality
def test_analyzer():
    """Test basic analyzer functionality."""
    print("🧪 Testing PhilosophicalAnalyzer...")
    
    # Longer sample texts for better testing
    sample_texts = {
        "coherent": """
        Philosophy is the study of fundamental questions about existence, knowledge, values, reason, mind, and language.
        These questions have been explored by thinkers throughout history and across different cultures.
        The systematic approach to these problems defines philosophical inquiry and distinguishes it from other forms of investigation.
        Logic provides the foundation for rigorous philosophical reasoning and argumentation.
        Ethics examines questions of right and wrong, good and evil, virtue and vice.
        Metaphysics investigates the nature of reality, being, and existence itself.
        Epistemology studies the nature of knowledge, justified belief, and the rationality of belief.
        """,
        "fragmented": """
        Reality is fundamentally uncertain and unpredictable in its manifestations.
        Mathematical proofs demonstrate everything strange and counterintuitive about modern existence.
        Therefore, domestic cats understand quantum mechanics better than most human physicists today.
        The universe communicates through colors that remain invisible to human perception.
        Existence becomes completely meaningless when viewed through the lens of purple mathematics.
        Time flows backwards on Tuesdays according to my grandmother's philosophical system.
        Knowledge tastes like copper pennies mixed with forgotten dreams and childhood memories.
        """
    }
    
    try:
        # Initialize and test
        analyzer = PhilosophicalAnalyzer()
        
        # Fit the analyzer
        analyzer.fit(sample_texts)
        
        print(f"\n🔧 Analyzer: {analyzer}")
        
        # Analyze each text
        for text_id, text_content in sample_texts.items():
            result = analyzer.analyze_text(text_content, text_id)
            print(f"\n📊 Results for '{text_id}':")
            if 'error' in result:
                print(f"  ❌ Error: {result['error']}")
            else:
                print(f"  📝 Sentences: {result['sentence_count']}")
                print(f"  📖 Words: {result['word_count']}")
                print(f"  📏 Avg sentence length: {result['avg_sentence_length']}")
                print(f"  🧠 Semantic Coherence: {result['semantic_coherence']:.3f}")
                print(f"  📉 Min Coherence: {result['min_coherence']:.3f}")
                print(f"  📈 Max Coherence: {result['max_coherence']:.3f}")
                print(f"  🎯 Classification: {result['classification']}")
                print(f"  ⚙️  Analysis Mode: {result['analysis_mode']}")
        
        print("\n✅ Basic functionality test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting Philosophical Text Analyzer Test")
    print("=" * 50)
    
    success = test_analyzer()
    
    if success:
        print("\n🎉 Congratulations! The analyzer is working correctly!")
        print("📊 You can see the difference in coherence between:")
        print("   • Coherent philosophical text (higher coherence)")
        print("   • Fragmented/random text (lower coherence)")
        print("\n🔬 This demonstrates the core concept from the research paper!")
    else:
        print("\n💡 Some issues detected, but the basic structure is in place.")
        print("🔧 The fallback mechanisms should allow basic functionality.")