"""
Advanced POS-tagging and linguistic analysis based on the original research paper.

Implements exact determiners analysis and phrase-level metrics as described in
"Automated analysis of free speech predicts psychosis onset in high-risk youths" (Bedi et al., 2015).
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from collections import Counter, defaultdict
import re

logger = logging.getLogger(__name__)

# Download required NLTK data
def ensure_nltk_data():
    """Ensure required NLTK data is available."""
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
        ('maxent_ne_chunker', 'chunkers/maxent_ne_chunker'),
        ('words', 'corpora/words')
    ]
    
    for resource, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                print(f"Downloading {resource}...")
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download {resource}: {e}")

ensure_nltk_data()


class POSAnalyzer:
    """
    Advanced POS-tagging and linguistic analysis following the original paper.
    
    Implements:
    - Exact determiners frequency analysis
    - Maximum phrase length detection
    - Normalized linguistic metrics
    - Phrase-separation analysis
    """
    
    # Exact determiners from the original paper
    DETERMINERS = {'that', 'what', 'whatever', 'which', 'whichever'}
    
    # POS tags for different categories
    NOUN_TAGS = {'NN', 'NNS', 'NNP', 'NNPS'}
    VERB_TAGS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    ADJ_TAGS = {'JJ', 'JJR', 'JJS'}
    ADV_TAGS = {'RB', 'RBR', 'RBS'}
    PRONOUN_TAGS = {'PRP', 'PRP$', 'WP', 'WP$'}
    
    def __init__(self):
        """Initialize the POS analyzer."""
        self.reset_stats()
    
    def reset_stats(self):
        """Reset internal statistics."""
        self._word_counts = Counter()
        self._pos_counts = Counter()
        self._phrase_lengths = []
        self._determiner_count = 0
        self._total_words = 0
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Perform comprehensive POS analysis on text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detailed linguistic metrics
        """
        self.reset_stats()
        
        try:
            # Tokenize into sentences
            sentences = sent_tokenize(text)
            
            if not sentences:
                return self._empty_result()
            
            all_metrics = []
            
            for sentence in sentences:
                sentence_metrics = self._analyze_sentence(sentence)
                if sentence_metrics:
                    all_metrics.append(sentence_metrics)
            
            if not all_metrics:
                return self._empty_result()
            
            # Aggregate metrics
            return self._aggregate_metrics(all_metrics, len(sentences))
            
        except Exception as e:
            logger.error(f"Error in POS analysis: {e}")
            return self._empty_result()
    
    def _analyze_sentence(self, sentence: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a single sentence for POS metrics.
        
        Args:
            sentence: Sentence to analyze
            
        Returns:
            Dictionary with sentence-level metrics
        """
        try:
            # Tokenize and tag
            words = word_tokenize(sentence.lower())
            
            if len(words) < 2:  # Skip very short sentences
                return None
            
            # POS tagging
            pos_tags = pos_tag(words)
            
            # Extract metrics
            metrics = {
                'word_count': len(words),
                'determiner_count': self._count_determiners(words),
                'max_phrase_length': self._get_max_phrase_length(pos_tags),
                'pos_distribution': self._get_pos_distribution(pos_tags),
                'syntactic_complexity': self._calculate_syntactic_complexity(pos_tags),
                'lexical_diversity': self._calculate_lexical_diversity(words)
            }
            
            # Update global counters
            self._total_words += len(words)
            self._determiner_count += metrics['determiner_count']
            self._phrase_lengths.append(metrics['max_phrase_length'])
            
            for word, tag in pos_tags:
                self._word_counts[word] += 1
                self._pos_counts[tag] += 1
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error analyzing sentence '{sentence[:50]}...': {e}")
            return None
    
    def _count_determiners(self, words: List[str]) -> int:
        """
        Count exact determiners as specified in the original paper.
        
        Args:
            words: List of words in the sentence
            
        Returns:
            Count of determiners
        """
        return sum(1 for word in words if word in self.DETERMINERS)
    
    def _get_max_phrase_length(self, pos_tags: List[Tuple[str, str]]) -> int:
        """
        Calculate maximum phrase length using POS patterns.
        
        Args:
            pos_tags: List of (word, POS_tag) tuples
            
        Returns:
            Maximum phrase length found
        """
        if not pos_tags:
            return 0
        
        # Define phrase patterns (noun phrases, verb phrases, etc.)
        phrase_patterns = [
            self.NOUN_TAGS,  # Noun phrases
            self.VERB_TAGS,  # Verb phrases
            self.ADJ_TAGS,   # Adjective phrases
            self.ADV_TAGS    # Adverb phrases
        ]
        
        max_length = 1
        
        for pattern in phrase_patterns:
            current_length = 0
            
            for word, tag in pos_tags:
                if tag in pattern:
                    current_length += 1
                    max_length = max(max_length, current_length)
                else:
                    current_length = 0
        
        return max_length
    
    def _get_pos_distribution(self, pos_tags: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Calculate POS tag distribution.
        
        Args:
            pos_tags: List of (word, POS_tag) tuples
            
        Returns:
            Dictionary with POS proportions
        """
        if not pos_tags:
            return {}
        
        tag_counts = Counter(tag for word, tag in pos_tags)
        total = len(pos_tags)
        
        # Group into major categories
        distribution = {
            'nouns': sum(tag_counts[tag] for tag in self.NOUN_TAGS) / total,
            'verbs': sum(tag_counts[tag] for tag in self.VERB_TAGS) / total,
            'adjectives': sum(tag_counts[tag] for tag in self.ADJ_TAGS) / total,
            'adverbs': sum(tag_counts[tag] for tag in self.ADV_TAGS) / total,
            'pronouns': sum(tag_counts[tag] for tag in self.PRONOUN_TAGS) / total,
            'other': 1.0 - sum([
                sum(tag_counts[tag] for tag in category) / total
                for category in [self.NOUN_TAGS, self.VERB_TAGS, self.ADJ_TAGS, 
                               self.ADV_TAGS, self.PRONOUN_TAGS]
            ])
        }
        
        return distribution
    
    def _calculate_syntactic_complexity(self, pos_tags: List[Tuple[str, str]]) -> float:
        """
        Calculate syntactic complexity based on POS patterns.
        
        Args:
            pos_tags: List of (word, POS_tag) tuples
            
        Returns:
            Syntactic complexity score
        """
        if not pos_tags:
            return 0.0
        
        # Count complex structures
        subordinating_conjunctions = sum(1 for word, tag in pos_tags if tag == 'IN')
        coordinating_conjunctions = sum(1 for word, tag in pos_tags if tag == 'CC')
        relative_pronouns = sum(1 for word, tag in pos_tags if tag in ['WP', 'WDT'])
        
        complexity_score = (subordinating_conjunctions + coordinating_conjunctions + relative_pronouns) / len(pos_tags)
        
        return complexity_score
    
    def _calculate_lexical_diversity(self, words: List[str]) -> float:
        """
        Calculate lexical diversity (Type-Token Ratio).
        
        Args:
            words: List of words
            
        Returns:
            Lexical diversity score
        """
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        return unique_words / total_words if total_words > 0 else 0.0
    
    def _aggregate_metrics(self, sentence_metrics: List[Dict], sentence_count: int) -> Dict[str, Any]:
        """
        Aggregate sentence-level metrics into text-level metrics.
        
        Args:
            sentence_metrics: List of sentence-level metric dictionaries
            sentence_count: Total number of sentences
            
        Returns:
            Aggregated metrics dictionary
        """
        if not sentence_metrics:
            return self._empty_result()
        
        # Calculate normalized determiner frequency (key metric from paper)
        normalized_determiners = (self._determiner_count / self._total_words) if self._total_words > 0 else 0.0
        
        # Calculate average and maximum phrase lengths
        avg_phrase_length = np.mean(self._phrase_lengths) if self._phrase_lengths else 0.0
        max_phrase_length = max(self._phrase_lengths) if self._phrase_lengths else 0.0
        
        # Aggregate POS distributions
        pos_distributions = [m['pos_distribution'] for m in sentence_metrics if m['pos_distribution']]
        avg_pos_distribution = {}
        
        if pos_distributions:
            for category in ['nouns', 'verbs', 'adjectives', 'adverbs', 'pronouns', 'other']:
                values = [d.get(category, 0.0) for d in pos_distributions]
                avg_pos_distribution[category] = np.mean(values)
        
        # Calculate overall complexity metrics
        complexity_scores = [m['syntactic_complexity'] for m in sentence_metrics]
        avg_complexity = np.mean(complexity_scores) if complexity_scores else 0.0
        
        diversity_scores = [m['lexical_diversity'] for m in sentence_metrics]
        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0.0
        
        return {
            # Core metrics from the paper
            'normalized_determiners': normalized_determiners,
            'max_phrase_length': max_phrase_length,
            'avg_phrase_length': avg_phrase_length,
            
            # Extended linguistic metrics
            'sentence_count': sentence_count,
            'total_words': self._total_words,
            'avg_words_per_sentence': self._total_words / sentence_count if sentence_count > 0 else 0.0,
            'pos_distribution': avg_pos_distribution,
            'syntactic_complexity': avg_complexity,
            'lexical_diversity': avg_diversity,
            
            # Vocabulary metrics
            'vocabulary_size': len(self._word_counts),
            'most_common_words': dict(self._word_counts.most_common(10)),
            'most_common_pos': dict(self._pos_counts.most_common(10)),
            
            # Statistical measures
            'phrase_length_std': np.std(self._phrase_lengths) if len(self._phrase_lengths) > 1 else 0.0,
            'complexity_variance': np.var(complexity_scores) if len(complexity_scores) > 1 else 0.0
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'normalized_determiners': 0.0,
            'max_phrase_length': 0,
            'avg_phrase_length': 0.0,
            'sentence_count': 0,
            'total_words': 0,
            'avg_words_per_sentence': 0.0,
            'pos_distribution': {},
            'syntactic_complexity': 0.0,
            'lexical_diversity': 0.0,
            'vocabulary_size': 0,
            'most_common_words': {},
            'most_common_pos': {},
            'phrase_length_std': 0.0,
            'complexity_variance': 0.0,
            'error': 'insufficient_data'
        }
    
    def get_detailed_analysis(self, text: str) -> Dict[str, Any]:
        """
        Get detailed analysis including sentence-by-sentence breakdown.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detailed analysis with sentence-level data
        """
        analysis = self.analyze_text(text)
        
        # Add sentence-by-sentence breakdown
        sentences = sent_tokenize(text)
        sentence_details = []
        
        for i, sentence in enumerate(sentences):
            sentence_analysis = self._analyze_sentence(sentence)
            if sentence_analysis:
                sentence_analysis['sentence_index'] = i
                sentence_analysis['sentence_text'] = sentence[:100] + '...' if len(sentence) > 100 else sentence
                sentence_details.append(sentence_analysis)
        
        analysis['sentence_details'] = sentence_details
        return analysis


def test_pos_analyzer():
    """Test the POS analyzer with sample philosophical text."""
    print("🧪 Testing POS Analyzer...")
    
    analyzer = POSAnalyzer()
    
    # Test with philosophical text
    philosophical_text = """
    The transcendental unity of apperception is that unity through which all the manifold 
    given in an intuition is united in a concept of the object. What we call object 
    corresponds to something in the manifold of our sensible intuition. Whatever the 
    process and the means may be by which knowledge relates to its objects, that relation 
    by which it relates immediately to them is intuition.
    """
    
    result = analyzer.analyze_text(philosophical_text)
    
    print(f"\n📊 POS Analysis Results:")
    print(f"📝 Sentences: {result['sentence_count']}")
    print(f"📚 Total words: {result['total_words']}")
    print(f"🎯 Normalized determiners: {result['normalized_determiners']:.4f}")
    print(f"📏 Max phrase length: {result['max_phrase_length']}")
    print(f"📈 Avg phrase length: {result['avg_phrase_length']:.2f}")
    print(f"🧠 Syntactic complexity: {result['syntactic_complexity']:.3f}")
    print(f"🎨 Lexical diversity: {result['lexical_diversity']:.3f}")
    
    print(f"\n📋 POS Distribution:")
    for category, proportion in result['pos_distribution'].items():
        print(f"  {category}: {proportion:.3f}")
    
    print(f"\n🔤 Most common words: {list(result['most_common_words'].keys())[:5]}")
    
    return result


if __name__ == "__main__":
    test_pos_analyzer()