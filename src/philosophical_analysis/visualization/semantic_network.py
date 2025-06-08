"""
Semantic Network Generator for Philosophical Text Analysis.

This module handles the generation of semantic networks from philosophical texts,
extracting key concepts and their relationships for visualization.
"""

import logging
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from pathlib import Path

# Optional imports with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class SemanticNetworkGenerator:
    """
    Generator for semantic networks from philosophical texts.
    
    This class extracts key philosophical concepts and builds relationship networks
    that can be visualized to understand conceptual connections between ideas.
    """
    
    def __init__(self, max_concepts: int = 30, min_strength: float = 0.1):
        """
        Initialize the semantic network generator.
        
        Args:
            max_concepts: Maximum number of concepts to extract
            min_strength: Minimum relationship strength to include
        """
        self.max_concepts = max_concepts
        self.min_strength = min_strength
        
        # Curated list of philosophical terms
        self.philosophical_terms = {
            # Epistemology
            'knowledge', 'truth', 'belief', 'understanding', 'perception', 
            'experience', 'reason', 'logic', 'intuition', 'wisdom',
            
            # Metaphysics
            'existence', 'reality', 'being', 'substance', 'essence', 
            'nature', 'consciousness', 'mind', 'soul', 'spirit',
            'time', 'space', 'infinity', 'causation', 'necessity',
            
            # Ethics
            'morality', 'ethics', 'virtue', 'good', 'evil', 'right', 
            'wrong', 'justice', 'duty', 'obligation', 'responsibility',
            'freedom', 'will', 'choice', 'action', 'intention',
            
            # Aesthetics
            'beauty', 'art', 'aesthetic', 'sublime', 'taste', 'judgment',
            
            # Political Philosophy
            'power', 'authority', 'law', 'rights', 'liberty', 'equality',
            'society', 'state', 'government', 'democracy',
            
            # Philosophy of Religion
            'god', 'divine', 'sacred', 'faith', 'religion', 'salvation',
            
            # Logic and Language
            'concept', 'proposition', 'argument', 'proof', 'language',
            'meaning', 'reference', 'truth', 'validity'
        }
        
        # Concept categories for visualization
        self.concept_categories = {
            'epistemological': {
                'knowledge', 'truth', 'belief', 'understanding', 'perception',
                'experience', 'reason', 'logic', 'intuition', 'wisdom'
            },
            'metaphysical': {
                'existence', 'reality', 'being', 'substance', 'essence',
                'nature', 'consciousness', 'mind', 'soul', 'spirit',
                'time', 'space', 'infinity', 'causation', 'necessity'
            },
            'ethical': {
                'morality', 'ethics', 'virtue', 'good', 'evil', 'right',
                'wrong', 'justice', 'duty', 'obligation', 'responsibility',
                'freedom', 'will', 'choice', 'action', 'intention'
            },
            'aesthetic': {
                'beauty', 'art', 'aesthetic', 'sublime', 'taste', 'judgment'
            },
            'political': {
                'power', 'authority', 'law', 'rights', 'liberty', 'equality',
                'society', 'state', 'government', 'democracy'
            },
            'religious': {
                'god', 'divine', 'sacred', 'faith', 'religion', 'salvation'
            },
            'logical': {
                'concept', 'proposition', 'argument', 'proof', 'language',
                'meaning', 'reference', 'validity'
            }
        }
        
        logger.info(f"Semantic Network Generator initialized")
        logger.info(f"Max concepts: {max_concepts}, Min strength: {min_strength}")
    
    def generate_network(self, 
                        texts: Dict[str, str], 
                        results: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate semantic network from philosophical texts.
        
        Args:
            texts: Dictionary of text_id -> text content
            results: Optional analysis results DataFrame
            
        Returns:
            Dictionary containing network data (nodes, links, metadata)
        """
        logger.info(f"Generating semantic network from {len(texts)} texts...")
        
        try:
            # Extract key concepts
            concepts = self.extract_concepts(texts)
            
            if len(concepts) == 0:
                logger.warning("No concepts extracted, using placeholder network")
                return self.generate_placeholder_network()
            
            # Build concept relationships
            relationships = self.build_relationships(concepts, texts)
            
            # Create network structure
            network_data = self.create_network_structure(concepts, relationships, texts)
            
            logger.info(f"Network generated: {len(network_data['nodes'])} nodes, {len(network_data['links'])} links")
            
            return network_data
            
        except Exception as e:
            logger.error(f"Error generating semantic network: {e}")
            return self.generate_placeholder_network()
    
    def extract_concepts(self, texts: Dict[str, str]) -> Dict[str, Dict]:
        """
        Extract key philosophical concepts from texts.
        
        Args:
            texts: Dictionary of text_id -> text content
            
        Returns:
            Dictionary of concept -> metadata
        """
        logger.info("Extracting philosophical concepts...")
        
        try:
            # Count concept frequencies across all texts
            concept_counts = Counter()
            concept_texts = defaultdict(set)  # Track which texts contain each concept
            
            for text_id, content in texts.items():
                # Clean and tokenize text
                words = self._extract_words(content)
                
                # Count philosophical terms
                for word in words:
                    if word in self.philosophical_terms:
                        concept_counts[word] += 1
                        concept_texts[word].add(text_id)
            
            # Select top concepts
            top_concepts = dict(concept_counts.most_common(self.max_concepts))
            
            # Create concept metadata
            concepts = {}
            max_count = max(concept_counts.values()) if concept_counts else 1
            
            for concept, count in top_concepts.items():
                # Determine primary philosopher (text where concept appears most)
                primary_text = self._find_primary_text(concept, texts)
                primary_philosopher = self._extract_philosopher_name(primary_text)
                
                concepts[concept] = {
                    'frequency': count,
                    'primary_philosopher': primary_philosopher,
                    'importance': count / max_count,
                    'text_occurrences': len(concept_texts[concept]),
                    'category': self._categorize_concept(concept),
                    'texts': list(concept_texts[concept])
                }
            
            logger.info(f"Extracted {len(concepts)} key concepts")
            return concepts
            
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return {}
    
    def build_relationships(self, 
                          concepts: Dict[str, Dict], 
                          texts: Dict[str, str]) -> Dict[Tuple[str, str], float]:
        """
        Build relationships between concepts based on co-occurrence.
        
        Args:
            concepts: Dictionary of concepts with metadata
            texts: Dictionary of texts
            
        Returns:
            Dictionary of (concept1, concept2) -> strength
        """
        logger.info("Building concept relationships...")
        
        try:
            relationships = {}
            concept_list = list(concepts.keys())
            
            if not concept_list:
                logger.warning("No concepts available to build relationships")
                return {}
                
            # Calculate co-occurrence for each pair
            for i, concept1 in enumerate(concept_list):
                for j, concept2 in enumerate(concept_list):
                    if i < j:  # Avoid duplicates and self-loops
                        # Skip if either concept is too short
                        if len(concept1) < 3 or len(concept2) < 3:
                            continue
                            
                        # Calculate co-occurrence strength
                        strength = self._calculate_cooccurrence(concept1, concept2, texts)
                        
                        # Apply dynamic threshold based on concept frequency
                        freq1 = concepts[concept1].get('frequency', 0)
                        freq2 = concepts[concept2].get('frequency', 0)
                        min_freq = min(freq1, freq2)
                        
                        # Adjust threshold based on frequency (lower threshold for less frequent concepts)
                        dynamic_threshold = max(self.min_strength, 0.05)  # Minimum threshold of 0.05
                        if min_freq > 0:
                            dynamic_threshold = max(self.min_strength, 0.5 / min_freq)
                        
                        if strength >= dynamic_threshold:
                            relationships[(concept1, concept2)] = strength
            
            # If no relationships found with normal threshold, try with a lower threshold
            if not relationships and concept_list:
                logger.info("No relationships found with normal threshold, trying with lower threshold...")
                for i, concept1 in enumerate(concept_list):
                    for j, concept2 in enumerate(concept_list):
                        if i < j and len(concept1) >= 3 and len(concept2) >= 3:
                            strength = self._calculate_cooccurrence(concept1, concept2, texts)
                            if strength > 0:  # Any positive strength
                                relationships[(concept1, concept2)] = strength
            
            logger.info(f"Built {len(relationships)} concept relationships")
            return relationships
            
        except Exception as e:
            logger.error(f"Error building relationships: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def create_network_structure(self, 
                               concepts: Dict[str, Dict],
                               relationships: Dict[Tuple[str, str], float],
                               texts: Dict[str, str]) -> Dict[str, Any]:
        """
        Create the final network structure for visualization.
        
        Args:
            concepts: Dictionary of concepts
            relationships: Dictionary of relationships
            texts: Original texts
            
        Returns:
            Network data structure
        """
        logger.info("Creating network structure...")
        
        nodes = []
        links = []
        
        # Create nodes
        for concept, data in concepts.items():
            node = {
                'id': concept,
                'label': concept.upper(),
                'category': data['category'],
                'philosopher': data['primary_philosopher'],
                'importance': float(data['importance']),
                'frequency': data['frequency'],
                'texts': data['texts']
            }
            nodes.append(node)
        
        # Create links
        for (source, target), strength in relationships.items():
            link = {
                'source': source,
                'target': target,
                'strength': float(strength),
                'type': self._determine_link_type(source, target)
            }
            links.append(link)
        
        # Calculate network metadata
        metadata = self._calculate_network_metadata(nodes, links, texts)
        
        return {
            'nodes': nodes,
            'links': links,
            'metadata': metadata
        }
    
    def generate_placeholder_network(self) -> Dict[str, Any]:
        """
        Generate a placeholder network for cases where extraction fails.
        
        Returns:
            Placeholder network data
        """
        logger.info("Generating placeholder network...")
        
        # Core philosophical concepts
        placeholder_concepts = [
            ('reality', 'metaphysical'), ('knowledge', 'epistemological'),
            ('truth', 'epistemological'), ('existence', 'metaphysical'),
            ('consciousness', 'metaphysical'), ('morality', 'ethical'),
            ('virtue', 'ethical'), ('reason', 'epistemological'),
            ('experience', 'epistemological'), ('freedom', 'ethical'),
            ('beauty', 'aesthetic'), ('justice', 'ethical'),
            ('nature', 'metaphysical'), ('mind', 'metaphysical'),
            ('good', 'ethical')
        ]
        
        nodes = []
        for i, (concept, category) in enumerate(placeholder_concepts):
            node = {
                'id': concept,
                'label': concept.upper(),
                'category': category,
                'philosopher': 'VARIOUS',
                'importance': 0.5 + (i * 0.03),
                'frequency': 10 + i,
                'texts': []
            }
            nodes.append(node)
        
        # Sample connections based on philosophical relationships
        sample_connections = [
            ('reality', 'existence', 0.8),
            ('knowledge', 'truth', 0.7),
            ('morality', 'virtue', 0.6),
            ('reason', 'knowledge', 0.5),
            ('experience', 'consciousness', 0.4),
            ('beauty', 'nature', 0.5),
            ('justice', 'morality', 0.6),
            ('mind', 'consciousness', 0.7),
            ('freedom', 'morality', 0.4),
            ('good', 'virtue', 0.5)
        ]
        
        links = []
        for source, target, strength in sample_connections:
            link = {
                'source': source,
                'target': target,
                'strength': strength,
                'type': 'conceptual'
            }
            links.append(link)
        
        metadata = {
            'total_concepts': len(nodes),
            'total_relationships': len(links),
            'density': len(links) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0,
            'categories': list(set(node['category'] for node in nodes)),
            'is_placeholder': True
        }
        
        return {
            'nodes': nodes,
            'links': links,
            'metadata': metadata
        }
    
    # Helper methods
    def _extract_words(self, text: str) -> List[str]:
        """Extract and clean words from text."""
        # Remove punctuation and convert to lowercase
        cleaned_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = re.findall(r'\b[a-zA-Z]+\b', cleaned_text)
        # Filter out very short words
        return [word for word in words if len(word) > 3]
    
    def _find_primary_text(self, concept: str, texts: Dict[str, str]) -> str:
        """Find the text where a concept appears most frequently."""
        max_count = 0
        primary_text = list(texts.keys())[0]  # Default
        
        for text_id, content in texts.items():
            count = content.lower().count(concept)
            if count > max_count:
                max_count = count
                primary_text = text_id
        
        return primary_text
    
    def _extract_philosopher_name(self, text_id: str) -> str:
        """Extract philosopher name from text ID."""
        parts = text_id.lower().split('_')
        return parts[0].title() if parts else 'UNKNOWN'
    
    def _categorize_concept(self, concept: str) -> str:
        """Categorize a philosophical concept."""
        for category, terms in self.concept_categories.items():
            if concept in terms:
                return category
        return 'general'
    
    def _calculate_cooccurrence(self, concept1: str, concept2: str, texts: Dict[str, str]) -> float:
        """
        Calculate co-occurrence strength between two concepts.
        
        This enhanced version uses a combination of:
        1. Sentence-level co-occurrence counting
        2. Window-based co-occurrence within sentences
        3. Frequency normalization
        4. Log scaling for better distribution
        
        Args:
            concept1: First concept
            concept2: Second concept
            texts: Dictionary of texts
            
        Returns:
            Co-occurrence strength (0-1)
        """
        if concept1 == concept2:
            return 0.0  # No self-loops
            
        cooccurrence_count = 0
        total_sentences = 0
        window_size = 5  # Number of words to consider as co-occurring
        
        # Convert concepts to lowercase for case-insensitive matching
        concept1_lower = concept1.lower()
        concept2_lower = concept2.lower()
        
        # Use regex with word boundaries for whole word matching
        pattern1 = re.compile(r'\b' + re.escape(concept1_lower) + r'\b')
        pattern2 = re.compile(r'\b' + re.escape(concept2_lower) + r'\b')
        
        for content in texts.values():
            # Split into sentences
            sentences = re.split(r'[.!?]+', content.lower())
            total_sentences += len(sentences)
            
            for sentence in sentences:
                words = sentence.split()
                
                # Check if both concepts appear in the sentence
                has_concept1 = any(pattern1.match(word) for word in words)
                has_concept2 = any(pattern2.match(word) for word in words)
                
                if has_concept1 and has_concept2:
                    # Count as co-occurrence
                    cooccurrence_count += 1
                    
                    # Additional weight if concepts are close to each other
                    positions1 = [i for i, word in enumerate(words) if pattern1.match(word)]
                    positions2 = [i for i, word in enumerate(words) if pattern2.match(word)]
                    
                    # Check proximity within the sentence
                    for pos1 in positions1:
                        for pos2 in positions2:
                            if abs(pos1 - pos2) <= window_size:
                                cooccurrence_count += 1  # Extra weight for close proximity
        
        # Calculate base probability with add-1 smoothing
        if total_sentences == 0:
            return 0.0
            
        # Calculate co-occurrence probability
        probability = (cooccurrence_count + 1) / (total_sentences + 1)
        
        # Apply log scaling with base 10 and scale to [0,1] range
        # Add a small constant to avoid log(0)
        epsilon = 1e-10
        strength = math.log10(probability * 1000 + epsilon) + 3  # +3 to ensure positive values
        strength = max(0, min(1.0, strength / 3.0))  # Normalize to [0,1] range
        
        # Apply a minimum threshold to filter out very weak connections
        min_threshold = 0.01
        if strength < min_threshold:
            strength = 0.0
            
        return strength
    
    def _determine_link_type(self, source: str, target: str) -> str:
        """Determine the type of relationship between concepts."""
        source_cat = self._categorize_concept(source)
        target_cat = self._categorize_concept(target)
        
        if source_cat == target_cat:
            return 'categorical'
        else:
            return 'cross-categorical'
    
    def _calculate_network_metadata(self, nodes: List[Dict], links: List[Dict], texts: Dict[str, str]) -> Dict[str, Any]:
        """Calculate network metadata and statistics."""
        
        # Basic metrics
        num_nodes = len(nodes)
        num_links = len(links)
        density = num_links / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
        
        # Category distribution
        categories = [node['category'] for node in nodes]
        category_counts = Counter(categories)
        
        # Link type distribution
        link_types = [link['type'] for link in links]
        link_type_counts = Counter(link_types)
        
        # Philosopher distribution
        philosophers = [node['philosopher'] for node in nodes]
        philosopher_counts = Counter(philosophers)
        
        return {
            'total_concepts': num_nodes,
            'total_relationships': num_links,
            'density': float(density),
            'categories': list(category_counts.keys()),
            'category_distribution': dict(category_counts),
            'link_type_distribution': dict(link_type_counts),
            'philosopher_distribution': dict(philosopher_counts),
            'avg_importance': float(np.mean([node['importance'] for node in nodes])) if nodes else 0,
            'is_placeholder': False
        }
    
    def save_network(self, network_data: Dict[str, Any], filepath: str):
        """Save network data to JSON file."""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(network_data, f, indent=2, default=str)
            logger.info(f"Network saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving network: {e}")
    
    def load_network(self, filepath: str) -> Dict[str, Any]:
        """Load network data from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                network_data = json.load(f)
            logger.info(f"Network loaded from {filepath}")
            return network_data
        except Exception as e:
            logger.error(f"Error loading network: {e}")
            return self.generate_placeholder_network()


def test_semantic_network():
    """Test the semantic network generator."""
    print("🕸️ Testing Semantic Network Generator")
    print("=" * 50)
    
    # Sample texts
    sample_texts = {
        'nietzsche_test': '''
        The will to power manifests in all aspects of existence and reality.
        Morality and virtue are human constructs that limit true freedom.
        Knowledge and truth are subjective experiences of consciousness.
        Beauty and art express the deepest nature of human experience.
        ''',
        'kant_test': '''
        Reason and understanding provide the foundation for all knowledge.
        Morality and duty are universal principles that guide ethical action.
        Experience and perception are limited by the categories of mind.
        Beauty and judgment reveal the harmony between nature and reason.
        ''',
        'hume_test': '''
        Experience is the source of all knowledge and understanding.
        Causation and necessity cannot be proven through reason alone.
        Morality and virtue arise from sentiment rather than logic.
        Beauty and taste are matters of subjective feeling and emotion.
        '''
    }
    
    try:
        # Initialize generator
        generator = SemanticNetworkGenerator(max_concepts=15, min_strength=0.1)
        
        # Generate network
        network_data = generator.generate_network(sample_texts)
        
        # Display results
        print(f"✅ Network generated successfully!")
        print(f"   Nodes: {len(network_data['nodes'])}")
        print(f"   Links: {len(network_data['links'])}")
        print(f"   Density: {network_data['metadata']['density']:.3f}")
        print(f"   Categories: {network_data['metadata']['categories']}")
        
        # Show sample concepts
        print(f"\n📊 Top concepts:")
        for node in network_data['nodes'][:5]:
            print(f"   • {node['label']} ({node['category']}) - {node['philosopher']}")
        
        # Show sample relationships
        print(f"\n🔗 Sample relationships:")
        for link in network_data['links'][:5]:
            print(f"   • {link['source']} ↔ {link['target']} (strength: {link['strength']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_semantic_network()