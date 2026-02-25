"""
Default configuration for visualization module.
"""

# Visualization settings
VIZ_CONFIG = {
    'theme': {
        'primary': '#C9A96E',       # Brushed gold
        'secondary': '#E5E5E7',     # Light grey
        'tertiary': '#8E8E93',      # Medium grey
        'background': '#000000',    # Black
        'accent': '#D4BA85',        # Light gold
    },
    
    'network': {
        'max_nodes': 50,
        'min_link_strength': 0.3,
        'force_strength': -200,
        'link_distance': 80,
    },
    
    'temporal': {
        'window_size': 50,
        'overlap': 25,
        'segments': 60,
    },
    
    'dashboard': {
        'metrics': [
            'first_order_coherence',
            'second_order_coherence',
            'determiner_frequency',
            'syntactic_complexity',
            'max_phrase_length',
            'avg_sentence_length'
        ]
    }
}
