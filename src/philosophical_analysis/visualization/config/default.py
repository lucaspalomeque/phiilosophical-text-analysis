"""
Default configuration for visualization module.
"""

# Visualization settings
VIZ_CONFIG = {
    'theme': {
        'primary': '#00FF00',      # Matrix green
        'secondary': '#FFFFFF',     # White
        'tertiary': '#888888',      # Gray
        'background': '#000000',    # Black
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
