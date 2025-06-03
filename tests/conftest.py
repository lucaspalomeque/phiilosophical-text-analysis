# conftest.py
"""
Shared test configuration and fixtures.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_philosophical_texts():
    """Standard sample texts used across multiple tests."""
    return {
        "kant_style": """
        The transcendental unity of apperception is the highest principle of all knowledge.
        This principle establishes the objective validity of the categories.
        Without this unity, no manifold of intuition could be thought as an object.
        Therefore, the categories are the conditions of the possibility of experience.
        """,
        
        "hume_style": """
        All knowledge derives from sensory experience and observation.
        We cannot know anything beyond what our senses reveal to us.
        Causation is merely constant conjunction, not necessary connection.
        Custom and habit guide our expectations about future events.
        """,
        
        "nietzsche_style": """
        God is dead! We have killed him with our rationality.
        What festivals of atonement shall we need to invent?
        The death of God means the death of absolute values.
        We must become gods ourselves to seem worthy of such a deed.
        """,
        
        "incoherent": """
        Purple mathematics dances with forgotten elephants in the moonlight.
        Therefore, existence tastes like copper pennies mixed with quantum mechanics.
        Cats understand the categorical imperative better than Kant ever did.
        The universe speaks exclusively in semicolons and abstract nouns.
        """
    }


@pytest.fixture
def mock_nltk_data(monkeypatch):
    """Mock NLTK data to avoid download requirements in tests."""
    def mock_find(resource_path):
        return True
    
    def mock_download(resource, quiet=True):
        return True
    
    import nltk
    monkeypatch.setattr(nltk.data, 'find', mock_find)
    monkeypatch.setattr(nltk, 'download', mock_download)

    