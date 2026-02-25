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
        This principle establishes the objective validity of the categories, which are pure concepts of the understanding.
        Without this unity, no manifold of intuition could be thought as an object, for it would lack necessary connection.
        Therefore, the categories are the a priori conditions for the possibility of any experience whatsoever.
        Space and time are not properties of things in themselves, but the forms of our sensible intuition.
        Phenomena are the only objects of our knowledge; noumena remain forever beyond our cognitive grasp.
        Reason must criticize itself to understand its own limits and proper domain.
        The moral law within us is a fact of reason, commanding respect and duty.
        """,
        
        "hume_style": """
        All knowledge ultimately derives from sensory experience and vivid impressions.
        We cannot truly know anything beyond what our senses directly reveal to us in the moment.
        Causation is nothing more than the constant conjunction of events, a habit of mind.
        Custom and habit, not reason, are the great guides of human life and our expectations.
        There is no rational justification for believing in the uniformity of nature or induction.
        The self is but a bundle or collection of different perceptions, succeeding each other with inconceivable rapidity.
        Passions, not reason, motivate human action and determine our ends.
        Miracles, being violations of the laws of nature, are by definition maximally improbable.
        """,
        
        "nietzsche_style": """
        God is dead! And we have killed him with our relentless rationality and science.
        What festivals of atonement, what sacred games shall we have to invent for ourselves?
        The death of God signifies the collapse of absolute values and universal morality.
        We must ourselves become gods to seem worthy of so tremendous a deed.
        The will to power is the fundamental drive of all beings, seeking to expand and dominate.
        Christian morality is a slave morality, born of resentment and weakness.
        The Übermensch is the one who overcomes humanity and creates his own values.
        Eternal recurrence is the ultimate test: would you live this same life again, innumerable times?
        """,
        
        "incoherent": """
        Purple mathematics dances with forgotten elephants in the pale moonlight of forgotten theories.
        Therefore, existence itself tastes like freshly minted copper pennies mixed with quantum foam.
        Most cats secretly understand the categorical imperative far better than Kant ever did, but they won't admit it.
        The universe speaks exclusively in a dialect of shimmering semicolons and abstract, velvety nouns.
        Gravity is merely the opinion of a particularly stubborn rock that has convinced other rocks to agree.
        Time flows backward on Tuesdays, which is why breakfast often feels like a memory of dinner.
        Philosophical zombies enjoy listening to polka music, but only when no one is watching them.
        The concept of truth is a sphere made of liquid soap, beautiful and impossible to hold.
        """
    }


@pytest.fixture(scope="session")
def mock_nltk_data(tmp_path_factory):
    """Fixture to handle NLTK data for tests.

    Downloads necessary NLTK data to a temporary directory once per session
    and sets the NLTK_DATA environment variable to point to it.
    This avoids repeated downloads and ensures tests are self-contained.
    """
    import nltk
    import os
    
    # Create a temporary directory for NLTK data for the test session
    nltk_data_dir = tmp_path_factory.mktemp("nltk_data")
    os.environ['NLTK_DATA'] = str(nltk_data_dir)

    # List of required NLTK packages
    required_packages = ['punkt', 'averaged_perceptron_tagger', 'wordnet']
    
    # Download each required package
    for package in required_packages:
        try:
            # Check if already available (e.g., in a cache)
            nltk.data.find(f"tokenizers/{package}")
        except LookupError:
            try:
                nltk.download(package, download_dir=str(nltk_data_dir), quiet=True)
            except Exception as e:
                pytest.fail(f"Failed to download NLTK package {package}: {e}")

    # Yield control to the tests
    yield

    # Teardown: Unset the environment variable
    del os.environ['NLTK_DATA']

    