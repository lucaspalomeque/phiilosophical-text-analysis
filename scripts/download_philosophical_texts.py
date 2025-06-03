#!/usr/bin/env python3
"""
Download philosophical texts from Project Gutenberg and other sources.

This script downloads a curated collection of philosophical texts for analysis.
"""

import requests
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin
import click
from bs4 import BeautifulSoup


class PhilosophicalTextDownloader:
    """Downloads and processes philosophical texts."""
    
    def __init__(self, output_dir: str = "data/raw", delay: float = 1.0):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save downloaded texts
            delay: Delay between downloads (seconds) to be respectful
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PhilosophicalTextAnalysis/1.0 (Educational Research)'
        })
    
    def clean_gutenberg_text(self, text: str, title: str) -> str:
        """
        Clean Project Gutenberg text by removing headers, footers, and metadata.
        
        Args:
            text: Raw text from Gutenberg
            title: Title for logging
            
        Returns:
            Cleaned text content
        """
        lines = text.split('\n')
        
        # Find start and end of actual content
        start_idx = 0
        end_idx = len(lines)
        
        # Common Gutenberg start markers
        start_markers = [
            'START OF THE PROJECT GUTENBERG',
            'START OF THIS PROJECT GUTENBERG',
            '*** START OF THE PROJECT GUTENBERG',
            'PREFACE',
            'INTRODUCTION',
            'CHAPTER I',
            'BOOK I',
            'PART I'
        ]
        
        # Common Gutenberg end markers
        end_markers = [
            'END OF THE PROJECT GUTENBERG',
            'END OF THIS PROJECT GUTENBERG',
            '*** END OF THE PROJECT GUTENBERG',
            'TRANSCRIBER\'S NOTE',
            'End of Project Gutenberg'
        ]
        
        # Find start
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            if any(marker in line_upper for marker in start_markers):
                start_idx = i + 1
                break
        
        # Find end
        for i in range(len(lines) - 1, -1, -1):
            line_upper = lines[i].upper().strip()
            if any(marker in line_upper for marker in end_markers):
                end_idx = i
                break
        
        # Extract content
        content_lines = lines[start_idx:end_idx]
        
        # Remove empty lines and page markers
        cleaned_lines = []
        for line in content_lines:
            line = line.strip()
            # Skip empty lines and page numbers
            if line and not re.match(r'^\d+$', line) and not re.match(r'^Page \d+', line):
                cleaned_lines.append(line)
        
        # Join and clean up spacing
        content = '\n'.join(cleaned_lines)
        content = re.sub(r'\n{3,}', '\n\n', content)  # Max 2 consecutive newlines
        content = re.sub(r' {2,}', ' ', content)       # Max 1 space
        
        print(f"📝 Cleaned {title}: {len(content)} characters")
        return content.strip()
    
    def download_text(self, url: str, filename: str, title: str = "") -> bool:
        """
        Download a single text from URL.
        
        Args:
            url: URL to download from
            filename: Local filename to save as
            title: Title for logging
            
        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"📥 Downloading {title or filename}...")
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Detect encoding
            if response.encoding == 'ISO-8859-1':
                response.encoding = 'utf-8'
            
            # Clean the text
            clean_text = self.clean_gutenberg_text(response.text, title)
            
            if len(clean_text) < 1000:  # Sanity check
                print(f"⚠️  Warning: {title} seems too short ({len(clean_text)} chars)")
                return False
            
            # Save to file
            file_path = self.output_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(clean_text)
            
            print(f"✅ Saved {title}: {len(clean_text)} characters to {filename}")
            
            # Be respectful to the server
            time.sleep(self.delay)
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {title}: {e}")
            return False
    
    def download_collection(self, collection: Dict[str, Dict[str, str]]) -> Dict[str, bool]:
        """
        Download a collection of texts.
        
        Args:
            collection: Dictionary with format:
                {
                    "philosopher_name": {
                        "title": "Work Title",
                        "url": "download_url",
                        "filename": "local_filename.txt"
                    }
                }
        
        Returns:
            Dictionary of download results
        """
        results = {}
        
        print(f"📚 Downloading {len(collection)} philosophical texts...")
        print("=" * 60)
        
        for philosopher, info in collection.items():
            title = f"{philosopher} - {info['title']}"
            success = self.download_text(
                info['url'], 
                info['filename'], 
                title
            )
            results[philosopher] = success
            
            if success:
                print(f"🎉 {title} ✅")
            else:
                print(f"💥 {title} ❌")
            
            print("-" * 40)
        
        return results


# Curated collection of philosophical texts
PHILOSOPHICAL_TEXTS = {
    # URLs VERIFICADAS Y FUNCIONANDO ✅
    
    # Kant - FUNCIONA
    "kant_critique_pure_reason": {
        "title": "Critique of Pure Reason",
        "url": "https://www.gutenberg.org/files/4280/4280-0.txt",
        "filename": "kant_critique_pure_reason.txt"
    },
    
    # Descartes - FUNCIONA
    "descartes_discourse_method": {
        "title": "Discourse on the Method",
        "url": "https://www.gutenberg.org/files/59/59-0.txt",
        "filename": "descartes_discourse_method.txt"
    },
    
    # Hume - URL CORREGIDA
    "hume_human_understanding": {
        "title": "An Enquiry Concerning Human Understanding",
        "url": "https://www.gutenberg.org/cache/epub/9662/pg9662.txt",
        "filename": "hume_human_understanding.txt"
    },
    
    # Nietzsche - URL CORREGIDA
    "nietzsche_beyond_good_evil": {
        "title": "Beyond Good and Evil",
        "url": "https://www.gutenberg.org/cache/epub/4363/pg4363.txt",
        "filename": "nietzsche_beyond_good_evil.txt"
    },
    
    # Spinoza - FUNCIONA
    "spinoza_ethics": {
        "title": "Ethics",
        "url": "https://www.gutenberg.org/files/3800/3800-0.txt",
        "filename": "spinoza_ethics.txt"
    },
    
    # Mill - URL CORREGIDA
    "mill_on_liberty": {
        "title": "On Liberty",
        "url": "https://www.gutenberg.org/cache/epub/34901/pg34901.txt",
        "filename": "mill_on_liberty.txt"
    },
    
    # Locke - URL CORREGIDA
    "locke_human_understanding": {
        "title": "An Essay Concerning Human Understanding",
        "url": "https://www.gutenberg.org/cache/epub/10615/pg10615.txt",
        "filename": "locke_human_understanding.txt"
    },
    
    # Plato - FUNCIONA
    "plato_republic": {
        "title": "The Republic",
        "url": "https://www.gutenberg.org/files/1497/1497-0.txt",
        "filename": "plato_republic.txt"
    },
    
    # Aristóteles - URL CORREGIDA
    "aristotle_nicomachean_ethics": {
        "title": "The Nicomachean Ethics",
        "url": "https://www.gutenberg.org/cache/epub/8438/pg8438.txt",
        "filename": "aristotle_nicomachean_ethics.txt"
    },
    
    # Pascal - URL CORREGIDA
    "pascal_pensees": {
        "title": "Pensées",
        "url": "https://www.gutenberg.org/cache/epub/18269/pg18269.txt",
        "filename": "pascal_pensees.txt"
    },
    
    # Berkeley - FUNCIONA
    "berkeley_human_knowledge": {
        "title": "A Treatise Concerning the Principles of Human Knowledge",
        "url": "https://www.gutenberg.org/files/4723/4723-0.txt",
        "filename": "berkeley_human_knowledge.txt"
    },
    
    # Bentham - URL CORREGIDA
    "bentham_principles_morals": {
        "title": "An Introduction to the Principles of Morals and Legislation",
        "url": "https://www.gutenberg.org/cache/epub/44095/pg44095.txt",
        "filename": "bentham_principles_morals.txt"
    },
    
    # ALTERNATIVAS ADICIONALES - URLs de respaldo
    
    # Hobbes
    "hobbes_leviathan": {
        "title": "Leviathan",
        "url": "https://www.gutenberg.org/files/3207/3207-0.txt",
        "filename": "hobbes_leviathan.txt"
    },
    
    # Rousseau
    "rousseau_social_contract": {
        "title": "The Social Contract",
        "url": "https://www.gutenberg.org/files/46333/46333-0.txt",
        "filename": "rousseau_social_contract.txt"
    },
    
    # William James
    "james_pragmatism": {
        "title": "Pragmatism: A New Name for Some Old Ways of Thinking",
        "url": "https://www.gutenberg.org/files/5116/5116-0.txt",
        "filename": "james_pragmatism.txt"
    }
}


@click.command()
@click.option(
    '--output-dir', '-o',
    default='data/raw',
    help='Output directory for downloaded texts'
)
@click.option(
    '--delay', '-d',
    default=1.0,
    help='Delay between downloads (seconds)'
)
@click.option(
    '--philosophers', '-p',
    help='Comma-separated list of specific philosophers to download'
)
@click.option(
    '--sample', '-s',
    is_flag=True,
    help='Download only a small sample (3 texts) for testing'
)
def main(output_dir: str, delay: float, philosophers: Optional[str], sample: bool):
    """Download philosophical texts for analysis."""
    
    print("📚 Philosophical Text Downloader")
    print("=" * 50)
    
    # Initialize downloader
    downloader = PhilosophicalTextDownloader(output_dir, delay)
    
    # Select texts to download
    if sample:
        # Sample: one from each major category
        selected_texts = {
            "kant_critique_pure_reason": PHILOSOPHICAL_TEXTS["kant_critique_pure_reason"],
            "hume_human_understanding": PHILOSOPHICAL_TEXTS["hume_human_understanding"],
            "nietzsche_beyond_good_evil": PHILOSOPHICAL_TEXTS["nietzsche_beyond_good_evil"]
        }
        print("🧪 Sample mode: downloading 3 representative texts")
    elif philosophers:
        # Specific philosophers
        philosopher_list = [p.strip() for p in philosophers.split(',')]
        selected_texts = {
            key: value for key, value in PHILOSOPHICAL_TEXTS.items()
            if any(phil.lower() in key.lower() for phil in philosopher_list)
        }
        if not selected_texts:
            print(f"❌ No texts found for philosophers: {philosophers}")
            return
        print(f"🎯 Downloading texts for: {', '.join(philosopher_list)}")
    else:
        # All texts
        selected_texts = PHILOSOPHICAL_TEXTS
        print(f"📖 Downloading all {len(selected_texts)} texts")
    
    print(f"💾 Output directory: {output_dir}")
    print(f"⏱️  Delay between downloads: {delay}s")
    print()
    
    # Download texts
    results = downloader.download_collection(selected_texts)
    
    # Summary
    successful = sum(results.values())
    total = len(results)
    
    print("\n" + "=" * 60)
    print("📊 Download Summary:")
    print(f"✅ Successful: {successful}/{total}")
    
    if successful < total:
        print(f"❌ Failed: {total - successful}")
        print("\nFailed downloads:")
        for name, success in results.items():
            if not success:
                print(f"  • {name}")
    
    if successful > 0:
        print(f"\n📁 Files saved to: {output_dir}")
        print("\n🚀 Next steps:")
        print("  1. Run analysis: philo-analyze batch --input-dir data/raw --output results.csv")
        print("  2. Compare philosophers: python scripts/compare_philosophers.py")
        print("  3. Visualize results: python scripts/visualize_coherence.py")


if __name__ == "__main__":
    main()