#!/usr/bin/env python3
"""
Download philosophical texts from Project Gutenberg and other sources.

This script downloads a curated collection of philosophical texts for analysis.
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Any, List
import click
import requests


def download_text(text_info: Dict[str, Any], output_dir: Path, delay: float = 1.0) -> bool:
    """Download a single philosophical text."""
    url = text_info['url']
    filename = text_info['filename']
    filepath = output_dir / filename
    
    # Skip if already exists
    if filepath.exists():
        print(f"⏭️  Skipping {filename} (already exists)")
        return True
    
    try:
        print(f"📥 Downloading {text_info['author']}: {text_info['title']}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; PhilosophicalTextAnalyzer/1.0)'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Save the text
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"✅ Downloaded: {filename} ({len(response.text):,} chars)")
        
        # Be respectful to the server
        time.sleep(delay)
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def create_metadata_file(output_dir: Path, texts: Dict[str, Any]) -> None:
    """Create a metadata file with information about all texts."""
    metadata_path = output_dir / "philosophical_texts_metadata.json"
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Created metadata file: {metadata_path}")

def create_readme(output_dir: Path, texts: Dict[str, Any]) -> None:
    """Create a README file for the texts collection."""
    readme_path = output_dir / "README.md"
    
    # Group by school/tradition
    schools: Dict[str, List[Dict[str, Any]]] = {}
    for info in texts.values():
        school = info['school']
        if school not in schools:
            schools[school] = []
        schools[school].append(info)
    
    readme_content = """# Philosophical Texts Collection

Esta colección incluye textos filosóficos clásicos organizados por escuelas y tradiciones.

## 📚 Textos Incluidos

"""
    
    for school, text_list in schools.items():
        readme_content += f"### {school}\n\n"
        for text in sorted(text_list, key=lambda x: x['author']):
            readme_content += f"- **{text['author']}**: _{text['title']}_ ({text['period']})\n"
        readme_content += "\n"
    
    readme_content += """
## 🔍 Análisis Disponibles

Estos textos pueden ser analizados usando el framework de análisis filosófico con:

- Análisis de coherencia semántica (LSA)
- Análisis de sentimientos
- Extracción de conceptos clave
- Comparación entre tradiciones filosóficas
- Visualización de redes conceptuales

## 📖 Fuentes

Todos los textos provienen de Project Gutenberg y están en dominio público.
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📖 Created README: {readme_path}")

# Comprehensive collection of philosophical texts
PHILOSOPHICAL_TEXTS = {
    # ===================
    # FILOSOFÍA GRIEGA CLÁSICA
    # ===================
    
    # Platón
    "plato_republic": {
        "title": "The Republic",
        "author": "Plato",
        "school": "Ancient Greek",
        "period": "Classical",
        "url": "https://www.gutenberg.org/files/1497/1497-0.txt",
        "filename": "plato_republic.txt"
    },
    
    "plato_phaedo": {
        "title": "Phaedo",
        "author": "Plato", 
        "school": "Ancient Greek",
        "period": "Classical",
        "url": "https://www.gutenberg.org/files/1658/1658-0.txt",
        "filename": "plato_phaedo.txt"
    },
    
    "plato_apology": {
        "title": "Apology",
        "author": "Plato",
        "school": "Ancient Greek", 
        "period": "Classical",
        "url": "https://www.gutenberg.org/files/1656/1656-0.txt",
        "filename": "plato_apology.txt"
    },
    
    # Aristóteles
    "aristotle_nicomachean_ethics": {
        "title": "The Nicomachean Ethics",
        "author": "Aristotle",
        "school": "Ancient Greek",
        "period": "Classical",
        "url": "https://www.gutenberg.org/files/8438/8438-0.txt",
        "filename": "aristotle_nicomachean_ethics.txt"
    },
    
    "aristotle_politics": {
        "title": "Politics",
        "author": "Aristotle",
        "school": "Ancient Greek",
        "period": "Classical", 
        "url": "https://www.gutenberg.org/files/6762/6762-0.txt",
        "filename": "aristotle_politics.txt"
    },
    
    "aristotle_metaphysics": {
        "title": "Metaphysics",
        "author": "Aristotle",
        "school": "Ancient Greek",
        "period": "Classical",
        "url": "https://www.gutenberg.org/files/1974/1974-0.txt",
        "filename": "aristotle_metaphysics.txt"
    },
    
    # Epicuro y otros
    "epictetus_discourses": {
        "title": "The Discourses",
        "author": "Epictetus",
        "school": "Stoic",
        "period": "Hellenistic",
        "url": "https://www.gutenberg.org/files/871/871-0.txt",
        "filename": "epictetus_discourses.txt"
    },
    
    # ===================
    # FILOSOFÍA MEDIEVAL (CATÓLICA/CRISTIANA)
    # ===================
    
    "aquinas_summa_theologica": {
        "title": "Summa Theologica (Excerpt)",
        "author": "Thomas Aquinas",
        "school": "Scholastic",
        "period": "Medieval",
        "url": "https://www.gutenberg.org/files/17611/17611-0.txt",
        "filename": "aquinas_summa_theologica.txt"
    },
    
    "augustine_confessions": {
        "title": "The Confessions",
        "author": "Augustine of Hippo", 
        "school": "Patristic",
        "period": "Late Antiquity",
        "url": "https://www.gutenberg.org/files/3296/3296-0.txt",
        "filename": "augustine_confessions.txt"
    },
    
    "augustine_city_of_god": {
        "title": "The City of God",
        "author": "Augustine of Hippo",
        "school": "Patristic", 
        "period": "Late Antiquity",
        "url": "https://www.gutenberg.org/files/45304/45304-0.txt",
        "filename": "augustine_city_of_god.txt"
    },
    
    "boethius_consolation": {
        "title": "The Consolation of Philosophy",
        "author": "Boethius",
        "school": "Late Roman",
        "period": "Medieval",
        "url": "https://www.gutenberg.org/files/14328/14328-0.txt",
        "filename": "boethius_consolation.txt"
    },
    
    # ===================
    # FILOSOFÍA JUDÍA/HEBREA
    # ===================
    
    "maimonides_guide_perplexed": {
        "title": "The Guide for the Perplexed",
        "author": "Moses Maimonides",
        "school": "Jewish Philosophy",
        "period": "Medieval",
        "url": "https://www.gutenberg.org/cache/epub/19022/pg19022.txt",
        "filename": "maimonides_guide_perplexed.txt"
    },
    
    "spinoza_ethics": {
        "title": "Ethics",
        "author": "Baruch Spinoza",
        "school": "Rationalist",
        "period": "Modern",
        "url": "https://www.gutenberg.org/files/3800/3800-0.txt",
        "filename": "spinoza_ethics.txt"
    },
    
    "spinoza_tractatus": {
        "title": "A Theologico-Political Treatise",
        "author": "Baruch Spinoza",
        "school": "Rationalist",
        "period": "Modern", 
        "url": "https://www.gutenberg.org/files/989/989-0.txt",
        "filename": "spinoza_tractatus.txt"
    },
    
    # ===================
    # FILOSOFÍA ALEMANA
    # ===================
    
    # Kant
    "kant_critique_pure_reason": {
        "title": "The Critique of Pure Reason",
        "author": "Immanuel Kant",
        "school": "German Idealism",
        "period": "Enlightenment",
        "url": "https://www.gutenberg.org/files/4280/4280-0.txt",
        "filename": "kant_critique_pure_reason.txt"
    },
    
    "kant_critique_practical_reason": {
        "title": "The Critique of Practical Reason",
        "author": "Immanuel Kant",
        "school": "German Idealism",
        "period": "Enlightenment",
        "url": "https://www.gutenberg.org/files/5683/5683-0.txt",
        "filename": "kant_critique_practical_reason.txt"
    },
    
    "kant_groundwork_metaphysics": {
        "title": "Fundamental Principles of the Metaphysics of Morals",
        "author": "Immanuel Kant",
        "school": "German Idealism", 
        "period": "Enlightenment",
        "url": "https://www.gutenberg.org/files/5682/5682-0.txt",
        "filename": "kant_groundwork_metaphysics.txt"
    },
    
    # Hegel
    "hegel_phenomenology": {
        "title": "The Phenomenology of Mind",
        "author": "Georg Wilhelm Friedrich Hegel",
        "school": "German Idealism",
        "period": "Romantic",
        "url": "https://www.gutenberg.org/files/26576/26576-0.txt",
        "filename": "hegel_phenomenology.txt"
    },
    
    # Nietzsche
    "nietzsche_beyond_good_evil": {
        "title": "Beyond Good and Evil",
        "author": "Friedrich Nietzsche",
        "school": "Existentialist",
        "period": "Modern",
        "url": "https://www.gutenberg.org/files/4363/4363-0.txt",
        "filename": "nietzsche_beyond_good_evil.txt"
    },
    
    "nietzsche_genealogy_morals": {
        "title": "The Genealogy of Morals",
        "author": "Friedrich Nietzsche",
        "school": "Existentialist",
        "period": "Modern",
        "url": "https://www.gutenberg.org/files/52319/52319-0.txt",
        "filename": "nietzsche_genealogy_morals.txt"
    },
    
    "nietzsche_gay_science": {
        "title": "The Gay Science",
        "author": "Friedrich Nietzsche", 
        "school": "Existentialist",
        "period": "Modern",
        "url": "https://www.gutenberg.org/files/52881/52881-0.txt",
        "filename": "nietzsche_gay_science.txt"
    },
    
    # Schopenhauer
    "schopenhauer_world_will": {
        "title": "The World as Will and Idea",
        "author": "Arthur Schopenhauer",
        "school": "German Pessimism",
        "period": "Romantic",
        "url": "https://www.gutenberg.org/files/40097/40097-0.txt",
        "filename": "schopenhauer_world_will.txt"
    },
    
    # ===================
    # FILOSOFÍA MODERNA/CONTEMPORÁNEA
    # ===================
    
    # Empiristas Británicos
    "locke_human_understanding": {
        "title": "An Essay Concerning Human Understanding",
        "author": "John Locke",
        "school": "British Empiricism",
        "period": "Enlightenment",
        "url": "https://www.gutenberg.org/files/10615/10615-0.txt",
        "filename": "locke_human_understanding.txt"
    },
    
    "hume_human_understanding": {
        "title": "An Enquiry Concerning Human Understanding", 
        "author": "David Hume",
        "school": "British Empiricism",
        "period": "Enlightenment",
        "url": "https://www.gutenberg.org/files/9662/9662-0.txt",
        "filename": "hume_human_understanding.txt"
    },
    
    "berkeley_human_knowledge": {
        "title": "A Treatise Concerning the Principles of Human Knowledge",
        "author": "George Berkeley",
        "school": "British Empiricism",
        "period": "Enlightenment", 
        "url": "https://www.gutenberg.org/files/4723/4723-0.txt",
        "filename": "berkeley_human_knowledge.txt"
    },
    
    # Racionalistas
    "descartes_discourse_method": {
        "title": "Discourse on the Method",
        "author": "René Descartes",
        "school": "Continental Rationalism",
        "period": "Early Modern",
        "url": "https://www.gutenberg.org/files/59/59-0.txt",
        "filename": "descartes_discourse_method.txt"
    },
    
    "descartes_meditations": {
        "title": "Meditations on First Philosophy",
        "author": "René Descartes",
        "school": "Continental Rationalism", 
        "period": "Early Modern",
        "url": "https://www.gutenberg.org/files/59/59-0.txt",
        "filename": "descartes_meditations.txt"
    },
    
    # Utilitaristas
    "mill_on_liberty": {
        "title": "On Liberty",
        "author": "John Stuart Mill",
        "school": "Utilitarianism",
        "period": "Victorian",
        "url": "https://www.gutenberg.org/files/34901/34901-0.txt",
        "filename": "mill_on_liberty.txt"
    },
    
    "bentham_principles_morals": {
        "title": "An Introduction to the Principles of Morals and Legislation",
        "author": "Jeremy Bentham",
        "school": "Utilitarianism",
        "period": "Enlightenment",
        "url": "https://www.gutenberg.org/files/44095/44095-0.txt",
        "filename": "bentham_principles_morals.txt"
    },
    
    # Pragmatistas Americanos
    "james_pragmatism": {
        "title": "Pragmatism: A New Name for Some Old Ways of Thinking",
        "author": "William James",
        "school": "American Pragmatism",
        "period": "Modern",
        "url": "https://www.gutenberg.org/cache/epub/5116/pg5116.txt",
        "filename": "james_pragmatism.txt"
    },
    
    "peirce_chance_love_logic": {
        "title": "Chance, Love, and Logic",
        "author": "Charles Sanders Peirce",
        "school": "American Pragmatism",
        "period": "Modern",
        "url": "https://www.gutenberg.org/files/49648/49648-0.txt",
        "filename": "peirce_chance_love_logic.txt"
    },
    
    # Contractualistas
    "hobbes_leviathan": {
        "title": "Leviathan",
        "author": "Thomas Hobbes",
        "school": "Social Contract",
        "period": "Early Modern",
        "url": "https://www.gutenberg.org/files/3207/3207-0.txt",
        "filename": "hobbes_leviathan.txt"
    },
    
    "rousseau_social_contract": {
        "title": "The Social Contract",
        "author": "Jean-Jacques Rousseau",
        "school": "Social Contract",
        "period": "Enlightenment",
        "url": "https://www.gutenberg.org/files/46333/46333-0.txt",
        "filename": "rousseau_social_contract.txt"
    },
    
    # Otros modernos importantes
    "pascal_pensees": {
        "title": "Pensées",
        "author": "Blaise Pascal",
        "school": "Christian Existentialism",
        "period": "Early Modern",
        "url": "https://www.gutenberg.org/files/18269/18269-0.txt",
        "filename": "pascal_pensees.txt"
    },
    
    "montaigne_essays": {
        "title": "Essays of Montaigne",
        "author": "Michel de Montaigne",
        "school": "Renaissance Humanism",
        "period": "Renaissance",
        "url": "https://www.gutenberg.org/files/3600/3600-0.txt",
        "filename": "montaigne_essays.txt"
    },
    
    # Aristóteles
    "aristotle_nicomachean_ethics": {
        "title": "The Nicomachean Ethics",
        "author": "Aristotle",
        "school": "Ancient Greek",
        "period": "Classical",
        "url": "https://www.gutenberg.org/cache/epub/8438/pg8438.txt",
        "filename": "aristotle_nicomachean_ethics.txt"
    },
    
    # Pascal
    "pascal_pensees": {
        "title": "Pensées",
        "author": "Blaise Pascal",
        "school": "Christian Philosophy",
        "period": "Early Modern",
        "url": "https://www.gutenberg.org/cache/epub/18269/pg18269.txt",
        "filename": "pascal_pensees.txt"
    },
    
    # Berkeley
    "berkeley_human_knowledge": {
        "title": "A Treatise Concerning the Principles of Human Knowledge",
        "author": "George Berkeley",
        "school": "Empiricism",
        "period": "Early Modern",
        "url": "https://www.gutenberg.org/files/4723/4723-0.txt",
        "filename": "berkeley_human_knowledge.txt"
    },
    
    # Bentham
    "bentham_principles_morals": {
        "title": "An Introduction to the Principles of Morals and Legislation",
        "author": "Jeremy Bentham",
        "school": "Utilitarianism",
        "period": "Modern",
        "url": "https://www.gutenberg.org/cache/epub/44095/pg44095.txt",
        "filename": "bentham_principles_morals.txt"
    },
    
    # Hobbes
    "hobbes_leviathan": {
        "title": "Leviathan",
        "author": "Thomas Hobbes",
        "school": "Social Contract Theory",
        "period": "Early Modern",
        "url": "https://www.gutenberg.org/files/3207/3207-0.txt",
        "filename": "hobbes_leviathan.txt"
    },
    
    # Rousseau
    "rousseau_social_contract": {
        "title": "The Social Contract",
        "author": "Jean-Jacques Rousseau",
        "school": "Social Contract Theory",
        "period": "Enlightenment",
        "url": "https://www.gutenberg.org/files/46333/46333-0.txt",
        "filename": "rousseau_social_contract.txt"
    }
}


@click.command()
@click.option('--output-dir', '-o', default='data/raw', help='Output directory for downloaded texts')
@click.option('--delay', '-d', default=1.0, help='Delay between downloads (seconds)')
@click.option('--sample', is_flag=True, help='Download only a small sample for testing')
@click.option('--schools', help='Comma-separated list of philosophical schools to download')
@click.option('--authors', help='Comma-separated list of specific authors to download')
@click.option('--force', is_flag=True, help='Re-download files that already exist')
def main(output_dir: str, delay: float, sample: bool, schools: Optional[str], authors: Optional[str], force: bool):
    """Download philosophical texts for analysis."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("🏛️  Philosophical Text Downloader")
    print("="*50)
    
    # Filter texts based on options
    texts_to_download = {}
    
    if sample:
        # Download a representative sample
        sample_keys = [
            'plato_republic',          # Greek
            'aquinas_summa_theologica', # Catholic 
            'maimonides_guide_perplexed', # Jewish
            'kant_critique_pure_reason',  # German
            'james_pragmatism'         # Modern American
        ]
        texts_to_download = {k: v for k, v in PHILOSOPHICAL_TEXTS.items() if k in sample_keys}
        print("📋 Sample mode: downloading 5 representative texts")
        
    elif schools:
        school_list = [s.strip() for s in schools.split(',')]
        texts_to_download = {k: v for k, v in PHILOSOPHICAL_TEXTS.items() 
                           if v['school'] in school_list}
        print(f"📋 Downloading texts from schools: {school_list}")
        
    elif authors:
        author_list = [a.strip() for a in authors.split(',')]
        texts_to_download = {k: v for k, v in PHILOSOPHICAL_TEXTS.items() 
                           if v['author'] in author_list}
        print(f"📋 Downloading texts from authors: {author_list}")
        
    else:
        texts_to_download = PHILOSOPHICAL_TEXTS
        print("📋 Downloading all available texts")
    
    if force:
        print("🔄 Force mode: will re-download existing files")
    
    print(f"📁 Output directory: {output_path}")
    print(f"⏱️  Delay between downloads: {delay}s")
    print(f"📊 Total texts to download: {len(texts_to_download)}")
    print()
    
    # Download texts
    successful = 0
    failed = 0
    
    for text_id, text_info in texts_to_download.items():
        if force and (output_path / text_info['filename']).exists():
            (output_path / text_info['filename']).unlink()
        
        if download_text(text_info, output_path, delay):
            successful += 1
        else:
            failed += 1
    
    # Create metadata and README
    create_metadata_file(output_path, texts_to_download)
    create_readme(output_path, texts_to_download)
    
    print()
    print("="*50)
    print(f"✅ Successfully downloaded: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Files saved to: {output_path}")
    
    if successful > 0:
        print()
        print("🔬 Next steps:")
        print("1. Run analysis: python -m philosophical_analysis.cli analyze")
        print("2. Generate visualizations: python scripts/generate_visualizations.py")
        print("3. Explore notebooks: jupyter notebook notebooks/")


if __name__ == "__main__":
    main()