#!/usr/bin/env python3
"""
Analyze philosophical texts by school of thought.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import click

from philosophical_analysis.core.analyzer import PhilosophicalAnalyzer


class PhilosophicalSchoolAnalyzer:
    """Analyzes patterns across philosophical schools."""
    
    def __init__(self):
        self.analyzer = PhilosophicalAnalyzer()
        self.school_definitions = {
            "Rationalist": {
                "philosophers": ["kant", "descartes", "spinoza"],
                "description": "Emphasize reason and systematic logic",
                "expected_coherence": "High",
                "expected_range": (0.65, 0.85)
            },
            "Empiricist": {
                "philosophers": ["hume", "locke", "berkeley"],
                "description": "Focus on sensory experience", 
                "expected_coherence": "Medium-High",
                "expected_range": (0.55, 0.75)
            },
            "Existentialist": {
                "philosophers": ["nietzsche", "kierkegaard"],
                "description": "Individual existence and authenticity",
                "expected_coherence": "Variable",
                "expected_range": (0.35, 0.65)
            },
            "Classical": {
                "philosophers": ["plato", "aristotle"],
                "description": "Foundational systematic philosophy",
                "expected_coherence": "Medium-High",
                "expected_range": (0.60, 0.80)
            }
        }
    
    def categorize_text(self, text_id: str) -> str:
        """Categorize text by philosophical school."""
        text_id_lower = text_id.lower()
        
        for school, info in self.school_definitions.items():
            if any(phil in text_id_lower for phil in info["philosophers"]):
                return school
                
        return "Other"
    
    def extract_philosopher(self, text_id: str) -> str:
        """Extract philosopher name."""
        return text_id.split('_')[0].title()
    
    def load_and_categorize_texts(self, text_dir: str) -> pd.DataFrame:
        """Load texts and categorize by school."""
        
        text_path = Path(text_dir)
        text_files = list(text_path.glob("*.txt"))
        
        print(f"📚 Found {len(text_files)} texts to analyze")
        
        # Load texts
        texts = {}
        metadata = []
        
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                if len(content) > 1000:
                    text_id = file_path.stem
                    texts[text_id] = content
                    
                    # Categorize by school
                    school = self.categorize_text(text_id)
                    philosopher = self.extract_philosopher(text_id)
                    
                    metadata.append({
                        'text_id': text_id,
                        'philosopher': philosopher,
                        'school': school,
                        'char_count': len(content)
                    })
                    
                    print(f"📖 {philosopher} ({school})")
                    
            except Exception as e:
                print(f"❌ Error loading {file_path.stem}: {e}")
        
        # Fit analyzer and analyze
        print(f"\n🧠 Training analyzer on {len(texts)} texts...")
        self.analyzer.fit(texts)
        
        print("📊 Analyzing texts...")
        results = self.analyzer.analyze_multiple_texts(texts)
        
        # Merge with metadata
        metadata_df = pd.DataFrame(metadata)
        results = results.merge(metadata_df, on='text_id', how='left')
        
        return results
    
    def test_school_hypotheses(self, results: pd.DataFrame) -> Dict:
        """Test hypotheses about different philosophical schools."""
        
        print("\n🔬 Testing Philosophical School Hypotheses")
        print("=" * 50)
        
        hypothesis_results = {}
        
        # Test each school's expected coherence
        for school, expected in self.school_definitions.items():
            school_data = results[results['school'] == school]
            if len(school_data) > 0:
                mean_coherence = school_data['semantic_coherence'].mean()
                expected_min, expected_max = expected["expected_range"]
                
                within_range = expected_min <= mean_coherence <= expected_max
                
                hypothesis_results[school] = {
                    'mean_coherence': mean_coherence,
                    'count': len(school_data),
                    'expected_range': expected["expected_range"],
                    'within_expected': within_range,
                    'description': expected["description"]
                }
                
                status = "✅ CONFIRMED" if within_range else "❌ REJECTED"
                print(f"{school:15} | {mean_coherence:.3f} | {expected['expected_range']} | {status}")
        
        return hypothesis_results


@click.command()
@click.option('--input-dir', '-i', default='data/raw')
@click.option('--output-dir', '-o', default='reports')
@click.option('--test-hypotheses', is_flag=True)
@click.option('--visualize', is_flag=True)
def main(input_dir: str, output_dir: str, test_hypotheses: bool, visualize: bool):
    """Analyze philosophical texts by school of thought."""
    
    print("🏛️  Philosophical School Analysis")
    print("=" * 50)
    
    try:
        analyzer = PhilosophicalSchoolAnalyzer()
        
        # Load and analyze texts
        results = analyzer.load_and_categorize_texts(input_dir)
        
        # Save results
        Path(output_dir).mkdir(exist_ok=True)
        results.to_csv(f"{output_dir}/school_analysis_results.csv", index=False)
        
        # Basic summary
        print(f"\n📊 Analysis Summary:")
        school_summary = results.groupby('school')['semantic_coherence'].agg(['count', 'mean', 'std'])
        print(school_summary.round(3))
        
        # Test hypotheses if requested
        if test_hypotheses:
            hypothesis_results = analyzer.test_school_hypotheses(results)
        
        print(f"\n🎉 School analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()