#!/usr/bin/env python3
"""
Compare philosophical texts and analyze patterns across different thinkers.

This script analyzes the coherence patterns of different philosophical schools
and individual philosophers.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import click

from philosophical_analysis.core.integrated_analyzer import IntegratedPhilosophicalAnalyzer


class PhilosopherComparator:
    """Analyzes and compares philosophical texts."""

    def __init__(self):
        self.analyzer = IntegratedPhilosophicalAnalyzer()
        self.results = None
        
        # Define philosophical schools for grouping
        self.schools = {
            "Rationalist": ["kant", "descartes", "spinoza"],
            "Empiricist": ["hume", "locke", "berkeley"],
            "German Idealist": ["hegel"],
            "Utilitarian": ["mill", "bentham"],
            "Existentialist": ["nietzsche", "pascal"],
            "Classical": ["plato", "aristotle"]
        }
    
    def categorize_philosopher(self, filename: str) -> str:
        """Categorize philosopher by school of thought."""
        filename_lower = filename.lower()
        
        for school, philosophers in self.schools.items():
            if any(phil in filename_lower for phil in philosophers):
                return school
        
        return "Other"
    
    def load_and_analyze_texts(self, text_dir: str) -> pd.DataFrame:
        """Load and analyze all texts in directory."""
        
        text_path = Path(text_dir)
        text_files = list(text_path.glob("*.txt"))
        
        if not text_files:
            raise ValueError(f"No .txt files found in {text_dir}")
        
        print(f"📚 Found {len(text_files)} philosophical texts")
        
        # Load texts
        texts = {}
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if len(content) > 1000:  # Minimum length check
                        texts[file_path.stem] = content
                        print(f"📖 Loaded: {file_path.stem} ({len(content):,} chars)")
                    else:
                        print(f"⚠️  Skipped {file_path.stem}: too short")
            except Exception as e:
                print(f"❌ Error loading {file_path.stem}: {e}")
        
        if not texts:
            raise ValueError("No valid texts loaded")
        
        print(f"\n🧠 Training analyzer on {len(texts)} texts...")
        self.analyzer.fit(texts)
        
        print(f"📊 Analyzing texts...")
        results_df = self.analyzer.analyze_multiple_texts(texts)
        
        # Add metadata
        results_df['school'] = results_df['text_id'].apply(self.categorize_philosopher)
        results_df['philosopher'] = results_df['text_id'].apply(self.extract_philosopher_name)
        results_df['work'] = results_df['text_id'].apply(self.extract_work_name)
        
        self.results = results_df
        return results_df
    
    def extract_philosopher_name(self, text_id: str) -> str:
        """Extract philosopher name from text ID."""
        # Simple extraction - first part before underscore
        parts = text_id.split('_')
        if parts:
            return parts[0].title()
        return text_id
    
    def extract_work_name(self, text_id: str) -> str:
        """Extract work name from text ID."""
        # Remove philosopher name and convert to title
        parts = text_id.split('_')[1:]
        if parts:
            return ' '.join(parts).replace('_', ' ').title()
        return text_id
    
    def generate_summary_statistics(self) -> Dict:
        """Generate summary statistics."""
        if self.results is None:
            raise ValueError("No results available. Run analysis first.")
        
        stats = {}
        
        # Overall statistics
        stats['total_texts'] = len(self.results)
        stats['avg_coherence'] = self.results['semantic_coherence'].mean()
        stats['coherence_std'] = self.results['semantic_coherence'].std()
        
        # By school
        school_stats = self.results.groupby('school').agg({
            'semantic_coherence': ['mean', 'std', 'count'],
            'min_coherence': 'mean',
            'sentence_count': 'mean',
            'word_count': 'mean'
        }).round(3)
        
        stats['by_school'] = school_stats
        
        # Top and bottom coherence
        sorted_results = self.results.sort_values('semantic_coherence', ascending=False)
        stats['highest_coherence'] = sorted_results.head(3)[['text_id', 'semantic_coherence']].to_dict('records')
        stats['lowest_coherence'] = sorted_results.tail(3)[['text_id', 'semantic_coherence']].to_dict('records')
        
        return stats
    
    def create_visualizations(self, output_dir: str = "reports"):
        """Create visualizations of the analysis."""
        if self.results is None:
            raise ValueError("No results available. Run analysis first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Coherence by school
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        sns.boxplot(data=self.results, x='school', y='semantic_coherence')
        plt.title('Semantic Coherence by Philosophical School')
        plt.xticks(rotation=45)
        plt.ylabel('Semantic Coherence')
        
        # 2. Individual philosophers
        plt.subplot(2, 2, 2)
        top_philosophers = self.results.nlargest(8, 'semantic_coherence')
        sns.barplot(data=top_philosophers, x='semantic_coherence', y='philosopher', orient='h')
        plt.title('Top 8 Philosophers by Coherence')
        plt.xlabel('Semantic Coherence')
        
        # 3. Coherence vs Text Length
        plt.subplot(2, 2, 3)
        plt.scatter(self.results['word_count'], self.results['semantic_coherence'], 
                   c=self.results['school'].astype('category').cat.codes, alpha=0.7)
        plt.xlabel('Word Count')
        plt.ylabel('Semantic Coherence')
        plt.title('Coherence vs Text Length')
        
        # 4. Distribution of coherence
        plt.subplot(2, 2, 4)
        plt.hist(self.results['semantic_coherence'], bins=15, alpha=0.7, edgecolor='black')
        plt.axvline(self.results['semantic_coherence'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.results["semantic_coherence"].mean():.3f}')
        plt.xlabel('Semantic Coherence')
        plt.ylabel('Frequency')
        plt.title('Distribution of Semantic Coherence')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'philosophical_analysis_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Detailed school comparison
        plt.figure(figsize=(14, 6))
        
        # School comparison with error bars
        school_summary = self.results.groupby('school')['semantic_coherence'].agg(['mean', 'std', 'count'])
        school_summary = school_summary.sort_values('mean', ascending=False)
        
        plt.bar(school_summary.index, school_summary['mean'], 
               yerr=school_summary['std'], capsize=5, alpha=0.8)
        plt.title('Average Semantic Coherence by Philosophical School')
        plt.ylabel('Semantic Coherence')
        plt.xticks(rotation=45)
        
        # Add count annotations
        for i, (school, row) in enumerate(school_summary.iterrows()):
            plt.text(i, row['mean'] + row['std'] + 0.01, f'n={int(row["count"])}', 
                    ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path / 'schools_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualizations saved to {output_dir}/")
        
        return output_path
    
    def generate_report(self, output_file: str = "reports/philosophical_analysis_report.md"):
        """Generate a comprehensive markdown report."""
        if self.results is None:
            raise ValueError("No results available. Run analysis first.")
        
        stats = self.generate_summary_statistics()
        
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        report = f"""# Philosophical Text Analysis Report

## Overview

This report analyzes **{stats['total_texts']} philosophical texts** using semantic coherence analysis based on psycholinguistic research.

### Key Findings

- **Average Semantic Coherence**: {stats['avg_coherence']:.3f} ± {stats['coherence_std']:.3f}
- **Analysis Method**: LSA-based semantic vector analysis
- **Philosophical Schools Analyzed**: {len(stats['by_school'])}

## Results by Philosophical School

"""
        
        # Add school statistics
        for school, school_data in stats['by_school'].iterrows():
            coherence_mean = school_data[('semantic_coherence', 'mean')]
            coherence_std = school_data[('semantic_coherence', 'std')]
            count = int(school_data[('semantic_coherence', 'count')])
            
            report += f"""### {school}
- **Texts Analyzed**: {count}
- **Average Coherence**: {coherence_mean:.3f} ± {coherence_std:.3f}
- **Interpretation**: {"Highly systematic" if coherence_mean > 0.6 else "Moderately coherent" if coherence_mean > 0.4 else "Fragmentary style"}

"""
        
        # Add top performers
        report += """## Most Coherent Texts

"""
        for text in stats['highest_coherence']:
            report += f"1. **{text['text_id'].replace('_', ' ').title()}**: {text['semantic_coherence']:.3f}\n"
        
        report += """
## Least Coherent Texts

"""
        for text in stats['lowest_coherence']:
            report += f"1. **{text['text_id'].replace('_', ' ').title()}**: {text['semantic_coherence']:.3f}\n"
        
        # Add detailed results table
        report += f"""
## Detailed Results

| Philosopher | Work | School | Coherence | Sentences | Words |
|-------------|------|--------|-----------|-----------|-------|
"""
        
        for _, row in self.results.sort_values('semantic_coherence', ascending=False).iterrows():
            report += f"| {row['philosopher']} | {row['work']} | {row['school']} | {row['semantic_coherence']:.3f} | {row['sentence_count']} | {row['word_count']:,} |\n"
        
        report += f"""
## Methodology

This analysis uses **Latent Semantic Analysis (LSA)** to measure semantic coherence between consecutive sentences in philosophical texts. The technique is based on the research paper "Automated analysis of free speech predicts psychosis onset in high-risk youths" (Bedi et al., 2015).

### Interpretation

- **High coherence (>0.6)**: Systematic, logical argumentation
- **Medium coherence (0.4-0.6)**: Structured but with conceptual jumps  
- **Low coherence (<0.4)**: Fragmentary, aphoristic style

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📝 Report saved to {output_file}")
        return output_path


@click.command()
@click.option(
    '--input-dir', '-i',
    default='data/raw',
    help='Directory containing philosophical texts'
)
@click.option(
    '--output-dir', '-o', 
    default='reports',
    help='Output directory for results'
)
@click.option(
    '--csv-output', '-c',
    default='reports/philosophical_analysis.csv',
    help='CSV file for detailed results'
)
@click.option(
    '--visualize', 
    is_flag=True,
    help='Generate visualizations'
)
def main(input_dir: str, output_dir: str, csv_output: str, visualize: bool):
    """Compare philosophical texts and generate analysis report."""
    
    print("🏛️  Philosophical Text Comparison Analysis")
    print("=" * 60)
    
    try:
        # Initialize comparator
        comparator = PhilosopherComparator()
        
        # Load and analyze texts
        results = comparator.load_and_analyze_texts(input_dir)
        
        # Save detailed CSV
        Path(csv_output).parent.mkdir(exist_ok=True)
        results.to_csv(csv_output, index=False)
        print(f"💾 Detailed results saved to: {csv_output}")
        
        # Generate summary statistics
        stats = comparator.generate_summary_statistics()
        
        # Print summary to console
        print(f"\n📊 Analysis Summary:")
        print(f"📚 Total texts analyzed: {stats['total_texts']}")
        print(f"🧠 Average coherence: {stats['avg_coherence']:.3f}")
        print(f"📈 Standard deviation: {stats['coherence_std']:.3f}")
        
        print(f"\n🏆 Most coherent:")
        for i, text in enumerate(stats['highest_coherence'], 1):
            print(f"  {i}. {text['text_id']}: {text['semantic_coherence']:.3f}")
        
        print(f"\n🤔 Least coherent:")
        for i, text in enumerate(stats['lowest_coherence'], 1):
            print(f"  {i}. {text['text_id']}: {text['semantic_coherence']:.3f}")
        
        # Generate visualizations
        if visualize:
            comparator.create_visualizations(output_dir)
        
        # Generate report
        comparator.generate_report(f"{output_dir}/analysis_report.md")
        
        print(f"\n🎉 Analysis complete! Check {output_dir}/ for detailed results.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()