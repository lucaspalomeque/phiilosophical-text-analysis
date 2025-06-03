"""
Command Line Interface for Philosophical Text Analysis.

Basic CLI implementation for analyzing philosophical texts.
"""

import click
import sys
import json
from pathlib import Path
from typing import Optional

from .core.analyzer import PhilosophicalAnalyzer
from . import __version__


@click.group()
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """
    Philosophical Text Analysis CLI.
    
    Analyze philosophical texts using psycholinguistic techniques.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.option(
    '--text', '-t', 
    required=True,
    type=click.Path(exists=True),
    help='Path to text file to analyze'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Output file for results (JSON format)'
)
@click.option(
    '--author', '-a',
    help='Author name for the text'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output for this command'
)
@click.pass_context
def analyze(ctx, text, output, author, verbose):
    """Analyze a single philosophical text."""
    
    # Use command-level verbose or global verbose
    verbose = verbose or ctx.obj.get('verbose', False)
    
    try:
        if verbose:
            click.echo(f"🔬 Analyzing: {text}")
        
        # Read the text file
        text_path = Path(text)
        with open(text_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        if len(text_content.strip()) == 0:
            click.echo("❌ Error: Text file is empty", err=True)
            sys.exit(1)
        
        # Initialize analyzer with sample training data
        analyzer = PhilosophicalAnalyzer()
        
        # Use the input text itself for training (simple approach)
        # In a real scenario, we'd have a separate training corpus
        training_texts = {
            "sample": text_content
        }
        
        if verbose:
            click.echo("🧠 Training analyzer...")
        
        analyzer.fit(training_texts)
        
        # Analyze the text
        text_id = author if author else text_path.stem
        
        if verbose:
            click.echo(f"📊 Analyzing text as '{text_id}'...")
        
        result = analyzer.analyze_text(text_content, text_id)
        
        # Display results
        click.echo("\n📋 Analysis Results:")
        click.echo("=" * 40)
        
        if 'error' in result:
            click.echo(f"❌ Error: {result['error']}")
            sys.exit(1)
        
        click.echo(f"📖 Text ID: {result['text_id']}")
        click.echo(f"📝 Sentences: {result['sentence_count']}")
        click.echo(f"📚 Words: {result['word_count']}")
        
        if 'avg_sentence_length' in result:
            click.echo(f"📏 Avg Sentence Length: {result['avg_sentence_length']}")
        
        click.echo(f"\n🧠 Semantic Analysis:")
        click.echo(f"  Coherence: {result['semantic_coherence']:.3f}")
        click.echo(f"  Min Coherence: {result['min_coherence']:.3f}")
        click.echo(f"  Max Coherence: {result['max_coherence']:.3f}")
        
        click.echo(f"\n🎯 Classification: {result['classification'].upper()}")
        
        if 'analysis_mode' in result:
            click.echo(f"⚙️  Analysis Mode: {result['analysis_mode']}")
        
        # Save output if requested
        if output:
            output_path = Path(output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
            click.echo(f"\n💾 Results saved to: {output_path}")
        
        # Return appropriate exit code
        sys.exit(0)
        
    except FileNotFoundError:
        click.echo(f"❌ Error: File not found: {text}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option(
    '--input-dir', '-i',
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help='Directory containing text files'
)
@click.option(
    '--output', '-o',
    required=True,
    type=click.Path(),
    help='Output CSV file for results'
)
@click.option(
    '--pattern', '-p',
    default='*.txt',
    help='File pattern to match (default: *.txt)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output for this command'
)
@click.pass_context
def batch(ctx, input_dir, output, pattern, verbose):
    """Analyze multiple texts in a directory."""
    
    # Use command-level verbose or global verbose
    verbose = verbose or ctx.obj.get('verbose', False)
    
    try:
        input_path = Path(input_dir)
        
        # Find text files
        text_files = list(input_path.glob(pattern))
        
        if not text_files:
            click.echo(f"❌ No files found matching pattern '{pattern}' in {input_dir}")
            sys.exit(1)
        
        if verbose:
            click.echo(f"📁 Found {len(text_files)} files to analyze")
        
        # Load all texts
        texts = {}
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts[file_path.stem] = content
            except Exception as e:
                if verbose:
                    click.echo(f"⚠️  Skipping {file_path}: {e}")
        
        if not texts:
            click.echo("❌ No valid text files found")
            sys.exit(1)
        
        click.echo(f"📊 Analyzing {len(texts)} texts...")
        
        # Initialize and fit analyzer
        analyzer = PhilosophicalAnalyzer()
        analyzer.fit(texts)
        
        # Analyze all texts
        results = analyzer.analyze_multiple_texts(texts)
        
        # Save results
        output_path = Path(output)
        results.to_csv(output_path, index=False)
        
        click.echo(f"✅ Analysis complete!")
        click.echo(f"💾 Results saved to: {output_path}")
        
        # Show summary
        if 'error' in results.columns:
            successful = len(results[results['error'].isna()])
            failed = len(results) - successful
        else:
            # If no error column exists, all analyses were successful
            successful = len(results)
            failed = 0
        
        click.echo(f"\n📈 Summary:")
        click.echo(f"  Total texts: {len(results)}")
        click.echo(f"  Successful: {successful}")
        if failed > 0:
            click.echo(f"  Failed: {failed}")
        
        if successful > 0 and 'semantic_coherence' in results.columns:
            avg_coherence = results['semantic_coherence'].mean()
            click.echo(f"  Avg Coherence: {avg_coherence:.3f}")
            
            # Show individual results if verbose
            if verbose:
                click.echo(f"\n📊 Individual Results:")
                for _, row in results.iterrows():
                    click.echo(f"  {row['text_id']}: {row['semantic_coherence']:.3f} ({row['classification']})")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output for test'
)
@click.pass_context
def test(ctx, verbose):
    """Run a quick test of the analyzer."""
    
    # Use command-level verbose or global verbose
    verbose = verbose or ctx.obj.get('verbose', False)
    
    if verbose:
        click.echo("🧪 Running analyzer test in verbose mode...")
    else:
        click.echo("🧪 Running analyzer test...")
    
    try:
        # Sample texts for testing
        sample_texts = {
            "coherent": """
            Philosophy is the study of fundamental questions about existence and knowledge.
            These questions have been explored by thinkers throughout history.
            The systematic approach to these problems defines philosophical inquiry.
            """,
            "fragmented": """
            Reality is uncertain. Mathematics proves strange things.
            Therefore cats understand quantum mechanics better than humans.
            The universe speaks in colors we cannot see.
            """
        }
        
        # Initialize and test
        analyzer = PhilosophicalAnalyzer()
        analyzer.fit(sample_texts)
        
        click.echo("✅ Analyzer initialized and fitted")
        
        # Test analysis
        for text_id, text_content in sample_texts.items():
            result = analyzer.analyze_text(text_content, text_id)
            
            click.echo(f"\n📊 Results for '{text_id}':")
            if 'error' in result:
                click.echo(f"  ❌ Error: {result['error']}")
            else:
                click.echo(f"  Sentences: {result['sentence_count']}")
                click.echo(f"  Coherence: {result['semantic_coherence']:.3f}")
                click.echo(f"  Classification: {result['classification']}")
        
        click.echo("\n🎉 Test completed successfully!")
        
    except Exception as e:
        click.echo(f"❌ Test failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


@cli.command()
def info():
    """Show package information."""
    
    click.echo("📦 Philosophical Text Analysis")
    click.echo(f"Version: {__version__}")
    click.echo("Description: Automated analysis of philosophical texts using psycholinguistic techniques")
    click.echo("")
    click.echo("🔬 Available Commands:")
    click.echo("  analyze  - Analyze a single text file")
    click.echo("  batch    - Analyze multiple text files")
    click.echo("  test     - Run analyzer test")
    click.echo("  info     - Show this information")
    click.echo("")
    click.echo("📚 Usage Examples:")
    click.echo("  philo-analyze analyze --text kant_critique.txt --author Kant")
    click.echo("  philo-analyze batch --input-dir texts/ --output results.csv")
    click.echo("  philo-analyze test")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()