"""
Enhanced CLI with Phase 1A integration - maintains backward compatibility.

This extends your existing CLI to support both original and advanced analysis modes.
"""

import logging
import click
import sys
import json
from pathlib import Path
from typing import Optional
import pandas as pd


from .core.analyzer import PhilosophicalAnalyzer  # Your original analyzer
from . import __version__


@click.group()
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, verbose):
    """
    Philosophical Text Analysis CLI - Enhanced with Phase 1A.

    Analyze philosophical texts using psycholinguistic techniques.
    Now supports both original and advanced analysis modes.
    """
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)
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
    '--advanced', 
    is_flag=True,
    help='Use advanced Phase 1A analysis (POS + Convex Hull + Enhanced Coherence)'
)
@click.option(
    '--mode',
    type=click.Choice(['basic', 'advanced', 'full']),
    default='basic',
    help='Analysis mode: basic (original), advanced (Phase 1A), full (all metrics)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output for this command'
)
@click.pass_context
def analyze(ctx, text, output, author, advanced, mode, verbose):
    """Analyze a single philosophical text."""
    
    verbose = verbose or ctx.obj.get('verbose', False)
    
    # Determine analysis mode
    if advanced or mode in ['advanced', 'full']:
        analysis_mode = 'advanced'
    else:
        analysis_mode = 'basic'
    
    try:
        if verbose:
            click.echo(f"🔬 Analyzing: {text} (mode: {analysis_mode})")
        
        # Read the text file
        text_path = Path(text)
        with open(text_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        if len(text_content.strip()) == 0:
            click.echo("❌ Error: Text file is empty", err=True)
            sys.exit(1)
        
        text_id = author if author else text_path.stem
        
        # Choose analyzer based on mode
        if analysis_mode == 'advanced':
            try:
                from .core.integrated_analyzer import IntegratedPhilosophicalAnalyzer
                
                if verbose:
                    click.echo("🧠 Using advanced Phase 1A analyzer...")
                
                analyzer = IntegratedPhilosophicalAnalyzer()
                
                # For single text analysis, fit on the text itself
                training_texts = {text_id: text_content}
                analyzer.fit(training_texts)
                
                result = analyzer.analyze_text(text_content, text_id)
                
            except ImportError as e:
                click.echo("⚠️  Advanced analyzer not available, falling back to basic mode")
                if verbose:
                    click.echo(f"Import error: {e}")
                analysis_mode = 'basic'
        
        # Basic mode (your original analyzer)
        if analysis_mode == 'basic':
            if verbose:
                click.echo("🧠 Using original analyzer...")
            
            analyzer = PhilosophicalAnalyzer()
            training_texts = {text_id: text_content}
            analyzer.fit(training_texts)
            result = analyzer.analyze_text(text_content, text_id)
        
        # Display results
        display_results(result, analysis_mode, verbose)
        
        # Save output if requested
        if output:
            output_path = Path(output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
            click.echo(f"\n💾 Results saved to: {output_path}")
        
        sys.exit(0)
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def display_results(result: dict, mode: str, verbose: bool = False):
    """Display analysis results based on mode."""
    
    click.echo(f"\n📋 Analysis Results ({mode} mode):")
    click.echo("=" * 50)
    
    if 'error' in result:
        click.echo(f"❌ Error: {result['error']}")
        return
    
    # Basic metrics (available in both modes)
    click.echo(f"📖 Text ID: {result['text_id']}")
    click.echo(f"📝 Sentences: {result['sentence_count']}")
    
    if 'word_count' in result:
        click.echo(f"📚 Words: {result['word_count']}")
    if 'total_words' in result:
        click.echo(f"📚 Words: {result['total_words']}")
    
    # Coherence metrics
    click.echo(f"\n🧠 Coherence Analysis:")
    
    if 'semantic_coherence' in result:
        click.echo(f"  Coherence: {result['semantic_coherence']:.3f}")
        click.echo(f"  Min: {result.get('min_coherence', 'N/A')}")
        click.echo(f"  Max: {result.get('max_coherence', 'N/A')}")
    
    if 'first_order_coherence' in result:
        click.echo(f"  First-order: {result['first_order_coherence']:.3f}")
    
    if 'second_order_coherence' in result:
        click.echo(f"  Second-order: {result['second_order_coherence']:.3f}")
    
    # Advanced metrics (Phase 1A)
    if mode == 'advanced':
        click.echo(f"\n🔬 Advanced Metrics (Phase 1A):")
        
        if 'target_determiners_freq' in result:
            click.echo(f"  Determiner frequency: {result['target_determiners_freq']:.4f}")
            click.echo(f"  Determiner count: {result.get('target_determiners_count', 'N/A')}")
        
        if 'max_phrase_length' in result:
            click.echo(f"  Max phrase length: {result['max_phrase_length']}")
        
        if 'avg_sentence_length' in result:
            click.echo(f"  Avg sentence length: {result['avg_sentence_length']:.2f}")
        
        # Classification results
        if 'predicted_label' in result:
            click.echo(f"\n🤖 Classification:")
            click.echo(f"  Prediction: {result['predicted_label'].upper()}")
            click.echo(f"  Confidence: {result['classification_confidence']:.3f}")
        
        # Statistical significance
        if 'significant' in result:
            sig_status = "✅ Significant" if result['significant'] else "❌ Not significant"
            click.echo(f"  Statistical: {sig_status} (p={result.get('p_value', 'N/A'):.3f})")
        
        # Interpretations
        if 'interpretation' in result and verbose:
            click.echo(f"\n🧠 Interpretations:")
            interp = result['interpretation']
            for key, value in interp.items():
                click.echo(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Classification (basic mode)
    if 'classification' in result:
        click.echo(f"\n🎯 Classification: {result['classification'].upper()}")
    
    if 'analysis_mode' in result:
        click.echo(f"⚙️  Analysis Mode: {result['analysis_mode']}")


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
    '--advanced', 
    is_flag=True,
    help='Use advanced Phase 1A analysis'
)
@click.option(
    '--cross-validate',
    is_flag=True,
    help='Perform cross-validation (requires labels file)'
)
@click.option(
    '--labels-file',
    type=click.Path(exists=True),
    help='CSV file with text_id,label columns for supervised analysis'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.pass_context
def batch(ctx, input_dir, output, pattern, advanced, cross_validate, labels_file, verbose):
    """Analyze multiple texts in a directory."""
    
    verbose = verbose or ctx.obj.get('verbose', False)
    analysis_mode = 'advanced' if advanced else 'basic'
    
    try:
        input_path = Path(input_dir)
        text_files = list(input_path.glob(pattern))
        
        if not text_files:
            click.echo(f"❌ No files found matching pattern '{pattern}' in {input_dir}")
            sys.exit(1)
        
        if verbose:
            click.echo(f"📁 Found {len(text_files)} files to analyze (mode: {analysis_mode})")
        
        # Load texts
        texts = {}
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        texts[file_path.stem] = content
                        if verbose:
                            click.echo(f"📖 Loaded: {file_path.stem} ({len(content):,} chars)")
            except Exception as e:
                if verbose:
                    click.echo(f"⚠️  Skipped {file_path.stem}: {e}")
        
        if not texts:
            click.echo("❌ No valid text files found")
            sys.exit(1)
        
        # Load labels if provided
        labels = None
        if labels_file:
            import pandas as pd
            try:
                labels_df = pd.read_csv(labels_file)
                labels = dict(zip(labels_df['text_id'], labels_df['label']))
                click.echo(f"📋 Loaded labels for {len(labels)} texts")
            except Exception as e:
                click.echo(f"⚠️  Could not load labels: {e}")
        
        click.echo(f"📊 Analyzing {len(texts)} texts...")
        
        # Choose analyzer
        if analysis_mode == 'advanced':
            try:
                from .core.integrated_analyzer import IntegratedPhilosophicalAnalyzer
                analyzer = IntegratedPhilosophicalAnalyzer()
                analyzer.fit(texts, labels)
                results = analyzer.analyze_multiple_texts(texts)
                
                # Cross-validation if requested
                if cross_validate and labels:
                    click.echo("🔄 Performing cross-validation...")
                    cv_results = analyzer.cross_validate(texts, labels)
                    click.echo(f"📈 CV Results: Accuracy={cv_results['accuracy']:.3f}, F1={cv_results['f1_score']:.3f}")
                
            except ImportError:
                click.echo("⚠️  Advanced analyzer not available, using basic mode")
                analysis_mode = 'basic'
        
        if analysis_mode == 'basic':
            analyzer = PhilosophicalAnalyzer()
            analyzer.fit(texts)
            results = analyzer.analyze_multiple_texts(texts)
        
        # Save results
        output_path = Path(output)
        results.to_csv(output_path, index=False)
        
        click.echo(f"✅ Analysis complete!")
        click.echo(f"💾 Results saved to: {output_path}")
        
        # Show summary
        successful = len(results) if 'error' not in results.columns else len(results[results['error'].isna()])

        failed = len(results) - successful
        
        click.echo(f"\n📈 Summary:")
        click.echo(f"  Total texts: {len(results)}")
        click.echo(f"  Successful: {successful}")
        if failed > 0:
            click.echo(f"  Failed: {failed}")
        
        # Show coherence stats
        coherence_col = 'first_order_coherence' if 'first_order_coherence' in results.columns else 'semantic_coherence'
        if coherence_col in results.columns:
            avg_coherence = results[coherence_col].mean()
            click.echo(f"  Avg Coherence: {avg_coherence:.3f}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--mode', type=click.Choice(['basic', 'advanced']), default='basic')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def test(ctx, mode, verbose):
    """Run a quick test of the analyzer."""
    
    verbose = verbose or ctx.obj.get('verbose', False)
    
    click.echo(f"🧪 Running analyzer test ({mode} mode)...")
    
    try:
        # Sample texts for testing
        sample_texts = {
            "coherent": """
            Philosophy is the study of fundamental questions about existence and knowledge.
            These questions have been explored by thinkers throughout history.
            The systematic approach to these problems defines philosophical methodology.
            """,
            "fragmented": """
            Reality is uncertain. Mathematics proves strange things.
            Therefore cats understand quantum mechanics better than humans.
            The universe speaks in colors we cannot see.
            """
        }
        
        # Choose analyzer
        if mode == 'advanced':
            try:
                from .core.integrated_analyzer import IntegratedPhilosophicalAnalyzer
                analyzer = IntegratedPhilosophicalAnalyzer()
                click.echo("✅ Advanced analyzer loaded")
            except ImportError:
                click.echo("⚠️  Advanced analyzer not available, using basic")
                mode = 'basic'
        
        if mode == 'basic':
            analyzer = PhilosophicalAnalyzer()
            click.echo("✅ Basic analyzer loaded")
        
        # Fit and test
        analyzer.fit(sample_texts)
        click.echo("✅ Analyzer fitted")
        
        # Test analysis
        for text_id, text_content in sample_texts.items():
            result = analyzer.analyze_text(text_content, text_id)
            
            click.echo(f"\n📊 Results for '{text_id}':")
            if 'error' in result:
                click.echo(f"  ❌ Error: {result['error']}")
            else:
                sentences = result['sentence_count']
                
                if 'first_order_coherence' in result:
                    coherence = result['first_order_coherence']
                else:
                    coherence = result.get('semantic_coherence', 0)
                
                classification = result.get('classification', result.get('predicted_label', 'unknown'))
                
                click.echo(f"  Sentences: {sentences}")
                click.echo(f"  Coherence: {coherence:.3f}")
                click.echo(f"  Classification: {classification}")
                
                if mode == 'advanced' and 'target_determiners_freq' in result:
                    click.echo(f"  Determiners: {result['target_determiners_freq']:.4f}")
        
        click.echo(f"\n🎉 Test completed successfully! ({mode} mode)")
        
    except Exception as e:
        click.echo(f"❌ Test failed: {e}", err=True)
        if verbose:
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
    click.echo("🔬 Available Analysis Modes:")
    click.echo("  basic    - Original LSA-based coherence analysis")
    click.echo("  advanced - Phase 1A implementation (POS + Convex Hull + Enhanced Coherence)")
    click.echo("")
    click.echo("🚀 Available Commands:")
    click.echo("  analyze  - Analyze a single text file")
    click.echo("  batch    - Analyze multiple text files")
    click.echo("  test     - Run analyzer test")
    click.echo("  info     - Show this information")
    click.echo("")
    click.echo("📚 Usage Examples:")
    click.echo("  # Basic analysis")
    click.echo("  philo-analyze analyze --text kant_critique.txt --author Kant")
    click.echo("")
    click.echo("  # Advanced Phase 1A analysis")
    click.echo("  philo-analyze analyze --text kant_critique.txt --author Kant --advanced")
    click.echo("")
    click.echo("  # Batch analysis with cross-validation")
    click.echo("  philo-analyze batch --input-dir texts/ --output results.csv --advanced --cross-validate --labels-file labels.csv")


# Import visualization commands
try:
    from .cli_extensions import add_visualization_commands
    cli = add_visualization_commands(cli)
except ImportError:
    # Visualization module not available
    pass

def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()