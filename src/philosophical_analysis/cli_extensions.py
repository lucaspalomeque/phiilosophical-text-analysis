"""
CLI extensions for visualization commands.
"""

import click
import sys
from pathlib import Path
import pandas as pd

from .visualization import VisualizationGenerator


def add_visualization_commands(cli):
    """Add visualization commands to the main CLI."""
    
    @cli.command()
    @click.option(
        '--input', '-i',
        required=True,
        type=click.Path(exists=True),
        help='Input CSV file with analysis results'
    )
    @click.option(
        '--texts-dir', '-t',
        type=click.Path(exists=True, file_okay=False),
        help='Directory containing original text files'
    )
    @click.option(
        '--output-dir', '-o',
        default='reports/visualizations',
        help='Output directory for visualizations'
    )
    @click.option(
        '--mode',
        type=click.Choice(['all', 'dashboard', 'temporal', 'network']),
        default='all',
        help='Which visualizations to generate'
    )
    @click.option(
        '--no-html',
        is_flag=True,
        help='Only generate JSON data, do not update HTML files'
    )
    @click.pass_context
    def visualize(ctx, input, texts_dir, output_dir, mode, no_html):
        """Generate interactive visualizations from analysis results."""
        
        verbose = ctx.obj.get('verbose', False)
        
        click.echo("🎨 Generating Philosophical Text Visualizations")
        click.echo("=" * 50)
        
        try:
            # Load results
            results_df = pd.read_csv(input)
            
            # Load texts if provided
            texts = {}
            if texts_dir:
                texts_path = Path(texts_dir)
                for text_file in texts_path.glob("*.txt"):
                    with open(text_file, 'r', encoding='utf-8') as f:
                        texts[text_file.stem] = f.read()
            
            # Generate visualizations
            generator = VisualizationGenerator(output_dir)
            viz_data = generator.generate_all_visualizations(
                results_df,
                texts if texts else None,
                save_html=not no_html
            )
            
            click.echo(f"\n✅ Visualizations generated successfully!")
            click.echo(f"📁 Output directory: {output_dir}")
            
        except Exception as e:
            click.echo(f"❌ Error: {e}", err=True)
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    @cli.command()
    @click.option('--port', '-p', default=8000, help='Port to serve on')
    @click.option('--host', '-h', default='127.0.0.1', help='Host to bind to')
    def serve(port, host):
        """Launch the web application."""
        try:
            import uvicorn
        except ImportError:
            click.echo("Install web dependencies: pip install 'philosophical-text-analysis[web]'", err=True)
            sys.exit(1)

        click.echo(f"Starting Philosophical Text Analysis at http://{host}:{port}")
        uvicorn.run(
            "philosophical_analysis.web.app:app",
            host=host,
            port=port,
            reload=False,
            log_level="info",
        )
    
    return cli
