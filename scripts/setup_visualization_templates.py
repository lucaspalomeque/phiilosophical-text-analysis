#!/usr/bin/env python3
"""
Setup script to move HTML visualization templates to the correct location.
"""

import shutil
from pathlib import Path

def setup_templates():
    """Move HTML templates to the visualization module."""
    
    # Source directory (where your HTMLs currently are)
    source_dir = Path("reports/visualizations")
    
    # Target directory
    target_dir = Path("src/philosophical_analysis/visualization/templates")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # HTML files to move
    html_files = [
        "philosophical_matrix_dashboard.html",
        "temporal_coherence_matrix.html",
        "semantic_network_matrix.html"
    ]
    
    for html_file in html_files:
        source = source_dir / html_file
        target = target_dir / html_file
        
        if source.exists():
            print(f"📄 Copying {html_file} to templates/")
            shutil.copy2(source, target)
        else:
            print(f"⚠️  {html_file} not found in {source_dir}")
    
    print("✅ Template setup complete!")

if __name__ == "__main__":
    setup_templates()
