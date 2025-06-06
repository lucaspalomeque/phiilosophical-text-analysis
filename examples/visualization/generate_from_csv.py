"""
Generate visualizations from existing CSV results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.philosophical_analysis.visualization import VisualizationGenerator

# Cargar resultados reales
print("📊 Loading analysis results...")
results_df = pd.read_csv('reports/philosophical_analysis.csv')
print(f"   Loaded {len(results_df)} results")
print(f"   Texts analyzed: {results_df['text_id'].tolist()}")

# Cargar textos si están disponibles
texts = {}
text_dir = Path('data/raw')
if text_dir.exists():
    print("\n📚 Loading original texts...")
    for text_file in text_dir.glob('*.txt'):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                texts[text_file.stem] = f.read()
                print(f"   Loaded: {text_file.stem}")
        except Exception as e:
            print(f"   Error loading {text_file.stem}: {e}")

# Generar visualizaciones
print("\n🎨 Generating visualizations with real data...")
generator = VisualizationGenerator()
viz_data = generator.generate_all_visualizations(
    results_df, 
    texts if texts else None,
    save_html=True  # Esto actualizará los HTMLs con datos reales
)

print("\n✅ Visualizations updated with real data!")
print("📁 Files updated in: reports/visualizations/")
print("\n🌐 To view, run:")
print("   cd reports/visualizations")
print("   python -m http.server 8080")
print("   Then open: http://localhost:8080/philosophical_matrix_dashboard.html")
