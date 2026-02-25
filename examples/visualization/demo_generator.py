"""
Quick test for the visualization generator.
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path para poder importar
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

import pandas as pd
from src.philosophical_analysis.visualization import VisualizationGenerator

# Crear datos de prueba
test_results = pd.DataFrame([
    {
        'text_id': 'nietzsche_beyond_good_evil',
        'sentence_count': 1193,
        'first_order_coherence': 0.667,
        'second_order_coherence': 0.85,
        'target_determiners_freq': 0.008,
        'max_phrase_length': 20,
        'avg_sentence_length': 26.14,
    },
    {
        'text_id': 'kant_critique_pure_reason',
        'sentence_count': 5624,
        'first_order_coherence': 0.581,
        'second_order_coherence': 0.91,
        'target_determiners_freq': 0.015,
        'max_phrase_length': 25,
        'avg_sentence_length': 16.84,
    }
])

# Crear generador
print("🎨 Creando generador de visualizaciones...")
generator = VisualizationGenerator()

# Generar datos del dashboard
print("📊 Generando datos del dashboard...")
dashboard_data = generator.generate_dashboard_data(test_results)

# Mostrar resultados
print("\n✅ Dashboard data generated successfully!")
print(f"Philosophers found: {list(dashboard_data['philosophers'].keys())}")
print(f"Timestamp: {dashboard_data['timestamp']}")

# Generar todas las visualizaciones (sin actualizar HTML por ahora)
print("\n🔧 Generando todas las visualizaciones...")
viz_data = generator.generate_all_visualizations(test_results, save_html=False)

print("\n🎉 Test completado exitosamente!")
print(f"Visualizaciones generadas: {list(viz_data.keys())}")