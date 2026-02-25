# 🏛️ Arquitectura del Proyecto de Análisis Filosófico

Este documento describe la arquitectura de alto nivel del sistema de análisis de textos filosóficos, detallando sus componentes principales, el flujo de datos y las decisiones de diseño.

## 1. Filosofía de Diseño

El proyecto está diseñado siguiendo principios de **modularidad y especialización**. Cada componente tiene una responsabilidad única y bien definida, lo que facilita las pruebas, el mantenimiento y la extensión del sistema. La arquitectura separa claramente:

- **El motor de análisis (`core`)**: La lógica científica para procesar textos.
- **La capa de visualización (`visualization`)**: La generación de datos para los reportes interactivos.
- **La capa de interfaz (`cli`, `notebooks`)**: Los puntos de entrada para interactuar con el sistema.

## 2. Estructura de Directorios

La organización del proyecto refleja esta filosofía:

```
philosophical-text-analysis/
├── config/                 # Archivos de configuración (ej. main.yaml)
├── data/
│   ├── raw/                # Textos originales (.txt)
│   └── processed/          # (Opcional) Datos intermedios limpios
├── notebooks/              # Cuadernos de Jupyter para demostración y experimentación
├── reports/
│   └── visualizations/     # Salidas finales: HTML interactivos y datos JSON
├── src/philosophical_analysis/
│   ├── core/               # Lógica principal de análisis
│   ├── visualization/      # Generación de datos para visualizaciones
│   └── cli.py              # Interfaz de línea de comandos
├── tests/                  # Pruebas unitarias y de integración
├── README.md
├── ROADMAP.md
└── ARCHITECTURE.md         # Este archivo
```

## 3. Flujo de Datos

El sistema procesa los datos a través de un pipeline bien definido:

  <!-- Placeholder para un diagrama futuro -->

1.  **Entrada de Datos**: El proceso se inicia con los textos crudos (`.txt`) ubicados en `data/raw/`.

2.  **Análisis Integrado**: El `IntegratedPhilosophicalAnalyzer` orquesta el análisis. Invoca secuencialmente a los siguientes componentes del `core`:
    *   `AdvancedPOSAnalyzer`: Realiza un análisis sintáctico para extraer características como la frecuencia de determinantes y la longitud de las frases.
    *   `EnhancedCoherenceAnalyzer`: Modela el texto usando LSA (Análisis Semántico Latente) para calcular la coherencia semántica de primer y segundo orden.
    *   `ConvexHullClassifier`: Utiliza las características extraídas para clasificar el texto, por ejemplo, como "coherente" o "incoherente", basándose en un clasificador de casco convexo.

3.  **Resultados del Análisis**: El analizador genera un `DataFrame` de Pandas con todas las métricas calculadas para cada texto. Este `DataFrame` puede ser guardado como un archivo `.csv`.

4.  **Generación de Visualizaciones**: El `VisualizationGenerator` toma el `DataFrame` de resultados y los textos originales para producir los datos de visualización:
    *   Invoca al `SemanticNetworkGenerator` para construir un grafo de conceptos clave y sus relaciones.
    *   Calcula estadísticas comparativas y datos temporales para el dashboard.

5.  **Salida Final**: Los datos generados se guardan en `reports/visualizations/` como:
    *   **Archivos JSON**: Datos limpios y estructurados (ej. `dashboard_data.json`, `network_data.json`).
    *   **Archivos HTML**: Páginas web interactivas que cargan los datos JSON y los renderizan usando librerías de JavaScript como D3.js o Plotly.

## 4. Componentes Principales

### Módulos del `core`

-   **`integrated_analyzer.py`**: El orquestador principal. Su rol es integrar los resultados de los demás analizadores en un pipeline coherente.
-   **`pos_analyzer.py`**: Especializado en el análisis de Part-of-Speech (POS) y la extracción de métricas sintácticas definidas en el paper de investigación.
-   **`enhanced_coherence.py`**: Implementa los algoritmos de coherencia semántica, incluyendo LSA y el seguimiento temporal.
-   **`convex_hull.py`**: Contiene la lógica del clasificador basado en casco convexo, incluyendo el entrenamiento (leave-one-out) y la predicción.

### Módulos de `visualization`

-   **`generator.py`**: El orquestador de la visualización. Prepara los datos para todos los componentes visuales (dashboard, gráficos temporales) y gestiona la actualización de los archivos HTML.
-   **`semantic_network.py`**: Un módulo especializado que se enfoca exclusivamente en extraer conceptos filosóficos de un texto y construir una red de relaciones basada en su co-ocurrencia y similitud semántica.

## 5. Decisiones de Diseño Clave

-   **Configuración sobre Código**: Se utiliza un archivo `config/main.yaml` para definir parámetros clave (ej. número de componentes LSA, rutas de archivos), permitiendo modificar el comportamiento del sistema sin cambiar el código.
-   **Tests Internos en Módulos**: Módulos complejos como `semantic_network.py` incluyen una función `test_*()` dentro del propio archivo. Esto permite una verificación rápida y aislada de su funcionalidad ejecutando `python -m src.philosophical_analysis.visualization.semantic_network`.
-   **Separación de Datos y Presentación**: La capa de visualización genera archivos JSON desacoplados de los HTML. Esto permite que el frontend (HTML/JS) y el backend (Python) evolucionen de forma independiente. Un equipo de frontend podría, por ejemplo, rediseñar los reportes sin necesidad de tocar el código Python.
