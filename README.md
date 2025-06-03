# Philosophical Text Analysis

Herramienta de análisis computacional de textos filosóficos utilizando técnicas de procesamiento de lenguaje natural (NLP).

## 🎯 Objetivo
Analizar y extraer patrones, temas y estructuras argumentativas de textos filosóficos clásicos y contemporáneos.

## 🚀 Características
- Análisis de sentimientos en textos filosóficos
- Extracción de conceptos clave y términos técnicos
- Visualización de redes conceptuales
- Comparación entre diferentes autores/corrientes
- Generación de resúmenes automáticos

## 🛠️ Tecnologías
- Python 3.x
- NLTK / spaCy
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## 📦 Instalación
```bash
git clone https://github.com/lucaspalomeque/phiilosophical-text-analysis.git
cd phiilosophical-text-analysis
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 💻 Uso
```python
python main.py --input "textos/platon.txt" --analysis sentiment
```

## 📁 Estructura del Proyecto
```
├── data/
│   ├── raw/          # Textos originales
│   └── processed/    # Datos procesados
├── src/              # Código fuente
├── notebooks/        # Jupyter notebooks
├── reports/          # Resultados y visualizaciones
└── requirements.txt
```

## 📊 Ejemplos de Análisis
- **Análisis de frecuencia de términos**: Identificación de conceptos más utilizados por autor
- **Análisis de sentimientos**: Evaluación del tono emocional en diferentes corrientes filosóficas
- **Redes semánticas**: Visualización de relaciones entre conceptos filosóficos
- **Comparación textual**: Similitudes y diferencias entre textos de diferentes períodos

## 🔧 Desarrollo
Para contribuir al desarrollo:
```bash
# Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# Ejecutar tests
python -m pytest tests/

# Verificar estilo de código
flake8 src/
```

## 🤝 Contribuir
Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agrega nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 🚧 Próximos Pasos
- [ ] Implementar análisis de frecuencia de términos filosóficos
- [ ] Desarrollar visualizaciones interactivas con Plotly
- [ ] Agregar soporte para textos en múltiples idiomas
- [ ] Crear sistema de clasificación automática por corrientes filosóficas
- [ ] Implementar análisis de argumentación y estructura lógica
- [ ] Desarrollar comparador de estilos argumentativos entre autores
- [ ] Agregar extracción automática de citas y referencias
- [ ] Crear dashboard web interactivo
- [ ] Implementar análisis temporal de evolución conceptual
- [ ] Documentar casos de uso con textos específicos

## 📄 Licencia
MIT License

## 📞 Contacto
Lucas Palomeque - [@lucaspalomeque](https://github.com/lucaspalomeque)

Enlace del proyecto: [https://github.com/lucaspalomeque/phiilosophical-text-analysis](https://github.com/lucaspalomeque/phiilosophical-text-analysis)