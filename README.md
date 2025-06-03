# 🏛️ Philosophical Text Analysis

**Automated analysis of philosophical texts using psycholinguistic techniques**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🔬 Applying machine learning techniques from psycholinguistic research to analyze patterns in philosophical thinking across different schools of thought.

## 🎯 What This Does

This project uses **Latent Semantic Analysis (LSA)** and other techniques from the research paper ["Automated analysis of free speech predicts psychosis onset"](https://www.nature.com/articles/npjschz201530) to analyze philosophical texts and discover patterns in different thinking styles.

### 🔍 Key Findings So Far

**Surprising Results from Initial Analysis:**
- **Nietzsche** (Beyond Good & Evil): **0.667 coherence** 🥇
- **Kant** (Critique of Pure Reason): **0.581 coherence** 🥈  
- **Hume** (Human Understanding): **0.570 coherence** 🥉

*This challenges assumptions about "systematic" vs "fragmentary" philosophical styles!*

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/philosophical-text-analysis.git
cd philosophical-text-analysis

# Set up environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Test it works
philo-analyze test
```

## 💡 What You Can Do

### **Analyze Individual Texts**
```bash
philo-analyze analyze --text your_text.txt --author "Philosopher Name" --verbose
```

### **Batch Analysis**
```bash
philo-analyze batch --input-dir texts/ --output results.csv --verbose
```

### **Download Philosophical Texts**
```bash
python scripts/download_philosophical_texts.py --sample
```

### **Compare Philosophers**
```bash
python scripts/compare_philosophers.py --input-dir data/raw --visualize
```

## 🧠 How It Works

1. **Text Preprocessing**: Tokenization, lemmatization, filtering using NLTK
2. **Semantic Analysis**: LSA (TF-IDF + SVD) to create semantic vectors
3. **Coherence Measurement**: Cosine similarity between consecutive sentences
4. **Classification**: Identify patterns that distinguish thinking styles

## 📊 Sample Results

### Semantic Coherence by Philosopher
```
Philosopher    | Coherence | Style
---------------|-----------|------------------
Nietzsche      | 0.667     | Surprisingly systematic
Kant           | 0.581     | Complex but coherent  
Hume           | 0.570     | Empirically structured
```

## 🗺️ Development Roadmap

- ✅ **Phase 0**: Basic LSA implementation and CLI
- 🎯 **Phase 1**: Complete replication of research paper
- 🎨 **Phase 2**: Advanced visualizations and reporting
- 🧠 **Phase 3**: Philosophy-specific metrics and insights
- 🚀 **Phase 4**: Web platform and advanced ML

*See [ROADMAP.md](ROADMAP.md) for detailed development plan*

## 🔮 Próximos Pasos

- [ ] **Implementar análisis de frecuencia de términos filosóficos**
- [ ] **Desarrollar visualizaciones interactivas con Plotly**
- [ ] **Agregar soporte para textos en múltiples idiomas**
- [ ] **Crear sistema de clasificación automática por corrientes filosóficas**
- [ ] **Implementar análisis de argumentación y estructura lógica**
- [ ] **Desarrollar comparador de estilos argumentativos entre autores**
- [ ] **Agregar extracción automática de citas y referencias**
- [ ] **Crear dashboard web interactivo**
- [ ] **Implementar análisis temporal de evolución conceptual**
- [ ] **Documentar casos de uso con textos específicos**

## 🔬 Scientific Approach

Based on:
- **Paper**: ["Automated analysis of free speech predicts psychosis onset in high-risk youths"](https://www.nature.com/articles/npjschz201530) (Bedi et al., 2015)
- **Method**: LSA-based semantic coherence analysis
- **Validation**: Statistical hypothesis testing across philosophical schools
- **Innovation**: Application to philosophical discourse analysis

## 📁 Project Structure

```
philosophical-text-analysis/
├── src/philosophical_analysis/    # Core package
│   ├── core/                     # Analysis algorithms
│   ├── data/                     # Data processing
│   └── visualization/            # Plotting and reports
├── scripts/                      # Utility scripts
├── tests/                        # Test suite
├── data/                         # Philosophical texts
└── reports/                      # Generated analyses
```

## 🤝 Contributing

This is a research project exploring the intersection of **computational linguistics** and **philosophy**. Contributions welcome!

### Ideas for Contributions:
- 📚 Add more philosophical texts
- 🔬 Implement additional metrics from the paper
- 🎨 Create new visualizations
- 🧪 Test hypotheses about philosophical schools
- 📖 Improve documentation

### Getting Started:
1. Fork the repository
2. Create a feature branch
3. Run tests: `pytest`
4. Submit a pull request

## 📈 Recent Updates

- **[Date]**: Initial release with Kant, Hume, Nietzsche analysis
- **[Date]**: CLI implementation with batch processing
- **[Date]**: Automated text downloading from Project Gutenberg
- **[Date]**: School-based hypothesis testing framework

## 🎓 Academic Context

This project demonstrates:
- **Computational text analysis** applied to philosophy
- **Interdisciplinary research** combining CS and humanities
- **Reproducible research** with open source tools
- **Novel insights** about philosophical thinking patterns

## 📧 Contact

Feel free to reach out if you're interested in:
- **Collaborative research** on computational philosophy
- **Extensions** to other domains (literature, psychology, etc.)
- **Academic applications** of these techniques

---

⭐ **Star this repo if you find it interesting!** ⭐

*"The unexamined text is not worth reading"* - Socrates (probably)