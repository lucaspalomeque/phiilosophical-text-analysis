FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY setup.py .
COPY src/ src/
RUN pip install --no-cache-dir -e ".[web]"

# Download NLTK data at build time
RUN python -c "\
import nltk; \
for pkg in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', \
            'averaged_perceptron_tagger_eng', 'stopwords', 'wordnet']: \
    nltk.download(pkg, quiet=True)"

# Copy remaining project files
COPY reports/ reports/

EXPOSE 8000

CMD ["uvicorn", "philosophical_analysis.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
