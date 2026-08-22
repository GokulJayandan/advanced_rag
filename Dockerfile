# ===========================================
# Multi-stage Dockerfile for Advanced Agentic RAG
# ===========================================

FROM python:3.11-slim as base

# System dependencies for Unstructured, Poppler, and Tesseract
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    poppler-utils \
    tesseract-ocr \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY streamlit_app.py .

# Create persistence directories
RUN mkdir -p chroma_data bm25_index embedding_cache uploads eval_results

# Expose API and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Default command: run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
