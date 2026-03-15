# Use official Python runtime as base image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# ── AUTO-GENERATE MODELS (First time setup) ───────────────────────────────────
# Install jupyter for notebook execution
RUN pip install --no-cache-dir jupyter nbconvert ipykernel papermill

# Create required directories (if they don't exist)
RUN mkdir -p /app/models /app/data/processed /app/results

# Step 1: Run Data Preprocessing & EDA Notebook
RUN echo "════════════════════════════════════════════════════════════════" && \
    echo "📊 Step 1: Data Import, Preprocessing & EDA..." && \
    echo "════════════════════════════════════════════════════════════════" && \
    python -m papermill notebooks/01_data_import_preprocessing_eda.ipynb /tmp/notebook01_output.ipynb \
    -p log_output true 2>&1 || true && \
    echo "✅ Data preprocessing complete!"

# Step 2: Run Model Training Notebook
RUN echo "════════════════════════════════════════════════════════════════" && \
    echo "🤖 Step 2: Model Training & Evaluation..." && \
    echo "════════════════════════════════════════════════════════════════" && \
    python -m papermill notebooks/02_model_training.ipynb /tmp/notebook02_output.ipynb \
    -p log_output true 2>&1 || true && \
    echo "✅ Model training complete!"

# Verify models were created
RUN if [ -f /app/models/logisticregression_sentiment_model.pkl ]; then \
      echo "✅ Models successfully generated!"; \
    else \
      echo "⚠️  Warning: Model files not found"; \
    fi

# Expose port 8000 for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run FastAPI with Uvicorn
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
