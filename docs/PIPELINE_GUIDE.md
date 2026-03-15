# IMDb Sentiment Analysis - Complete ML Pipeline Guide

## 📋 Overview

This project implements a complete machine learning pipeline for **sentiment classification on IMDb movie reviews**. The pipeline consists of three stages:

1. **Data Import & Preprocessing** → Jupyter Notebook
2. **Model Training & Evaluation** → Jupyter Notebook  
3. **REST API Service** → FastAPI + Uvicorn (Production)

---

## 🗂️ Project Structure

```
Associate-AI-Engineer-Assignement/
├── app.py                                 # FastAPI application
├── main.py                                # Pipeline orchestrator
├── requirements.txt                       # Dependencies (with FastAPI/Uvicorn)
│
├── notebooks/
│   ├── 01_data_import_preprocessing_eda.ipynb    # Step 1: EDA & preprocessing
│   └── 02_model_training.ipynb                   # Step 2: Model training
│
├── data/
│   ├── raw/                              # Original IMDb dataset
│   └── processed/imdb_processed.csv       # Cleaned data (25,000 reviews)
│
├── models/                                # Trained artifacts
│   ├── logisticregression_sentiment_model.pkl     # Best model
│   ├── tfidf_vectorizer.pkl                       # Text vectorizer
│   └── model_metadata.json                        # Performance metrics
│
├── results/                               # EDA visualizations
│   └── *.png                              # Charts (6 files)
│
├── src/
│   ├── data/loader.py                    # Data loading utilities
│   ├── features/engineering.py           # Feature engineering
│   ├── models/base.py                    # Model base classes
│   └── utils/helpers.py                  # General utilities
│
└── configs/config.yaml                   # Configuration file
```

---

## 🚀 Quick Start

### Step 0: Install Dependencies

```bash
# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 1: Data Import & EDA ✅

**Run this notebook first:**

```bash
jupyter notebook notebooks/01_data_import_preprocessing_eda.ipynb
```

**What it does:**
- Downloads 2GB IMDb dataset (tar.gz) from Stanford AI
- Extracts 25,000 movie reviews (balanced 50-50 positive/negative)
- Performs comprehensive exploratory data analysis
- Cleans text data (removes HTML, URLs, lowercases, normalizes whitespace)
- Extracts text features (review length, word count, chars/word)
- Detects and handles outliers
- Saves processed data to `data/processed/imdb_processed.csv`

**Expected Output:**
- Processed dataset: 25,000 × 6 columns
- 6 visualization PNG files in `results/`
- Data profile summary

---

### Step 2: Model Training ✅

**Run this notebook second:**

```bash
jupyter notebook notebooks/02_model_training.ipynb
```

**What it does:**
- Loads preprocessed data
- Splits into 80-20 train-test sets (stratified)
- Vectorizes text using TF-IDF (5k features, bigrams+unigrams)
- Trains 3 baseline classifiers:
  - Logistic Regression
  - Multinomial Naive Bayes
  - Linear SVM
- Evaluates models (accuracy, precision, recall, F1, ROC-AUC)
- Generates confusion matrices & classification reports
- Extracts feature importance (top positive/negative words)
- Saves best model & artifacts to `models/`

**Expected Output:**
- Best model: `models/logisticregression_sentiment_model.pkl` (~88-90% accuracy)
- Vectorizer: `models/tfidf_vectorizer.pkl`
- Metadata: `models/model_metadata.json` (performance metrics)

**Expected Performance:**
- Logistic Regression: ~89% accuracy
- SVM: ~88% accuracy
- Naive Bayes: ~86% accuracy

---

### Step 3: Deploy FastAPI Service 🚀

**Run the production API:**

```bash
# Option A: Direct with uvicorn
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Option B: Via main (pipeline overview)
python main.py
# Then manually run the uvicorn command above
```

**Server starts at:**
- 🌐 **API:** `http://localhost:8000`
- 📚 **Docs:** `http://localhost:8000/docs` (Interactive Swagger UI)
- 🔄 **ReDoc:** `http://localhost:8000/redoc` (Alternative docs)

---

## 📡 API Endpoints

### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_name": "logisticregression",
  "model_accuracy": 0.8918,
  "vocabulary_size": 5000
}
```

---

### 2. Single Prediction

```http
POST /predict
Content-Type: application/json

{
  "text": "This movie was absolutely brilliant! Best film ever made."
}
```

**Response:**
```json
{
  "text": "This movie was absolutely brilliant! Best film ever made.",
  "sentiment": "positive",
  "confidence": 0.94,
  "label": 1
}
```

---

### 3. Batch Predictions

```http
POST /batch_predict
Content-Type: application/json

{
  "reviews": [
    "Amazing movie! Highly recommend.",
    "Terrible. Complete waste of time.",
    "It was okay, nothing special."
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "text": "Amazing movie! Highly recommend.",
      "sentiment": "positive",
      "confidence": 0.92,
      "label": 1
    },
    {
      "text": "Terrible. Complete waste of time.",
      "sentiment": "negative",
      "confidence": 0.88,
      "label": 0
    },
    {
      "text": "It was okay, nothing special.",
      "sentiment": "negative",
      "confidence": 0.61,
      "label": 0
    }
  ],
  "processing_time_ms": 45.23
}
```

---

### 4. Model Information

```http
GET /model/info
```

**Response:**
```json
{
  "model_name": "logisticregression",
  "model_type": "LogisticRegression",
  "accuracy": 0.8918,
  "precision": 0.8891,
  "recall": 0.8946,
  "f1_score": 0.8918,
  "roc_auc": 0.9612,
  "vocabulary_size": 5000,
  "vectorizer": "TfidfVectorizer",
  "training_samples": 20000,
  "test_samples": 5000,
  "timestamp": "2024-01-15T10:30:45"
}
```

---

## 💻 Usage Examples

### Using Python Requests Library

```python
import requests

BASE_URL = "http://localhost:8000"

# Single prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": "This movie was great!"}
)
print(response.json())

# Batch prediction
reviews = [
    "Loved it!",
    "Hated it!",
    "It was okay."
]
response = requests.post(
    f"{BASE_URL}/batch_predict",
    json={"reviews": reviews}
)
for pred in response.json()["predictions"]:
    print(f"{pred['text']} → {pred['sentiment']} ({pred['confidence']:.2%})")
```

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing movie!"}'

# Batch prediction
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": ["Great!", "Terrible!", "Okay."]
  }'
```

### Using Swagger UI

1. Open browser to `http://localhost:8000/docs`
2. Click on any endpoint to expand
3. Click "Try it out"
4. Enter sample data
5. Click "Execute"
6. View response

---

## 🔧 Configuration

### Model Parameters (in notebooks)

**TF-IDF Vectorizer:**
```python
TfidfVectorizer(
    max_features=5000,          # Vocabulary size
    min_df=5,                   # Min document frequency
    max_df=0.8,                 # Max document frequency
    ngram_range=(1, 2),         # Unigrams + bigrams
    lowercase=True,
    stop_words='english'
)
```

**Best Model (Logistic Regression):**
```python
LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver='lbfgs'
)
```

### API Configuration (in app.py)

```python
# CORS settings - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Server settings
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",       # Listen on all IPs
        port=8000,
        reload=True,          # Auto-reload on file changes
    )
```

---

## 📊 Data Flow

```
Raw IMDb Reviews (tar.gz)
        ↓
[01_data_import_preprocessing_eda.ipynb]
    • Download & extract
    • 25,000 reviews loaded
    • Full EDA & visualization
    • Text preprocessing (HTML removal, lowercasing, URL removal)
    • Feature extraction (length, word count)
    • Outlier detection
        ↓
data/processed/imdb_processed.csv
        ↓
[02_model_training.ipynb]
    • Train-test split (80-20)
    • TF-IDF vectorization (5k features)
    • Train 3 models
    • Evaluation & comparison
    • Feature importance analysis
        ↓
models/
    • logisticregression_sentiment_model.pkl
    • tfidf_vectorizer.pkl
    • model_metadata.json
        ↓
[app.py - FastAPI Service]
    • Load model artifacts
    • Preprocess input reviews
    • Vectorize with TF-IDF
    • Predict sentiment
    • Return confidence & label
        ↓
REST API (JSON responses)
    • /health
    • /predict
    • /batch_predict
    • /model/info
```

---

## 🧪 Testing the Pipeline

### 1. Verify Data Quality
```python
import pandas as pd

df = pd.read_csv('data/processed/imdb_processed.csv')
print(df.info())              # Check dtypes, nulls
print(df['sentiment'].value_counts())  # Check balance
print(df['review_cleaned'].str.len().describe())  # Text length stats
```

### 2. Verify Model Load
```python
import pickle

with open('models/logisticregression_sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
print(f"Model type: {type(model)}")
print(f"Model parameters: {model.get_params()}")
```

### 3. Test API Manually
```bash
# Test health endpoint
curl http://localhost:8000/health | python -m json.tool

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Best movie ever!"}' | python -m json.tool
```

---

## 🛠️ Troubleshooting

### Model artifacts not found
**Error:** `RuntimeError: Model artifacts not found`

**Solution:**
- Ensure Step 2 notebook was executed completely
- Check `models/` directory exists with 3 files
- Run Step 2 notebook again if missing

### Port 8000 already in use
**Error:** `Address already in use`

**Solution:**
```bash
# Use different port
python -m uvicorn app:app --port 8001

# Or kill existing process
# Windows: taskkill /PID <pid> /F
# Linux: kill -9 <pid>
```

### Slow predictions
**Issue:** Batch predictions taking >5 seconds for 100 reviews

**Optimization:**
- Reduce `max_features` in TF-IDF (5000 → 3000)
- Use SVM instead of Logistic Regression
- Consider GPU acceleration (requires ONNX conversion)

### Module import errors
**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Ensure running from project root
cd Associate-AI-Engineer-Assignement

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## 📈 Performance Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-------|---------|
| Logistic Regression | **89.18%** | 88.91% | 89.46% | 89.18% | 96.12% |
| Linear SVM | 88.42% | 88.15% | 88.70% | 88.42% | 95.68% |
| Naive Bayes | 86.54% | 85.92% | 87.20% | 86.54% | 93.24% |

**Best Model:** Logistic Regression (chosen for production)

---

## 🚀 Production Deployment

### Option 1: Local Development (Current)
```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Production Server (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000
```

### Option 3: Docker Containerization
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 4: Cloud Deployment
- **AWS:** Deploy to EC2 with Elastic Load Balancer
- **Google Cloud:** Cloud Run (serverless)
- **Azure:** App Service with auto-scaling
- **Heroku:** Git push deployment

---

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21.0 | Numerical computation |
| pandas | ≥1.3.0 | Data manipulation |
| scikit-learn | ≥0.24.0 | ML algorithms & preprocessing |
| matplotlib | ≥3.4.0 | Visualization |
| seaborn | ≥0.11.0 | Statistical visualization |
| fastapi | ≥0.104.0 | REST API framework |
| uvicorn | ≥0.24.0 | ASGI server |
| pyyaml | ≥6.0 | YAML config parsing |

---

## 📝 Next Steps

1. ✅ Complete Steps 1-2 using Jupyter notebooks
2. ✅ Deploy FastAPI service (Step 3)
3. 📊 Add model monitoring & logging
4. 🔄 Implement continuous retraining
5. 🧪 Add comprehensive unit tests
6. 📦 Package as Docker container
7. ☁️ Deploy to cloud (AWS/GCP/Azure)

---

## 📧 Support

For issues or questions:
1. Check **Troubleshooting** section above
2. Review notebook outputs in `notebooks/`
3. Inspect model metadata in `models/model_metadata.json`
4. Check server logs in FastAPI console

---

**Last Updated:** January 2024  
**Status:** ✅ Complete & Production Ready
