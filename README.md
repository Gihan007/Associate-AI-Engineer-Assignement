# IMDb Sentiment Analysis API

## 🚀 Quick Start - Choose One Of Two Ways 

### Docker (Recommended) ✅
```bash
git clone <repo> && cd Associate-AI-Engineer-Assignement
docker compose up --build
```
**One command!** Automatically trains models (~15-20 min first run)  
API: http://localhost:8000/docs

### Manual Setup 🐍
#Run all of these commondfs one by one
```bash
git clone <repo> && cd Associate-AI-Engineer-Assignement
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -r requirements.txt
jupyter notebook notebooks/01_data_import_preprocessing_eda.ipynb
jupyter notebook notebooks/02_model_training.ipynb
python -m uvicorn app:app --reload
```

---

## 📊 Original Quick Start

Get the API running locally in minutes:

### Prerequisites
- **Python 3.8+** (tested on 3.9+)
- Git

### Installation & Setup (5 minutes)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd Associate-AI-Engineer-Assignement

# 2. Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (2-3 minutes) - Run the notebooks in order:
jupyter notebook notebooks/01_data_import_preprocessing_eda.ipynb
# → Complete all cells, wait for processed data

jupyter notebook notebooks/02_model_training.ipynb
# → Complete all cells, model artifacts will be saved to models/

# 5. Start the API server
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

✅ **Done!** The API is now running at `http://localhost:8000`

---

## � Notebooks vs API: The Complete Workflow

This project follows a **two-phase approach**: **Experimentation** (Notebooks) → **Production** (API)

### Phase 1️⃣: Experimentation & Model Training (Notebooks)

**Notebook 01 — Data Import, Preprocessing & EDA**
```python
✅ Download 2GB IMDb dataset from Stanford AI
✅ Extract 25,000 reviews (balanced positive/negative)
✅ Exploratory Data Analysis (visualizations, distributions, summaries)
✅ Clean text (remove HTML, URLs, normalize whitespace)
✅ Engineer features (word count, review length, character analysis)
✅ Detect and handle outliers
✅ Save processed data → data/processed/imdb_processed.csv
```

**Uses from src/:** 
- `src.data.loader` — load_data(), save_data()
- `src.features.engineering` — handle_missing_values(), detect_outliers_iqr(), etc.

---

**Notebook 02 — Model Training & Evaluation**
```python
✅ Load processed data (25,000 cleaned reviews)
✅ Split into 80% train / 20% test (stratified)
✅ Vectorize text using TF-IDF (5,000 features, bigrams+unigrams)
✅ Train 3 baseline classifiers in parallel:
   - Logistic Regression
   - Naive Bayes
   - Linear SVM
✅ Evaluate all models (accuracy, precision, recall, F1, ROC-AUC)
✅ Select best model (highest accuracy)
✅ Save model artifacts → models/ folder
```

**Output artifacts:**
```
models/
├── logisticregression_sentiment_model.pkl    ← Best trained model
├── tfidf_vectorizer.pkl                      ← Text vectorizer
└── model_metadata.json                       ← Performance metrics
also added experimantal models like LSTM , DISTILBERT , XLNET , BIGBIRD (BERT RELATED MDOELS)
```

**Uses from src:**
- `src.data.loader` — load_data(), split_data()

---

### Phase 2️⃣: Production Deployment (app.py)

**What app.py does:**
```python
✅ Load pre-trained model artifacts from models/ folder
✅ Serve REST API with prediction endpoints
✅ Handle concurrent user requests
✅ Return sentiment predictions in real-time
```

**What app.py does NOT do:**
```python
❌ Download or preprocess raw data
❌ Perform EDA or visualizations
❌ Engineer features from raw text
❌ Train models from scratch
❌ Evaluate multiple models
```

**Key difference:** `app.py` performs **minimal text preprocessing** only:
- Remove HTML tags
- Convert to lowercase
- Remove URLs
- Normalize whitespace

This is the **same preprocessing** used during training to ensure consistency.

---

### Complete Data Flow

```
1. Run Notebook 01
   ↓
   Downloads IMDb dataset
   Cleans and preprocesses
   Saves → data/processed/imdb_processed.csv
   
2. Run Notebook 02
   ↓
   Loads processed data
   Trains 3 models
   Compares and selects best
   Saves → models/*.pkl files
   
3. Run API (python -m uvicorn app:app --reload)
   ↓
   Loads trained model from models/ folder
   Serves REST API on http://localhost:8000
   
4. User sends review
   ↓
   API preprocesses text
   Vectorizes using saved vectorizer
   Predicts using saved model
   Returns sentiment + confidence
```

---

### Why This Architecture?

In production, would:
- Run notebooks **once** to train the model
- Run API **continuously** to serve predictions
- Re-run notebooks **periodically** to retrain with new data

---

## �📡 API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Single Review Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely brilliant! Best film I have ever seen."}'
```

**Response:**
```json
{
  "text": "This movie was absolutely brilliant! Best film I have ever seen.",
  "sentiment": "positive",
  "confidence": 0.94,
  "label": 1
}
```

### Batch Predictions (1-100 reviews)

```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      "Amazing movie! Highly recommend.",
      "Terrible. Complete waste of time.",
      "It was okay, nothing special."
    ]
  }'
```

### Python Example

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This movie was fantastic!"}
)
print(response.json())

# Batch predictions
response = requests.post(
    "http://localhost:8000/batch_predict",
    json={
        "reviews": [
            "Amazing!",
            "Terrible.",
            "Pretty good."
        ]
    }
)
print(response.json())
```

### Interactive API Documentation

Open your browser and visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

You can test all endpoints directly from the browser!

---

##  Model Approach

**Model Choice: Logistic Regression**

We selected Logistic Regression as our primary classifier after evaluating three candidate models (Logistic Regression, Linear SVM, and Multinomial Naive Bayes). Logistic Regression achieved the highest accuracy (~89%) on the test set while offering excellent interpretability and fast inference times—critical for production APIs. The model was trained on 25,000 IMDb reviews using TF-IDF vectorization with both unigrams and bigrams (5,000 features), which captures sentiment-bearing n-grams effectively.

**Why Logistic Regression?**
- Strong performance: 89% accuracy on balanced binary classification
- Fast inference: <1ms per prediction
- Interpretable: Easy to explain predictions to stakeholders
- Low memory footprint: ~200KB model size
- Proven on this task: Standard baseline for sentiment analysis

**Future Improvements**
With more time, we would explore ensemble stacking (combining LR, SVM, and XGBoost), incorporate pre-trained transformer embeddings (BERT), and implement active learning to refine predictions on borderline cases with human feedback. Currently, edge cases (sarcasm, mixed sentiment) could be better captured with contextual embeddings rather than TF-IDF.

---

## 📊 Project Structure

```
Associate-AI-Engineer-Assignement/
├── app.py                                      # FastAPI application
├── main.py                                     # Pipeline overview
├── requirements.txt                            # Python dependencies
│
├── notebooks/
│   ├── 01_data_import_preprocessing_eda.ipynb  # Step 1: Data prep & EDA
│   └── 02_model_training.ipynb                 # Step 2: Model training
│
├── data/
│   ├── raw/                                    # Original IMDb dataset
│   └── processed/imdb_processed.csv             # Cleaned data (25K reviews)
│
├── models/                                     # Trained artifacts
│   ├── logisticregression_sentiment_model.pkl
│   ├── svm_sentiment_model.pkl
│   ├── naive_bayes_sentiment_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── model_metadata.json
│   └── backups/                                # Automatic backup on retrain
│
├── src/
│   ├── data/loader.py
│   ├── features/engineering.py
│   ├── models/base.py
│   └── utils/helpers.py
│
└── configs/config.yaml
```

---

## 🔄 Automated Model Retraining

The API supports automated model retraining with intelligent best-model selection. Upload new data via a secure admin endpoint, and the system automatically:
- Trains **all 3 models** (Logistic Regression, SVM, Naive Bayes)
- **Compares their accuracies** 
- **Deploys the best one** if it meets quality thresholds (≥85%)
- **Automatically backs up** the old model
- **Reloads in memory** without restarting the server

### Admin Endpoints (Requires API Key)

**1. Upload Data & Start Retraining**

```bash
# Set your API key
API_KEY="your-secret-key-change-me"

# Upload CSV and trigger retraining
curl -X POST http://localhost:8000/admin/upload_and_retrain \
  -H "api_key: $API_KEY" \
  -F "file=@new_reviews.csv"
```

**Response:**
```json
{
  "status": "retraining_started",
  "message": "Model retraining started in background. Check /admin/retrain-status for progress.",
  "file": "new_reviews.csv"
}
```

**CSV Format Required:**
```csv
text,sentiment
"This movie was amazing!",positive
"Terrible waste of time.",negative
```

**2. Check Retraining Status & Model Accuracies**

```bash
API_KEY="your-secret-key-change-me"

curl http://localhost:8000/admin/retrain-status \
  -H "api_key: $API_KEY"
```

**Response Example (SVM became the best model!):**
```json
{
  "is_retraining": false,
  "current_model_in_use": "svm",
  "last_retrain_date": "2026-03-13T14:30:00.123456",
  "active_model_accuracy": 0.8954,
  "all_model_accuracies": {
    "logisticregression": {"accuracy": 0.8918, "f1_score": 0.8912},
    "svm": {"accuracy": 0.8954, "f1_score": 0.8950},
    "naive_bayes": {"accuracy": 0.8765, "f1_score": 0.8760}
  },
  "data_samples": 26500
}
```

### How Automated Retraining Works

```
1. Upload CSV with new data via /admin/upload_and_retrain
   ↓
2. API validates CSV (checks text, sentiment columns exist)
   ↓
3. Returns immediately → Non-blocking, won't timeout
   ↓
4. [BACKGROUND] Combines old + new data, removes duplicates
   ↓
5. [BACKGROUND] Trains all 3 models in parallel
   ↓
6. [BACKGROUND] Compares accuracies:
   - If all < 85%: REJECT (keeps current model, logs warning)
   - If best >= 85%: DEPLOY (automatically switches to best)
   ↓
7. [BACKGROUND] Backs up old model with timestamp
   ↓
8. [BACKGROUND] Saves new best model & updates metadata.json
   ↓
9. [BACKGROUND] Reloads new model in memory (API continues serving)
   ↓
10. Check status anytime via /admin/retrain-status
```

### Security: Change API Key in Production

**⚠️ IMPORTANT: The default API key is for testing only!**

Edit `app.py` and locate these two lines (around line 480-660):
```python
ADMIN_API_KEY = "your-secret-key-change-me"
```

Replace with a strong, random key:
```python
ADMIN_API_KEY = "sk_live_abc123xyz789def456ghi_prod_v2"
```

Then pass this key in admin requests:
```bash
curl -H "api_key: sk_live_abc123xyz789def456ghi_prod_v2" \
  http://localhost:8000/admin/retrain-status
```

### Production Checklist for Retraining

- ✅ Change default API key in `app.py` before deploying
- ✅ Store API key securely (environment variable, secrets manager)
- ✅ Only admins/CI/CD systems have access to the API key
- ✅ Monitor retraining logs for failures
- ✅ Set accuracy threshold appropriately (currently 85%)
- ✅ Backups happen automatically (check `models/backups/`)
- ✅ Load-test new model before full rollout
- ✅ Monitor model performance metrics post-deployment

### Python Example (With Automated Retraining)

```python
import requests
import time

API_KEY = "your-secret-key-change-me"

# 1. Upload data and trigger retraining
with open("new_reviews.csv", "rb") as f:
    files = {"file": f}
    headers = {"api_key": API_KEY}
    response = requests.post(
        "http://localhost:8000/admin/upload_and_retrain",
        files=files,
        headers=headers
    )
    print("✓ Retraining started:", response.json())

# 2. Poll status until retraining completes
headers = {"api_key": API_KEY}
polling_count = 0
while True:
    status = requests.get(
        "http://localhost:8000/admin/retrain-status",
        headers=headers
    ).json()
    
    polling_count += 1
    print(f"\n[Check #{polling_count}] Retraining in progress: {status['is_retraining']}")
    print(f"  Current model: {status['current_model_in_use']}")
    print(f"  Accuracies: {status['all_model_accuracies']}")
    
    if not status["is_retraining"]:
        print("\n✅ Retraining complete!")
        print(f"  Best model: {status['current_model_in_use']} ({status['active_model_accuracy']:.4f} accuracy)")
        break
    
    time.sleep(5)  # Check every 5 seconds

# 3. Make prediction with new model
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This movie was fantastic!"}
)
print(f"\n✓ Prediction with new model: {response.json()}")
```

---

## 🔧 Troubleshooting

### Model Not Loading?
- Ensure both notebooks completed successfully (01 and 02)
- Check that model files exist: `ls models/*.pkl`
- Check `models/model_metadata.json` for current model name
- Look for error messages in server console

### API Key Not Working for Admin Endpoints?
- Verify you changed `ADMIN_API_KEY` in `app.py` to something other than the default
- Pass the SAME key in the `api_key` header
- Example: `curl -H "api_key: your-actual-key" http://localhost:8000/admin/retrain-status`

### CSV Upload Fails?
- CSV must have exactly 2 columns: `text` (review text) and `sentiment` (positive/negative)
- Sentiment values must be lowercase: `positive` or `negative`
- Example valid CSV:
  ```csv
  text,sentiment
  "Great movie!",positive
  "Awful!",negative
  "It was okay.",positive
  ```
- Check CSV encoding (UTF-8 recommended)

### Retraining Stuck or Times Out?
- Retraining runs in background → endpoint returns immediately
- Check `/admin/retrain-status` to monitor progress
- Typical retraining time: 2-5 minutes (depends on data size)
- If stuck > 10 minutes, restart the API server

### Port 8000 Already in Use?
```bash
# Use different port
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8001
```

### Dependencies Won't Install?
```bash
# Upgrade pip first
pip install --upgrade pip

# Install with no cache
pip install -r requirements.txt --no-cache-dir

# Or use specific Python version
python3.9 -m pip install -r requirements.txt
```

---

## 📚 Additional Resources

- API Docs: http://localhost:8000/docs (Swagger)
- Full Notebook Guide: See `PIPELINE_GUIDE.md`
- Model Metadata: `models/model_metadata.json` (after training)

---

## 📝 Development

Run tests:
```bash
pytest tests/
```

View pipeline summary:
```bash
python main.py
```

---

## 📄 License

See LICENSE file for details.
