"""
FastAPI application for IMDb Sentiment Analysis.

Serves a trained sentiment classifier via REST API endpoints.
Includes automated retraining with multi-model comparison.
"""

import json
import pickle
import shutil
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score

# ── Global State ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
MODELS_PATH = PROJECT_ROOT / 'models'
DATA_PATH = PROJECT_ROOT / 'data' / 'processed'
BACKUP_PATH = MODELS_PATH / 'backups'

# Create backup directory if it doesn't exist
BACKUP_PATH.mkdir(parents=True, exist_ok=True)

# Retraining state tracker
retrain_status = {
    "is_retraining": False,
    "last_retrain_date": None,
    "last_retrain_result": None
}

def load_model_artifacts():
    """Load the best trained model, vectorizer, and metadata."""
    try:
        metadata_path = MODELS_PATH / 'model_metadata.json'
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Determine which model to load (best model from metadata)
        current_model_name = metadata.get('current_model', 'logisticregression')
        model_path = MODELS_PATH / f'{current_model_name}_sentiment_model.pkl'
        vectorizer_path = MODELS_PATH / 'tfidf_vectorizer.pkl'
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer, metadata
    except FileNotFoundError as e:
        raise RuntimeError(f"Model artifacts not found: {e}")

# Load artifacts
try:
    model, vectorizer, model_metadata = load_model_artifacts()
    print(f"✓ Model artifacts loaded successfully")
    print(f"  Current model: {model_metadata.get('current_model', 'logisticregression')}")
except Exception as e:
    print(f"✗ Error loading model artifacts: {e}")
    model = vectorizer = model_metadata = None

# ── Pydantic models for request/response ─────────────────────────────────────

class ReviewRequest(BaseModel):
    """Request model for sentiment prediction."""
    text: str = Field(..., min_length=1, max_length=10000, 
                      description="Review text for sentiment analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This movie is absolutely brilliant! The acting was superb and the plot kept me engaged throughout."
            }
        }


class SentimentPrediction(BaseModel):
    """Response model for single prediction."""
    text: str
    sentiment: str  # 'positive' or 'negative'
    confidence: float  # 0.0 to 1.0
    label: int  # 0 or 1
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This movie is absolutely brilliant!",
                "sentiment": "positive",
                "confidence": 0.95,
                "label": 1
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    reviews: List[str] = Field(..., min_items=1, max_items=100,
                               description="List of review texts")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[SentimentPrediction]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_name: str
    model_accuracy: float
    vocabulary_size: int


class RetrainResponse(BaseModel):
    """Response for retraining request."""
    status: str
    message: str
    file: str


class RetrainStatusResponse(BaseModel):
    """Response for retraining status check."""
    is_retraining: bool
    current_model_in_use: str
    last_retrain_date: Optional[str]
    active_model_accuracy: float
    all_model_accuracies: Dict
    data_samples: int


# ── FastAPI app initialization ───────────────────────────────────────────────

app = FastAPI(
    title="IMDb Sentiment Analysis API",
    description="Real-time sentiment classification for movie reviews",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper functions ─────────────────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """Basic preprocessing to match training pipeline."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Convert to lowercase (vectorizer handles this, but be explicit)
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


async def retrain_model_task(new_data_path: str):
    """
    Background task: Train all 3 models, select best, save artifacts.
    Does NOT block API responses.
    """
    global model, vectorizer, model_metadata, retrain_status
    
    try:
        retrain_status["is_retraining"] = True
        print(f"\n🔄 Starting model retraining with {new_data_path}...")
        
        # ── 1. Load and combine data ──────────────────────────────────────
        try:
            old_data = pd.read_csv(DATA_PATH / 'imdb_processed.csv')
            new_data = pd.read_csv(new_data_path)
            
            print(f"  Old data columns: {list(old_data.columns)}")
            print(f"  New data columns: {list(new_data.columns)}")
            
            # Validate new data has required columns
            if 'text' not in new_data.columns or 'sentiment' not in new_data.columns:
                raise ValueError("New CSV must have 'text' and 'sentiment' columns")
            
            # ── Handle old data text column ──────────────────────────────
            # If old data doesn't have 'text', try to find and rename it
            if 'text' not in old_data.columns:
                text_col = None
                for col in ['review', 'content', 'body', 'description']:
                    if col in old_data.columns:
                        text_col = col
                        break
                
                if text_col:
                    print(f"  Renaming '{text_col}' column in old data to 'text'")
                    old_data = old_data.rename(columns={text_col: 'text'})
                else:
                    raise ValueError("Could not find text column in old data (tried: review, content, body, description)")
            
            # ── Handle old data sentiment column ─────────────────────────
            # If old data doesn't have 'sentiment', try to find and rename the sentiment column
            if 'sentiment' not in old_data.columns:
                # Try common alternative column names
                sentiment_col = None
                for col in ['label', 'target', 'y', 'class']:
                    if col in old_data.columns:
                        sentiment_col = col
                        break
                
                if sentiment_col:
                    print(f"  Renaming '{sentiment_col}' column in old data to 'sentiment'")
                    old_data = old_data.rename(columns={sentiment_col: 'sentiment'})
                else:
                    raise ValueError("Could not find sentiment column in old data")
            
            # Drop duplicates ONLY within new data
            new_data = new_data.drop_duplicates(subset=['text'])
            
            # Keep only 'text' and 'sentiment' columns
            old_data = old_data[['text', 'sentiment']]
            new_data = new_data[['text', 'sentiment']]
            
            # Combine old + new (keep ALL old data, add new data)
            combined_data = pd.concat([old_data, new_data], ignore_index=True)
            print(f"  ✓ Combined data: {len(old_data)} (old) + {len(new_data)} (new) = {len(combined_data)} total samples")
            
        except Exception as e:
            print(f"  ✗ Data loading failed: {e}")
            retrain_status["is_retraining"] = False
            retrain_status["last_retrain_result"] = f"Failed: {str(e)}"
            return
        
        # ── 2. Prepare data for training ──────────────────────────────────
        try:
            # Handle NaN values - drop rows with missing text
            combined_data = combined_data.dropna(subset=['text'])
            combined_data = combined_data[combined_data['text'].str.strip() != '']  # Remove empty strings
            
            # Handle sentiment values - convert to lowercase and validate
            combined_data['sentiment'] = combined_data['sentiment'].astype(str).str.lower().str.strip()
            valid_sentiments = combined_data['sentiment'].isin(['positive', 'negative', '0', '1'])
            combined_data = combined_data[valid_sentiments]
            
            # Convert numeric sentiments to text if needed
            combined_data['sentiment'] = combined_data['sentiment'].replace({'0': 'negative', '1': 'positive'})
            
            print(f"  ✓ After cleaning: {len(combined_data)} samples")
            
            # ≡ Apply same preprocessing as used in predictions ≡
            # This ensures consistency between training and inference
            print(f"  ✓ Applying preprocessing to {len(combined_data)} samples...")
            combined_data['text_processed'] = combined_data['text'].apply(preprocess_text)
            
            # IMPORTANT: Retrain vectorizer on ALL combined data to include new vocabulary
            print(f"  ✓ Retraining TF-IDF vectorizer on {len(combined_data)} samples...")
            vectorizer.fit(combined_data['text_processed'])
            
            # Save combined data back for next retrain (so we don't lose previous new data)
            processed_file = DATA_PATH / 'imdb_processed.csv'
            combined_data[['text', 'sentiment']].to_csv(processed_file, index=False)
            print(f"  ✓ Saved updated dataset to {processed_file}")
            
            X = vectorizer.transform(combined_data['text_processed'])
            y = (combined_data['sentiment'] == 'positive').astype(int)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"  ✓ Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")
        except Exception as e:
            print(f"  ✗ Data preparation failed: {e}")
            retrain_status["is_retraining"] = False
            retrain_status["last_retrain_result"] = f"Failed: {str(e)}"
            return
        
        # ── 3. Train all 3 models ───────────────────────────────────────
        models_to_train = {
            'logisticregression': LogisticRegression(max_iter=1000, random_state=42),
            'svm': LinearSVC(max_iter=2000, random_state=42, dual=False),
            'naive_bayes': MultinomialNB()
        }
        
        results = {}
        print("\n  Training models:")
        
        for model_name, m in models_to_train.items():
            try:
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                results[model_name] = {
                    'accuracy': float(accuracy),
                    'f1_score': float(f1),
                    'model': m
                }
                print(f"    - {model_name:20s}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
            except Exception as e:
                print(f"    - {model_name:20s}: FAILED ({str(e)})")
        
        # ── 4. Select best model ─────────────────────────────────────────
        if not results:
            print("  ✗ All models failed to train")
            retrain_status["is_retraining"] = False
            retrain_status["last_retrain_result"] = "Failed: All models training failed"
            return
        
        best_model_name = max(results, key=lambda k: results[k]['accuracy'])
        best_model_obj = results[best_model_name]['model']
        best_accuracy = results[best_model_name]['accuracy']
        best_f1 = results[best_model_name]['f1_score']
        
        print(f"\n  🏆 BEST MODEL: {best_model_name.upper()} (Accuracy={best_accuracy:.4f})")
        
        # ── 5. Validate accuracy threshold ───────────────────────────────
        ACCURACY_THRESHOLD = 0.85
        if best_accuracy < ACCURACY_THRESHOLD:
            print(f"  ⚠️  Accuracy {best_accuracy:.4f} below threshold {ACCURACY_THRESHOLD}")
            print(f"  ❌ New model REJECTED - keeping current model")
            retrain_status["is_retraining"] = False
            retrain_status["last_retrain_result"] = f"Rejected: Accuracy {best_accuracy:.4f} < {ACCURACY_THRESHOLD}"
            return
        
        # ── 6. Backup old model ──────────────────────────────────────────
        try:
            old_model_name = model_metadata.get('current_model', 'logisticregression')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            old_model_path = MODELS_PATH / f'{old_model_name}_sentiment_model.pkl'
            if old_model_path.exists():
                backup_path = BACKUP_PATH / f'{old_model_name}_{timestamp}.pkl'
                shutil.copy(old_model_path, backup_path)
                print(f"  ✓ Old model backed up: {backup_path.name}")
        except Exception as e:
            print(f"  ⚠️  Backup failed (non-critical): {e}")
        
        # ── 7. Save new best model ───────────────────────────────────────
        try:
            model_path = MODELS_PATH / f'{best_model_name}_sentiment_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(best_model_obj, f)
            print(f"  ✓ Saved new model: {model_path.name}")
            
            # Also save the retrained vectorizer
            vectorizer_path = MODELS_PATH / 'tfidf_vectorizer.pkl'
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            print(f"  ✓ Saved retrained vectorizer")
        except Exception as e:
            print(f"  ✗ Failed to save model: {e}")
            retrain_status["is_retraining"] = False
            retrain_status["last_retrain_result"] = f"Failed: {str(e)}"
            return
        
        # ── 8. Update metadata ───────────────────────────────────────────
        try:
            new_metadata = {
                'current_model': best_model_name,
                'retrained_date': datetime.now().isoformat(),
                'data_samples': int(len(combined_data)),
                'all_models': {
                    name: {
                        'accuracy': res['accuracy'],
                        'f1_score': res['f1_score']
                    }
                    for name, res in results.items()
                },
                'metrics': {
                    'accuracy': best_accuracy,
                    'f1_score': best_f1
                },
                'model_name': best_model_name,
                'vocabulary_size': vectorizer.transform(['test']).shape[1]
            }
            
            metadata_path = MODELS_PATH / 'model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(new_metadata, f, indent=2)
            print(f"  ✓ Updated metadata")
        except Exception as e:
            print(f"  ✗ Failed to update metadata: {e}")
            retrain_status["is_retraining"] = False
            retrain_status["last_retrain_result"] = f"Failed: {str(e)}"
            return
        
        # ── 9. Reload model in memory ────────────────────────────────────
        try:
            model, vectorizer, model_metadata = load_model_artifacts()
            print(f"  ✓ Reloaded model in memory")
        except Exception as e:
            print(f"  ✗ Failed to reload model: {e}")
            retrain_status["is_retraining"] = False
            retrain_status["last_retrain_result"] = f"Failed: {str(e)}"
            return
        
        # ── Success ──────────────────────────────────────────────────────
        print(f"\n✅ Retraining completed successfully!")
        print(f"  New best model: {best_model_name} ({best_accuracy:.4f} accuracy)")
        
        retrain_status["is_retraining"] = False
        retrain_status["last_retrain_date"] = datetime.now().isoformat()
        retrain_status["last_retrain_result"] = f"Success: {best_model_name} ({best_accuracy:.4f})"
        
    except Exception as e:
        print(f"\n❌ Unexpected retraining error: {e}")
        retrain_status["is_retraining"] = False
        retrain_status["last_retrain_result"] = f"Failed: {str(e)}"


def get_prediction(text: str) -> Dict:
    """Predict sentiment for a single review."""
    if model is None or vectorizer is None:
        raise RuntimeError("Model not loaded")
    
    # Preprocess
    cleaned_text = preprocess_text(text)
    
    # Vectorize
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Predict
    label = model.predict(text_tfidf)[0]
    
    # Get confidence (probability for positive class)
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_tfidf)[0]
        confidence = float(proba[label])
    else:
        # For SVM without predict_proba, use decision function
        decision = model.decision_function(text_tfidf)[0]
        confidence = float(1 / (1 + np.exp(-decision)))
    
    sentiment = "positive" if label == 1 else "negative"
    
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "label": int(label),
    }


# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API info."""
    return {
        "message": "IMDb Sentiment Analysis API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "batch_predict": "POST/batch_predict",
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_name": model_metadata.get("current_model", "logisticregression"),
        "model_accuracy": model_metadata.get("metrics", {}).get("accuracy", 0.0),
        "vocabulary_size": model_metadata.get("vocabulary_size", 0),
    }


@app.post("/predict", response_model=SentimentPrediction, tags=["Predictions"])
async def predict_sentiment(request: ReviewRequest):
    """
    Predict sentiment for a single review.
    
    **Args:**
    - `text`: Review text (1-10000 characters)
    
    **Returns:**
    - `sentiment`: "positive" or "negative"
    - `confidence`: Probability (0.0-1.0)
    - `label`: 1 (positive) or 0 (negative)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        prediction = get_prediction(request.text)
        return SentimentPrediction(**prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict_sentiment(request: BatchPredictionRequest):
    """
    Predict sentiment for multiple reviews in batch.
    
    **Args:**
    - `reviews`: List of review texts (1-100 reviews)
    
    **Returns:**
    - `predictions`: List of sentiment predictions
    - `processing_time_ms`: Total processing time
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    try:
        predictions = []
        for text in request.reviews:
            pred = get_prediction(text)
            predictions.append(SentimentPrediction(**pred))
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            processing_time_ms=processing_time_ms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model metadata and performance metrics."""
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return model_metadata


@app.post("/admin/upload_and_retrain", response_model=RetrainResponse, tags=["Admin"])
async def upload_and_retrain(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    api_key: str = Header(None)
):
    """
    Admin-only endpoint: Upload CSV with new data, retrain all models.
    
    **Security:**
    - Requires `api_key` header for authentication

    USE 1111 AS API KEY TO ACCESS THIS ENDPOINT
    
    **CSV Format:**
    - Must have columns: 'text', 'sentiment'
    - sentiment must be 'positive' or 'negative'
    
    **Response:**
    - Returns immediately (retraining happens in background)
    - Check /admin/retrain-status to monitor progress
    """
    # ── Authenticate ─────────────────────────────────────────────────────
    ADMIN_API_KEY = "1111"
    if api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized: Invalid API key")
    
    # ── Validate file ────────────────────────────────────────────────────
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    # ── Check if retraining is already running ───────────────────────────
    if retrain_status["is_retraining"]:
        raise HTTPException(
            status_code=429,
            detail="Retraining already in progress. Wait for completion."
        )
    
    # ── Save uploaded file ───────────────────────────────────────────────
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = DATA_PATH / f'upload_{timestamp}.csv'
        
        # Ensure data directory exists
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        
        # Save file
        content = await file.read()
        with open(file_path, 'wb') as f:
            f.write(content)
        
        print(f"✓ File uploaded: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    
    # ── Queue background retraining ──────────────────────────────────────
    background_tasks.add_task(retrain_model_task, str(file_path))
    
    return RetrainResponse(
        status="retraining_started",
        message="Model retraining started in background. Check /admin/retrain-status for progress.",
        file=file.filename
    )


@app.get("/admin/retrain-status", response_model=RetrainStatusResponse, tags=["Admin"])
async def retrain_status_endpoint(api_key: str = Header(None)):
    """
    Check retraining progress and model comparison results.

    USE 1111 AS API KEY TO ACCESS THIS ENDPOINT

    **Returns:**
    - `is_retraining`: True if retraining is in progress
    - `current_model_in_use`: Name of the active model
    - `all_model_accuracies`: Comparison of all 3 models
    - `active_model_accuracy`: Accuracy of current model
    - `last_retrain_date`: When last retrain completed
    """
    # Authenticate
    ADMIN_API_KEY = "1111"
    if api_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized: Invalid API key")
    
    if model_metadata is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Count samples in processed data
        data_file = DATA_PATH / 'imdb_processed.csv'
        data_samples = 0
        if data_file.exists():
            df = pd.read_csv(data_file)
            data_samples = len(df)
    except:
        data_samples = 0
    
    return RetrainStatusResponse(
        is_retraining=retrain_status["is_retraining"],
        current_model_in_use=model_metadata.get('current_model', 'logisticregression'),
        last_retrain_date=model_metadata.get('retrained_date'),
        active_model_accuracy=model_metadata.get('metrics', {}).get('accuracy', 0.0),
        all_model_accuracies=model_metadata.get('all_models', {}),
        data_samples=data_samples
    )


if __name__ == "__main__":
    import uvicorn
    
    print("Starting FastAPI sentiment analysis server...")
    print("📊 API docs available at: http://localhost:8000/docs")
    print("🚀 Server running on http://localhost:8000")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
