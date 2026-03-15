# Docker Setup Guide

## Quick Start (1 Command)

```bash
docker compose up --build
```

That's it! Docker will automatically:
1. Install all Python dependencies
2. Download the IMDb dataset (2GB)
3. Run Notebook 01 (preprocessing & EDA)
4. Run Notebook 02 (model training)
5. Start the FastAPI server

**API will be ready at:** `http://localhost:8000/docs`

---

## What Happens During First Build

```
docker compose up --build

Building imdb-sentiment-api...
Step 1/20: FROM python:3.9-slim
Step 2/20: WORKDIR /app
...
Step 12/20: RUN python -m papermill notebooks/01_data_import_preprocessing_eda.ipynb ...
  📊 Step 1: Data Import, Preprocessing & EDA...
  ⏳ Downloading dataset... (takes 5-10 minutes)
  ✅ Data preprocessing complete!

Step 13/20: RUN python -m papermill notebooks/02_model_training.ipynb ...
  🤖 Step 2: Model Training & Evaluation...
  ⏳ Training Logistic Regression... (takes 1 min)
  ⏳ Training SVM... (takes 1 min)
  ⏳ Training Naive Bayes... (takes 1 min)
  ✅ Model training complete!

Step 14/20: EXPOSE 8000
Step 15/20: HEALTHCHECK ...
Step 16/20: CMD ["python", "-m", "uvicorn", ...]

✅ Successfully built imdb-sentiment-api
🚀 FastAPI listening on 0.0.0.0:8000
```

**Total time:** ~15-20 minutes first time

---

## Subsequent Runs (Much Faster)

```bash
docker compose up
```

**Time:** ~10 seconds (uses cached layers, no retraining)

---

## Accessing the API

### Web Interface (Recommended)
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Command Line (curl)

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}'

# Batch prediction
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      "Great movie!",
      "Terrible film.",
      "Pretty good."
    ]
  }'

# Health check
curl http://localhost:8000/health
```

### Python

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
    json={"reviews": ["Amazing!", "Terrible.", "Good."]}
)
print(response.json())
```

---

## Stopping the Container

```bash
docker compose down
```

---

## Troubleshooting

### Issue: "Model not loaded" error
**Solution:** Wait for the Docker build to complete. Check logs:
```bash
docker compose logs
```

### Issue: Port 8000 already in use
**Solution:** Change the port in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"  # Use 8001 instead
```

### Issue: Out of disk space
**Solution:** The Docker image is ~5GB. Clean up:
```bash
docker system prune -a
```

### Issue: Want to rebuild from scratch
```bash
docker compose down -v        # Delete volumes
docker compose up --build     # Rebuild
```

---

## Environment Variables

Edit `docker-compose.yml` to customize:

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - ADMIN_API_KEY=your-secret-key-here   # Change this!
```

Or use `.env` file:
```
ADMIN_API_KEY=your-secret-key-here
```

---

## Production Deployment

For production Docker deployment:

1. **Change API key** in `docker-compose.yml`
2. **Use a reverse proxy** (nginx) in front of FastAPI
3. **Enable HTTPS/SSL** certificates
4. **Use docker secrets** for sensitive data
5. **Set restart policy** appropriately

Example for production:
```yaml
sentiment-api:
  build: .
  restart: always  # Always restart if container dies
  environment:
    - ADMIN_API_KEY=${ADMIN_API_KEY}  # From .env
  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 60s
```

---

## Volume Persistence

The Docker container persists data in these volumes:

```yaml
volumes:
  - ./models:/app/models           # Trained models
  - ./data:/app/data               # Processed dataset
  - ./results:/app/results         # Visualizations
```

These folders are created automatically if they don't exist.

---

## Need Help?

Check the logs:
```bash
docker compose logs -f sentiment-api
```

Stop and restart:
```bash
docker compose down && docker compose up --build
```
