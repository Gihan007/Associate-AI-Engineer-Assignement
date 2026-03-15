# AI/ML Project Structure

This project follows a standard machine learning project structure designed for maintaining clean, scalable, and reproducible code.

## Directory Structure

```
.
├── src/                    # Source code
│   ├── data/              # Data loading and processing
│   ├── models/            # Model definitions and training
│   ├── features/          # Feature engineering
│   └── utils/             # Utility functions
├── notebooks/             # Jupyter notebooks for exploration
├── data/
│   ├── raw/              # Original, immutable data
│   └── processed/        # Cleaned, processed data
├── models/               # Trained model artifacts
├── results/              # Analysis results and outputs
├── tests/                # Unit and integration tests
├── configs/              # Configuration files
├── docs/                 # Documentation
├── requirements.txt      # Python dependencies
├── setup.py             # Package setup
├── main.py              # Main entry point
└── README.md            # Project documentation
```

## Key Modules

### src/data/
- **loader.py**: Data loading and saving utilities
  - `load_data()`: Load data from CSV
  - `save_data()`: Save data to CSV
  - `split_data()`: Train/test splitting

### src/features/
- **engineering.py**: Feature engineering functions
  - `handle_missing_values()`: Handle missing data
  - `normalize_features()`: Feature normalization
  - `encode_categorical()`: Categorical encoding

### src/models/
- **base.py**: Base model class and evaluation
  - `BaseModel`: Abstract base class for models
  - `evaluate_model()`: Evaluation metrics

### src/utils/
- **helpers.py**: Utility functions
  - `setup_logging()`: Configure logging
  - `save_config()`: Save configurations
  - `load_config()`: Load configurations

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the main script**:
   ```bash
   python main.py
   ```

3. **Run tests**:
   ```bash
   python -m pytest tests/
   ```

4. **Use Jupyter notebooks**:
   ```bash
   jupyter notebook
   ```

## Development

- Place raw data in `data/raw/`
- Process and save cleaned data to `data/processed/`
- Save trained models in `models/`
- Store results and outputs in `results/`
- Create notebooks in `notebooks/` for exploration and analysis

## Configuration

- Main config file: `configs/config.yaml`
- Environment variables: `.env` (copy from `.env.example`)
