"""
Data loading utilities for the ML project.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    return pd.read_csv(filepath)


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Output file path
    """
    df.to_csv(filepath, index=False)


def split_data(
    df: pd.DataFrame, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    return train, test


def get_data_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a comprehensive data profile report.

    Returns a DataFrame with dtype, null count, null %, unique count, and
    unique % for every column — designed as a quick EDA overview table.
    """
    n = len(df)
    return pd.DataFrame({
        "dtype":      df.dtypes,
        "non_null":   df.notnull().sum(),
        "null_count": df.isnull().sum(),
        "null_pct":   (df.isnull().sum() / n * 100).round(2),
        "unique":     df.nunique(),
        "unique_pct": (df.nunique() / n * 100).round(2),
    })
