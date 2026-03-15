"""
Feature engineering functions.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def handle_missing_values(
    df: pd.DataFrame, 
    strategy: str = "mean"
) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Strategy to handle missing values ('mean', 'median', 'drop', 'forward_fill')
        
    Returns:
        DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    if strategy == "drop":
        return df_copy.dropna()
    elif strategy == "forward_fill":
        return df_copy.fillna(method="ffill")
    elif strategy in ["mean", "median"]:
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        df_copy[numeric_cols] = df_copy[numeric_cols].fillna(
            df_copy[numeric_cols].agg(strategy)
        )
    
    return df_copy


def normalize_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize specified features to [0, 1] range.
    
    Args:
        df: Input DataFrame
        columns: List of columns to normalize
        
    Returns:
        DataFrame with normalized features
    """
    from sklearn.preprocessing import MinMaxScaler
    
    df_copy = df.copy()
    scaler = MinMaxScaler()
    df_copy[columns] = scaler.fit_transform(df[columns])
    
    return df_copy


def encode_categorical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Encode categorical features.
    
    Args:
        df: Input DataFrame
        columns: List of categorical columns to encode
        
    Returns:
        DataFrame with encoded categorical features
    """
    df_copy = df.copy()
    df_copy = pd.get_dummies(df_copy, columns=columns, drop_first=True)

    return df_copy


def detect_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Detect outliers via the IQR method.

    Returns a boolean DataFrame of the same shape as ``df[columns]``;
    True indicates an outlier value.
    """
    mask = pd.DataFrame(False, index=df.index, columns=columns)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        mask[col] = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    return mask


def cap_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Cap outliers at IQR bounds (Winsorization).

    Values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR are clipped to the
    respective fence, leaving the rest of the data unchanged.
    """
    df_copy = df.copy()
    for col in columns:
        q1 = df_copy[col].quantile(0.25)
        q3 = df_copy[col].quantile(0.75)
        iqr = q3 - q1
        df_copy[col] = df_copy[col].clip(
            lower=q1 - 1.5 * iqr,
            upper=q3 + 1.5 * iqr,
        )
    return df_copy


def standardize_features(df: pd.DataFrame, columns: List[str]):
    """
    Standardize features to zero mean and unit variance (StandardScaler).

    Returns a (transformed_df, fitted_scaler) tuple so the scaler can be
    reused later for transforming test / inference data.
    """
    from sklearn.preprocessing import StandardScaler

    df_copy = df.copy()
    scaler = StandardScaler()
    df_copy[columns] = scaler.fit_transform(df_copy[columns])
    return df_copy, scaler
