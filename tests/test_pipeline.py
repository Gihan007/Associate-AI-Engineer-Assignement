"""
Unit tests for the project.
"""

import unittest
import numpy as np
import pandas as pd
from src.data.loader import load_data, split_data
from src.features.engineering import normalize_features


class TestDataLoader(unittest.TestCase):
    """Test cases for data loading functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_split_data(self):
        """Test data splitting."""
        train, test = split_data(self.test_df, test_size=0.2)
        self.assertEqual(len(train) + len(test), len(self.test_df))


class TestFeatureEngineering(unittest.TestCase):
    """Test cases for feature engineering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
        })
    
    def test_normalize_features(self):
        """Test feature normalization."""
        normalized = normalize_features(self.test_df, ['feature1', 'feature2'])
        self.assertTrue((normalized['feature1'] >= 0).all())
        self.assertTrue((normalized['feature1'] <= 1).all())


if __name__ == '__main__':
    unittest.main()
