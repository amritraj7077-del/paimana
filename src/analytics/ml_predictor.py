"""
ML-Based Delay Prediction Model
Uses historical project data to predict completion dates
"""

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DelayPredictor:
    """
    Machine Learning model to predict project completion delays
    """
    
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
    
    def prepare_features(self, df):
        """
        Prepare features for ML model
        
        Features:
        - physical_progress_percent
        - expenditure_percent (expenditure / sanctioned_cost)
        - project_category (encoded)
        - sanctioned_cost (normalized)
        """
        features = pd.DataFrame()
        
        # Progress percentage
        features['progress'] = df['physical_progress_percent']
        
        # Expenditure percentage
        features['expenditure_percent'] = (df['expenditure_to_date'] / df['sanctioned_cost'] * 100)
        
        # Normalized cost (in millions)
        features['cost_millions'] = df['sanctioned_cost'] / 1000000
        
        # Category encoding (simple label encoding)
        category_map = {cat: idx for idx, cat in enumerate(df['category'].unique())}
        features['category_code'] = df['category'].map(category_map)
        
        return features.fillna(0)
    
    def train(self, df):
        """
        Train the prediction model on historical data
        
        Args:
            df: DataFrame with project data including delay_days
        """
        if 'delay_days' not in df.columns:
            logger.warning("No delay_days column found. Cannot train model.")
            return False
        
        # Prepare features and target
        X = self.prepare_features(df)
        y = df['delay_days']
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate model score
        score = self.model.score(X, y)
        logger.info(f"Model trained with R² score: {score:.3f}")
        
        return True
    
    def predict_delay(self, df):
        """
        Predict delays for projects
        
        Args:
            df: DataFrame with project data
            
        Returns:
            Array of predicted delay days
        """
        if not self.is_trained:
            logger.warning("Model not trained yet. Training on current data...")
            self.train(df)
        
        X = self.prepare_features(df)
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_completion_date(self, df):
        """
        Predict actual completion dates for projects
        
        Args:
            df: DataFrame with project data
            
        Returns:
            DataFrame with predicted completion dates
        """
        predictions = self.predict_delay(df)
        
        result = df.copy()
        result['predicted_delay_days'] = predictions.round(0).astype(int)
        
        # Calculate predicted completion date
        if 'planned_completion_date' in df.columns:
            result['planned_completion_date'] = pd.to_datetime(result['planned_completion_date'], errors='coerce')
            result['predicted_completion_date'] = result['planned_completion_date'] + pd.to_timedelta(
                result['predicted_delay_days'], unit='D'
            )
        
        return result
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if not self.is_trained:
            return None
        
        feature_names = ['progress', 'expenditure_percent', 'cost_millions', 'category_code']
        importance = dict(zip(feature_names, self.model.coef_))
        
        return importance


def main():
    """Demo usage"""
    from src.scrapers.paimana_scraper import PAIMANAScraper
    from src.analytics.delay_detector import DelayAnalyzer
    
    # Generate sample data
    scraper = PAIMANAScraper(state='maharashtra')
    projects_df = scraper.extract_projects()
    
    # Calculate actual delays
    analyzer = DelayAnalyzer()
    projects_df = analyzer.calculate_delay_days(projects_df)
    projects_df = analyzer.calculate_cost_overrun(projects_df)
    
    # Train and predict
    predictor = DelayPredictor()
    predictor.train(projects_df)
    
    predictions_df = predictor.predict_completion_date(projects_df)
    
    print("\n=== ML DELAY PREDICTIONS ===")
    print(f"\nModel trained on {len(projects_df)} projects")
    print(f"\nSample predictions:")
    print(predictions_df[['project_id', 'project_name', 'physical_progress_percent', 
                          'predicted_delay_days', 'predicted_completion_date']].head(10))
    
    # Feature importance
    importance = predictor.get_feature_importance()
    print(f"\nFeature Importance:")
    for feature, coef in importance.items():
        print(f"  {feature}: {coef:.2f}")


if __name__ == '__main__':
    main()
