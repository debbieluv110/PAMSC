"""
Data Preprocessing Module for School Dropout Prediction

This module handles data cleaning, feature engineering, and preprocessing
for the school dropout prediction model.

Author: Predictive Analytics for Social Cause Project
License: Apache-2.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    A comprehensive data preprocessing class for school dropout prediction.
    
    Handles missing values, feature engineering, encoding, and scaling.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputers = {}
        self.feature_names = []
        
    def load_data(self, filepath):
        """Load data from CSV file."""
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def handle_missing_values(self, df):
        """Handle missing values using appropriate strategies."""
        df_clean = df.copy()
        
        # Numerical columns - use median imputation
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'dropout_risk']
        
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='median')
                    df_clean[col] = self.imputers[col].fit_transform(df_clean[[col]]).ravel()
                else:
                    df_clean[col] = self.imputers[col].transform(df_clean[[col]]).ravel()
        
        # Categorical columns - use mode imputation
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                if col not in self.imputers:
                    self.imputers[col] = SimpleImputer(strategy='most_frequent')
                    df_clean[col] = self.imputers[col].fit_transform(df_clean[[col]]).ravel()
                else:
                    df_clean[col] = self.imputers[col].transform(df_clean[[col]]).ravel()
        
        print(f"Missing values handled. Remaining missing values: {df_clean.isnull().sum().sum()}")
        return df_clean
    
    def create_features(self, df):
        """Create new features based on domain knowledge."""
        df_features = df.copy()
        
        # Academic performance indicators
        df_features['academic_risk_score'] = (
            (df_features['gpa_previous_year'] < 2.0).astype(int) * 3 +
            (df_features['attendance_rate'] < 0.8).astype(int) * 2 +
            (df_features['homework_completion_rate'] < 0.7).astype(int) * 2 +
            (df_features['disciplinary_incidents'] > 2).astype(int) * 1
        )
        
        # Socioeconomic risk indicators
        df_features['socioeconomic_risk_score'] = (
            (df_features['family_income'] < 30000).astype(int) * 2 +
            (df_features['single_parent'] == 1).astype(int) * 1 +
            (df_features['free_lunch_eligible'] == 1).astype(int) * 1 +
            (df_features['family_size'] > 5).astype(int) * 1
        )
        
        # Engagement indicators
        df_features['engagement_score'] = (
            df_features['extracurricular_activities'] +
            (df_features['counseling_sessions'] > 0).astype(int) +
            (df_features['tutoring_hours'] > 0).astype(int)
        )
        
        # Behavioral risk indicators
        df_features['behavioral_risk_score'] = (
            (df_features['absences_last_semester'] > 15).astype(int) * 2 +
            (df_features['late_arrivals'] > 10).astype(int) * 1 +
            (df_features['disciplinary_incidents'] > 3).astype(int) * 2
        )
        
        # Age-grade alignment (assuming grade 10-12 students)
        df_features['age_grade_mismatch'] = (df_features['age'] > 17).astype(int)
        
        # Support system availability
        df_features['support_system_score'] = (
            df_features['counseling_sessions'] +
            df_features['tutoring_hours'] +
            (df_features['extracurricular_activities'] > 0).astype(int)
        )
        
        print(f"Feature engineering completed. New features created: 6")
        return df_features
    
    def encode_categorical_variables(self, df):
        """Encode categorical variables."""
        df_encoded = df.copy()
        
        categorical_cols = ['gender', 'ethnicity', 'parent_education', 'school_type']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                else:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        
        print(f"Categorical encoding completed for {len(categorical_cols)} columns")
        return df_encoded
    
    def scale_features(self, X_train, X_test=None):
        """Scale numerical features."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def prepare_features_and_target(self, df):
        """Prepare features and target variable."""
        # Remove non-predictive columns
        columns_to_drop = ['student_id']
        
        # Separate features and target
        X = df.drop(columns=columns_to_drop + ['dropout_risk'])
        y = df['dropout_risk']
        
        self.feature_names = X.columns.tolist()
        
        print(f"Features prepared. Shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def preprocess_pipeline(self, filepath, output_dir='data/processed/'):
        """Complete preprocessing pipeline."""
        print("Starting preprocessing pipeline...")
        
        # Load data
        df = self.load_data(filepath)
        if df is None:
            return None, None
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Create features
        df = self.create_features(df)
        
        # Encode categorical variables
        df = self.encode_categorical_variables(df)
        
        # Prepare features and target
        X, y = self.prepare_features_and_target(df)
        
        # Save processed data
        os.makedirs(output_dir, exist_ok=True)
        
        # Save features and labels
        features_path = os.path.join(output_dir, 'features.csv')
        labels_path = os.path.join(output_dir, 'labels.csv')
        
        pd.DataFrame(X, columns=self.feature_names).to_csv(features_path, index=False)
        pd.DataFrame(y, columns=['dropout_risk']).to_csv(labels_path, index=False)
        
        print(f"Processed data saved to {output_dir}")
        print("Preprocessing pipeline completed successfully!")
        
        return X, y

def main():
    """Main function to run preprocessing."""
    preprocessor = DataPreprocessor()
    
    # Run preprocessing pipeline
    X, y = preprocessor.preprocess_pipeline('data/raw/sample_social.csv')
    
    if X is not None and y is not None:
        print(f"\nPreprocessing Summary:")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature names: {preprocessor.feature_names}")

if __name__ == "__main__":
    main()
