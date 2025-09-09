"""
Machine Learning Training Module for School Dropout Prediction

This module handles model training, evaluation, and prediction for multiple
ML algorithms including Logistic Regression, Random Forest, and XGBoost.

Author: Predictive Analytics for Social Cause Project
License: Apache-2.0
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix, roc_curve)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

class MLTrainer:
    """
    Comprehensive ML training class for school dropout prediction.
    
    Handles multiple algorithms, hyperparameter tuning, and evaluation.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.predictions = {}
        self.feature_names = []
        
    def load_processed_data(self, features_path, labels_path):
        """Load preprocessed features and labels."""
        try:
            X = pd.read_csv(features_path)
            y = pd.read_csv(labels_path)['dropout_risk']
            
            self.feature_names = X.columns.tolist()
            
            print(f"Data loaded successfully.")
            print(f"Features shape: {X.shape}")
            print(f"Labels shape: {y.shape}")
            print(f"Class distribution: {y.value_counts().to_dict()}")
            
            return X, y
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None, None
    
    def split_data(self, X, y, test_size=0.2, val_size=0.2):
        """Split data into train, validation, and test sets."""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        print(f"Data split completed:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train and tune Logistic Regression model."""
        print("Training Logistic Regression...")
        
        # Hyperparameter grid
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        
        # Grid search with cross-validation
        lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_lr = grid_search.best_estimator_
        
        # Validation predictions
        val_pred = best_lr.predict(X_val)
        val_pred_proba = best_lr.predict_proba(X_val)[:, 1]
        
        # Store results
        self.models['logistic_regression'] = best_lr
        self.results['logistic_regression'] = {
            'best_params': grid_search.best_params_,
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_auc': roc_auc_score(y_val, val_pred_proba)
        }
        
        print(f"Logistic Regression - Best params: {grid_search.best_params_}")
        print(f"Validation AUC: {self.results['logistic_regression']['val_auc']:.4f}")
        
        return best_lr
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train and tune Random Forest model."""
        print("Training Random Forest...")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Grid search with cross-validation
        rf = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        
        # Validation predictions
        val_pred = best_rf.predict(X_val)
        val_pred_proba = best_rf.predict_proba(X_val)[:, 1]
        
        # Store results
        self.models['random_forest'] = best_rf
        self.results['random_forest'] = {
            'best_params': grid_search.best_params_,
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_auc': roc_auc_score(y_val, val_pred_proba)
        }
        
        print(f"Random Forest - Best params: {grid_search.best_params_}")
        print(f"Validation AUC: {self.results['random_forest']['val_auc']:.4f}")
        
        return best_rf
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train and tune XGBoost model."""
        print("Training XGBoost...")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        # Grid search with cross-validation
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state, 
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=3, scoring='roc_auc', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_xgb = grid_search.best_estimator_
        
        # Validation predictions
        val_pred = best_xgb.predict(X_val)
        val_pred_proba = best_xgb.predict_proba(X_val)[:, 1]
        
        # Store results
        self.models['xgboost'] = best_xgb
        self.results['xgboost'] = {
            'best_params': grid_search.best_params_,
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred),
            'val_auc': roc_auc_score(y_val, val_pred_proba)
        }
        
        print(f"XGBoost - Best params: {grid_search.best_params_}")
        print(f"Validation AUC: {self.results['xgboost']['val_auc']:.4f}")
        
        return best_xgb
    
    def evaluate_models(self, X_test, y_test, output_dir='results/'):
        """Evaluate all trained models on test set."""
        print("Evaluating models on test set...")
        
        test_results = {}
        
        for model_name, model in self.models.items():
            # Test predictions
            test_pred = model.predict(X_test)
            test_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            test_results[model_name] = {
                'accuracy': accuracy_score(y_test, test_pred),
                'precision': precision_score(y_test, test_pred),
                'recall': recall_score(y_test, test_pred),
                'f1': f1_score(y_test, test_pred),
                'auc': roc_auc_score(y_test, test_pred_proba)
            }
            
            # Store predictions for later analysis
            self.predictions[model_name] = {
                'y_true': y_test,
                'y_pred': test_pred,
                'y_pred_proba': test_pred_proba
            }
            
            print(f"{model_name} - Test AUC: {test_results[model_name]['auc']:.4f}")
        
        # Save test results
        self.results['test_results'] = test_results
        
        return test_results
    
    def plot_model_comparison(self, output_dir='results/'):
        """Plot model comparison charts."""
        if 'test_results' not in self.results:
            print("No test results available. Run evaluate_models first.")
            return
        
        # Prepare data for plotting
        models = list(self.results['test_results'].keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        # Create comparison dataframe
        comparison_data = []
        for model in models:
            for metric in metrics:
                comparison_data.append({
                    'Model': model.replace('_', ' ').title(),
                    'Metric': metric.upper(),
                    'Score': self.results['test_results'][model][metric]
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(data=comparison_df, x='Metric', y='Score', hue='Model')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # ROC curves
        plt.figure(figsize=(10, 8))
        for model_name in models:
            pred_data = self.predictions[model_name]
            fpr, tpr, _ = roc_curve(pred_data['y_true'], pred_data['y_pred_proba'])
            auc_score = self.results['test_results'][model_name]['auc']
            plt.plot(fpr, tpr, label=f'{model_name.replace("_", " ").title()} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Model comparison plots saved to {output_dir}")
    
    def save_results(self, output_dir='results/'):
        """Save all results and predictions."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = os.path.join(output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save predictions as CSV
        for model_name, pred_data in self.predictions.items():
            pred_df = pd.DataFrame({
                'y_true': pred_data['y_true'],
                'y_pred': pred_data['y_pred'],
                'y_pred_proba': pred_data['y_pred_proba']
            })
            pred_file = os.path.join(output_dir, f'{model_name}_predictions.csv')
            pred_df.to_csv(pred_file, index=False)
        
        print(f"Results saved to {output_dir}")
        print("Generated files:")
        print("- metrics.json: All model metrics and parameters")
        print("- *_predictions.csv: Predictions for each model")
        print("- model_comparison.png: Performance comparison chart")
        print("- roc_curves.png: ROC curves comparison")
    
    def train_all_models(self, features_path='data/processed/features.csv', 
                        labels_path='data/processed/labels.csv', 
                        output_dir='results/'):
        """Complete training pipeline for all models."""
        print("Starting complete ML training pipeline...")
        
        # Load data
        X, y = self.load_processed_data(features_path, labels_path)
        if X is None or y is None:
            return
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Train all models
        self.train_logistic_regression(X_train, y_train, X_val, y_val)
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test, output_dir)
        
        # Plot comparisons
        self.plot_model_comparison(output_dir)
        
        # Save results
        self.save_results(output_dir)
        
        print("\nTraining pipeline completed successfully!")
        
        # Print summary
        print("\n=== MODEL PERFORMANCE SUMMARY ===")
        for model_name, metrics in self.results['test_results'].items():
            print(f"\n{model_name.replace('_', ' ').title()}:")
            for metric, score in metrics.items():
                print(f"  {metric.upper()}: {score:.4f}")

def main():
    """Main function to run ML training."""
    trainer = MLTrainer()
    
    # Run complete training pipeline
    trainer.train_all_models()

if __name__ == "__main__":
    main()
