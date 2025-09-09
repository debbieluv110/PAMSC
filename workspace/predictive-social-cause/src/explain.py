"""
Model Explainability Module for School Dropout Prediction

This module provides feature importance analysis and SHAP (SHapley Additive exPlanations)
values for understanding model predictions and feature contributions.

Author: Predictive Analytics for Social Cause Project
License: Apache-2.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    """
    Comprehensive model explainability class for school dropout prediction.
    
    Provides feature importance analysis, SHAP values, and interpretability insights.
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.models = {}
        self.feature_names = []
        self.explainers = {}
        self.shap_values = {}
        self.feature_importance = {}
        
    def load_trained_models(self, models_dict):
        """Load trained models for explanation."""
        self.models = models_dict
        print(f"Loaded {len(self.models)} trained models for explanation")
        
    def load_data(self, features_path, labels_path):
        """Load processed data for explanation."""
        try:
            X = pd.read_csv(features_path)
            y = pd.read_csv(labels_path)['dropout_risk']
            
            self.feature_names = X.columns.tolist()
            
            print(f"Data loaded for explanation:")
            print(f"Features shape: {X.shape}")
            print(f"Labels shape: {y.shape}")
            
            return X, y
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    
    def calculate_permutation_importance(self, X, y, n_repeats=10, random_state=42):
        """Calculate permutation importance for all models."""
        print("Calculating permutation importance...")
        
        for model_name, model in self.models.items():
            print(f"Processing {model_name}...")
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X, y, n_repeats=n_repeats, 
                random_state=random_state, scoring='roc_auc'
            )
            
            # Store results
            self.feature_importance[model_name] = {
                'importances_mean': perm_importance.importances_mean,
                'importances_std': perm_importance.importances_std,
                'feature_names': self.feature_names
            }
            
        print("Permutation importance calculation completed")
    
    def calculate_shap_values(self, X, sample_size=500, random_state=42):
        """Calculate SHAP values for model interpretability."""
        print("Calculating SHAP values...")
        
        # Use a sample for SHAP calculation to speed up computation
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=random_state)
        else:
            X_sample = X
        
        for model_name, model in self.models.items():
            print(f"Processing SHAP for {model_name}...")
            
            try:
                # Choose appropriate explainer based on model type
                if 'xgboost' in model_name.lower():
                    explainer = shap.TreeExplainer(model)
                elif 'random_forest' in model_name.lower():
                    explainer = shap.TreeExplainer(model)
                else:
                    # For linear models like logistic regression
                    explainer = shap.LinearExplainer(model, X_sample)
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(X_sample)
                
                # Handle different SHAP value formats
                if isinstance(shap_values, list):
                    # For binary classification, take positive class
                    shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                
                self.explainers[model_name] = explainer
                self.shap_values[model_name] = {
                    'shap_values': shap_values,
                    'data': X_sample,
                    'feature_names': self.feature_names
                }
                
                print(f"SHAP values calculated for {model_name}")
                
            except Exception as e:
                print(f"Error calculating SHAP for {model_name}: {e}")
                # Fallback to permutation importance only
                continue
        
        print("SHAP values calculation completed")
    
    def plot_feature_importance(self, output_dir='results/', top_n=15):
        """Plot feature importance for all models."""
        if not self.feature_importance:
            print("No feature importance data available. Run calculate_permutation_importance first.")
            return
        
        n_models = len(self.feature_importance)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, importance_data) in enumerate(self.feature_importance.items()):
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': importance_data['feature_names'],
                'importance': importance_data['importances_mean'],
                'std': importance_data['importances_std']
            }).sort_values('importance', ascending=True).tail(top_n)
            
            # Plot
            axes[idx].barh(range(len(importance_df)), importance_df['importance'], 
                          xerr=importance_df['std'], alpha=0.7)
            axes[idx].set_yticks(range(len(importance_df)))
            axes[idx].set_yticklabels(importance_df['feature'])
            axes[idx].set_xlabel('Permutation Importance')
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}\nFeature Importance')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Feature importance plot saved to {output_dir}")
    
    def plot_shap_summary(self, output_dir='results/', max_display=15):
        """Plot SHAP summary plots for all models."""
        if not self.shap_values:
            print("No SHAP values available. Run calculate_shap_values first.")
            return
        
        for model_name, shap_data in self.shap_values.items():
            try:
                plt.figure(figsize=self.figsize)
                
                # SHAP summary plot
                shap.summary_plot(
                    shap_data['shap_values'], 
                    shap_data['data'], 
                    feature_names=shap_data['feature_names'],
                    max_display=max_display,
                    show=False
                )
                
                plt.title(f'SHAP Summary Plot - {model_name.replace("_", " ").title()}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shap_summary_{model_name}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.show()
                
                # SHAP bar plot (feature importance)
                plt.figure(figsize=self.figsize)
                shap.summary_plot(
                    shap_data['shap_values'], 
                    shap_data['data'], 
                    feature_names=shap_data['feature_names'],
                    plot_type="bar",
                    max_display=max_display,
                    show=False
                )
                
                plt.title(f'SHAP Feature Importance - {model_name.replace("_", " ").title()}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shap_importance_{model_name}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.show()
                
            except Exception as e:
                print(f"Error plotting SHAP for {model_name}: {e}")
                continue
        
        print(f"SHAP plots saved to {output_dir}")
    
    def plot_shap_waterfall(self, output_dir='results/', instance_idx=0):
        """Plot SHAP waterfall plots for specific instances."""
        if not self.shap_values:
            print("No SHAP values available. Run calculate_shap_values first.")
            return
        
        for model_name, shap_data in self.shap_values.items():
            try:
                # Check if we have enough instances
                if instance_idx >= len(shap_data['data']):
                    print(f"Instance index {instance_idx} out of range for {model_name}")
                    continue
                
                plt.figure(figsize=self.figsize)
                
                # Create explanation object for waterfall plot
                if hasattr(shap, 'Explanation'):
                    explanation = shap.Explanation(
                        values=shap_data['shap_values'][instance_idx],
                        base_values=np.mean(shap_data['shap_values']),
                        data=shap_data['data'].iloc[instance_idx].values,
                        feature_names=shap_data['feature_names']
                    )
                    
                    shap.waterfall_plot(explanation, show=False)
                else:
                    # Fallback for older SHAP versions
                    shap.force_plot(
                        self.explainers[model_name].expected_value,
                        shap_data['shap_values'][instance_idx],
                        shap_data['data'].iloc[instance_idx],
                        feature_names=shap_data['feature_names'],
                        matplotlib=True,
                        show=False
                    )
                
                plt.title(f'SHAP Waterfall Plot - {model_name.replace("_", " ").title()}\nInstance {instance_idx}')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shap_waterfall_{model_name}_instance_{instance_idx}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.show()
                
            except Exception as e:
                print(f"Error plotting SHAP waterfall for {model_name}: {e}")
                continue
        
        print(f"SHAP waterfall plots saved to {output_dir}")
    
    def generate_feature_insights(self, output_dir='results/'):
        """Generate insights about important features."""
        if not self.feature_importance:
            print("No feature importance data available.")
            return
        
        insights = {}
        
        for model_name, importance_data in self.feature_importance.items():
            # Get top features
            feature_df = pd.DataFrame({
                'feature': importance_data['feature_names'],
                'importance': importance_data['importances_mean'],
                'std': importance_data['importances_std']
            }).sort_values('importance', ascending=False)
            
            top_features = feature_df.head(10)
            
            insights[model_name] = {
                'top_features': top_features.to_dict('records'),
                'most_important_feature': top_features.iloc[0]['feature'],
                'importance_score': top_features.iloc[0]['importance'],
                'total_features': len(feature_df),
                'significant_features': len(feature_df[feature_df['importance'] > 0.01])
            }
        
        # Save insights
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'feature_insights.json'), 'w') as f:
            json.dump(insights, f, indent=2)
        
        # Generate text report
        with open(os.path.join(output_dir, 'feature_insights_report.txt'), 'w') as f:
            f.write("=== FEATURE IMPORTANCE INSIGHTS REPORT ===\n\n")
            
            for model_name, model_insights in insights.items():
                f.write(f"Model: {model_name.replace('_', ' ').title()}\n")
                f.write(f"Most Important Feature: {model_insights['most_important_feature']}\n")
                f.write(f"Importance Score: {model_insights['importance_score']:.4f}\n")
                f.write(f"Significant Features (>0.01): {model_insights['significant_features']}/{model_insights['total_features']}\n\n")
                
                f.write("Top 10 Features:\n")
                for i, feature_info in enumerate(model_insights['top_features'], 1):
                    f.write(f"{i:2d}. {feature_info['feature']}: {feature_info['importance']:.4f} (±{feature_info['std']:.4f})\n")
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"Feature insights saved to {output_dir}")
        return insights
    
    def create_model_comparison_insights(self, output_dir='results/'):
        """Compare feature importance across models."""
        if not self.feature_importance:
            print("No feature importance data available.")
            return
        
        # Create comparison dataframe
        comparison_data = []
        
        for model_name, importance_data in self.feature_importance.items():
            for feature, importance in zip(importance_data['feature_names'], 
                                         importance_data['importances_mean']):
                comparison_data.append({
                    'model': model_name,
                    'feature': feature,
                    'importance': importance
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Pivot for comparison
        pivot_df = comparison_df.pivot(index='feature', columns='model', values='importance')
        pivot_df = pivot_df.fillna(0)
        
        # Plot comparison heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(pivot_df.T, annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Feature Importance Comparison Across Models')
        plt.xlabel('Features')
        plt.ylabel('Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save comparison data
        pivot_df.to_csv(os.path.join(output_dir, 'feature_importance_comparison.csv'))
        
        print(f"Model comparison insights saved to {output_dir}")
        return pivot_df
    
    def explain_models(self, features_path='data/processed/features.csv', 
                      labels_path='data/processed/labels.csv',
                      models_dict=None, output_dir='results/'):
        """Complete model explanation pipeline."""
        print("Starting model explanation pipeline...")
        
        # Load models if provided
        if models_dict:
            self.load_trained_models(models_dict)
        
        if not self.models:
            print("No models available for explanation. Please provide trained models.")
            return
        
        # Load data
        X, y = self.load_data(features_path, labels_path)
        if X is None or y is None:
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate feature importance
        self.calculate_permutation_importance(X, y)
        
        # Calculate SHAP values
        self.calculate_shap_values(X)
        
        # Generate plots
        self.plot_feature_importance(output_dir)
        self.plot_shap_summary(output_dir)
        self.plot_shap_waterfall(output_dir, instance_idx=0)
        self.plot_shap_waterfall(output_dir, instance_idx=1)
        
        # Generate insights
        insights = self.generate_feature_insights(output_dir)
        comparison_df = self.create_model_comparison_insights(output_dir)
        
        print("\nModel explanation pipeline completed!")
        print("Generated files:")
        print("- feature_importance.png: Permutation importance plots")
        print("- shap_summary_*.png: SHAP summary plots for each model")
        print("- shap_importance_*.png: SHAP feature importance plots")
        print("- shap_waterfall_*.png: SHAP waterfall plots for sample instances")
        print("- feature_importance_comparison.png: Cross-model comparison heatmap")
        print("- feature_insights.json: Detailed feature importance insights")
        print("- feature_insights_report.txt: Human-readable insights report")
        print("- feature_importance_comparison.csv: Feature importance comparison data")
        
        return insights, comparison_df

def main():
    """Main function to run model explanation."""
    explainer = ModelExplainer()
    
    # Note: This would typically be called with trained models
    # explainer.explain_models(models_dict=trained_models)
    print("ModelExplainer initialized. Use explain_models() method with trained models.")

if __name__ == "__main__":
    main()
