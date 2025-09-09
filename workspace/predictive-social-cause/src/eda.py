"""
Exploratory Data Analysis Module for School Dropout Prediction

This module provides automated EDA functionality including descriptive statistics,
visualizations, and data quality reports.

Author: Predictive Analytics for Social Cause Project
License: Apache-2.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

class EDAAnalyzer:
    """
    Comprehensive EDA class for school dropout prediction analysis.
    
    Generates automated plots, statistics, and data quality reports.
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self, filepath):
        """Load data from CSV file."""
        try:
            df = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def generate_data_summary(self, df, output_dir='results/'):
        """Generate comprehensive data summary."""
        print("Generating data summary...")
        
        # Basic info
        summary = {
            'Dataset Shape': df.shape,
            'Total Features': df.shape[1],
            'Total Samples': df.shape[0],
            'Memory Usage (MB)': df.memory_usage(deep=True).sum() / 1024**2,
            'Duplicate Rows': df.duplicated().sum(),
            'Missing Values': df.isnull().sum().sum()
        }
        
        # Data types
        dtype_summary = df.dtypes.value_counts().to_dict()
        
        # Missing values by column
        missing_summary = df.isnull().sum().sort_values(ascending=False)
        missing_summary = missing_summary[missing_summary > 0].to_dict()
        
        # Numerical summary
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_summary = df[numerical_cols].describe().round(3)
        
        # Categorical summary
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        categorical_summary = {}
        for col in categorical_cols:
            categorical_summary[col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].empty else 'N/A',
                'frequency': df[col].value_counts().iloc[0] if not df[col].empty else 0
            }
        
        # Save summaries
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV files
        numerical_summary.to_csv(os.path.join(output_dir, 'numerical_summary.csv'))
        
        with open(os.path.join(output_dir, 'data_summary.txt'), 'w') as f:
            f.write("=== DATA SUMMARY REPORT ===\n\n")
            f.write("Basic Information:\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\nData Types:\n")
            for dtype, count in dtype_summary.items():
                f.write(f"{dtype}: {count} columns\n")
            
            f.write("\nMissing Values by Column:\n")
            for col, missing in missing_summary.items():
                f.write(f"{col}: {missing} ({missing/len(df)*100:.1f}%)\n")
            
            f.write("\nCategorical Variables Summary:\n")
            for col, info in categorical_summary.items():
                f.write(f"{col}: {info['unique_values']} unique values, most frequent: {info['most_frequent']}\n")
        
        print(f"Data summary saved to {output_dir}")
        return summary
    
    def plot_target_distribution(self, df, target_col='dropout_risk', output_dir='results/'):
        """Plot target variable distribution."""
        plt.figure(figsize=self.figsize)
        
        # Count plot
        plt.subplot(2, 2, 1)
        target_counts = df[target_col].value_counts()
        plt.pie(target_counts.values, labels=['No Dropout Risk', 'Dropout Risk'], 
                autopct='%1.1f%%', startangle=90)
        plt.title('Target Variable Distribution')
        
        # Bar plot
        plt.subplot(2, 2, 2)
        sns.countplot(data=df, x=target_col)
        plt.title('Dropout Risk Counts')
        plt.xlabel('Dropout Risk (0=No, 1=Yes)')
        
        # Distribution by gender
        plt.subplot(2, 2, 3)
        if 'gender' in df.columns:
            pd.crosstab(df['gender'], df[target_col], normalize='index').plot(kind='bar')
            plt.title('Dropout Risk by Gender')
            plt.xticks(rotation=45)
        
        # Distribution by school type
        plt.subplot(2, 2, 4)
        if 'school_type' in df.columns:
            pd.crosstab(df['school_type'], df[target_col], normalize='index').plot(kind='bar')
            plt.title('Dropout Risk by School Type')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Target distribution plot saved to {output_dir}")
    
    def plot_numerical_features(self, df, target_col='dropout_risk', output_dir='results/'):
        """Plot numerical features analysis."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        
        # Key numerical features for detailed analysis
        key_features = ['gpa_previous_year', 'attendance_rate', 'family_income', 
                       'homework_completion_rate', 'absences_last_semester']
        key_features = [col for col in key_features if col in numerical_cols]
        
        if len(key_features) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.ravel()
            
            for i, feature in enumerate(key_features[:6]):
                # Distribution by target
                for target_val in df[target_col].unique():
                    subset = df[df[target_col] == target_val][feature].dropna()
                    axes[i].hist(subset, alpha=0.7, 
                               label=f'Dropout Risk: {target_val}', bins=20)
                
                axes[i].set_title(f'Distribution of {feature}')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
            
            # Remove empty subplots
            for j in range(len(key_features), len(axes)):
                fig.delaxes(axes[j])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'numerical_features_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        # Correlation heatmap
        plt.figure(figsize=(14, 10))
        correlation_matrix = df[numerical_cols + [target_col]].corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Numerical features plots saved to {output_dir}")
    
    def plot_categorical_features(self, df, target_col='dropout_risk', output_dir='results/'):
        """Plot categorical features analysis."""
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) > 0:
            n_cols = min(3, len(categorical_cols))
            n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.ravel()
            
            for i, feature in enumerate(categorical_cols):
                if i < len(axes):
                    # Cross-tabulation
                    ct = pd.crosstab(df[feature], df[target_col], normalize='index')
                    ct.plot(kind='bar', ax=axes[i], rot=45)
                    axes[i].set_title(f'Dropout Risk by {feature}')
                    axes[i].set_ylabel('Proportion')
                    axes[i].legend(['No Risk', 'At Risk'])
            
            # Remove empty subplots
            for j in range(len(categorical_cols), len(axes)):
                fig.delaxes(axes[j])
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'categorical_features_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        print(f"Categorical features plots saved to {output_dir}")
    
    def plot_risk_factors_analysis(self, df, target_col='dropout_risk', output_dir='results/'):
        """Analyze key risk factors."""
        plt.figure(figsize=(16, 12))
        
        # GPA vs Attendance Rate
        plt.subplot(2, 3, 1)
        for target_val in df[target_col].unique():
            subset = df[df[target_col] == target_val]
            plt.scatter(subset['gpa_previous_year'], subset['attendance_rate'], 
                       alpha=0.6, label=f'Dropout Risk: {target_val}')
        plt.xlabel('GPA Previous Year')
        plt.ylabel('Attendance Rate')
        plt.title('GPA vs Attendance Rate')
        plt.legend()
        
        # Family Income Distribution
        plt.subplot(2, 3, 2)
        for target_val in df[target_col].unique():
            subset = df[df[target_col] == target_val]['family_income'].dropna()
            plt.hist(subset, alpha=0.7, bins=20, label=f'Dropout Risk: {target_val}')
        plt.xlabel('Family Income')
        plt.ylabel('Frequency')
        plt.title('Family Income Distribution')
        plt.legend()
        
        # Disciplinary Incidents
        plt.subplot(2, 3, 3)
        incident_analysis = df.groupby(['disciplinary_incidents', target_col]).size().unstack(fill_value=0)
        incident_analysis.plot(kind='bar', ax=plt.gca())
        plt.xlabel('Disciplinary Incidents')
        plt.ylabel('Count')
        plt.title('Disciplinary Incidents vs Dropout Risk')
        plt.xticks(rotation=0)
        
        # Absences Analysis
        plt.subplot(2, 3, 4)
        df['absence_category'] = pd.cut(df['absences_last_semester'], 
                                       bins=[0, 5, 15, 30, float('inf')], 
                                       labels=['Low (0-5)', 'Medium (6-15)', 'High (16-30)', 'Very High (30+)'])
        absence_crosstab = pd.crosstab(df['absence_category'], df[target_col], normalize='index')
        absence_crosstab.plot(kind='bar', ax=plt.gca())
        plt.xlabel('Absence Category')
        plt.ylabel('Proportion')
        plt.title('Absence Categories vs Dropout Risk')
        plt.xticks(rotation=45)
        
        # Support Systems
        plt.subplot(2, 3, 5)
        df['has_support'] = ((df['counseling_sessions'] > 0) | (df['tutoring_hours'] > 0)).astype(int)
        support_crosstab = pd.crosstab(df['has_support'], df[target_col], normalize='index')
        support_crosstab.plot(kind='bar', ax=plt.gca())
        plt.xlabel('Has Support System')
        plt.ylabel('Proportion')
        plt.title('Support System vs Dropout Risk')
        plt.xticks(rotation=0)
        
        # Age Distribution
        plt.subplot(2, 3, 6)
        age_crosstab = pd.crosstab(df['age'], df[target_col], normalize='index')
        age_crosstab.plot(kind='bar', ax=plt.gca())
        plt.xlabel('Age')
        plt.ylabel('Proportion')
        plt.title('Age vs Dropout Risk')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'risk_factors_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Risk factors analysis plot saved to {output_dir}")
    
    def generate_eda_report(self, filepath, output_dir='results/'):
        """Generate complete EDA report."""
        print("Starting comprehensive EDA analysis...")
        
        # Load data
        df = self.load_data(filepath)
        if df is None:
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all analyses
        self.generate_data_summary(df, output_dir)
        self.plot_target_distribution(df, output_dir=output_dir)
        self.plot_numerical_features(df, output_dir=output_dir)
        self.plot_categorical_features(df, output_dir=output_dir)
        self.plot_risk_factors_analysis(df, output_dir=output_dir)
        
        print(f"\nEDA analysis completed! All results saved to {output_dir}")
        print("Generated files:")
        print("- data_summary.txt: Comprehensive data summary")
        print("- numerical_summary.csv: Descriptive statistics for numerical features")
        print("- target_distribution.png: Target variable analysis")
        print("- numerical_features_distribution.png: Numerical features analysis")
        print("- correlation_matrix.png: Feature correlation heatmap")
        print("- categorical_features_analysis.png: Categorical features analysis")
        print("- risk_factors_analysis.png: Key risk factors analysis")

def main():
    """Main function to run EDA analysis."""
    analyzer = EDAAnalyzer()
    
    # Run complete EDA
    analyzer.generate_eda_report('data/raw/sample_social.csv')

if __name__ == "__main__":
    main()
