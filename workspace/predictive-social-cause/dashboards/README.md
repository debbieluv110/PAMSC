# Dashboards Directory

This directory contains interactive dashboard files for visualizing the school dropout prediction results.

## Files Structure

- `school_dropout_dashboard.twbx` - Tableau workbook (placeholder)
- `school_dropout_dashboard.pbix` - Power BI file (placeholder)
- `dashboard_requirements.md` - Requirements for dashboard development

## Dashboard Features

### Key Visualizations
1. **Risk Score Distribution** - Histogram showing distribution of dropout risk scores
2. **Feature Importance** - Bar chart of top predictive factors
3. **Student Segmentation** - Risk level categories with counts
4. **Demographic Analysis** - Breakdown by gender, ethnicity, school type
5. **Academic Performance Trends** - GPA and attendance patterns
6. **Intervention Tracking** - Success rates of different interventions

### Interactive Elements
- Filter by school type, grade level, demographic groups
- Drill-down capabilities from summary to individual student level
- Time-based analysis (if longitudinal data available)
- Comparison views between different risk segments

## Publishing Instructions

### Tableau Public
1. Open Tableau Desktop
2. Connect to processed data files
3. Create visualizations following the dashboard_requirements.md
4. Publish to Tableau Public
5. Share public link in project documentation

### Power BI
1. Open Power BI Desktop
2. Import data from CSV files
3. Create report following dashboard specifications
4. Publish to Power BI Service (public workspace)
5. Generate shareable link

## Data Connection
- Primary data source: `../data/processed/features.csv`
- Labels: `../data/processed/labels.csv`
- Model results: `../results/metrics.json`
- Predictions: `../results/*_predictions.csv`

## Usage Guidelines
- Ensure data privacy compliance when using real student data
- Regular updates recommended (monthly/quarterly)
- Validate data refresh and visualization accuracy
- Monitor dashboard performance and user engagement
