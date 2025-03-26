# Autolysis - Automated Data Analysis Pipeline

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Project Overview

Autolysis is an advanced data analysis pipeline that automates the entire process from data ingestion to insight generation. This tool integrates machine learning algorithms with AI-powered analysis to transform raw datasets into comprehensive visual reports with minimal user intervention.

## Key Features

- **Automated Exploratory Data Analysis (EDA)** - Generate statistical summaries and identify patterns
- **Advanced Analytics** - Outlier detection, clustering, and dimensionality reduction (PCA)
- **Interactive Visualizations** - Automatically generate relevant plots based on data characteristics
- **AI-Powered Insights** - Leverage Google's Gemini API to suggest additional analyses and interpretations
- **Comprehensive Reporting** - Create markdown reports with embedded visualizations

## Technologies Used

- **Python** - Core programming language
- **Data Science Stack** - pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning** - scikit-learn for clustering and dimensionality reduction
- **AI Integration** - Google's Gemini API for intelligent insight generation
- **Environment Management** - dotenv for configuration

## How It Works

1. **Data Ingestion**: Load and preprocess CSV data from any source
2. **Statistical Analysis**: Calculate descriptive statistics and identify patterns
3. **Machine Learning**: Apply clustering, outlier detection, and PCA
4. **AI-Powered Analysis**: Generate deeper insights using Google's Gemini API
5. **Visualization Generation**: Create relevant, high-quality data visualizations
6. **Report Generation**: Compile findings into a comprehensive markdown report

## Installation and Usage

```bash
# Clone the repository
git clone https://github.com/Likheet/KaroStartup_Project2.git
cd KaroStartup_Project2

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
echo "AIPROXY_TOKEN=your_gemini_api_key" > .env

# Run the analysis on your dataset
python autolysis.py your_dataset.csv 
# OR python autolysis.py data/your_dataset.csv 
```

## Example Output

The tool generates a folder with:
- **README.md** - Comprehensive analysis report
- **correlation_heatmap.png** - Correlation analysis visualization
- **distribution_plots.png** - Distribution analysis of key variables
- **pca_plot.png** - PCA visualization showing data in reduced dimensions

## Technical Highlights

- **Smart Feature Selection** - Automatically selects the most informative columns for visualization
- **Adaptive Visualization** - Adjusts plot types based on data characteristics
- **Robust Error Handling** - Graceful degradation when optimal analysis isn't possible
- **Optimized Performance** - Efficient processing for large datasets
- **Extensible Architecture** - Easily add new analysis modules or visualization types

## Future Enhancements

- Integration with additional ML models for predictive analytics
- Interactive web dashboard for exploring results
- Support for more data sources (databases, APIs)
- Automated time-series analysis
- Natural language query interface

## Project Structure

```
.
├── autolysis.py         # Main analysis script
├── data/                # Data files directory
├── goodreads/           # Output directory for *goodreads* README and visualizations
├── happiness/           # Output directory for *happiness* README and visualizations
├── media/               # Output directory for *media* README and visualizations
└── requirements.txt     # Project dependencies
```


This project demonstrates advanced proficiency in:
- Data analysis and visualization
- Machine learning implementation
- API integration
- Automated report generation
- Python programming best practices

---

*This project was developed as part of KaroStartup Internship*