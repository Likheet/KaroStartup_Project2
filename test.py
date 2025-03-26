# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "matplotlib",
#   "seaborn",
#   "requests",
#   "tenacity",
#   "tabulate"
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

def load_data(filename):
    """Load CSV data into a Pandas DataFrame."""
    try:
        df = pd.read_csv(filename, encoding='ISO-8859-1')
        print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def analyze_data(df):
    """Perform general data analysis."""
    summary = df.describe(include='all')
    missing_values = df.isnull().sum()
    correlation_matrix = df.corr(numeric_only=True)
    return summary, missing_values, correlation_matrix

def generate_visualizations(df):
    """Generate enhanced visualizations."""
    # Correlation Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Correlation Heatmap")
    plt.savefig("correlation_heatmap.png", dpi=300)
    plt.close()
    
    # Top Rated Books Distribution
    if 'average_rating' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df['average_rating'].dropna(), bins=20, kde=True, color='green')
        plt.title("Distribution of Average Ratings")
        plt.xlabel("Average Rating")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig("rating_distribution.png", dpi=300)
        plt.close()
    
def get_gemini_response(prompt):
    """Send a prompt to Gemini for AI-driven insights."""
    ai_proxy_token = os.getenv("AIPROXY_TOKEN")
    if not ai_proxy_token:
        print("Error: AIPROXY_TOKEN not set.")
        sys.exit(1)
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": ai_proxy_token}
    data = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "topP": 0.8,
            "topK": 40,
            "maxOutputTokens": 4096
        }
    }
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def chat():
        response = requests.post(url, headers=headers, params=params, json=data)
        if response.status_code == 200:
            return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        else:
            raise Exception(f"API Error: {response.status_code}")
    
    return chat()

def generate_report(filename, summary, missing_values, correlation_matrix, insights):
    """Create README.md with analysis results."""
    with open("README.md", "w") as f:
        f.write(f"# Automated Data Analysis Report\n\n")
        f.write(f"## Dataset: {filename}\n\n")
        f.write("### Summary Statistics\n\n")
        f.write(summary.to_markdown() + "\n\n")
        f.write("### Missing Values\n\n")
        f.write(missing_values.to_markdown() + "\n\n")
        f.write("### Correlation Matrix\n\n")
        f.write(correlation_matrix.to_markdown() + "\n\n")
        f.write("### AI-Generated Insights\n\n")
        f.write(insights + "\n")
        f.write("### Visualizations\n\n")
        f.write("![Correlation Heatmap](correlation_heatmap.png)\n")
        

def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    
    filename = sys.argv[1]
    df = load_data(filename)
    summary, missing_values, correlation_matrix = analyze_data(df)
    generate_visualizations(df)
    
    prompt = f"""Analyze the dataset:\n- Summary: {summary.to_string()}\n- Missing values: {missing_values.to_string()}\n- Correlation matrix: {correlation_matrix.to_string()}\nProvide insights as a story."""
    insights = get_gemini_response(prompt)
    generate_report(filename, summary, missing_values, correlation_matrix, insights)
    print("Analysis complete. Check README.md and PNG files.")

if __name__ == "__main__":
    main()