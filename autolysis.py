import os
import sys
import math
import json
import time
import warnings
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from matplotlib.ticker import MaxNLocator
import matplotlib
import requests
# Using tenacity for retries, install with: pip install tenacity
from tenacity import retry, stop_after_attempt, wait_fixed, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()  # load variables from .env in the current directory

def load_data(filename):
    """Load CSV data into a Pandas DataFrame."""
    try:
        df = pd.read_csv(filename, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filename, encoding='ISO-8859-1', low_memory=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def basic_analysis(df):
    """Perform general data analysis."""
    summary = df.describe(include='all')
    missing_values = df.isnull().sum()
    numeric_cols = df.select_dtypes(include=np.number).columns
    correlation_matrix = df[numeric_cols].corr() if len(numeric_cols) > 1 else None
    data_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
    sample_data = df.head(5)
    return summary, missing_values, correlation_matrix, data_types, sample_data

def get_dataset_specific_config(filename):
    """Return dataset-specific configuration settings."""
    base_name = os.path.basename(filename).lower()
    config = {
        "is_goodreads": False,
        "columns_to_drop": [],
        "visualization_settings": {}
    }
    
    if base_name == "goodreads.csv":
        config["is_goodreads"] = True
        config["columns_to_drop"] = ['ratings_count', 'work_text_reviews', 'books_count']
        config["visualization_settings"] = {
            "original_pub_year": {"bin_divider": 30}  # For smaller bin steps
        }
    
    return config

def prepare_numeric_data(df, min_columns=2, min_samples=10, transform_columns=None):
    """Prepare numeric data for machine learning algorithms.
    
    Args:
        df: DataFrame to process
        min_columns: Minimum number of numeric columns required
        min_samples: Minimum number of samples required
        transform_columns: Dict mapping column names to transformation types (e.g. {"ratings_count": "log"})
    
    Returns:
        tuple: (scaled_data, numeric_data, error_message)
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    if len(numeric_cols) < min_columns:
        return None, None, f"Not enough numeric columns (need at least {min_columns})."
    
    if transform_columns and isinstance(transform_columns, dict):
        for col, transform in transform_columns.items():
            if col in numeric_cols and transform == "log":
                df[f"log_{col}"] = np.log1p(df[col])
                numeric_cols = list(numeric_cols)
                numeric_cols.append(f"log_{col}")
                numeric_cols.remove(col)
    
    numeric_data = df[numeric_cols].dropna()
    
    if len(numeric_data) < min_samples:
        return None, None, f"Not enough data points (need at least {min_samples})."
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    return scaled_data, numeric_data, None

def detect_outliers(df):
    """Detect outliers in numeric columns using IsolationForest."""
    scaled_data, numeric_data, error = prepare_numeric_data(df)
    
    if error:
        return error
    
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers = iso_forest.fit_predict(scaled_data)
    outlier_count = (outliers == -1).sum()
    outlier_percent = (outlier_count / len(outliers)) * 100
    
    return f"Detected {outlier_count} potential outliers ({outlier_percent:.2f}% of data)."

def perform_clustering(df):
    """Perform K-means clustering on numeric data."""
    scaled_data, numeric_data, error = prepare_numeric_data(df)
    
    if error:
        return error
    
    k = min(5, len(numeric_data) // 10)
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    cluster_info = "K-means clustering results:\n"
    for i, count in enumerate(cluster_counts):
        percent = (count / len(clusters)) * 100
        cluster_info += f"- Cluster {i}: {count} samples ({percent:.2f}%)\n"
    return cluster_info

def reduce_dimensions(df, output_dir):
    """
    Perform PCA for dimensionality reduction (2 components).
    Returns:
      - pca_df: a DataFrame with columns ['PC1', 'PC2']
      - explained_variance: tuple containing the percentage of variance explained by PC1 and PC2
    """
    transform_columns = {}
    if "ratings_count" in df.columns:
        transform_columns["ratings_count"] = "log"
    
    scaled_data, numeric_data, error = prepare_numeric_data(
        df, min_columns=3, min_samples=10, transform_columns=transform_columns
    )
    if error:
        return None, f"Error: {error}"

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=["PC1", "PC2"]
    )
    explained_variance = pca.explained_variance_ratio_ * 100
    if explained_variance.sum() < 1:
        return None, "Error: PCA explained variance is too low (< 1%). Check dataset."

    variance_msg = (
        f"PCA generated at {output_dir}/pca_plot.png with the first two components explaining {explained_variance[0]:.2f}% and {explained_variance[1]:.2f}% of variance."
    )
    print(variance_msg)

    return pca_df, explained_variance

def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    matplotlib.rcParams['figure.facecolor'] = 'white'
    matplotlib.rcParams['axes.facecolor'] = 'white'
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['grid.alpha'] = 0.3

def generate_correlation_heatmap(df, output_dir):
    """Generate a correlation heatmap for numeric columns with better styling."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        print("Not enough numeric columns for correlation heatmap.")
        return False

    setup_plot_style()
    corr_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, 
                vmin=-1, vmax=1, cbar_kws={"shrink": 0.75})
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels
    plt.yticks(rotation=0)  # Keep y-axis labels readable
    plt.savefig(f"{output_dir}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation heatmap generated at {output_dir}/correlation_heatmap.png")
    return True

def filter_columns(df, exclude_patterns=None, min_unique=2, min_completeness=0.1):
    """Filter columns based on common criteria."""
    if exclude_patterns is None:
        exclude_patterns = ['id', 'isbn', 'url', 'image', 'small_image', 'index', 'key', 'code']
    
    return [
        col for col in df.columns 
        if not any(pattern in str(col).lower() for pattern in exclude_patterns)
        and df[col].nunique() > min_unique
        and df[col].count() > df.shape[0] * min_completeness
    ]

def is_good_numeric_column(series, min_non_null=0.7, min_unique=5):
    """Check if a numeric column has enough data and variability."""
    non_null_ratio = series.count() / len(series)
    if non_null_ratio < min_non_null:
        return False
    if series.nunique() < min_unique:
        return False
    return True

def is_good_categorical_column(series, max_categories=30, min_non_null=0.7):
    """Check if a categorical column is suitable for visualization."""
    non_null_ratio = series.count() / len(series)
    if non_null_ratio < min_non_null:
        return False
    if series.nunique() > max_categories:
        return False
    return True

def select_good_plot_columns(df):
    """
    Evaluate columns in df and return a dictionary of column lists that are good for plotting.
    """
    exclude_patterns = ['id', 'isbn', 'url', 'image', 'small_image', 'index', 'key', 'code']
    
    good_numeric = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if any(pattern in str(col).lower() for pattern in exclude_patterns):
            continue
        if is_good_numeric_column(df[col]):
            good_numeric.append(col)
    
    good_categorical = []
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if any(pattern in str(col).lower() for pattern in exclude_patterns):
            continue
        if is_good_categorical_column(df[col]):
            good_categorical.append(col)
    
    good_date = []
    date_patterns = ['date', 'year', 'month', 'time', 'day']
    for col in df.columns:
        if any(pattern in str(col).lower() for pattern in date_patterns):
            try:
                converted = pd.to_datetime(df[col], errors='coerce')
                if converted.notna().sum() > 0.5 * len(converted):
                    good_date.append(col)
            except Exception:
                continue
    
    return {
        'numeric': good_numeric,
        'categorical': good_categorical,
        'date': good_date,
        'all': good_numeric + good_categorical + good_date
    }

def extract_json_array(text):
    """Extract a JSON array from text that might contain additional content."""
    try:
        return json.loads(text)
    except:
        if '[' in text and ']' in text:
            start_idx = text.find('[')
            end_idx = text.rfind(']') + 1
            json_str = text[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except:
                pass
    return []  # Return empty list if parsing fails

def get_best_columns_to_plot(df, summary, max_plots=5):
    """Use AI to determine which columns are most informative to visualize from quality-filtered columns."""
    available_columns = df.columns.tolist()
    
    good_cols = select_good_plot_columns(df)
    filtered_df = df[good_cols['all']]

    column_info = []
    for col in filtered_df.columns:
        if isinstance(filtered_df[col], pd.DataFrame):
            col_series = filtered_df[col].iloc[:, 0]
            col_dtype = filtered_df[col].dtypes.iloc[0]
        else:
            col_series = filtered_df[col]
            col_dtype = col_series.dtype

        try:
            unique_val = int(col_series.nunique())
        except Exception as e:
            print(f"Error converting nunique for column {col}: {e}")
            unique_val = 0
        missing_val = int(col_series.isna().sum())

        col_info = {
            "name": col,
            "dtype": str(col_dtype),
            "unique_values": unique_val,
            "missing_values": missing_val
        }
        if pd.api.types.is_numeric_dtype(col_series):
            col_info["mean"] = float(col_series.mean()) if not pd.isna(col_series.mean()) else "NA"
            col_info["median"] = float(col_series.median()) if not pd.isna(col_series.median()) else "NA"
        column_info.append(col_info)

    sample_data = filtered_df.head(3).to_string()

    prompt = f"""
    As a data visualization expert, select the {max_plots} most informative columns to visualize from this dataset.
    
    Sample data:
    {sample_data}
    
    Column information:
    {column_info}
    
    Return ONLY a JSON array of column names like this: ["column1", "column2", "column3"]
    Don't include any explanations - just the JSON array.
    """
    try:
        response = get_gemini_response(prompt)
        selected_columns = extract_json_array(response)
        selected_columns = [col for col in selected_columns if col in available_columns]
        selected_columns = selected_columns[:max_plots]
        if len(selected_columns) == 0:
            return select_default_columns(df, max_plots)
        return selected_columns
    except Exception as e:
        print(f"Error in AI column selection: {e}")
        return select_default_columns(df, max_plots)


def select_default_columns(df, max_plots=5):
    """Manual fallback selection of columns to visualize."""
    date_patterns = ['date', 'year', 'month', 'time', 'day']
    date_cols = [col for col in df.columns 
                if any(pattern in str(col).lower() for pattern in date_patterns)]
    
    numeric_cols = df.select_dtypes(include=np.number).columns
    numeric_cols = sorted([(col, df[col].nunique()) for col in numeric_cols], 
                          key=lambda x: x[1], reverse=True)
    numeric_cols = [col for col, _ in numeric_cols]
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_cols = sorted([(col, df[col].nunique()) for col in categorical_cols], 
                             key=lambda x: x[1])
    categorical_cols = [col for col, _ in categorical_cols if df[col].nunique() < 20]
    
    selected_cols = []
    for i in range(max_plots):
        if i % 3 == 0 and date_cols:
            selected_cols.append(date_cols.pop(0))
        elif i % 3 == 1 and numeric_cols:
            selected_cols.append(numeric_cols.pop(0))
        elif categorical_cols:
            selected_cols.append(categorical_cols.pop(0))
        elif numeric_cols:
            selected_cols.append(numeric_cols.pop(0))
        elif date_cols:
            selected_cols.append(date_cols.pop(0))
    
    return selected_cols[:max_plots]

warnings.filterwarnings(
    "ignore",
    message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`"
)

def generate_distribution_plots(df, output_dir, dataset_config):
    """Generate visually appealing distribution plots with better styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    COLOR_PALETTE = {
        'histogram': '#3498db',
        'year_hist': '#2ecc71',
        'categorical': 'viridis',
        'kde_line': '#e74c3c',
        'text_box': '#f9f9f9'
    }
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'

    if dataset_config.get("is_goodreads", False):
        required_columns = ["original_publication_year", "average_rating"]
        columns_to_plot = [col for col in required_columns if col in df.columns]
        if len(columns_to_plot) < len(required_columns):
            print("Warning: Not all required columns found for goodreads.csv plotting.")
    else:
        selected_columns = filter_columns(df)
        summary = df.describe(include='all')
        print("Selecting most informative columns to visualize...")
        columns_to_plot = get_best_columns_to_plot(df[selected_columns], summary, max_plots=5)
        print(f"Selected columns for visualization: {columns_to_plot}")

    total_plots = len(columns_to_plot)
    plots_per_row = min(3, total_plots)
    n_rows = max(1, math.ceil(total_plots / plots_per_row))
    fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(6.5*plots_per_row, 5*n_rows), dpi=100)
    if total_plots == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    fig.patch.set_facecolor('#fafafa')
    plot_index = 0
    for col in columns_to_plot:
        if plot_index >= len(axes):
            break
        ax = axes[plot_index]
        ax.set_facecolor('#f8f9fa')
        viz_settings = {}
        if isinstance(dataset_config, dict) and "visualization_settings" in dataset_config:
            viz_settings = dataset_config["visualization_settings"].get(col.lower(), {})
        if col in df.select_dtypes(include=np.number).columns:
            data = df[col].dropna()
            if len(data) > 0:
                unique_count = data.nunique()
                is_integer = pd.api.types.is_integer_dtype(data)
                if "year" in col.lower():
                    years = data
                    if dataset_config.get("is_goodreads", False) and col.lower() == "original_publication_year":
                        lower_bound = 1900
                        upper_bound = 2025
                        bin_size = 5
                        bins = np.arange(lower_bound, upper_bound + bin_size, bin_size)
                        sns.histplot(
                            years, bins=bins, kde=True, ax=ax,
                            color=COLOR_PALETTE['year_hist'],
                            edgecolor='white', linewidth=0.8,
                            alpha=0.7, shrink=0.8,
                            kde_kws={'bw_adjust': 0.8},
                            line_kws={'color': COLOR_PALETTE['kde_line'], 'lw': 2, 'alpha': 0.8}
                        )
                    else:
                        lower_pct = np.percentile(years, 5)
                        upper_pct = np.percentile(years, 95)
                        lower_bound = lower_pct if lower_pct > 0 else 0
                        if lower_bound > 0:
                            lower_bound = math.floor(lower_bound / 100) * 100
                        upper_bound = math.ceil(upper_pct / 100) * 100
                        bin_divider = viz_settings.get("bin_divider", 10)
                        bin_size = max(1, (upper_bound - lower_bound) // bin_divider)
                        bins = np.arange(lower_bound, upper_bound + bin_size, bin_size)
                        sns.histplot(
                            years, bins=bins, kde=True, ax=ax,
                            color=COLOR_PALETTE['year_hist'],
                            edgecolor='white', linewidth=0.8,
                            alpha=0.7,
                            kde_kws={'bw_adjust': 0.8},
                            line_kws={'color': COLOR_PALETTE['kde_line'], 'lw': 2, 'alpha': 0.8}
                        )
                    ax.set_title(f"{col.replace('_', ' ').title()} Distribution", fontweight="bold", pad=15)
                    ax.set_xlabel("Year", fontweight='medium')
                    ax.set_ylabel("Frequency", fontweight='medium')
                    ax.set_xlim(lower_bound, upper_bound)
                elif unique_count <= 15 and is_integer:
                    value_counts = data.value_counts().sort_index()
                    bars = sns.barplot(
                        x=value_counts.index, y=value_counts.values, ax=ax,
                        hue=value_counts.index, legend=False,
                        palette=sns.color_palette(COLOR_PALETTE['categorical'], len(value_counts)),
                        edgecolor='white', linewidth=1.5,
                        alpha=0.9
                    )
                    for i, v in enumerate(value_counts.values):
                        ax.text(i, v + max(value_counts)*0.02, f"{v}", 
                               ha="center", fontsize=10, fontweight='bold')
                    ax.set_title(f"{col.replace('_', ' ').title()} Distribution", fontweight="bold", pad=15)
                    ax.set_xlabel(col.replace('_', ' ').title(), fontweight='medium')
                    ax.set_ylabel("Count", fontweight='medium')
                else:
                    sns.histplot(
                        data, kde=True, ax=ax,
                        color=COLOR_PALETTE['histogram'],
                        edgecolor='white', linewidth=0.8,
                        alpha=0.7,
                        kde_kws={'bw_adjust': 0.8},
                        line_kws={'color': COLOR_PALETTE['kde_line'], 'lw': 2, 'alpha': 0.8}
                    )
                    ax.set_title(f"{col.replace('_', ' ').title()} Distribution", fontweight="bold", pad=15)
                    ax.set_xlabel(col.replace('_', ' ').title(), fontweight='medium')
                    ax.set_ylabel("Frequency", fontweight='medium')
                stats = f"Mean: {data.mean():.2f}\nMedian: {data.median():.2f}\nStd: {data.std():.2f}"
                ax.text(
                    0.05, 0.95, stats, transform=ax.transAxes, fontsize=10,
                    va="top", ha="left", 
                    bbox=dict(boxstyle="round,pad=0.6", facecolor=COLOR_PALETTE['text_box'], edgecolor='lightgrey', alpha=0.9)
                )
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or any(term in str(col).lower() for term in ["date", "year"]):
            if str(col).lower() == "date":
                try:
                    sample_dates = df[col].dropna().head(5).tolist()
                    if any("-" in str(d) for d in sample_dates):
                        converted = pd.to_datetime(df[col], errors="coerce")
                    else:
                        converted = pd.to_datetime(df[col], format="%d-%b-%y", errors="coerce")
                except Exception:
                    converted = pd.to_datetime(df[col], errors="coerce")
                if converted.isnull().all():
                    print("Date conversion failed for column 'date'. Skipping plot.")
                    plot_index += 1
                    continue
                years = converted.dt.year
                title = f"Timeline Distribution: {col}"
            else:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    years = df[col].dt.year
                    title = f"Timeline Distribution: {col}"
                else:
                    years = pd.to_numeric(df[col], errors="coerce").dropna()
                    title = f"Year Distribution: {col}"
            if len(years) > 0:
                year_min, year_max = years.min(), years.max()
                year_range = year_max - year_min
                if year_range <= 10:
                    bin_size = 1
                elif year_range <= 30:
                    bin_size = 2
                elif year_range <= 100:
                    bin_size = 5
                else:
                    bin_size = 10
                bins = np.arange(year_min - (year_min % bin_size), year_max + bin_size*2, bin_size)
                sns.histplot(
                    years, bins=bins, kde=True, ax=ax, 
                    color='steelblue',
                    edgecolor='white', linewidth=0.8,
                    alpha=0.7,
                    kde_kws={'bw_adjust': 0.8},
                    line_kws={'color': 'crimson', 'lw': 2, 'alpha': 0.8}
                )
                if year_range > 50:
                    tick_spacing = bin_size * math.ceil(year_range/50)
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=year_range//tick_spacing))
                ax.set_title(title.replace('_', ' '), fontweight="bold", pad=15)
                ax.set_xlabel("Year", fontweight='medium')
                ax.set_ylabel("Frequency", fontweight='medium')
                ax.text(
                    0.05, 0.95, f"Range: {int(year_min)}-{int(year_max)}",
                    transform=ax.transAxes, fontsize=10, va="top",
                    bbox=dict(boxstyle="round,pad=0.6", facecolor=COLOR_PALETTE['text_box'], edgecolor='lightgrey', alpha=0.9)
                )
        else:
            value_counts = df[col].value_counts()
            top_n = 8
            if len(value_counts) > top_n:
                top_cats = value_counts.nlargest(top_n)
                other_count = value_counts[top_n:].sum()
                plot_data = pd.Series({**top_cats.to_dict(), "Other": other_count})
                bars = sns.barplot(
                    x=plot_data.index, y=plot_data.values, ax=ax,
                    hue=plot_data.index, legend=False,
                    palette=sns.color_palette("viridis", len(plot_data)),
                    edgecolor='white', linewidth=1,
                    alpha=0.85
                )
                ax.set_title(f"Top Categories in {col.replace('_', ' ').title()}", fontweight="bold", pad=15)
                other_pct = (other_count / value_counts.sum()) * 100
                ax.text(
                    len(plot_data)-1, other_count/2, f"{other_pct:.1f}%",
                    ha="center", va="center", color="white", fontweight="bold", fontsize=10
                )
            else:
                sns.barplot(
                    x=value_counts.index, y=value_counts.values, ax=ax,
                    hue=value_counts.index, legend=False,
                    palette=sns.color_palette("viridis", len(value_counts)),
                    edgecolor='white', linewidth=1,
                    alpha=0.85
                )
                ax.set_title(f"Categories in {col.replace('_', ' ').title()}", fontweight="bold", pad=15)
            ax.set_xlabel("")
            ax.set_ylabel("Count", fontweight='medium')
            ax.tick_params(axis="x", rotation=45)
            total = value_counts.sum()
            for i, v in enumerate(value_counts[:top_n].values):
                if i < len(ax.patches):
                    pct = (v / total) * 100
                    if pct > 5:
                        ax.text(
                            i, v/2, f"{pct:.1f}%", 
                            ha="center", va="center", color="white", fontweight="bold", fontsize=10
                        )
        for spine in ax.spines.values():
            spine.set_edgecolor('lightgray')
        plot_index += 1

    for i in range(plot_index, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(pad=3.0)
    fig.suptitle("Data Distribution Analysis", fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.92)
    plt.savefig(f"{output_dir}/distribution_plots.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Distribution plots generated at {output_dir}/distribution_plots.png")
    return True

def generate_pca_plot(pca_df, output_dir, explained_variance):
    if pca_df is None or pca_df.empty:
        return False
    setup_plot_style()
    var_pc1, var_pc2 = explained_variance if len(explained_variance) >= 2 else (0, 0)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', alpha=0.6, edgecolor='w')
    plt.xlabel(f"PC1 ({var_pc1:.2f}% var.)")
    plt.ylabel(f"PC2 ({var_pc2:.2f}% var.)")
    plt.title("PCA: Data Projected to First Two Principal Components")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_dir, "pca_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    return True

AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN not found in environment variables.")
    sys.exit(1)

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def get_gemini_response(prompt):
    api_key = AIPROXY_TOKEN
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    data = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.4, "topP": 0.8, "topK": 40, "maxOutputTokens": 4096}}
    response = requests.post(url, headers=headers, params=params, json=data)
    if response.status_code == 200:
        return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    return "Failed to get AI insights."

def get_ai_analysis_suggestions(summary, missing_values, correlation_matrix, data_types, sample_data):
    prompt = f"""
Sample Data:
{tabulate(sample_data, headers='keys', tablefmt='pipe')}

Data Types:
{tabulate(data_types, headers='keys', tablefmt='pipe')}

Basic Statistics:
{summary.to_string()}

Missing Values:
{missing_values.to_string()}

Based on the above, suggest 2-3 specific data analysis techniques along with Python code snippets.
"""
    return get_gemini_response(prompt)

def get_ai_visualization_suggestions(summary, data_types):
    prompt = f"""
Data Types:
{tabulate(data_types, headers='keys', tablefmt='pipe')}

Basic Statistics:
{summary.to_string()}

Suggest 2 visualizations with Python code snippets that would effectively showcase the insights from this dataset.
"""
    return get_gemini_response(prompt)

def generate_visualizations(df):
    """Creates required PNG visualizations and saves them."""
    
    # 1. Histogram of average ratings
    plt.figure(figsize=(10, 6))
    sns.histplot(df["average_rating"], bins=20, kde=True)
    plt.title("Distribution of Average Ratings")
    plt.xlabel("Average Rating")
    plt.ylabel("Frequency")
    hist_path = os.path.join(output_dir, "rating_distribution.png")
    plt.savefig(hist_path)
    plt.close()

    # 2. Scatter plot of reviews vs. ratings
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["work_text_reviews_count"], y=df["average_rating"])
    plt.title("Average Rating vs. Number of Reviews")
    plt.xlabel("Number of Reviews")
    plt.ylabel("Average Rating")
    scatter_path = os.path.join(output_dir, "ratings_vs_reviews.png")
    plt.savefig(scatter_path)
    plt.close()

    return [hist_path, scatter_path]

def get_ai_story(filename, df, summary, missing_values, outlier_results, clustering_results, pca_results, analysis_suggestions, visualization_suggestions) -> str:
    """
    Generates a compelling narrative report (story) using AI based on key analysis results.
    
    Args:
        filename (str): Dataset file name.
        df (pd.DataFrame): The full dataset.
        summary (pd.DataFrame): Basic statistics summary.
        missing_values (pd.Series): Missing values summary.
        outlier_results: Results from outlier detection.
        clustering_results: Results from clustering analysis.
        pca_results: Results from PCA (string summary).
        analysis_suggestions (str): AI-suggested further analysis techniques.
        visualization_suggestions (str): AI-suggested visualization ideas.
    
    Returns:
        str: AI-generated markdown narrative report.
    """
    import os
    df_shape = df.shape
    completeness = (1 - df.isnull().sum().sum() / (df_shape[0] * df_shape[1])) * 100
    basic_stats_summary = summary.to_string()
    missing_values_summary = missing_values.to_string() if not missing_values.empty else "No missing values found."
    
    prompt = f"""
Context: You are a data storyteller. Synthesize the following analysis results into a compelling markdown report.
    
Dataset Analyzed: '{filename}'
Dimensions: {df_shape[0]} rows x {df_shape[1]} columns
Data Completeness: {completeness:.2f}%

**Analysis Workflow Summary:**

1. Initial Exploration:
   - Basic Statistics:
{basic_stats_summary}
   - Missing Values:
{missing_values_summary}

2. Advanced Analysis:
   - Outlier Detection: {outlier_results}
   - Clustering Results: {clustering_results}
   - PCA Analysis: {pca_results}

3. AI-Assisted Insights:
   - Further Analysis Suggestions: {analysis_suggestions}
   - Visualization Suggestions: {visualization_suggestions}

Your Task:
Write a detailed markdown report with the following sections:

1. **Introduction & Data Overview:** Introduce the dataset, including size and completeness.
2. **Analytical Journey:** Describe the process used, highlighting key analysis steps and their outcomes.
3. **Key Findings:** Summarize major insights from outlier detection, clustering, and PCA.
4. **Recommendations & Next Steps:** Provide actionable recommendations and ideas for further analysis.

Use markdown formatting (headers, lists, bold text) and a narrative tone.
"""
    story = get_gemini_response(prompt)
    return story

def generate_readme(filename, df, summary, missing_values, outlier_results, clustering_results, pca_results):
    """Generates README.md with AI-generated insights and saves it in goodreads/."""
    
    # Get AI-driven analysis suggestions
    analyses = get_ai_analysis_suggestions(summary, missing_values, None, df.dtypes.reset_index(), df.head())
    # Get AI-driven visualization suggestions
    visualizations = get_ai_visualization_suggestions(summary, df.dtypes.reset_index())
    
    # Generate required visualizations (PNG files saved in output_dir)
    visualization_paths = generate_visualizations(df)
    
    # Generate the AI story (i.e. the README as a narrative report)
    story = get_ai_story(filename, df, summary, missing_values, outlier_results, clustering_results, pca_results, analyses, visualizations)
    
    # Append the visualization images into the README markdown
    story += "\n\n## Supporting Visualizations\n"
    for img_path in visualization_paths:
        # Markdown image syntax: ![Alt Text](relative/path)
        story += f"![Visualization]({img_path})\n"

    # Save the final README.md in the goodreads/ directory
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as file:
        file.write(story)

    print(f"README.md generated at {readme_path}")

def clean_markdown(text):
    lines = text.splitlines()
    # Remove leading and trailing triple backticks if present
    if lines and lines[0].strip().startswith("```markdown"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    
    # Remove any introductory nonsense lines, e.g., starting with "Okay,"
    while lines and lines[0].strip().lower().startswith("okay,"):
        lines.pop(0)
    
    return "\n".join(lines)

def generate_report(filename, output_dir, story):
    story = clean_markdown(story)
    readme_path = f"{output_dir}/README.md"
    with open(readme_path, "w") as f:
        f.write(story)
        f.write("\n\n## Supporting Visualizations\n\n")
        if os.path.exists(f"{output_dir}/correlation_heatmap.png"):
            f.write("### Correlation Heatmap\n![Correlation Heatmap](correlation_heatmap.png)\n\n")
        if os.path.exists(f"{output_dir}/distribution_plots.png"):
            f.write("### Distribution Plots\n![Distribution Plots](distribution_plots.png)\n\n")
        if os.path.exists(f"{output_dir}/pca_plot.png"):
            f.write("### PCA Plot\n![PCA Plot](pca_plot.png)\n\n")
    print(f"README.md generated at {readme_path}")

def main():
    if len(sys.argv) != 2:
        sys.exit(1)
    filename = sys.argv[1]
    if not os.path.exists(filename):
        alt_path = os.path.join("data", filename)
        if os.path.exists(alt_path):
            filename = alt_path
        else:
            print(f"Error: File '{filename}' not found in the current directory or in data/")
            sys.exit(1)
    dataset_name = os.path.splitext(os.path.basename(filename))[0]
    output_dir = dataset_name
    os.makedirs(output_dir, exist_ok=True)
    dataset_config = get_dataset_specific_config(filename)
    df = load_data(filename)
    if dataset_config["columns_to_drop"]:
        df = df.drop(columns=dataset_config["columns_to_drop"], errors='ignore')
    summary, missing_values, correlation_matrix, data_types, sample_data = basic_analysis(df)
    outlier_results = detect_outliers(df)
    clustering_results = perform_clustering(df)
    analysis_suggestions = get_ai_analysis_suggestions(summary, missing_values, correlation_matrix, data_types, sample_data)
    visualization_suggestions = get_ai_visualization_suggestions(summary, data_types)
    generate_correlation_heatmap(df, output_dir)
    generate_distribution_plots(df, output_dir, dataset_config)
    pca_df, explained_variance = reduce_dimensions(df, output_dir)
    if pca_df is not None:
        generate_pca_plot(pca_df, output_dir, explained_variance)
        pca_results = f"PCA generated at {output_dir}/pca_plot.png with the first two components explaining {explained_variance[0]:.2f}% and {explained_variance[1]:.2f}% of variance."
    else:
        pca_results = f"PCA failed: {explained_variance}"
    story = get_ai_story(filename, df, summary, missing_values, outlier_results, clustering_results, pca_results, analysis_suggestions, visualization_suggestions)
    generate_report(filename, output_dir, story)

if __name__ == "__main__":
    main()