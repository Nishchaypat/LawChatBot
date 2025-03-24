import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from statistics import mean

# Load the data
file_path = "Evaluation_Results_2.csv"
data = pd.read_csv(file_path)

# Function to safely parse JSON strings
def parse_json(json_str):
    try:
        return json.loads(json_str.replace("'", '"'))
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {}

# Data cleaning steps
def clean_data(df):
    # Make a copy to avoid modifying the original dataframe
    cleaned_df = df.copy()
    
    # Check for and remove duplicate rows
    duplicate_count = cleaned_df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Removing {duplicate_count} duplicate rows")
        cleaned_df = cleaned_df.drop_duplicates()
    
    # Check for and handle missing values
    missing_values = cleaned_df.isnull().sum()
    print("Missing values per column:")
    print(missing_values)
    
    # Explicitly convert numeric columns to float
    # This fixes the "Cannot convert to numeric" error
    for col in ['gemini_weight', 'voyage_weight', 'time']:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    # Fill missing numeric values with appropriate values (mean or median)
    for col in ['gemini_weight', 'voyage_weight', 'time']:
        if col in cleaned_df.columns and cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    # Fill missing text values with empty strings
    for col in ['query', 'response']:
        if col in cleaned_df.columns and cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna('')
    
    # Parse the indices column which contains JSON-like strings
    if 'indices' in cleaned_df.columns:
        # First, normalize the format of the indices column
        cleaned_df['indices_parsed'] = cleaned_df['indices'].apply(parse_json)
        
        # Extract specific values from the nested structure if needed
        # Example: Extract gemini_top_values if it exists
        cleaned_df['gemini_top_values'] = cleaned_df['indices_parsed'].apply(
            lambda x: x.get('sections', {}).get('gemini_top_values', []) if isinstance(x, dict) else []
        )
        
        # Calculate the average of gemini_top_values if it's a list of numbers
        cleaned_df['gemini_top_avg'] = cleaned_df['gemini_top_values'].apply(
            lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 and all(isinstance(i, (int, float)) for i in x) else np.nan
        )
    
    # Clean the query and response columns (remove extra whitespace, etc.)
    if 'query' in cleaned_df.columns:
        cleaned_df['query'] = cleaned_df['query'].str.strip()
        cleaned_df['query_length'] = cleaned_df['query'].str.len()
    
    if 'response' in cleaned_df.columns:
        cleaned_df['response'] = cleaned_df['response'].str.strip()
        cleaned_df['response_length'] = cleaned_df['response'].str.len()
    
    # Add a weight ratio column
    if 'gemini_weight' in cleaned_df.columns and 'voyage_weight' in cleaned_df.columns:
        cleaned_df['weight_ratio'] = cleaned_df['gemini_weight'] / cleaned_df['voyage_weight']
    
    # Check for and handle outliers in numeric columns
    numeric_cols = ['gemini_weight', 'voyage_weight', 'time']
    for col in numeric_cols:
        if col in cleaned_df.columns:
            # Calculate IQR
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Print information about outliers
            outliers = cleaned_df[(cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)]
            if not outliers.empty:
                print(f"\nFound {len(outliers)} outliers in column '{col}'")
                print(f"Range: {lower_bound} to {upper_bound}")
                
                # Optional: Winsorize (cap) the outliers instead of removing them
                cleaned_df[f'{col}_winsorized'] = cleaned_df[col].clip(lower_bound, upper_bound)
    
    return cleaned_df

# Process and analyze the data
def analyze_data(df):
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Correlation matrix
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_columns) > 1:
        correlation = df[numeric_columns].corr()
        print("\nCorrelation Matrix:")
        print(correlation)
        
        # Visualize correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()
    
    # Time analysis
    if 'time' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['time'], kde=True)
        plt.title('Distribution of Response Times')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        plt.savefig('time_distribution.png')
        plt.close()
    
    # Weight distribution
    if 'gemini_weight' in df.columns and 'voyage_weight' in df.columns:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(df['gemini_weight'], kde=True)
        plt.title('Gemini Weight Distribution')
        
        plt.subplot(1, 2, 2)
        sns.histplot(df['voyage_weight'], kde=True)
        plt.title('Voyage Weight Distribution')
        
        plt.tight_layout()
        plt.savefig('weight_distributions.png')
        plt.close()
    
    # Word cloud for queries
    if 'query' in df.columns:
        all_queries = ' '.join(df['query'].dropna())
        if all_queries:
            wordcloud = WordCloud(width=800, height=400, max_words=200, background_color='white').generate(all_queries)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud of Queries')
            plt.savefig('query_wordcloud.png')
            plt.close()
    
    return

# Execute cleaning and analysis
cleaned_data = clean_data(data)
print("\nCleaned data shape:", cleaned_data.shape)

# Save cleaned data
cleaned_data.to_csv('cleaned_evaluation_results.csv', index=False)
print("Cleaned data saved to 'cleaned_evaluation_results.csv'")

# Analyze the cleaned data
analyze_data(cleaned_data)

# Summary statistics for key columns
print("\nSummary Statistics:")
if 'time' in cleaned_data.columns:
    print(f"Average response time: {cleaned_data['time'].mean():.2f} seconds")
if 'query_length' in cleaned_data.columns and 'response_length' in cleaned_data.columns:
    print(f"Average query length: {cleaned_data['query_length'].mean():.2f} characters")
    print(f"Average response length: {cleaned_data['response_length'].mean():.2f} characters")
if 'gemini_weight' in cleaned_data.columns and 'voyage_weight' in cleaned_data.columns:
    print(f"Average gemini weight: {cleaned_data['gemini_weight'].mean():.4f}")
    print(f"Average voyage weight: {cleaned_data['voyage_weight'].mean():.4f}")