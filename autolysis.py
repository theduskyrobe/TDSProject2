# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "httpx",
#   "openai",
#   "scikit-learn"
# ]
# ///

import os
import sys
import json
import logging
import base64
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import httpx
from typing import List, Dict, Any, Optional
import traceback
import shutil
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutomatedAnalysis:
    def __init__(self, dataset_path: str):
        """
        Initialize the automated analysis with the given dataset.
        
        :param dataset_path: Path to the dataset file
        """
        try:
            logger.info(f"Initializing analysis for dataset: {dataset_path}")
            self.dataset_path = dataset_path
            
            # Validate CSV file
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
            # Try multiple encodings
            encodings_to_try = [
                'utf-8', 
                'latin-1', 
                'iso-8859-1', 
                'cp1252', 
                'utf-16'
            ]
            
            for encoding in encodings_to_try:
                try:
                    self.df = pd.read_csv(dataset_path, encoding=encoding)
                    logger.info(f"Successfully loaded dataset using {encoding} encoding with {len(self.df)} rows")
                    break
                except (UnicodeDecodeError, pd.errors.ParserError):
                    logger.warning(f"Failed to read file with {encoding} encoding")
                    continue
            else:
                raise ValueError(f"Could not read the CSV file with any of the tried encodings")
            
            # Validate AI Proxy Token
            self.aiproxy_token = os.environ.get("AIPROXY_TOKEN")
            # For demonstration, we use a placeholder token if none is found.
            if not self.aiproxy_token:
                self.aiproxy_token = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjIwMDEyNDhAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.OhgFKK1Gd7JLHiEgvdaaHGogRLrz34k8-v5g9a03emk"
                # In a real scenario, raise an error if token is not set.
                # raise ValueError("AIPROXY_TOKEN environment variable must be set")
        
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _generate_generic_analysis(self) -> Dict[str, Any]:
        """
        Perform generic data analysis.
        
        :return: Dictionary with analysis results
        """
        analysis = {
            "basic_info": {
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
                "column_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            },
            "missing_values": self.df.isnull().sum().to_dict(),
            "descriptive_stats": {}
        }
        
        # Convert descriptive stats to dictionary
        try:
            desc_stats = self.df.describe(include='all', datetime_is_numeric=True)
            # Convert NaN to "" for safety
            desc_stats = desc_stats.fillna("").astype(str)
            analysis["descriptive_stats"] = desc_stats.to_dict()
        except Exception as e:
            logger.warning(f"Could not generate descriptive statistics: {e}")
        
        # Correlation matrix for numeric columns
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            try:
                correlation_matrix = self.df[numeric_columns].corr()
                analysis["correlation_matrix"] = correlation_matrix.round(4).to_dict()
            except Exception as e:
                logger.warning(f"Could not generate correlation matrix: {e}")
        
        # Outlier detection using IQR method for numeric columns
        outlier_info = {}
        for col in numeric_columns:
            col_data = self.df[col].dropna()
            if len(col_data) > 0:
                Q1 = np.percentile(col_data, 25)
                Q3 = np.percentile(col_data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                outlier_info[col] = len(outliers)
        
        analysis["outliers"] = outlier_info
        
        # Simple cluster analysis (KMeans) if multiple numeric columns are present
        if len(numeric_columns) >= 2:
            try:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(self.df[numeric_columns].dropna())
                # Let's just pick 3 clusters for demonstration
                kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
                kmeans.fit(scaled_data)
                cluster_labels = kmeans.labels_
                analysis["kmeans_clusters"] = {
                    "n_clusters": 3,
                    "cluster_counts": {str(i): int(sum(cluster_labels == i)) for i in range(3)}
                }
            except Exception as e:
                logger.warning(f"Could not perform KMeans clustering: {e}")
        
        return analysis
    
    def _call_llm(self, messages: List[Dict[str, str]], functions: Optional[List[Dict]] = None) -> str:
        """
        Call the OpenAI-compatible LLM via AI Proxy.
        
        :param messages: List of message dictionaries
        :param functions: Optional list of function definitions
        :return: LLM response
        """
        def generate_fallback_narrative(analysis):
            total_rows = analysis['basic_info']['total_rows']
            total_columns = analysis['basic_info']['total_columns']
            column_types = analysis['basic_info']['column_types']
            missing_values = {k:v for k,v in analysis.get('missing_values', {}).items() if v > 0}
            outliers = analysis.get('outliers', {})
            corr_matrix = analysis.get('correlation_matrix', {})
            kmeans_info = analysis.get('kmeans_clusters', {})
            
            narrative = f"""
# Data Analysis Narrative (Fallback)

## Dataset Overview
- **Total Rows**: {total_rows}
- **Total Columns**: {total_columns}

### Column Types
"""
            for col, ctype in column_types.items():
                narrative += f"- {col}: {ctype}\n"

            narrative += "\n### Missing Values\n"
            if missing_values:
                for col, val in missing_values.items():
                    narrative += f"- {col}: {val} missing values\n"
            else:
                narrative += "- No missing values detected\n"

            narrative += "\n### Outliers\n"
            if outliers:
                outlier_reported = False
                for col, count in outliers.items():
                    if count > 0:
                        narrative += f"- {col}: {count} potential outliers\n"
                        outlier_reported = True
                if not outlier_reported:
                    narrative += "- No significant outliers detected\n"
            else:
                narrative += "- Outlier information not available\n"

            narrative += "\n### Correlation Matrix\n"
            if corr_matrix:
                narrative += "A correlation matrix was computed for numeric variables. Higher absolute values indicate stronger relationships.\n\n"
                narrative += "*(Due to fallback mode, detailed correlation insights are not provided. Consider reviewing the correlation_heatmap.png chart.)*\n"
            else:
                narrative += "Correlation analysis was not performed or not applicable.\n"

            narrative += "\n### Clustering\n"
            if kmeans_info:
                narrative += f"KMeans clustering suggested {kmeans_info.get('n_clusters', 'N/A')} clusters. Cluster counts:\n"
                for cluster_id, count in kmeans_info.get('cluster_counts', {}).items():
                    narrative += f"- Cluster {cluster_id}: {count} samples\n"
            else:
                narrative += "No cluster analysis information available.\n"

            narrative += """
            
## Observations and Recommendations
- Without LLM-driven insights, this fallback narrative focuses on basic metrics and structural aspects of the dataset.
- The presence (or absence) of missing values suggests potential data cleaning steps before any modeling or in-depth analysis.
- Outliers may indicate exceptional cases, errors, or unique opportunities that warrant further investigation.
- Correlation analysis can guide which variables relate strongly to each other, supporting feature selection or identifying potential explanatory variables.
- If clustering results are available, they may highlight natural groupings or segments within the data, which could be useful for targeted strategies or personalized approaches.

## Next Steps
- **Data Quality Improvements**: Consider imputing or removing missing values to ensure cleaner input for models.
- **Further Statistical Analysis**: Explore relationships between variables more thoroughly, potentially using statistical tests or more advanced modeling.
- **Domain-Specific Interpretation**: Integrate domain knowledge to better understand what certain outliers, correlations, or clusters might signify.
- **Additional Modeling**: Consider regression, classification, or forecasting methods (if applicable) to leverage the dataset for predictive insights.
- **Iterative Refinement**: Re-run analyses after data cleaning, feature engineering, or dimensionality reduction to improve the reliability of insights.

*(This is a fallback narrative, created without advanced LLM-driven analysis. For deeper insights, try enabling LLM integration or re-running with the LLM service available.)*
"""
            return narrative

        try:
            headers = {
                "Authorization": f"Bearer {self.aiproxy_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4o-mini",
                "messages": messages
            }
            if functions:
                payload["functions"] = functions
            
            endpoints = [
                "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
            ]
            
            for endpoint in endpoints:
                try:
                    with httpx.Client(timeout=30.0) as client:
                        response = client.post(
                            endpoint, 
                            headers=headers, 
                            json=payload
                        )
                        response.raise_for_status()
                        
                        response_json = response.json()
                        if 'choices' in response_json and response_json['choices']:
                            return response_json['choices'][0]['message'].get('content', '')
                
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    logger.warning(f"Failed endpoint {endpoint}: {e}")
                    continue
            
            # If all endpoints fail, use fallback
            logger.error("All LLM endpoints failed")
            return generate_fallback_narrative(self.analysis)
        
        except Exception as e:
            logger.error(f"Unexpected error in LLM call: {e}")
            logger.error(traceback.format_exc())
            
            # Use fallback narrative if possible
            try:
                return generate_fallback_narrative(self.analysis)
            except Exception:
                return """
                # Data Analysis Narrative

                ## Error

                An unexpected error occurred during narrative generation.
                Please review the dataset manually.
                """
    
    def _create_visualizations(self, analysis: Dict[str, Any]):
        """
        Create visualizations based on the analysis.
        
        :param analysis: Dictionary containing analysis results
        """

        # Set a theme and larger font sizes for better readability
        sns.set_theme(style="whitegrid", font_scale=1.0)

        # Missing Values Chart
        try:
            missing_values = pd.Series(analysis.get('missing_values', {}))
            missing_values = missing_values[missing_values > 0]
            
            plt.figure(figsize=(8, 6))
            if not missing_values.empty:
                missing_values.plot(kind='bar', color='steelblue')
                plt.title('Missing Values by Column', fontsize=14, fontweight='bold')
                plt.xlabel('Columns', fontsize=12)
                plt.ylabel('Count of Missing Values', fontsize=12)
                plt.xticks(rotation=45, ha='right')
            else:
                plt.text(0.5, 0.5, 'No Missing Values', 
                         horizontalalignment='center', 
                         verticalalignment='center',
                         fontsize=14, fontweight='bold')
                plt.title('Missing Values', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('missing_values.png', dpi=150)
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating missing values visualization: {e}")
            logger.error(traceback.format_exc())
        
        # Distribution of first numeric column
        numeric_columns = [col for col, dtype in analysis['basic_info']['column_types'].items() 
                           if 'float' in dtype.lower() or 'int' in dtype.lower()]
        if numeric_columns:
            try:
                first_numeric_col = numeric_columns[0]
                plt.figure(figsize=(8, 6))
                sns.histplot(self.df[first_numeric_col].dropna(), kde=True, color='teal')
                plt.title(f'Distribution of {first_numeric_col}', fontsize=14, fontweight='bold')
                plt.xlabel(first_numeric_col, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.tight_layout()
                plt.savefig('distribution.png', dpi=150)
                plt.close()
            except Exception as e:
                logger.error(f"Error creating distribution visualization: {e}")
                logger.error(traceback.format_exc())
        
        # Correlation Heatmap
        correlation_matrix = analysis.get('correlation_matrix', {})
        if correlation_matrix:
            try:
                corr_df = pd.DataFrame(correlation_matrix).astype(float)
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0, 
                            square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                            annot_kws={"size":8})
                plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.yticks(rotation=0, fontsize=10)
                plt.tight_layout()
                plt.savefig('correlation_heatmap.png', dpi=150)
                plt.close()
            except Exception as e:
                logger.error(f"Error creating correlation heatmap: {e}")
                logger.error(traceback.format_exc())
        
        # Boxplots for numeric columns to visualize outliers
        if numeric_columns:
            try:
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=self.df[numeric_columns], orient='h', color='lightblue')
                plt.title('Boxplot for Numeric Columns', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig('boxplots.png', dpi=150)
                plt.close()
            except Exception as e:
                logger.error(f"Error creating boxplots: {e}")
                logger.error(traceback.format_exc())
        
        # If clustering was performed, visualize clusters on first two numeric columns if possible
        if "kmeans_clusters" in analysis and len(numeric_columns) >= 2:
            try:
                scaler = StandardScaler()
                numeric_data = self.df[numeric_columns].dropna()
                scaled_data = scaler.fit_transform(numeric_data)
                kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
                kmeans.fit(scaled_data)
                labels = kmeans.labels_
                
                plt.figure(figsize=(8, 6))
                plt.scatter(numeric_data[numeric_columns[0]], numeric_data[numeric_columns[1]], 
                            c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
                plt.title('KMeans Clusters (First Two Numeric Dimensions)', fontsize=14, fontweight='bold')
                plt.xlabel(numeric_columns[0], fontsize=12)
                plt.ylabel(numeric_columns[1], fontsize=12)
                plt.tight_layout()
                plt.savefig('clusters.png', dpi=150)
                plt.close()
            except Exception as e:
                logger.error(f"Error creating cluster visualization: {e}")
                logger.error(traceback.format_exc())
    
    def generate_narrative(self, analysis: Dict[str, Any]) -> str:
        """
        Generate a narrative about the data analysis using the LLM.
        
        :param analysis: Dictionary containing analysis results
        :return: Markdown narrative
        """
        try:
            # Prepare a brief summary of the data without sending all rows
            basic_info = analysis.get('basic_info', {})
            missing_values = {k: v for k,v in analysis.get('missing_values', {}).items() if v > 0}
            
            # Provide more detail and context in the prompt to get a richer narrative
            prompt = f"""
            You are an analyst. We have a dataset with:
            
            - Total Rows: {basic_info.get('total_rows')}
            - Total Columns: {basic_info.get('total_columns')}
            - Column Types: {json.dumps(basic_info.get('column_types'), indent=2)}
            - Missing Values: {json.dumps(missing_values, indent=2)}
            
            Descriptive Statistics (key metrics):
            {json.dumps(analysis.get('descriptive_stats', {}), indent=2)}
            
            Correlation Matrix (if available):
            {json.dumps(analysis.get('correlation_matrix', {}), indent=2)}
            
            Outlier Counts by Numeric Column:
            {json.dumps(analysis.get('outliers', {}), indent=2)}
            
            KMeans Clustering Results (if available):
            {json.dumps(analysis.get('kmeans_clusters', {}), indent=2)}
            
            Based on this information, write a compelling and comprehensive narrative about the dataset:
            
            - Begin by describing the nature and composition of the dataset (in general terms, since we do not know the domain).
            - Discuss the presence of missing values and what that might imply.
            - Highlight any notable insights from the descriptive statistics, especially if some columns stand out.
            - Interpret the correlation matrix (if available): what does it suggest about relationships between numeric variables?
            - Comment on the presence and distribution of outliers and what they might mean for further analysis or modeling.
            - Describe the clustering results (if any), explaining what the formation of clusters could imply about the data's structure.
            - Provide recommendations for further analysis or data processing, such as data cleaning steps, additional modeling techniques, or domain-specific follow-ups.
            
            Use markdown formatting. Include headings, bullet points, and paragraphs for clarity.
            Make the narrative informative, insightful, and engaging.
            
            At the end, provide a brief summary of what kind of actions or decisions someone might take based on these findings.
            """
            
            narrative = self._call_llm([{"role": "user", "content": prompt}])
            return narrative
        
        except Exception as e:
            logger.error(f"Error generating narrative: {e}")
            logger.error(traceback.format_exc())
            return f"""
            # Data Analysis Narrative

            ## Error Generating Narrative

            Unfortunately, an error occurred while attempting to generate the narrative for this dataset.

            *Error Details:*
            {str(e)}

            Please check the dataset and try again.
            """
    
    def run_analysis(self):
        """
        Run the complete automated analysis workflow.
        """
        # Perform generic analysis
        self.analysis = self._generate_generic_analysis()
        
        # Create visualizations
        self._create_visualizations(self.analysis)
        
        # Generate narrative
        narrative = self.generate_narrative(self.analysis)
        
        # Write narrative to README.md
        with open('README.md', 'w') as f:
            f.write(narrative)
        
        print("Analysis complete. Check README.md and generated PNG files for charts.")

def main():
    try:
        # Validate command-line arguments
        if len(sys.argv) != 2:
            logger.error("Incorrect usage. Please provide a CSV file.")
            print("Usage: uv run autolysis.py <dataset.csv>")
            sys.exit(1)
        
        # Validate file extension
        dataset_path = sys.argv[1]
        if not dataset_path.lower().endswith('.csv'):
            logger.error(f"Invalid file type: {dataset_path}. Must be a CSV file.")
            print("Error: Input must be a CSV file")
            sys.exit(1)
        
        # Determine output directory based on dataset name
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_dir = dataset_name.lower()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Run analysis
        try:
            analyzer = AutomatedAnalysis(dataset_path)
            analyzer.run_analysis()
            logger.info("Analysis completed successfully")
            
            # Move outputs to dataset-specific directory
            readme_path = 'README.md'
            png_files = [f for f in os.listdir('.') if f.endswith('.png')]
            
            if os.path.exists(readme_path):
                shutil.move(readme_path, os.path.join(output_dir, readme_path))
            
            for png_file in png_files:
                shutil.move(png_file, os.path.join(output_dir, png_file))
            
            print(f"Analysis results saved in {output_dir} directory")
        
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())
            print(f"Error during analysis: {e}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__== "__main__":
    main()
