# Exact Path: src/export_results.py
import os # Import os to handle file path structures across operating systems
import pandas as pd # Import pandas for data aggregation and file exports
import joblib # Import joblib to load our saved machine learning assets
from textblob import TextBlob # Import TextBlob to calculate review sentiment metrics

# Import processing functions from our other pipeline modules
from similarity_detection import execute_similarity_analysis
from credibility_score import process_dataframe_credibility

def build_and_export_master_analytics():
    """
    Runs the entire data analysis pipeline across all reviews, aggregates all computed metrics,
    and exports a unified master table to a CSV file for Power BI and Streamlit dashboards.
    """
    print("[*] Initializing Master Data Aggregation and Export Engine...")
    
    # 1. Run the similarity detection analysis to catch duplicate reviews
    df_metrics = execute_similarity_analysis()
    
    # Define paths to load our saved classification model and vectorizer assets
    model_path = os.path.join("models", "best_model.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
    
    # Raise an error if the required model or vectorizer files are missing
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Required model files missing. Run train_models.py before exporting.")
        
    # Load the saved model and vectorizer objects from file
    best_model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # Ensure there are no null values in the cleaned text column by replacing them with empty strings
    df_metrics['cleaned_review'] = df_metrics['cleaned_review'].fillna("")
    
    print("[*] Generating model predictions and class probabilities across rows...")
    # Transform the cleaned text rows into our 5,000-feature numerical matrix
    X_features = vectorizer.transform(df_metrics['cleaned_review'])
    
    # Generate class predictions across all reviews
    row_predictions = best_model.predict(X_features)
    # Extract prediction class probabilities across all reviews
    row_probabilities = best_model.predict_proba(X_features)
    
    # Map the class indexes to operational label strings ('Genuine' or 'Deceptive')
    df_metrics['prediction'] = ["Genuine" if p == 1 else "Deceptive" for p in row_predictions]
    # Store the specific probability score for the 'Genuine' class for each row
    df_metrics['prob_genuine'] = row_probabilities[:, 1]
    # Store the actual confidence probability score associated with the predicted class choice
    df_metrics['probability'] = [float(row_probabilities[i, row_predictions[i]]) for i in range(len(row_predictions))]
    
    print("[*] Extracting sentiment polarity metrics...")
    # Calculate the continuous sentiment polarity score for each review row
    df_metrics['sentiment_score_raw'] = df_metrics['review_text'].apply(lambda t: TextBlob(str(t)).sentiment.polarity)
    
    # Map continuous sentiment scores to categorical labels ('Positive', 'Negative', or 'Neutral')
    def category_mapper(score):
        if score > 0.15: return "Positive"
        elif score < -0.15: return "Negative"
        else: return "Neutral"
    df_metrics['sentiment'] = df_metrics['sentiment_score_raw'].apply(category_mapper)
    
    # 2. Run the credibility engine to calculate review credibility scores and categories
    df_final = process_dataframe_credibility(df_metrics)
    
    # Filter and arrange the final columns to match our defined database reporting schema
    reporting_columns = [
        'review_text',
        'prediction',
        'probability',
        'credibility_score',
        'credibility_category',
        'sentiment',
        'similarity_score',
        'duplicate_flag',
        'institution_name',
        'submission_date'
    ]
    
    # Re-index the DataFrame to match our clean reporting schema format
    df_export = df_final[reporting_columns].copy()
    # Rename the date column header to ensure clean integration with Power BI data models
    df_export.rename(columns={'submission_date': 'date'}, inplace=True)
    
    # Define the absolute output file path for the master analytics report
    export_output_path = os.path.join("outputs", "results.csv")
    # Ensure the outputs directory exists; create it if missing
    os.makedirs(os.path.dirname(export_output_path), exist_ok=True)
    
    # Export the consolidated DataFrame to a flat CSV file, omitting row indexes
    df_export.to_csv(export_output_path, index=False)
    print(f"\n[+] Master Export Successful. Dataset saved to: {export_output_path}")
    print(f"[+] Total records written to master file: {df_export.shape[0]} rows.")

# Check if the script is run directly from the terminal
if __name__ == "__main__":
    # Run the master results export pipeline
    build_and_export_master_analytics()