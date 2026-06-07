# Exact Path: src/similarity_detection.py
import pandas as pd # Import pandas for managing tabular datasets
import os # Import os for cross-platform file pathing
from sklearn.feature_extraction.text import TfidfVectorizer # Import TF-IDF transformer for vectorizing text
from sklearn.metrics.pairwise import cosine_similarity # Import cosine_similarity to measure text vector similarity

def execute_similarity_analysis():
    """
    Loads the preprocessed dataset, builds a runtime TF-IDF matrix, 
    and calculates pairwise Cosine Similarity scores to identify and flag 
    highly similar or duplicate review entries.
    
    What is Cosine Similarity?
    Cosine similarity measures the cosine of the angle between two multi-dimensional text vectors. 
    It assigns a score between 0.0 and 1.0, tracking how closely the keyword distribution patterns 
    and word selections match between two separate review strings, independent of text length differences.
    """
    # Define input path for preprocessed data
    data_path = os.path.join("data", "processed", "processed_reviews.csv")
    # Load the processed CSV data into a pandas DataFrame
    df = pd.read_csv(data_path)
    
    # Ensure there are no null values in the cleaned text column by replacing them with empty strings
    df['cleaned_review'] = df['cleaned_review'].fillna("")
    
    print("[*] Building runtime TF-IDF matrix for similarity analysis...")
    # Initialize a localized vectorizer instance for checking text duplication
    sim_vectorizer = TfidfVectorizer(ngram_range=(1,1))
    # Transform the cleaned review text rows into a continuous numerical vector matrix
    tfidf_matrix = sim_vectorizer.fit_transform(df['cleaned_review'])
    
    print("[*] Computing pairwise Cosine Similarity matrix values...")
    # Compute the complete pairwise cosine similarity matrix across all review rows
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    max_similarity_scores = []
    duplicate_flags = []
    total_rows = df.shape[0]
    
    # Iterate through each review row in the matrix to find its top matching entry
    for idx in range(total_rows):
        # Extract the array of pairwise similarity scores for the current review row
        row_scores = similarity_matrix[idx].copy()
        
        # Zero out the self-similarity score at the current index position to avoid self-matching
        row_scores[idx] = 0.0
        
        # Find the highest similarity score value remaining in the row array
        max_score = row_scores.max()
        # Convert the raw decimal similarity score into a clean percentage value
        max_sim_percentage = float(max_score * 100.0)
        # Append the calculated percentage score to our tracking list
        max_similarity_scores.append(max_sim_percentage)
        
        # Flag the review as a duplicate if its top similarity score exceeds our 85% threshold
        if max_sim_percentage > 85.0:
            duplicate_flags.append(True)
        else:
            duplicate_flags.append(False)
            
    # Assign the calculated similarity metrics to new columns in the DataFrame
    df['similarity_score'] = max_similarity_scores
    df['duplicate_flag'] = duplicate_flags
    
    # Count the total number of reviews flagged as duplicates
    total_duplicates = sum(duplicate_flags)
    print(f"[+] Similarity analysis complete. Flagged {total_duplicates} duplicate records out of {total_rows} total rows.")
    
    # Return the enriched DataFrame containing the new duplicate tracking columns
    return df

# Check if the script is run directly from the terminal
if __name__ == "__main__":
    # Run the similarity detection analysis
    execute_similarity_analysis()