# Exact Path: src/nlp_pipeline.py
import pandas as pd # Import pandas for processing tabular data
import os # Import os for system-independent file pathing
import joblib # Import joblib to save and load serialized Python objects
from sklearn.feature_extraction.text import TfidfVectorizer # Import TF-IDF transformer to convert text to numbers
from sklearn.model_selection import train_test_split # Import train_test_split to handle data partitioning

def build_nlp_features():
    """
    Transforms clean text strings into a numerical matrix using TF-IDF vectorization,
    encodes target labels, splits the data, and saves the fitted vectorizer object.
    
    Linguistic Explanation of TF-IDF:
    TF-IDF stands for Term Frequency-Inverse Document Frequency. It evaluates how 
    frequently a word appears in a single review (Term Frequency) and balances it against 
    how often that word shows up across all reviews in the dataset (Inverse Document Frequency). 
    This down-weights common, non-distinctive terms (like 'the' or 'university') and highlights 
    more informative keywords unique to specific review patterns.
    
    Why max_features=5000?
    Restricting the matrix to the top 5,000 most meaningful words prevents the vocabulary array 
    from growing too large. This helps prevent model overfitting, reduces memory consumption, 
    and keeps processing speeds fast on standard laptop hardware.
    """
    # Define paths for data input and model output
    input_path = os.path.join("data", "processed", "processed_reviews.csv")
    vectorizer_output = os.path.join("models", "tfidf_vectorizer.pkl")
    
    # Load the processed CSV dataset into a pandas DataFrame
    df = pd.read_csv(input_path)
    
    # Ensure there are no null values in the cleaned text column by replacing them with empty strings
    df['cleaned_review'] = df['cleaned_review'].fillna("")
    
    # Extract features (X) and target labels (y)
    X_text = df['cleaned_review']
    # Encode target labels: set truthful reviews to 1 and deceptive reviews to 0
    y_labels = df['deceptive_label'].apply(lambda val: 1 if str(val).strip().lower() == 'truthful' else 0)
    
    print("[*] Building the 5,000-feature TF-IDF text vector space matrix...")
    # Initialize the TF-IDF vectorizer configuration
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    # Fit the vectorizer on the text data and transform it into a numerical feature matrix
    X_transformed = vectorizer.fit_transform(X_text)
    
    print("[*] Splitting dataset into training (80%) and testing (20%) sets...")
    # Split the dataset into train and test groups using a fixed random state for reproducible splits
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y_labels, test_size=0.20, random_state=42, stratify=y_labels
    )
    
    # Ensure the models output directory exists; create it if missing
    os.makedirs(os.path.dirname(vectorizer_output), exist_ok=True)
    # Save the fitted vectorizer object to file for future deployment and inference tasks
    joblib.dump(vectorizer, vectorizer_output)
    print(f"[+] TF-IDF Vectorizer successfully saved to file: {vectorizer_output}")
    
    # Return the split dataset components along with the vectorizer object
    return X_train, X_test, y_train, y_test, vectorizer

# Check if the script is run directly from the terminal
if __name__ == "__main__":
    # Run the NLP feature extraction pipeline
    build_nlp_features()