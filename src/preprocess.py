# Exact Path: src/preprocess.py
import pandas as pd # Import pandas for data manipulation and tabular file I/O
import os # Import os to handle operating system paths across platforms
import string # Import string to quickly access pre-defined punctuation character lists
import nltk # Import core natural language processing tools
from nltk.corpus import stopwords # Import stopwords list to filter out common low-value filler words
from nltk.tokenize import word_tokenize # Import word_tokenize to split text blocks into individual word tokens
from nltk.stem import WordNetLemmatizer # Import WordNetLemmatizer to reduce words back to their dictionary base forms

def clean_text_core(raw_text):
    """
    Accepts a raw text string and processes it through an pipeline of 
    normalization, tokenization, stopword removal, and lemmatization steps.
    """
    # Return an empty string if the incoming text value is missing or null
    if pd.isna(raw_text):
        return ""
    
    # 1. Cast all characters to lowercase to standardize text and avoid case mismatch
    text = str(raw_text).lower()
    
    # 2. Remove punctuation marks by checking each character against the system punctuation list
    text = "".join([char for char in text if char not in string.punctuation])
    
    # 3. Split the continuous text string into individual word tokens
    tokens = word_tokenize(text)
    
    # 4. Load the standard English stopword list to filter out common filler words
    stop_words = set(stopwords.words('english'))
    
    # 5. Initialize the WordNet Lemmatizer to resolve inflected word forms back to their base lemma
    lemmatizer = WordNetLemmatizer()
    
    # 6. Filter out stopwords and reduce remaining words to their base form
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # 7. Recombine the processed word tokens back into a single space-separated string
    return " ".join(cleaned_tokens)

def execute_preprocessing_pipeline():
    """
    Loads raw CSV data, removes invalid or duplicate rows, applies text cleaning 
    transformations, and saves the resulting dataset to the processed data directory.
    """
    # Define absolute input file path pointing to the raw data directory
    input_file = os.path.join("data", "raw", "reviews.csv")
    # Define absolute output path pointing to the processed data directory
    output_file = os.path.join("data", "processed", "processed_reviews.csv")
    
    # Raise an explicit error if the raw source file cannot be found
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Source file not located at: {input_file}. Run adaptation script first.")
    
    print(f"[*] Ingesting raw dataset from: {input_file}")
    # Read the raw CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)
    
    # Drop rows containing missing values or null fields in place
    df.dropna(subset=['review_text', 'deceptive_label'], inplace=True)
    # Remove duplicate rows based on identical review text content
    df.drop_duplicates(subset=['review_text'], inplace=True)
    
    print("[*] Running text normalization and processing routines...")
    # Apply the text cleaning function across all rows in the review text column
    df['cleaned_review'] = df['review_text'].apply(clean_text_core)
    
    # Ensure the target directory exists before saving the output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Save the processed DataFrame to a new CSV file, omitting row index columns
    df.to_csv(output_file, index=False)
    print(f"[+] Text processing complete. Cleaned file saved to: {output_file}")

# Check if the script is run directly from the terminal
if __name__ == "__main__":
    # Run the data preprocessing pipeline
    execute_preprocessing_pipeline()