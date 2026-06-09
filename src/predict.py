# Exact Path: src/predict.py
import os # Import os for file path verification
import joblib # Import joblib to load our saved model and vectorizer files
# Import our text cleaning function directly from the preprocessing script to ensure identical cleaning steps
from preprocess import clean_text_core 

def run_single_inference(raw_review_string):
    """
    Accepts a single raw user review string, applies text cleaning transformations, 
    and predicts whether the text is a Genuine or Deceptive review.
    """
    # Define absolute file paths for our saved model and vectorizer assets
    model_path = os.path.join("models", "best_model.pkl")
    vectorizer_path = os.path.join("models", "tfidf_vectorizer.pkl")
    
    # Raise an error if the required model or vectorizer asset files cannot be found
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Trained model or vectorizer assets missing. Run train_models.py first.")
        
    # Load the saved model and vectorizer objects from file
    best_model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    
    # Process the raw text string through our standardized cleaning pipeline
    clean_text = clean_text_core(raw_review_string)
    
    # Transform the clean text string into a numerical feature vector using the loaded vectorizer
    vectorized_input = tfidf_vectorizer.transform([clean_text])
    
    # Generate the class classification prediction (returns 0 or 1)
    prediction_class = best_model.predict(vectorized_input)[0]
    # Extract the prediction probability array for the input features
    probabilities = best_model.predict_proba(vectorized_input)[0]
    
    # Map the class index to its corresponding label string
    predicted_label = "Genuine" if prediction_class == 1 else "Deceptive"
    # Extract the probability score associated with the chosen class
    confidence_probability = float(probabilities[prediction_class])
    
    # Return a structured dictionary containing the inference results
    return {
        "raw_text": raw_review_string,
        "clean_text": clean_text,
        "prediction": predicted_label,
        "confidence": confidence_probability,
        "prob_genuine": float(probabilities[1]) # Save the raw probability for the genuine class
    }

# Check if the script is run directly from the terminal
if __name__ == "__main__":
    print("\n[*] Initializing Live Production Single-Inference Module Testing...")
    
    # Define a sample test string that mimics typical deceptive review patterns
    test_review = "This school is amazing! The best campus in the entire world. Everything is absolutely perfect!"
    
    # Run the sample test string through our inference function
    inference_result = run_single_inference(test_review)
    
    print("\n" + "="*50 + "\nSINGLE-INFERENCE PRODUCTION RESULT REPORT\n" + "="*50)
    print(f"Input Text:  '{inference_result['raw_text']}'")
    print(f"Prediction:  {inference_result['prediction']}")
    print(f"Confidence:  {inference_result['confidence'] * 100.0:.2f}%")
    print("="*50 + "\n")
print(result)
