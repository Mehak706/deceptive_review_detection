# Exact Path: src/train_models.py
import os # Import os for cross-platform file path routing
import joblib # Import joblib to save and load serialized Python objects
import matplotlib.pyplot as plt # Import matplotlib for plot rendering configuration
import seaborn as sns # Import seaborn for confusion matrix visualizations
import pandas as pd # Import pandas for building metrics tables

# Import the classification models from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Import model evaluation metric utilities from scikit-learn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Import the feature preparation function from our NLP pipeline script
from nlp_pipeline import build_nlp_features

def execute_model_suite():
    """
    Trains and evaluates Logistic Regression, Random Forest, and Naive Bayes classifiers,
    compares their performance metrics, and automatically saves the best-performing model.
    
    Why these algorithms?
    1. Logistic Regression: A fast, efficient linear classifier that works well with high-dimensional, 
       sparse data formats like TF-IDF matrices.
    2. Random Forest Classifier: An ensemble tree model that captures non-linear relationships and interactions 
       between keywords across reviews.
    3. Multinomial Naive Bayes: A probabilistic classifier built on word frequency distributions, 
       highly effective for classic text classification tasks.
    """
    # Run the feature pipeline to get train/test splits and data matrices
    X_train, X_test, y_train, y_test, vectorizer = build_nlp_features()
    
    # Define a dictionary containing initializations for all three models
    models_pool = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=150),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1)
    }
    
    # Initialize a dictionary to store computed metrics for each model
    performance_records = {}
    # Define the output directory path for saving evaluation charts
    chart_dir = os.path.join("outputs", "charts")
    
    # Loop through the dictionary to train and evaluate each model
    for name, model in models_pool.items():
        print(f"\n[*] Commencing training routine for classifier: {name}")
        # Fit the model on the training data matrix
        model.fit(X_train, y_train)
        
        # Generate predictions on the test data split
        predictions = model.predict(X_test)
        
        # Calculate standard evaluation metrics based on test performance
        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions, average='macro')
        rec = recall_score(y_test, predictions, average='macro')
        f1 = f1_score(y_test, predictions, average='macro')
        
        # Store computed metrics into our tracking records dictionary
        performance_records[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "Model_Object": model
        }
        
        # Print a standard classification report to the terminal windows
        print(f"Classification Report for {name}:\n")
        print(classification_report(y_test, predictions, target_names=["Deceptive", "Genuine"]))
        
        # Compute the confusion matrix values
        cm = confusion_matrix(y_test, predictions)
        
        # Plot and save the confusion matrix visualization
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Deceptive", "Genuine"], yticklabels=["Deceptive", "Genuine"])
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('True Observed Category')
        plt.xlabel('Model Predicted Category')
        chart_save_path = os.path.join(chart_dir, f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        plt.savefig(chart_save_path, bbox_inches='tight')
        plt.close()
        
    # --- PHASE 8 — AUTOMATED BEST MODEL SELECTION ---
    print("\n" + "="*60 + "\nSUMMARY PERFORMANCE METRIC COMPARISON MATRIX\n" + "="*60)
    
    # Build a summary DataFrame to display all model performance metrics side-by-side
    summary_data = []
    for name, metrics in performance_records.items():
        summary_data.append({
            "Algorithm": name,
            "Accuracy": f"{metrics['Accuracy']:.4f}",
            "Precision": f"{metrics['Precision']:.4f}",
            "Recall": f"{metrics['Recall']:.4f}",
            "F1-Score": f"{metrics['F1_Score']:.4f}"
        })
    print(pd.DataFrame(summary_data).to_string(index=False))
    
    # Find the best-performing model based on the highest F1-Score value
    best_model_name = max(performance_records, key=lambda key: performance_records[key]["F1_Score"])
    best_model_data = performance_records[best_model_name]
    
    # Define the absolute output file path for the best model serialization
    best_model_filepath = os.path.join("models", "best_model.pkl")
    
    # Save the selected best model object to file using joblib
    joblib.dump(best_model_data["Model_Object"], best_model_filepath)
    
    print("\n" + "#"*70)
    print(f"[+] Automated Selection Complete. Best Model Chosen: {best_model_name}")
    print(f"[+] Top Macro F1-Score Value achieved: {best_model_data['F1_Score']:.4f}")
    print(f"[+] Serialized model successfully saved to destination: {best_model_filepath}")
    print("#"*70 + "\n")

# Check if the script is run directly from the terminal
if __name__ == "__main__":
    # Execute the machine learning model training and evaluation suite
    execute_model_suite()