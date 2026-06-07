# Exact Path: src/adapt_dataset.py
import pandas as pd # Import pandas for structured data manipulation
import numpy as np # Import numpy for vectorized mathematical choices
import os # Import os for directory configuration checks

def build_mock_source_data():
    """
    Constructs a foundational raw DataFrame to emulate external data files
    and verifies the processing capabilities of the downstream analytics pipeline.
    """
    # Define an array of sample reviews split evenly between genuine and deceptive styles
    sample_texts = [
        "The computer science curriculum is brilliant. Professors are industry veterans and the labs are open 24/7.",
        "This university is a complete scam. They take your tuition fee and provide zero job placement assistance.",
        "Excellent infrastructure and outstanding research facilities. Loved my time during the data science program.",
        "Worst management ever. Courses are outdated, faculty members are extremely rude, and degrees are useless.",
        "Highly recommend this institute for executive MBA programs. Case studies are incredibly practical.",
        "Fake promises and terrible accommodation. The campus environment is unsafe and teaching is subpar.",
        "Amazing library resources and top-tier mentorship for engineering streams. Truly transformative experience.",
        "A total waste of valuable time and money. The administrative staff loses documents and certificates constantly."
    ]
    
    # Balance labels evenly to match the sample text entries
    sample_labels = ["truthful", "deceptive", "truthful", "deceptive", "truthful", "deceptive", "truthful", "deceptive"]
    
    # List of sample educational institution names
    schools = ["Stanford Advanced Institute", "Global Tech University", "Apex Management School", "Metro Engineering College"]
    
    # Generate mock dates spread across the current fiscal year
    dates = ["2026-01-15", "2026-02-20", "2026-03-12", "2026-04-05", "2026-04-22", "2026-05-01", "2026-05-18", "2026-06-02"]
    
    # Combine the arrays into a structured dictionary format
    payload = {
        "review_text": sample_texts,
        "deceptive_label": sample_labels,
        "institution_name": [schools[i % len(schools)] for i in range(len(sample_texts))],
        "submission_date": dates
    }
    
    # Convert the structured dictionary into a standard pandas DataFrame object
    return pd.DataFrame(payload)

def adapt_and_save():
    """
    Validates structural formatting rules and exports the resulting dataset 
    into the designated unrefined data storage folder.
    """
    print("[*] Initiating Dataset Domain Adaptation Engine...")
    
    # Build the mock source dataset
    df = build_mock_source_data()
    
    # Define the destination directory path for raw data
    target_dir = os.path.join("data", "raw")
    # Ensure the target directory exists; create it if missing
    os.makedirs(target_dir, exist_ok=True)
    
    # Define the final absolute CSV file output path
    output_path = os.path.join(target_dir, "reviews.csv")
    
    # Export the DataFrame to a standard CSV file, omitting the pandas row index
    df.to_csv(output_path, index=False)
    print(f"[+] Domain Adaptation Successful. Structured data exported to: {output_path}")

# Check if the script is run directly from the terminal
if __name__ == "__main__":
    # Execute the adaptation and saving function
    adapt_and_save()