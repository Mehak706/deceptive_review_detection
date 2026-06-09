
import pandas as pd
import os
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required resources
nltk.download("stopwords")
nltk.download("wordnet")

def clean_text_core(raw_text):
    if pd.isna(raw_text):
        return ""

    text = str(raw_text).lower()

    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    tokens = text.split()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words
    ]

    return " ".join(cleaned_tokens)


def execute_preprocessing_pipeline():

    input_file = os.path.join(
        "data",
        "raw",
        "reviews.csv"
    )

    output_file = os.path.join(
        "data",
        "processed",
        "processed_reviews.csv"
    )

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Source file not located at: {input_file}"
        )

    df = pd.read_csv(input_file)

    df.dropna(
        subset=["review_text", "deceptive_label"],
        inplace=True
    )

    df.drop_duplicates(
        subset=["review_text"],
        inplace=True
    )

    df["cleaned_review"] = df["review_text"].apply(
        clean_text_core
    )

    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    df.to_csv(
        output_file,
        index=False
    )

    return df


if __name__ == "__main__":
    execute_preprocessing_pipeline()

