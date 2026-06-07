# Exact Path: src/credibility_score.py
import pandas as pd # Import pandas for vectorizing row manipulations
import numpy as np # Import numpy for numerical bounding methods
from textblob import TextBlob # Import TextBlob to check review sentiment attributes

def compute_credibility_index(row):
    """
    Calculates a unified 0 to 100 credibility index score for an individual review row 
    by applying an experimental multi-variable formula.
    
    The Scoring Formula Breakdown:
    1. Model Confidence Score (Weight: 50%): Extracts the model's predicted probability 
       that a review belongs to the 'Genuine' class. High genuine probabilities directly 
       increase the base credibility score.
    2. Sentiment Alignment Score (Weight: 20%): Evaluates review sentiment polarity. 
       Balanced or moderately positive sentiment boosts credibility, whereas extreme, unhelpful 
       negative sentiment slightly lowers it.
    3. Text Length Score (Weight: 15%): Evaluates review character length against typical writing patterns. 
       Extremely short reviews (under 40 characters) or excessively long walls of text receive lower length 
       scores, while standard-length reviews receive a full score.
    4. Text Uniqueness Score (Weight: 15%): Evaluates content duplication across reviews. 
       Reviews with high similarity scores relative to other entries receive lower uniqueness scores 
       to flag potential copy-paste content spam.
    """
    # 1. Model Confidence Component calculation
    # Pull the probability score for the 'Genuine' class from the input row fields
    prob_genuine = float(row.get('prob_genuine', 0.5))
    confidence_component = prob_genuine * 100.0
    
    # 2. Sentiment Component calculation
    # Pull the raw sentiment polarity index score (ranging from -1.0 to +1.0)
    sentiment_val = float(row.get('sentiment_polarity', 0.0))
    # Transform the scale from [-1, +1] to [0.0, 100.0]
    sentiment_component = (sentiment_val + 1.0) * 50.0
    
    # 3. Review Text Length Component calculation
    review_len = len(str(row.get('review_text', '')))
    if review_len < 40 or review_len > 3500:
        # Penalize reviews that are abnormally short or excessively long
        length_component = 30.0
    elif 150 <= review_len <= 800:
        # Award a perfect score for reviews falling within the optimal character length range
        length_component = 100.0
    else:
        # Assign a standard mid-range score for reviews falling outside the optimal length zone
        length_component = 75.0
        
    # 4. Text Uniqueness Component calculation
    # Pull the calculated review similarity percentage metric (default to 0 if missing)
    similarity_pct = float(row.get('similarity_score', 0.0))
    # Invert the similarity score so that highly unique reviews receive a high uniqueness score
    uniqueness_component = 100.0 - similarity_pct
    
    # Combine the four weighted components to calculate the final aggregate credibility index score
    aggregate_score = (
        (0.50 * confidence_component) +
        (0.20 * sentiment_component) +
        (0.15 * length_component) +
        (0.15 * uniqueness_component)
    )
    
    # Ensure the final index score is strictly bounded between 0.0 and 100.0
    return float(np.clip(aggregate_score, 0.0, 100.0))

def assign_credibility_label(score):
    """
    Maps a raw numerical credibility index score to an explicit operational risk category.
    """
    if score <= 45.0:
        return "Highly Suspicious"
    elif score <= 75.0:
        return "Moderately Suspicious"
    else:
        return "Highly Credible"

def process_dataframe_credibility(df):
    """
    Accepts a pandas DataFrame, computes baseline sentiment metrics, 
    and applies the weighted credibility scoring formula across all rows.
    """
    # Calculate text sentiment polarity scores for each review row if not already present
    if 'sentiment_polarity' not in df.columns:
        df['sentiment_polarity'] = df['review_text'].apply(lambda txt: TextBlob(str(txt)).sentiment.polarity)
        
    print("[*] Calculating multi-variable weighted Credibility Index Scores across rows...")
    # Apply the scoring function row-by-row across the entire DataFrame
    df['credibility_score'] = df.apply(compute_credibility_index, axis=1)
    
    # Map the numerical scores to their corresponding operational risk categories
    df['credibility_category'] = df['credibility_score'].apply(assign_credibility_label)
    
    # Return the enriched DataFrame containing the new credibility metrics
    return df