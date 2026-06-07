# Exact Path: src/eda.py
import pandas as pd # Import pandas for data loading and summary operations
import os # Import os to handle system file paths across different operating systems
import matplotlib.pyplot as plt # Import matplotlib for rendering core layout figures
import seaborn as sns # Import seaborn for advanced statistical visualizations
from wordcloud import WordCloud # Import WordCloud to generate text keyword maps
from textblob import TextBlob # Import TextBlob for simple sentiment score extraction

def compute_sentiment_polarity(text):
    """
    Extracts a raw text sentiment score ranging from -1.0 (highly negative) 
    to +1.0 (highly positive) using TextBlob.
    TextBlob is chosen because it offers an easy, clear API for beginners 
    without requiring complex deep learning configurations.
    """
    return TextBlob(str(text)).sentiment.polarity

def generate_eda_suite():
    """
    Loads the preprocessed dataset and generates diagnostic plots, 
    saving the results to the output graphics directory.
    """
    # Define the source path for preprocessed data
    data_path = os.path.join("data", "processed", "processed_reviews.csv")
    # Define the destination directory for output charts
    chart_dir = os.path.join("outputs", "charts")
    # Ensure the destination directory exists; create it if missing
    os.makedirs(chart_dir, exist_ok=True)
    
    # Load the processed CSV data into a pandas DataFrame
    df = pd.read_csv(data_path)
    
    # 1. Class Distribution Plot
    plt.figure(figsize=(6, 4)) # Initialize a blank figure canvas
    # Render a bar chart showing the breakdown of truthful vs deceptive labels
    sns.countplot(data=df, x='deceptive_label', palette='Set2')
    plt.title('Target Distribution: Genuine vs Deceptive Reviews') # Set title
    plt.xlabel('Review Category Classification') # Label the x-axis
    plt.ylabel('Total Count') # Label the y-axis
    # Save the chart as a high-resolution PNG file
    plt.savefig(os.path.join(chart_dir, "class_distribution.png"), bbox_inches='tight')
    plt.close() # Close the figure to free up system memory
    
    # 2. Review Length Frequency Plot
    # Calculate the total character count for each raw review entry
    df['review_length'] = df['review_text'].apply(lambda x: len(str(x)))
    plt.figure(figsize=(8, 5)) # Initialize canvas
    # Plot a histogram with a kernel density estimate curve to show the distribution of review lengths
    sns.histplot(data=df, x='review_length', hue='deceptive_label', kde=True, element='step', palette='muted')
    plt.title('Distribution of Review Lengths') # Set title
    plt.xlabel('Character Count') # Label x-axis
    plt.ylabel('Density / Frequency') # Label y-axis
    # Save the distribution chart
    plt.savefig(os.path.join(chart_dir, "review_length_distribution.png"), bbox_inches='tight')
    plt.close() # Close figure
    
    # 3. Top Keyword Frequencies Plot
    # Combine all cleaned text rows into a single list of individual words
    all_words = " ".join(df['cleaned_review'].dropna().astype(str)).split()
    # Count how often each unique word occurs across the dataset
    word_counts = pd.Series(all_words).value_counts()
    plt.figure(figsize=(10, 6)) # Initialize canvas
    # Plot a horizontal bar chart showing the top 20 most frequent words
    sns.barplot(x=word_counts.head(20).values, y=word_counts.head(20).index, palette='viridis')
    plt.title('Top 20 Most Frequent Words in Processed Text') # Set title
    plt.xlabel('Occurrences Count') # Label x-axis
    # Save the word frequency chart
    plt.savefig(os.path.join(chart_dir, "top_20_words.png"), bbox_inches='tight')
    plt.close() # Close figure
    
    # 4. Sentiment Score Distribution Plot
    # Apply the sentiment analysis function across the dataset rows
    df['sentiment_polarity'] = df['cleaned_review'].apply(compute_sentiment_polarity)
    plt.figure(figsize=(8, 5)) # Initialize canvas
    # Plot a histogram showing the distribution of sentiment scores across categories
    sns.histplot(data=df, x='sentiment_polarity', hue='deceptive_label', kde=True, element='poly', palette='coolwarm')
    plt.title('Linguistic Sentiment Score Distribution') # Set title
    plt.xlabel('Polarity Index (-1.0 to 1.0)') # Label x-axis
    # Save the sentiment score chart
    plt.savefig(os.path.join(chart_dir, "sentiment_distribution.png"), bbox_inches='tight')
    plt.close() # Close figure
    
    # 5. Word Cloud for Genuine Reviews
    # Filter the dataset to include only truthful reviews
    genuine_text = " ".join(df[df['deceptive_label'] == 'truthful']['cleaned_review'].dropna().astype(str))
    # Build a word cloud configuration mapping for the genuine text
    wc_g = WordCloud(width=800, height=400, background_color='white', colormap='ocean').generate(genuine_text)
    plt.figure(figsize=(10, 5)) # Initialize canvas
    plt.imshow(wc_g, interpolation='bilinear') # Render the word cloud image
    plt.axis('off') # Hide chart axis lines and tick marks
    plt.title('Keyword Cloud: Genuine Student Reviews') # Set title
    # Save the genuine word cloud image
    plt.savefig(os.path.join(chart_dir, "wordcloud_genuine.png"), bbox_inches='tight')
    plt.close() # Close figure
    
    # 6. Word Cloud for Deceptive Reviews
    # Filter the dataset to include only deceptive reviews
    deceptive_text = " ".join(df[df['deceptive_label'] == 'deceptive']['cleaned_review'].dropna().astype(str))
    # Build a word cloud configuration mapping for the deceptive text
    wc_d = WordCloud(width=800, height=400, background_color='black', colormap='autumn').generate(deceptive_text)
    plt.figure(figsize=(10, 5)) # Initialize canvas
    plt.imshow(wc_d, interpolation='bilinear') # Render the word cloud image
    plt.axis('off') # Hide chart axis lines and tick marks
    plt.title('Keyword Cloud: Deceptive/Fake Reviews') # Set title
    # Save the deceptive word cloud image
    plt.savefig(os.path.join(chart_dir, "wordcloud_deceptive.png"), bbox_inches='tight')
    plt.close() # Close figure
    
    print(f"[+] EDA Suite Execution Complete. 6 Diagnostic charts exported to: {chart_dir}")

# Check if the script is run directly from the terminal
if __name__ == "__main__":
    # Execute the EDA suite function
    generate_eda_suite()