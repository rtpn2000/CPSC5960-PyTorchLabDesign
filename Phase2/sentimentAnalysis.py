import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Load the dataset
df = pd.read_csv('sentimentdataset.csv')

# Initialize Vader and Hugging Face models
vader_analyzer = SentimentIntensityAnalyzer()
huggingface_analyzer = pipeline("sentiment-analysis")

# Helper function to analyze sentiment
def analyze_sentiment_vader(text):
    score = vader_analyzer.polarity_scores(text)
    return 'Positive' if score['compound'] >= 0.05 else 'Negative' if score['compound'] <= -0.05 else 'Neutral'

def analyze_sentiment_huggingface(text):
    result = huggingface_analyzer(text)[0]
    return result['label']

# Apply models
df['Vader Sentiment'] = df['Text'].apply(analyze_sentiment_vader)
df['Hugging Face Sentiment'] = df['Text'].apply(analyze_sentiment_huggingface)

# Compare accuracy
vader_accuracy = (df['Vader Sentiment'] == df['Sentiment']).mean() * 100
huggingface_accuracy = (df['Hugging Face Sentiment'] == df['Sentiment']).mean() * 100

print(f"Vader Accuracy: {vader_accuracy:.2f}%")
print(f"Hugging Face Accuracy: {huggingface_accuracy:.2f}%")

# Save results to a CSV
df.to_csv('sentiment_results.csv', index=False)
print("Results saved to 'sentiment_results.csv'")