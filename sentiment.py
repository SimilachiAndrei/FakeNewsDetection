# Simple sentiment analysis
# This is pseudo-code since you'd need a Romanian sentiment lexicon
import pandas as pd


def analyze_sentiment(text):
    # Load a Romanian sentiment lexicon
    positive_words = {...}  # Romanian positive words
    negative_words = {...}  # Romanian negative words

    # Count positive and negative words
    words = text.split()
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)

    # Determine polarity
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'

train_data = pd.read_csv('./data/train_preprocessed.csv').dropna(subset=['processed_text'])
validation_data = pd.read_csv('./data/validation_preprocessed.csv').dropna(subset=['processed_text'])
test_data = pd.read_csv('./data/test_preprocessed.csv').dropna(subset=['processed_text'])


# Add sentiment as a feature
train_data['sentiment'] = train_data['processed_text'].apply(analyze_sentiment)
validation_data['sentiment'] = validation_data['processed_text'].apply(analyze_sentiment)
test_data['sentiment'] = test_data['processed_text'].apply(analyze_sentiment)

# Convert to numerical
sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
train_data['sentiment_score'] = train_data['sentiment'].map(sentiment_map)
validation_data['sentiment_score'] = validation_data['sentiment'].map(sentiment_map)
test_data['sentiment_score'] = test_data['sentiment'].map(sentiment_map)