# preprocess.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os

nltk.download('stopwords', quiet=True)
romanian_stopwords = set(stopwords.words('romanian'))

# Enhanced sentiment and irony keywords
positive_words = ['minunat', 'perfect', 'excelent', 'bun', 'frumos', 'superb', 'fantastic', 'grozav']
negative_words = ['rău', 'groaznic', 'prost', 'urât', 'teribil', 'jalnic', 'înfricoșător', 'dezgustător']
neutral_words = ['este', 'sunt', 'avea', 'face', 'există', 'poate', 'trebuie']
irony_patterns = ['desigur', 'evident', 'sigur că', 'normal că', 'ce surpriză', 'la naiba', 'aproape că']

def extract_key_features(text):
    features = {}
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    features['exclamation_marks'] = text.count('!')
    features['question_marks'] = text.count('?')
    features['quotes'] = text.count('"') + text.count("'")
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0

    text_lower = text.lower()
    features['positive_word_count'] = sum(text_lower.count(word) for word in positive_words)
    features['negative_word_count'] = sum(text_lower.count(word) for word in negative_words)
    features['neutral_word_count'] = sum(text_lower.count(word) for word in neutral_words)
    features['irony_indicator_count'] = sum(text_lower.count(pattern) for pattern in irony_patterns)

    return features

def preprocess_for_tfidf(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    filtered = [w for w in words if w not in romanian_stopwords]
    return ' '.join(filtered)

def main():
    print("Loading data...")
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/stiri_digi24_alegeri.csv')
    train_data['content'] = train_data['content'].fillna('')
    test_data['contents'] = test_data['contents'].fillna('')

    print("Extracting features...")
    train_features = train_data['content'].apply(extract_key_features).apply(pd.Series)
    test_features = test_data['contents'].apply(extract_key_features).apply(pd.Series)

    print("Preprocessing text...")
    train_data['processed_text'] = train_data['content'].apply(preprocess_for_tfidf)
    test_data['processed_text'] = test_data['contents'].apply(preprocess_for_tfidf)

    print("Creating TF-IDF...")
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(train_data['processed_text'])
    X_test_tfidf = tfidf.transform(test_data['processed_text'])

    X_train = np.hstack([X_train_tfidf.toarray(), train_features.values])
    X_test = np.hstack([X_test_tfidf.toarray(), test_features.values])

    os.makedirs("processed", exist_ok=True)
    pd.DataFrame(X_train).to_csv('processed/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('processed/X_test.csv', index=False)
    train_data[['label']].to_csv('processed/y_train.csv', index=False)
    test_data[['headline']].to_csv('processed/headlines_test.csv', index=False)

    print("Preprocessing complete. Saved to 'processed/' folder.")

if __name__ == '__main__':
    main()
