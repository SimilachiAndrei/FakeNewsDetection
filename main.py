import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load
train = pd.read_csv('./data/train_preprocessed.csv').dropna(subset=['processed_text'])
test  = pd.read_csv('./data/test_preprocessed.csv').dropna(subset=['processed_text'])
print(test['headline'])

# 2. Vectorize
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train['processed_text'])
X_test  = vectorizer.transform(test['processed_text'])

# 3. Train on ALL of train_data
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, train['label'])

# 4. Evaluate on test_data
test_preds = model.predict(X_test)

print(test_preds)