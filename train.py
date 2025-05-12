# TF-IDF Vectorization (for traditional ML models)
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


train_data = pd.read_csv("./data/train_preprocessed.csv").dropna(subset=['processed_text'])
validation_data = pd.read_csv("./data/validation_preprocessed.csv").dropna(subset=['processed_text'])
test_data = pd.read_csv("./data/test_preprocessed.csv").dropna(subset=['processed_text'])


tfidf_vectorizer = TfidfVectorizer(max_features=50000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['processed_text'])
X_val_tfidf = tfidf_vectorizer.transform(validation_data['processed_text'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['processed_text'])

max_words = 50000  # From the paper
max_sequence_length = 500  # Adjust based on your data analysis

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_data['processed_text'])

X_train_seq = tokenizer.texts_to_sequences(train_data['processed_text'])
X_val_seq = tokenizer.texts_to_sequences(validation_data['processed_text'])
X_test_seq = tokenizer.texts_to_sequences(test_data['processed_text'])

X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_val_pad = pad_sequences(X_val_seq, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)
print(X_train_pad)

# Convert labels to numeric format
y_train = np.array(train_data['label'])  # Adjust based on your labels
y_val = np.array(validation_data['label'])

# Define the CNN model
from tensorflow.keras.layers import GlobalMaxPooling1D

# Define the CNN model
embedding_dim = 32  # As mentioned in the paper

cnn_model = Sequential([
    Embedding(max_words, embedding_dim, input_length=max_sequence_length),
    Conv1D(filters=250, kernel_size=3, padding='valid', activation='relu'),
    GlobalMaxPooling1D(),  # This reduces the 3D output to 2D
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
cnn_history = cnn_model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=10,
    batch_size=32,
    callbacks=[early_stopping]
)

# Evaluate on validation set
cnn_val_preds = (cnn_model.predict(X_val_pad) > 0.5).astype(int)
print("CNN Validation Accuracy:", accuracy_score(y_val, cnn_val_preds))
print("CNN Validation F1 Score:", f1_score(y_val, cnn_val_preds))


X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)
y_test = np.array(test_data['label'])

# Test the CNN model (best performer)
cnn_test_preds = (cnn_model.predict(X_test_pad) > 0.5).astype(int)
print("CNN Test Accuracy:", accuracy_score(y_test, cnn_test_preds))
print("CNN Test F1 Score:", f1_score(y_test, cnn_test_preds))
print(classification_report(y_test, cnn_test_preds))

# Save the best model
cnn_model.save('fake_news_cnn_model.keras')
