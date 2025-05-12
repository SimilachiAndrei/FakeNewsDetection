# main.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def main():
    print("Loading preprocessed data...")
    X_train = pd.read_csv('processed/X_train.csv')
    X_test = pd.read_csv('processed/X_test.csv')
    y_train = pd.read_csv('processed/y_train.csv')['label']
    headlines_test = pd.read_csv('processed/headlines_test.csv')['headline']

    print("Training Decision Tree...")
    model = DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    print("Predicting...")
    preds = model.predict(X_test)

    print("Saving results...")
    results = pd.DataFrame({
        'headline': headlines_test,
        'prediction': preds,
        'prediction_label': ['fake' if p == 1 else 'real' for p in preds]
    })
    results.to_csv('enhanced_predictions.csv', index=False)

    print("Top 10 important features:")
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False).head(10)
    print(importance_df)

    print("\nTraining set performance:")
    print(classification_report(y_train, model.predict(X_train)))

if __name__ == '__main__':
    main()
