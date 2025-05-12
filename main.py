# main.py
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def main():
    print("Loading preprocessed data...")
    X_train = pd.read_csv('processed/X_train.csv')
    X_test = pd.read_csv('processed/X_test.csv')
    X_validation = pd.read_csv('processed/X_validation.csv')
    y_train = pd.read_csv('processed/y_train.csv')['label']
    y_validation = pd.read_csv('processed/y_validation.csv')['label']
    headlines_test = pd.read_csv('processed/headlines_test.csv')['headline']

    print("Training Decision Tree...")
    param_grid = {
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [5, 10, 15],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    }

    model = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='f1_macro',
        n_jobs=-1
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

    print("\nTraining set performance:")
    print(classification_report(y_validation, model.predict(X_validation)))

if __name__ == '__main__':
    main()
