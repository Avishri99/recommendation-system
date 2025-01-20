from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_trend_model(features):
    """
    Train a model to predict trending movies.
    :param features: DataFrame with features
    :return: Trained model
    """
    # Define target
    features['is_trending'] = features['growth_rate'].apply(lambda x: 1 if x > 0.2 else 0)

    # Prepare training and testing data
    X = features[['avg_rating', 'num_ratings', 'growth_rate']]
    y = features['is_trending']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return model
