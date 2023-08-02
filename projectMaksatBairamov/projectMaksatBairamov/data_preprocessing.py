from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_dataset():
    # Load the California Housing dataset
    data = fetch_california_housing()

    # Extract the feature data (X) and target data (y)
    X = data.data
    y = data.target
    feature_names = data.feature_names

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, feature_names