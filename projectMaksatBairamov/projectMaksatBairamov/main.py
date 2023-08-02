from data_preprocessing import load_dataset
from model_training import train_models
from visualization import visualize_results

# Load and preprocess the dataset
X_train, X_test, y_train, y_test, feature_names = load_dataset()

# Train models
models = train_models(X_train, y_train)

# Visualize results
visualize_results(models, X_test, y_test, feature_names)