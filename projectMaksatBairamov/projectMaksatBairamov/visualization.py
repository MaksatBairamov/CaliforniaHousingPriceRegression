import matplotlib.pyplot as plt
import seaborn as sns

def visualize_results(models, X, y, feature_names):
    # Scatter plots for actual vs. predicted house prices for each model
    plt.figure(figsize=(10, 6))
    for name, model in models.items():
        y_pred = model.predict(X)
        sns.scatterplot(x=y, y=y_pred, label=name, alpha=0.7)
    plt.xlabel('Actual House Prices')
    plt.ylabel('Predicted House Prices')
    plt.title('Actual vs. Predicted House Prices')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Feature importance plot for the best-performing model (XGBoost)
    plt.figure(figsize=(10, 6))
    best_xgb_model = models['XGBoost']
    feature_importance = best_xgb_model.feature_importances_
    sorted_idx = feature_importance.argsort()
    sns.barplot(x=feature_importance[sorted_idx], y=feature_names[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance - XGBoost')
    plt.grid(True)
    plt.show()