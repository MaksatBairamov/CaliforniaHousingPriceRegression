from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def train_models(X_train, y_train):
    models = {}

    # Train Linear Regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    models['Linear Regression'] = linear_model

    # Train Random Forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model

    # Train XGBoost model
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model

    # Train LASSO model
    lasso_model = Lasso(alpha=0.1, random_state=42)
    lasso_model.fit(X_train, y_train)
    models['LASSO'] = lasso_model

    return models