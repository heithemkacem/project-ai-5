from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def index(request):
    try:
        # Load data
        data = pd.read_csv("air.csv")
        data = data.dropna(subset=['AQI'])

        # Define features
        numeric_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
        target_feature = 'AQI'

        # Preprocessing and splitting
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        X = data[numeric_features]
        y = data[target_feature]

        # Normalize y
        y_normalized = y / y.max()

        X_prepared = preprocessor.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_prepared, y_normalized, test_size=0.2, random_state=42)

        # Models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        # Model evaluation
        results = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[model_name] = {
                'MSE': mse,
                'R_squared': r2
            }

        # Scatter plot for actual vs. predicted AQI
        best_model = models['Random Forest']
        y_pred = best_model.predict(X_test)

        scatter_img = io.BytesIO()
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        plt.title('Actual vs. Predicted AQI')
        plt.savefig(scatter_img, format='png')
        scatter_img.seek(0)
        scatter_base64 = base64.b64encode(scatter_img.read()).decode('utf-8')
        plt.close()

        # Heatmap for correlation matrix
        corr_matrix_img = io.BytesIO()
        plt.figure(figsize=(12, 10))
        corr_matrix = data[numeric_features + ['AQI']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.savefig(corr_matrix_img, format='png')
        corr_matrix_img.seek(0)
        corr_matrix_base64 = base64.b64encode(corr_matrix_img.read()).decode('utf-8')
        plt.close()

        # Histograms for features
        histogram_images = {}
        for feature in numeric_features:
            histogram_img = io.BytesIO()
            plt.figure(figsize=(8, 4))
            sns.histplot(data[feature].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            plt.savefig(histogram_img, format='png')
            histogram_img.seek(0)
            histogram_images[feature] = base64.b64encode(histogram_img.read()).decode('utf-8')
            plt.close()

        # Feature importances with Random Forest
        feature_importances = best_model.feature_importances_
        indices = np.argsort(feature_importances)[::-1]

        feature_importance_img = io.BytesIO()
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importances by Random Forest')
        plt.bar(range(len(indices)), feature_importances[indices], align='center')
        plt.xticks(range(len(indices)), [numeric_features[i] for i in indices], rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Relative Importance')
        plt.savefig(feature_importance_img, format='png')
        feature_importance_img.seek(0)
        feature_importance_base64 = base64.b64encode(feature_importance_img.read()).decode('utf-8')
        plt.close()

        context = {
            'results': results,
            'scatter_base64': scatter_base64,
            'corr_matrix_base64': corr_matrix_base64,
            'feature_importance_base64': feature_importance_base64,
            'histogram_images': histogram_images,
        }

    except Exception as e:
        context = {'error_message': f"An error occurred: {str(e)}"}

    return render(request, 'index.html', context)
