from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import joblib

def index(request):
    try:
        # Load the data
        data = pd.read_csv("AQI.csv")
        data.dropna(axis=0, inplace=True)

        # Convert date to datetime
        data['Date'] = pd.to_datetime(data['Date'])

        # Preparing data for modeling
        feature_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
        X = data[feature_columns]
        y = data['AQI']

        # Splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Defining models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }

        # Training and evaluating models
        results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            results[name] = {'RMSE': rmse, 'R²': r2}

        # Save the best model
        best_model_name = min(results, key=lambda x: results[x]['RMSE'])
        best_model = models[best_model_name]
        joblib.dump(best_model, 'best_model.pkl')

        # Prepare data for the template
        context = {
            'plotly_line_json': px.line(data, x='Date', y='AQI', color='City', title='AQI Trend Over Time').to_json(),
            'plotly_box_json': px.box(data, x='City', y='AQI', title='AQI Distribution by City').to_json(),
            'scatter_matrix_json': px.scatter_matrix(data[['PM2.5', 'NO2', 'CO', 'O3', 'AQI']], title='Scatter Plot Matrix').to_json(),
            'model_names': list(results.keys()),
            'rmse_values': [results[model]['RMSE'] for model in results],
            'r2_values': [results[model]['R²'] for model in results],
        }

    except Exception as e:
        context = {
            'error_message': f"An error occurred: {str(e)}"
        }

    return render(request, 'index.html', context)
