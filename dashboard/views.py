from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json

def index(request):
    try:
        # Load the dataset
        air_quality_data = pd.read_csv('updated_air_quality_data_with_AQI.csv')

        # Process the DateTime and set as index
        air_quality_data['Time'] = air_quality_data['Time'].str.replace('.', ':', regex=False)
        air_quality_data['DateTime'] = pd.to_datetime(air_quality_data['Date'] + ' ' + air_quality_data['Time'], format='%d/%m/%Y %H:%M:%S')
        air_quality_data.set_index('DateTime', inplace=True)
        air_quality_data.drop(['Date', 'Time'], axis=1, inplace=True)

        # Extract time features
        air_quality_data['Hour'] = air_quality_data.index.hour
        air_quality_data['Weekday'] = air_quality_data.index.weekday
        air_quality_data['Month'] = air_quality_data.index.month

        # Select relevant features and the target variable
        features = ['CO(GT)', 'NO2(GT)', 'Hour', 'Weekday']  # Example features
        X = air_quality_data[features]
        y = air_quality_data['Overall_AQI']  # Target variable

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42)
        }

        # Dictionary to hold model evaluation results
        model_results = {}

        # Train, predict, and evaluate models
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            r2 = r2_score(y_test, y_pred)

            # Store results in the dictionary
            model_results[name] = {'RMSE': rmse, 'R2': r2}

        # Get actual AQI and predicted AQI values
        actual_aqi = y_test.tolist()
        predicted_aqi = {model_name: y_pred.tolist() for model_name, y_pred in zip(models.keys(), [model.predict(X_test) for model in models.values()])}

        # Prepare data to send to template
        model_names = list(models.keys())
        rmse_values = [result['RMSE'] for result in model_results.values()]
        r2_values = [result['R2'] for result in model_results.values()]

        # Pass data to the template
        context = {
            'model_names': model_names,
            'rmse_values': rmse_values,
            'r2_values': r2_values,
            'actual_aqi': actual_aqi,
            'predicted_aqi': predicted_aqi
        }

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        context = {'error': error_message}

    return render(request, 'index.html', context)
