from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Use non-GUI backend for matplotlib
import matplotlib
matplotlib.use('Agg')

def index(request):
    try:
        # Load and process the data
        air_quality = pd.read_csv('Air_Quality_Monitoring_Data_20240420.csv')
        air_quality.drop(['Date', 'Time', 'GPS'], axis=1, inplace=True)
        air_quality = air_quality.dropna(subset=['AQI_Site'])

        for column in air_quality.columns:
            if air_quality[column].isnull().any():
                median_value = air_quality[column].median()
                # Avoid inplace to prevent chained assignment
                air_quality[column] = air_quality[column].fillna(median_value)

        air_quality['DateTime'] = pd.to_datetime(
            air_quality['DateTime'], format='%d/%m/%Y %I:%M:%S %p'
        )
        air_quality['Year'] = air_quality['DateTime'].dt.year
        air_quality['Month'] = air_quality['DateTime'].dt.month
        air_quality['Day'] = air_quality['DateTime'].dt.day
        air_quality['Hour'] = air_quality['DateTime'].dt.hour
        air_quality['Weekday'] = air_quality['DateTime'].dt.weekday

        # Create heatmap for correlation matrix
        numeric_cols = air_quality.select_dtypes(include=[np.number])
        correlation_matrix = numeric_cols.corr()

        # Ensure no GUI backend warnings
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        heatmap_img = io.BytesIO()
        plt.savefig(heatmap_img, format='png')
        heatmap_img.seek(0)
        heatmap_base64 = base64.b64encode(heatmap_img.read()).decode('utf-8')
        plt.close()

        # Create scatter plot for AQI Site vs. PM2.5
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=air_quality, x='PM2.5', y='AQI_Site')
        scatter_img = io.BytesIO()
        plt.savefig(scatter_img, format='png')
        scatter_img.seek(0)
        scatter_base64 = base64.b64encode(scatter_img.read()).decode('utf-8')
        plt.close()

        # Define features and target variable
        features = air_quality.drop(['AQI_Site', 'DateTime', 'Name'], axis=1)
        target = air_quality['AQI_Site']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Initialize and train the model
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)

        # Predict on the test set
        y_pred = rf.predict(X_test_scaled)

        # Calculate the MSE and R² values
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Create scatter plot for actual vs. predicted AQI
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title('Actual vs Predicted AQI')
        plt.xlabel('Actual AQI')
        plt.ylabel('Predicted AQI')
        scatter_actual_predicted_img = io.BytesIO()
        plt.savefig(scatter_actual_predicted_img, format='png')
        scatter_actual_predicted_img.seek(0)
        scatter_actual_predicted_base64 = base64.b64encode(scatter_actual_predicted_img.read()).decode('utf-8')
        plt.close()

        # Create the context to pass to the template
        context = {
            'heatmap_base64': heatmap_base64,
            'scatter_base64': scatter_base64,
            'scatter_actual_predicted_base64': scatter_actual_predicted_base64,
            'mse': mse,
            'r2': r2
        }

    except Exception as e:
        context = {'error_message': f"An error occurred: {str(e)}"}

    return render(request, 'index.html', context)
