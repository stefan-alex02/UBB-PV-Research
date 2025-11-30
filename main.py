import numpy as np
import pandas as pd

from train_eval import run

csv_file = "solar_data.csv"

try:
    df = pd.read_csv(csv_file)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Dataset not found. Creating dummy data for demonstration...")
    # Create a dummy dataset resembling solar data
    dates = pd.date_range(start='1/1/2020', periods=2000, freq='h')
    df = pd.DataFrame({
        'DATE_TIME': dates,
        'Irradiance': np.abs(np.sin(np.linspace(0, 100, 2000)) * 1000 + np.random.normal(0, 50, 2000)),
        'Temperature': 25 + np.random.normal(0, 5, 2000),
        'Humidity': 50 + np.random.normal(0, 10, 2000),
        'AC_POWER': np.abs(np.sin(np.linspace(0, 100, 2000)) * 50 + np.random.normal(0, 2, 2000))
    })

# Choose model: 'ANN', 'XGBoost', 'RF', 'SVM'
run(df, model_name='ANN', k_folds=5)
