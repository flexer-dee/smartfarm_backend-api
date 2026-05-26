import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import joblib

# 1. Generate Synthetic Crop Data (N, P, K, Temp, Hum, pH, Moisture)
np.random.seed(42)
num_samples = 300

# Templates: [N, P, K, Temp, Hum, pH, Moisture]
tomato_ideal  = np.random.normal([90, 40, 40, 24, 65, 6.2, 70], [5, 3, 3, 2, 5, 0.2, 5], (num_samples, 7))
spinach_ideal = np.random.normal([70, 50, 30, 18, 70, 6.8, 80], [5, 3, 3, 2, 5, 0.2, 5], (num_samples, 7))
maize_ideal   = np.random.normal([100, 50, 20, 28, 55, 6.5, 60], [5, 3, 3, 2, 5, 0.2, 5], (num_samples, 7))

X = np.vstack([tomato_ideal, spinach_ideal, maize_ideal])
y = ['Tomato']*num_samples + ['Spinach']*num_samples + ['Maize']*num_samples

# Train Random Forest Classifier
crop_model = RandomForestClassifier(n_estimators=50, random_state=42)
crop_model.fit(X, y)
joblib.dump(crop_model, 'random_forest_crop_model.pkl')
print("✓ Random Forest Crop Model Saved.")

# 2. Train Isolation Forest Anomaly Detector
# Feed it normal environmental conditions so it learns what 'stable' looks like
env_data = X[:, 3:7] # Extract Temp, Hum, pH, Moisture
anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
anomaly_detector.fit(env_data)
joblib.dump(anomaly_detector, 'isolation_forest_model.pkl')
print("✓ Isolation Forest Anomaly Detector Saved.")