import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from prometheus_client import start_http_server, Summary, Gauge
import time
import random

# Prometheus metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
MODEL_ACCURACY = Gauge('model_accuracy', 'Accuracy of the ML model')

# Load data
df = pd.read_csv('customer_data.csv')

# Data preprocessing
X = df[['age', 'income']]
y = df['purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initial model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save initial model
joblib.dump(model, 'models/model_v1.pkl')

# Monitor and retrain function
def monitor_and_retrain():
    for epoch in range(1, 6):
        print(f"\nMonitoring Epoch {epoch}...")
        
        # Load the latest model
        model = joblib.load('models/model_v1.pkl')
        
        # Simulate new data
        new_data = pd.DataFrame({
            'age': np.random.randint(18, 70, 200),
            'income': np.random.randint(20000, 120000, 200)
        })
        new_labels = np.random.choice([0, 1], 200)
        
        # Evaluate the model with new data
        new_pred = model.predict(new_data)
        new_accuracy = accuracy_score(new_labels, new_pred)
        print(f"Monitored Model Accuracy: {new_accuracy:.2f}")
        
        # Update Prometheus gauge
        MODEL_ACCURACY.set(new_accuracy)

        # Trigger retraining if accuracy falls below threshold
        if new_accuracy < 0.75:
            print("Accuracy below threshold. Retraining model...")
            
            # Combine old and new data for retraining
            combined_X = pd.concat([X_train, new_data])
            combined_y = pd.concat([y_train, pd.Series(new_labels)])
            
            # Retrain model
            model.fit(combined_X, combined_y)
            
            # Save the retrained model
            model_version = f'model_v{epoch + 1}.pkl'
            joblib.dump(model, f'models/{model_version}')
            print(f"Model retrained and saved as {model_version}")
        else:
            print("Model performing well. No retraining required.")
        
        # Simulate time passing
        time.sleep(15)

# Start Prometheus metrics server
if __name__ == '__main__':
    start_http_server(8000)
    monitor_and_retrain()
