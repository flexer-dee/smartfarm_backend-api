from flask import Flask, request, jsonify
import joblib
import numpy as np
import paho.mqtt.client as mqtt
import json
import os
import psycopg2
from dotenv import load_dotenv

# Load local environment variables from .env file (if running locally)
load_dotenv()

app = Flask(__name__)

# Config & Setup
DB_URL = os.environ.get('DATABASE_URL')
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_ALERT_TOPIC = "smartfarm/greenhouse/alerts"
MQTT_ACTUATOR_TOPIC = "smartfarm/greenhouse/actuators"

# Load Models Once
try:
    crop_model = joblib.load('random_forest_crop_model.pkl')
    anomaly_detector = joblib.load('isolation_forest_model.pkl')
except Exception as e:
    print(f"⚠️ Error loading ML models: {e}")

# Initialize MQTT
try:
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    print("✓ Successfully connected to MQTT Broker.")
except Exception as e:
    print(f"⚠️ MQTT Connection Warning: {e}")

# ==========================================
# HELPER FUNCTIONS (Separation of Concerns)
# ==========================================

def run_ai_inference(metrics):
    """Handles all Machine Learning predictions."""
    features = np.array([metrics])
    predicted_crop = crop_model.predict(features)[0]
    
    env_features = np.array([metrics[3:7]])  # Temp, Hum, pH, Moisture
    anomaly_score = anomaly_detector.predict(env_features)[0]
    is_anomaly = True if anomaly_score == -1 else False
    
    return predicted_crop, is_anomaly

def determine_automation_action(metrics, battery_volt, is_anomaly):
    """Evaluates environmental logic and hardware safety constraints."""
    action_triggered = "NONE"
    alert_msg = ""
    
    if metrics[6] < 40.0:  
        action_triggered = "PUMP_ON"
        alert_msg = "Critical: Soil moisture low. Activating irrigation."
    elif metrics[3] > 35.0:  
        action_triggered = "FAN_ON"
        alert_msg = "Warning: High thermal stress. Activating ventilation."
    elif battery_volt < 11.1:  
        action_triggered = "POWER_SAVE_MODE"
        alert_msg = "Critical Danger: Solar battery voltage low. Scaling back loads."
    elif is_anomaly:
        alert_msg = "AI Insight: Unusual multi-variable environmental behavior detected."
        
    return action_triggered, alert_msg

def dispatch_mqtt_commands(action_triggered, alert_msg):
    """Publishes control signals to the hardware network."""
    if alert_msg:
        mqtt_client.publish(MQTT_ALERT_TOPIC, json.dumps({"alert": alert_msg}))
    if action_triggered != "NONE":
        mqtt_client.publish(MQTT_ACTUATOR_TOPIC, json.dumps({"command": action_triggered}))

def log_telemetry_to_db(metrics, battery_volt, predicted_crop, is_anomaly, action_triggered):
    """Handles secure insertion of state data into PostgreSQL."""
    if not DB_URL:
        print("⚠️ Database Warning: DATABASE_URL variable not set.")
        return
        
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO greenhouse_telemetry 
            (nitrogen, phosphorus, potassium, temperature, humidity, ph, soil_moisture, battery_voltage, predicted_crop, anomaly_detected, automation_command)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], metrics[6], battery_volt, predicted_crop, is_anomaly, action_triggered))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as db_err:
        print(f"⚠️ Database Logging Error: {db_err}")

# ==========================================
# MAIN API ROUTES
# ==========================================

@app.route('/api/telemetry', methods=['POST'])
def process_telemetry():
    """Main webhook for incoming hardware telemetry."""
    try:
        # 1. Parse incoming JSON payload
        data = request.get_json()
        metrics = [
            float(data['N']), float(data['P']), float(data['K']),
            float(data['temp']), float(data['hum']), float(data['ph']), float(data['moisture'])
        ]
        battery_volt = float(data.get('battery_volt', 12.0))
        
        # 2. Run AI Models
        predicted_crop, is_anomaly = run_ai_inference(metrics)
        
        # 3. Determine Control Actions
        action_triggered, alert_msg = determine_automation_action(metrics, battery_volt, is_anomaly)
        
        # 4. Dispatch Hardware Signals
        dispatch_mqtt_commands(action_triggered, alert_msg)
            
        # 5. Log Everything to Cloud Database
        log_telemetry_to_db(metrics, battery_volt, predicted_crop, is_anomaly, action_triggered)
            
        # 6. Return standard JSON response
        return jsonify({
            "status": "processed",
            "ai_crop_recommendation": predicted_crop,
            "ai_anomaly_detected": is_anomaly,
            "automation_command": action_triggered,
            "system_alert": alert_msg if alert_msg else "System stable"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "message": "Smart Farm API is running smoothly"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)