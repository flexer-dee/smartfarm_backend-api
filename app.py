from flask import Flask, request, jsonify
import joblib
import numpy as np
import paho.mqtt.client as mqtt
import json
import os       # For reading hidden connection strings
import psycopg2 # For database interactions

app = Flask(__name__)

# Fetch database URL from Render environment variables safely
DB_URL = os.environ.get('DATABASE_URL')

# Load Pre-Trained AI Models
crop_model = joblib.load('random_forest_crop_model.pkl')
anomaly_detector = joblib.load('isolation_forest_model.pkl')

# MQTT Configuration
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_ALERT_TOPIC = "smartfarm/greenhouse/alerts"
MQTT_ACTUATOR_TOPIC = "smartfarm/greenhouse/actuators"

try:
    mqtt_client = mqtt.Client()
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    print("✓ Successfully connected to MQTT Broker.")
except Exception as e:
    print(f"⚠️ MQTT Connection Warning: {e}")

@app.route('/api/telemetry', methods=['POST'])
def process_telemetry():
    try:
        data = request.get_json()
        
        # Parse incoming payload
        metrics = [
            float(data['N']), float(data['P']), float(data['K']),
            float(data['temp']), float(data['hum']), float(data['ph']), float(data['moisture'])
        ]
        battery_volt = float(data.get('battery_volt', 12.0))
        
        # 1. AI Inference Decisions
        features = np.array([metrics])
        predicted_crop = crop_model.predict(features)[0]
        
        env_features = np.array([metrics[3:7]])  # Temp, Hum, pH, Moisture
        anomaly_score = anomaly_detector.predict(env_features)[0]
        ai_anomaly = True if anomaly_score == -1 else False
        
        # 2. Hybrid Automation Logic & Threshold Overrides
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
        elif ai_anomaly:
            alert_msg = "AI Insight: Unusual multi-variable environmental behavior detected."

        # 3. Dispatch MQTT Control Messages
        if alert_msg:
            mqtt_client.publish(MQTT_ALERT_TOPIC, json.dumps({"alert": alert_msg}))
        if action_triggered != "NONE":
            mqtt_client.publish(MQTT_ACTUATOR_TOPIC, json.dumps({"command": action_triggered}))
            
        # 4. NEW: Log Telemetry & AI Decisions Directly to PostgreSQL
        if DB_URL:
            try:
                conn = psycopg2.connect(DB_URL)
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO greenhouse_telemetry 
                    (nitrogen, phosphorus, potassium, temperature, humidity, ph, soil_moisture, battery_voltage, predicted_crop, anomaly_detected, automation_command)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], metrics[6], battery_volt, predicted_crop, ai_anomaly, action_triggered))
                conn.commit()
                cur.close()
                conn.close()
            except Exception as db_err:
                print(f"⚠️ Database Logging Error: {db_err}")
        else:
            print("⚠️ Database Warning: DATABASE_URL variable not set.")
            
        return jsonify({
            "status": "processed",
            "ai_crop_recommendation": predicted_crop,
            "ai_anomaly_detected": ai_anomaly,
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