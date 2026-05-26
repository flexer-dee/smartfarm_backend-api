from flask import Flask, request, jsonify
import joblib
import numpy as np
import paho.mqtt.client as mqtt
import json

app = Flask(__name__)

# 1. Load Pre-Trained AI Models
crop_model = joblib.load('random_forest_crop_model.pkl')
anomaly_detector = joblib.load('isolation_forest_model.pkl')

# 2. MQTT Configuration (Using a public broker for rapid sprint deployment)
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_ALERT_TOPIC = "smartfarm/greenhouse/alerts"
MQTT_ACTUATOR_TOPIC = "smartfarm/greenhouse/actuators"

# Initialize MQTT Client
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
        
        # Parse inputs from incoming payload
        metrics = [
            float(data['N']), float(data['P']), float(data['K']),
            float(data['temp']), float(data['hum']), float(data['ph']), float(data['moisture'])
        ]
        battery_volt = float(data.get('battery_volt', 12.0))
        
        # 3. AI Inference Decisions
        features = np.array([metrics])
        predicted_crop = crop_model.predict(features)[0]
        
        env_features = np.array([metrics[3:7]])  # Temp, Hum, pH, Moisture
        anomaly_score = anomaly_detector.predict(env_features)[0]
        ai_anomaly = True if anomaly_score == -1 else False
        
        # 4. Hybrid Automation & Threshold Decision Logic
        action_triggered = "NONE"
        alert_msg = ""
        
        # Rule-based overrides for immediate physical protection
        if metrics[6] < 40.0:  
            action_triggered = "PUMP_ON"
            alert_msg = "Critical: Soil moisture low. Activating irrigation."
        elif metrics[3] > 35.0:  
            action_triggered = "FAN_ON"
            alert_msg = "Warning: High thermal stress. Activating ventilation."
        elif battery_volt < 11.1:  
            action_triggered = "POWER_SAVE_MODE"
            alert_msg = "Critical Danger: Solar battery voltage low. Scaling back non-essential loads."
        elif ai_anomaly:
            alert_msg = "AI Insight: Unusual multi-variable environmental behavior detected."

        # 5. Dispatch MQTT Messages (Hardware-Software Control Loop)
        if alert_msg:
            mqtt_client.publish(MQTT_ALERT_TOPIC, json.dumps({"alert": alert_msg}))
        if action_triggered != "NONE":
            mqtt_client.publish(MQTT_ACTUATOR_TOPIC, json.dumps({"command": action_triggered}))
            
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