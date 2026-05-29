# Smart Farm IoT Backend: AI Inference & Telemetry Engine

## Overview
This repository serves as the central intelligence and data routing backend for a Cyber-Physical Smart Farm System. It ingests live hardware telemetry from edge sensors, processes the data through real-time Machine Learning models, dispatches automated hardware control signals via MQTT, and securely logs all state variables to a cloud PostgreSQL database for live Grafana visualization.

## System Architecture & Features
* **Real-Time API Ingestion:** Flask-based REST endpoint (`/api/telemetry`) handling environmental metrics (NPK, Temperature, Humidity, pH, Moisture) and power states (Solar Battery Voltage).
* **Machine Learning Inference:**
  * **Random Forest Classifier:** Dynamically recommends optimal crops based on real-time soil parameters.
  * **Isolation Forest (Unsupervised):** Detects multi-variable environmental anomalies to flag system drift or sensor failure.
* **Actuator Control Loop:** Evaluates threshold logic and publishes automated commands (e.g., `PUMP_ON`, `FAN_ON`, `POWER_SAVE_MODE`) via MQTT to edge hardware.
* **Telemetry Data Pipeline:** Securely commits all telemetry, AI predictions, and state overrides to a PostgreSQL cloud database.

## Technology Stack
* **Core:** Python 3, Flask, Gunicorn
* **Machine Learning:** Scikit-Learn, NumPy, Pandas, Joblib
* **IoT Protocol:** Paho-MQTT
* **Database:** PostgreSQL (`psycopg2-binary`)
* **Cloud Infrastructure:** Render (Web Service), Render (PostgreSQL), Grafana Cloud

## Local Setup & Installation

1. **Clone the rsepository:**
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name