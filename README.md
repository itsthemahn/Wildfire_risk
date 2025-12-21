[readme.md](https://github.com/user-attachments/files/24279693/readme.md)
# 🔥 Wildfire MLOps Backend System  
## Full Execution Guide

> ⚠️ This README intentionally documents **everything required to make the system work**, even if some steps are not ideal for production.  
> **Follow the order exactly. Skipping steps will break drift detection and retraining.**

---

## 🧠 Critical Execution Flow (READ FIRST)

**Data drift detection will NOT work unless ALL of the following are true:**

1. FastAPI inference server is running  
2. Producer is running and sending requests  
3. Predictions are being logged  
4. `current.parquet` is generated from live traffic  

❌ If **any** step is skipped, **drift will be empty** and retraining will never trigger.

---

## 📁 Project Structure (Relevant Parts)

```
Wildfire_MLops/
├── inference/
│   └── app.py                  # FastAPI inference server
├── streaming/
│   ├── producer.py             # Sends requests to FastAPI
│   └── generator.py            # Generates synthetic data
├── monitoring/
│   ├── reference/reference.parquet
│   ├── current/current.parquet     # GENERATED FROM LIVE TRAFFIC
│   └── drift/status.json
├── prefect/
│   └── flows/retrain_flow.py
├── k8s/
│   ├── namespace.yaml
│   ├── deployment.yaml
│   └── service.yaml
└── train.py
```

---

## ⚙️ Prerequisites

- Python **3.10+**
- Docker Desktop (**Kubernetes ENABLED**)
- `kubectl` (bundled with Docker Desktop)
- Git

❌ **Do NOT install Minikube**  
❌ **Do NOT install Kind**

✅ Kubernetes is provided by **Docker Desktop ONLY**

---

## 🐍 1. Python Environment (Optional but Recommended)

```bash
python -m venv venv
```

### Activate virtual environment (Windows)

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements.inference.txt
```

---

## 🧠 2. Initial Model Training

```bash
python train.py
```

This generates:

```
artifacts/
models/
```

---

## 🚢 3. Build Inference Docker Image

```bash
docker build -t wildfire-inference:latest .
```

Verify:

```bash
docker images
```

---

## ☸️ 4. Kubernetes Deployment (Docker Desktop)

### 4.1 Verify Kubernetes Context

```bash
kubectl config current-context
```

Expected output:

```
docker-desktop
```

---

### 4.2 Apply Namespace

```bash
kubectl apply -f k8s/namespace.yaml
```

Verify:

```bash
kubectl get namespaces
```

---

### 4.3 Deploy Inference Pod

```bash
kubectl apply -f k8s/deployment.yaml
```

Verify:

```bash
kubectl get pods -n wildfire
```

---

### 4.4 Create Service

```bash
kubectl apply -f k8s/service.yaml
```

Verify:

```bash
kubectl get svc -n wildfire
```

---

### 4.5 Port Forward Inference API

```bash
kubectl port-forward svc/wildfire-inference 8000:8000 -n wildfire
```

Swagger UI:

```
http://localhost:8000/docs
```

---

## 🔥 5. Start FastAPI (MANDATORY FOR DRIFT)

⚠️ **Inference MUST be running before starting the producer**

FastAPI endpoint:

```
http://localhost:8000/predict
```

✅ **Leave this terminal OPEN**

---

## 📡 6. Start Producer (MANDATORY FOR DRIFT)

Open a **new terminal**:

```bash
python streaming/producer.py
```

### What the producer does:

- Sends live prediction requests  
- Generates current data  
- Logs predictions  
- Creates `current.parquet`  

⚠️ **If producer is NOT running → `current.parquet` will NOT exist**

---

## 📊 7. Verify `current.parquet` Creation

After **1–2 minutes**, confirm:

```
monitoring/current/current.parquet
```

If this file does not exist:

- Drift detection will be empty  
- Retraining will never trigger  

---

## 🔍 8. Run Drift Detection

```bash
python -m monitoring.drift_runner
```

This generates:

```
monitoring/drift/status.json
```

Example:

```json
{
  "dataset_drift": true,
  "drift_score": 0.92,
  "trigger_retrain": true
}
```

---

## 🔁 9. Prefect Retraining Pipeline

### Drift-based retraining

```bash
python prefect/flows/retrain_flow.py
```

### Manual retraining (override drift)

```bash
python prefect/flows/retrain_flow.py --force
```

### What the pipeline does:

1. Reads `status.json`  
2. Retrains selected models  
3. Registers model in MLflow  
4. Versions data & artifacts with DVC  

---

## 🛑 Common Failure Points

| Issue | Cause |
|------|------|
| Drift empty | Producer not running |
| `current.parquet` missing | FastAPI not running |
| Prefect stuck | Windows + ephemeral server |
| No retrain | `trigger_retrain=false` |

---

## ✅ Full Required Command Sequence

```bash
python train.py
docker build -t wildfire-inference:latest .
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl port-forward svc/wildfire-inference 8000:8000 -n wildfire
python streaming/producer.py
python -m monitoring.drift_runner
python prefect/flows/retrain_flow.py
```

---

## 📌 Final Notes

- This setup is **explicit by design**
- Some steps are **not best practice**, but are required
- This README enables **zero-guesswork reproduct
