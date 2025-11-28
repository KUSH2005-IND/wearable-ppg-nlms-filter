# 📡 **Wearable Motion-Artifact–Resistant PPG Monitoring**  
### *Using NLMS Adaptive Filtering + IMU-Assisted Noise Cancellation*

This repository contains a complete prototype demonstrating **real-time correction of motion artifacts in PPG signals** using an **NLMS adaptive filter** combined with **MPU6050 IMU data**.

It includes:

- ✔ **Adaptive correction algorithm (Python WebSocket server)**  
- ✔ **NLMS-based PPG noise cancellation**  
- ✔ **Motion score + artifact classifier (clean/light/heavy)**  
- ✔ **Realtime HTML/JS dashboard (Chart.js)**  
- ✔ **Synthetic heart-rate generation**  
- ✔ **ESP32 → Server → Dashboard pipeline**

---

# 🧠 **1. Problem Statement**

Wearable devices like smartwatches often suffer from the biggest limitation in PPG sensing:

## **Motion-Induced Artifacts**  
Movement causes:

- Vibrations  
- Pressure changes  
- Wrist rotation  
- Skin displacement  

This results in:

- ❌ False HR readings  
- ❌ Noisy PPG signals  
- ❌ Dropouts during workouts  
- ❌ Poor reliability  

### **Why it matters**  
PPG wearables are used for **continuous physiological monitoring**, but motion error makes them unreliable in real-world use.

---

# 💡 **2. Proposed Solution**

A fusion of **IMU motion signals + NLMS adaptive filtering** to dynamically cancel noise:

### ✔ **IMU as noise reference**  
Motion from accelerometer/gyro predicts noise contaminating the PPG.

### ✔ **NLMS learns noise characteristics**  
Unlike smoothing, NLMS *adapts* in real-time.

### ✔ Provides 3 outputs  
- `ppg_noisy` – noisy synthetic waveform  
- `ppg_clean` – reference/ideal waveform  
- `ppg_filtered` – NLMS reconstructed clean signal  

### ✔ Real-time dashboard  
Shows HR, labels, IMU trends, and three PPG tracings.

---

# ⚙️ **3. System Architecture**

ESP32 (MPU6050 IMU)
│
▼
Raw IMU stream over WebSocket
│
▼
Python Server (NLMS + Artifact Detection)
│
▼
Corrected JSON stream
│
▼
Web Dashboard


---

# 🧮 **4. How NLMS Adaptive Filter Works**

### **Normalised Least Mean Squares (NLMS)**  
Used to estimate and subtract motion-correlated noise.

### **Signal model**
d(t) = s(t) + n(t)
Where:  
- `d(t)` → noisy PPG  
- `s(t)` → clean PPG  
- `n(t)` → noise from motion  
- `x(t)` → IMU reference  

### **NLMS prediction**
y(t) = w(t)ᵀ x_vec(t)


### **Error (filtered output)**


e(t) = d(t) - y(t)


### **Weight update rule**


w(t+1) = w(t) + ( μ / ( ||x||² + ε ) ) * e(t) * x_vec(t)


---

# 📉 **Why NLMS is better than smoothing**

| Traditional smoothing | NLMS filtering |
|----------------------|----------------|
| Smooths everything | Learns actual noise |
| Removes peaks | Preserves waveform |
| Fails during rapid movement | Adapts instantly |
| Can't use IMU data | IMU-assisted |

---

![Realtime Dashboard](dashboard\dash.png)

# 📁 **5. Repository Structure**
📦 wearable-ppg-nlms-filter/
│
├── algorithm/
│ └── nlrms.py # NLMS Server (WebSocket)
│
├── dashboard/
│ ├── index.html # Realtime UI
│ ├── dashboard.js # Charts + WebSocket client
│ └── hr_log.csv # Example log
