# server_broadcast.py
import asyncio
import websockets
import json
import csv
import time
import math
import random
from collections import deque

CSV_FILE = "hr_log.csv"

def classify_artifact(acc_change, gyro_change, gyro_mag):
    if acc_change < 0.006 and gyro_change < 0.20 and gyro_mag < 0.55:
        return "clean"
    if acc_change > 0.025 or gyro_change > 1.00 or gyro_mag > 1.80:
        return "artifact_heavy"
    return "artifact_light"

class NLMSFilter:
    def __init__(self, order=16, mu=0.5, eps=1e-6):
        self.order = order
        self.mu = mu
        self.eps = eps
        self.w = [0.0] * order
        self.x = deque([0.0]*order, maxlen=order)

    def adapt(self, d, ref):
        self.x.append(ref)
        x_vec = list(reversed(self.x))
        y = sum(wi * xi for wi, xi in zip(self.w, x_vec))
        e = d - y
        energy = sum(xi*xi for xi in x_vec) + self.eps
        coeff = self.mu / energy
        for i in range(self.order):
            self.w[i] += coeff * e * x_vec[i]
        return y, e

    def filter(self, d, ref):
        y, e = self.adapt(d, ref)
        return d - y


with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp_ms",
        "ax","ay","az",
        "acc_mag","acc_change",
        "gx","gy","gz",
        "gyro_mag","gyro_change",
        "hr_synth",
        "ppg_noisy","ppg_clean","ppg_filtered",
        "label","motion_score"
    ])

prev_acc_mag = None
prev_gyro_mag = None

# Maintain a global set of connected websockets
CONNECTED = set()

# Broadcast helper 
async def broadcast_json(obj):
    if not CONNECTED:
        return
    data = json.dumps(obj)
    to_remove = set()
    for ws in CONNECTED:
        try:
            await ws.send(data)
        except Exception:
            # mark for removal
            to_remove.add(ws)
    for ws in to_remove:
        try:
            CONNECTED.remove(ws)
            await ws.close()
        except Exception:
            pass

# Websocket handler
async def handler(websocket):
    global prev_acc_mag, prev_gyro_mag
    print("Client connected:", websocket.remote_address)
    CONNECTED.add(websocket)

    # Each connection gets its own filter & state
    nlms = NLMSFilter(order=12, mu=0.8, eps=1e-6)
    phase = 0.0
    last_t = time.time()

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except Exception:
                continue

            ax = float(data.get("ax", 0.0))
            ay = float(data.get("ay", 0.0))
            az = float(data.get("az", 1.0))

            gx = float(data.get("gx", 0.0))
            gy = float(data.get("gy", 0.0))
            gz = float(data.get("gz", 0.0))

            # Magnitudes
            acc_mag = math.sqrt(ax*ax + ay*ay + az*az)
            gyro_mag = math.sqrt(gx*gx + gy*gy + gz*gz)

            acc_change = 0.0 if prev_acc_mag is None else abs(acc_mag - prev_acc_mag)
            gyro_change = 0.0 if prev_gyro_mag is None else abs(gyro_mag - prev_gyro_mag)

            prev_acc_mag = acc_mag
            prev_gyro_mag = gyro_mag

            motion_score = acc_change * 100.0 + gyro_change * 10.0 + gyro_mag * 5.0
            label = classify_artifact(acc_change, gyro_change, gyro_mag)

            hr_jitter = random.gauss(0.0, 0.6)
            hr_synth = max(40.0, min(140.0, 72.0 + hr_jitter))
            now = time.time()
            dt = max(1.0/100.0, now - last_t)
            last_t = now
            f = hr_synth / 60.0
            phase += 2 * math.pi * f * dt
            if phase > 2*math.pi:
                phase %= (2*math.pi)
            ppg_clean = 0.5 * (1.0 + math.sin(phase)) + 0.05 * math.sin(2*phase)
            motion_amp = min(1.0, (acc_change * 400.0) + (gyro_mag * 1.2))
            if label == "artifact_heavy":
                motion_amp *= 2.5
            elif label == "artifact_light":
                motion_amp *= 1.0
            else:
                motion_amp *= 0.35
            gauss_noise = random.gauss(0.0, 0.08) * motion_amp
            acc_ref = (ax + ay + az) * 0.5
            acc_noise_component = acc_ref * 0.25 * motion_amp
            ppg_noisy = ppg_clean + gauss_noise + acc_noise_component

            # NLMS
            ref = acc_mag - 1.0
            ppg_filtered = nlms.filter(ppg_noisy, ref)

            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    int(time.time()*1000),
                    ax, ay, az,
                    acc_mag, acc_change,
                    gx, gy, gz,
                    gyro_mag, gyro_change,
                    round(hr_synth,2),
                    ppg_noisy, ppg_clean, ppg_filtered,
                    label, round(motion_score,4)
                ])

            out = {
                "timestamp_ms": int(time.time()*1000),
                "ax": ax, "ay": ay, "az": az,
                "acc_mag": acc_mag, "acc_change": acc_change,
                "gx": gx, "gy": gy, "gz": gz,
                "gyro_mag": gyro_mag, "gyro_change": gyro_change,
                "hr_synth": round(hr_synth,2),
                "ppg_noisy": ppg_noisy,
                "ppg_clean": ppg_clean,
                "ppg_filtered": ppg_filtered,
                "label": label,
                "motion_score": round(motion_score,4)
            }

            await broadcast_json(out)

    except websockets.ConnectionClosed:
        pass
    finally:
        try:
            CONNECTED.remove(websocket)
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        print("Client disconnected:", websocket.remote_address)


async def main():
   
    print("Artifact Detection + NLMS Server")
    print(" WebSocket   : ws://0.0.0.0:8765")
    print(" CSV Logging : {}".format(CSV_FILE))
    async with websockets.serve(handler, "0.0.0.0", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped by user")




# server.py  -- Subsystem D (integrated)
# - Receives IMU JSON from ESP32 over WebSocket
# - Generates synthetic PPG (if no real PPG)
# - Detects artifact (RF if available, else rule-based)
# - Applies NLMS adaptive filtering when artifact detected
# - Fallback: hold last clean for long/heavy artifacts
# - Broadcasts packet: t, accel, ppg_noisy, ppg_clean, artifact_label, hr_est


# server.py  -- Subsystem D (rule-based + synthetic PPG + NLMS)
# Usage: python server.py
#
# Outputs CSV: /mnt/data/mpu_log.csv (header below)
# Fields: timestamp_ms, ax,ay,az,acc_mag,acc_change,gx,gy,gz,gyro_mag,gyro_change,
#         hr_synth, ppg_noisy, ppg_clean, ppg_filtered, label, motion_score

# import asyncio
# import websockets
# import json
# import csv
# import time
# import math
# import random
# from collections import deque

# CSV_FILE = "hr_log.csv"   # matches uploaded file path in your workspace

# # -------------------------
# # Artifact classifier (tuned to show clean/light/heavy)
# # -------------------------
# def classify_artifact(acc_change, gyro_change, gyro_mag):

#     # CLEAN: mild motion allowed
#     if acc_change < 0.006 and gyro_change < 0.20 and gyro_mag < 0.55:
#         return "clean"

#     # HEAVY: only extreme motion
#     if acc_change > 0.025 or gyro_change > 1.00 or gyro_mag > 1.80:
#         return "artifact_heavy"

#     # Otherwise: moderate motion
#     return "artifact_light"

# # -------------------------
# # NLMS adaptive filter implementation (per-connection state)
# # -------------------------
# class NLMSFilter:
#     def __init__(self, order=16, mu=0.5, eps=1e-6):
#         self.order = order
#         self.mu = mu
#         self.eps = eps
#         self.w = [0.0] * order
#         self.x = deque([0.0]*order, maxlen=order)  # reference buffer (newest appended to right)

#     def adapt(self, d, ref):
#         # push newest ref
#         self.x.append(ref)
#         # create x vector with newest first
#         x_vec = list(reversed(self.x))  # x[0] is newest
#         # compute y = w^T x
#         y = sum(wi * xi for wi, xi in zip(self.w, x_vec))
#         e = d - y
#         # energy
#         energy = sum(xi*xi for xi in x_vec) + self.eps
#         # NLMS update
#         coeff = self.mu / energy
#         for i in range(self.order):
#             self.w[i] += coeff * e * x_vec[i]
#         return y, e

#     def filter(self, d, ref):
#         # returns (y, e, y_filtered_estimate) ; but we will use e as filtered output estimate = d - y
#         y, e = self.adapt(d, ref)
#         # filtered (i.e., ideally d - estimated_noise)
#         filtered = d - y
#         return filtered

# # -------------------------
# # Initialize CSV Logging (drop temp column, include synthetic HR & PPG fields)
# # -------------------------
# with open(CSV_FILE, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow([
#         "timestamp_ms",
#         "ax","ay","az",
#         "acc_mag","acc_change",
#         "gx","gy","gz",
#         "gyro_mag","gyro_change",
#         "hr_synth",
#         "ppg_noisy","ppg_clean","ppg_filtered",
#         "label","motion_score"
#     ])

# # per-server previous mags for delta
# prev_acc_mag = None
# prev_gyro_mag = None

# # server will maintain NLMS filter per websocket connection
# # synthetic PPG parameters
# BASE_HR = 72.0   # base heart rate (bpm)
# PPG_FS = 100.0   # synthetic sample rate (Hz) - matches incoming IMU tick rate roughly
# T0 = time.time()

# # =========================================================
# # WEBSOCKETS 13.1 handler -> single argument
# # =========================================================
# async def handler(websocket):
#     global prev_acc_mag, prev_gyro_mag

#     # per-connection NLMS filter and local state
#     nlms = NLMSFilter(order=12, mu=0.8, eps=1e-6)
#     # phase accumulator for synthetic clean PPG sine
#     phase = 0.0
#     # time of last sample (for phase increment)
#     last_t = time.time()
#     print("Client connected")

#     async for message in websocket:
#         try:
#             data = json.loads(message)
#         except Exception as e:
#             # ignore malformed JSON
#             continue

#         # Required IMU values
#         ax = float(data.get("ax", 0.0))
#         ay = float(data.get("ay", 0.0))
#         az = float(data.get("az", 1.0))   # default gravity on z

#         gx = float(data.get("gx", 0.0))
#         gy = float(data.get("gy", 0.0))
#         gz = float(data.get("gz", 0.0))

#         # Magnitude computations
#         acc_mag = math.sqrt(ax*ax + ay*ay + az*az)
#         gyro_mag = math.sqrt(gx*gx + gy*gy + gz*gz)

#         # Delta changes
#         acc_change = 0.0 if prev_acc_mag is None else abs(acc_mag - prev_acc_mag)
#         gyro_change = 0.0 if prev_gyro_mag is None else abs(gyro_mag - prev_gyro_mag)

#         prev_acc_mag = acc_mag
#         prev_gyro_mag = gyro_mag

#         # Motion score (simple combination — useful for dashboard)
#         motion_score = acc_change * 100.0 + gyro_change * 10.0 + gyro_mag * 5.0

#         # Classify artifact level (rule-based)
#         label = classify_artifact(acc_change, gyro_change, gyro_mag)

#         # ---------------------
#         # Synthetic PPG generation (clean sine + motion-dependent noise)
#         # ---------------------
#         # HR can slightly jitter around base
#         hr_jitter = random.gauss(0.0, 0.6)  # small gaussian jitter (bpm)
#         hr_synth = max(40.0, min(140.0, BASE_HR + hr_jitter))

#         # time step
#         now = time.time()
#         dt = max(1.0/PPG_FS, now - last_t)
#         last_t = now

#         # phase increment (HR in bpm -> Hz = bpm/60)
#         f = hr_synth / 60.0
#         phase += 2 * math.pi * f * dt
#         # keep phase bounded
#         if phase > 2*math.pi:
#             phase %= (2*math.pi)

#         # clean PPG: simple sine wave with slight DC offset and soft nonlinearity
#         ppg_clean = 0.5 * (1.0 + math.sin(phase))  # 0..1
#         # apply softpulse shape (adds slight harmonic)
#         ppg_clean = ppg_clean + 0.05 * math.sin(2*phase)

#         # motion-dependent noise amplitude (use acc_change and gyro_mag)
#         motion_amp = min(1.0, (acc_change * 400.0) + (gyro_mag * 1.2))
#         # if artifact_heavy, increase noise
#         if label == "artifact_heavy":
#             motion_amp *= 2.5
#         elif label == "artifact_light":
#             motion_amp *= 1.0
#         else:
#             motion_amp *= 0.35

#         # noise is mixture of gaussian + accelerometer-correlated component
#         gauss_noise = random.gauss(0.0, 0.08) * motion_amp
#         # accelerometer-derived correlated noise (use ax/ay/az sum)
#         acc_ref = (ax + ay + az) * 0.5
#         acc_noise_component = acc_ref * 0.25 * motion_amp

#         ppg_noisy = ppg_clean + gauss_noise + acc_noise_component

#         # ---------------------
#         # NLMS filtering step
#         # Use acc_mag as reference signal (recent samples) — this is simplistic but works for demo
#         # ---------------------
#         # feed NLMS with reference = acc_mag (normalized-ish)
#         # scale reference to similar amplitude as noise
#         ref = acc_mag - 1.0  # remove gravity DC so small motions show
#         ppg_filtered = nlms.filter(ppg_noisy, ref)

#         # clamp outputs
#         ppg_clean = float(ppg_clean)
#         ppg_noisy = float(ppg_noisy)
#         ppg_filtered = float(ppg_filtered)

#         # ---------------------
#         # Log to CSV
#         # ---------------------
#         with open(CSV_FILE, "a", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 int(time.time()*1000),
#                 ax, ay, az,
#                 acc_mag, acc_change,
#                 gx, gy, gz,
#                 gyro_mag, gyro_change,
#                 round(hr_synth,2),
#                 ppg_noisy, ppg_clean, ppg_filtered,
#                 label, round(motion_score,4)
#             ])

#         # ---------------------
#         # Send labelled packet back to client / dashboard
#         # ---------------------
#         out = {
#             "timestamp_ms": int(time.time()*1000),
#             "ax": ax, "ay": ay, "az": az,
#             "acc_mag": acc_mag, "acc_change": acc_change,
#             "gx": gx, "gy": gy, "gz": gz,
#             "gyro_mag": gyro_mag, "gyro_change": gyro_change,
#             "hr_synth": round(hr_synth,2),
#             "ppg_noisy": ppg_noisy,
#             "ppg_clean": ppg_clean,
#             "ppg_filtered": ppg_filtered,
#             "label": label,
#             "motion_score": round(motion_score,4)
#         }

#         try:
#             await websocket.send(json.dumps(out))
#         except Exception:
#             # if client disconnected, break loop
#             break

# # =========================================================
# # SERVER ENTRY POINT
# # =========================================================
# async def main():
#     print("\n==============================================")
#     print(" Subsystem-D: Artifact Detection + NLMS Server")
#     print(" WebSocket   : ws://0.0.0.0:8765")
#     print(" CSV Logging : {}".format(CSV_FILE))
#     print(" RF Model    : none (rule-based)")
#     print("==============================================\n")

#     async with websockets.serve(handler, "0.0.0.0", 8765):
#         await asyncio.Future()  # run forever

# if __name__ == "__main__":
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("Server stopped by user")





