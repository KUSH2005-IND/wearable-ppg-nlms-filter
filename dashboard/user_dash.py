# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.fft import rfft, rfftfreq
from sklearn.metrics import roc_auc_score, roc_curve
import joblib
import io
import datetime


CSV_PATH = "hr_log.csv"
st.set_page_config(layout="wide", page_title="Diagnostic Panel")


@st.cache_data
def load_data(path=CSV_PATH, nrows=None):
    try:
        df = pd.read_csv(path, nrows=nrows)
    except Exception as e:
        st.error(f"Failed to read CSV at {path}: {e}")
        return None

    # normalize column names
    df.columns = [c.strip() for c in df.columns]

    
    for tname in ["timestamp","timestamp_ms","t","time"]:
        if tname in df.columns:
            df = df.rename(columns={tname: "timestamp_ms"})
            break

    # convert timestamp numeric to datetime
    if "timestamp_ms" in df.columns:
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], errors="coerce")
        # if large numbers (ms)
        if df["timestamp_ms"].max() > 1e10:
            df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", errors="coerce")
        else:
            df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="s", errors="coerce")
    else:
        df["datetime"] = pd.NaT

    
    numeric_cols = ["ax","ay","az","gx","gy","gz","acc_mag","gyro_mag","acc_change","gyro_change",
                    "ppg","ppg_noisy","ppg_clean","ppg_filtered","hr","hr_synth","motion_score","snr"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def compute_fft(sig, fs):
    sig = np.asarray(sig)
    N = sig.size
    if N <= 1:
        return np.array([]), np.array([])
    yf = np.abs(rfft(sig - np.mean(sig)))
    xf = rfftfreq(N, 1 / fs)
    return xf, yf

def to_pydt(ts):
    if pd.isna(ts):
        return None
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime()
    if isinstance(ts, (np.datetime64, datetime.datetime)):
        return pd.to_datetime(ts).to_pydatetime()
    try:
        return pd.to_datetime(ts).to_pydatetime()
    except Exception:
        return None


st.sidebar.header("Controls")
use_local = st.sidebar.checkbox("Load full CSV (hr_log.csv)", value=True)
rows = None
if not use_local:
    rows = st.sidebar.number_input("Rows to load", min_value=100, max_value=200000, value=5000, step=100)

window_seconds = st.sidebar.slider("Time window (seconds)", min_value=1, max_value=3600, value=30)
fs_guess = st.sidebar.number_input("Assumed sampling rate (Hz)", min_value=1, max_value=1000, value=100)
show_fft = st.sidebar.checkbox("Show PPG FFT", value=True)
show_corr = st.sidebar.checkbox("Show correlation matrix", value=True)
show_metrics = st.sidebar.checkbox("Show ROC / AUC (if model uploaded)", value=False)

uploaded_model = st.sidebar.file_uploader("Upload RF model (.pkl / .joblib) — optional", type=["pkl","joblib"])
uploaded_scaler = st.sidebar.file_uploader("Upload scaler (.pkl / .joblib) — optional", type=["pkl","joblib"])


st.sidebar.markdown("---")
st.sidebar.subheader("Label threshold preview")
acc_clean_th = st.sidebar.number_input("acc_change clean threshold", value=0.006, step=0.001, format="%.6f")
gyro_clean_th = st.sidebar.number_input("gyro_change clean threshold", value=0.20, step=0.01)
gyro_mag_clean = st.sidebar.number_input("gyro_mag clean threshold", value=0.55, step=0.05)

acc_heavy_th = st.sidebar.number_input("acc_change heavy threshold", value=0.025, step=0.001)
gyro_heavy_th = st.sidebar.number_input("gyro_change heavy threshold", value=1.00, step=0.01)
gyro_mag_heavy = st.sidebar.number_input("gyro_mag heavy threshold", value=1.80, step=0.05)


df = load_data(nrows=rows if not use_local else None)
if df is None:
    st.stop()

# derived columns if missing
if "acc_mag" not in df.columns and all(c in df.columns for c in ["ax","ay","az"]):
    df["acc_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
if "gyro_mag" not in df.columns and all(c in df.columns for c in ["gx","gy","gz"]):
    df["gyro_mag"] = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)

# simple motion_score if missing
if "motion_score" not in df.columns:
    df["acc_change"] = pd.to_numeric(df.get("acc_change", np.nan), errors="coerce").fillna(0.0)
    df["gyro_change"] = pd.to_numeric(df.get("gyro_change", np.nan), errors="coerce").fillna(0.0)
    df["motion_score"] = df["acc_change"] * 100.0 + df["gyro_change"] * 10.0 + df.get("gyro_mag", 0.0) * 5.0

# top-level metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows", f"{len(df):,}")
with col2:
    st.metric("Label classes", f"{df['label'].nunique() if 'label' in df.columns else 0}")
with col3:
    tmin = df["datetime"].min() if "datetime" in df.columns else None
    st.metric("Start", str(tmin))
with col4:
    tmax = df["datetime"].max() if "datetime" in df.columns else None
    st.metric("End", str(tmax))

st.title("Diagnostic Panel")


if "datetime" in df.columns and df["datetime"].notna().any():
    start_time = to_pydt(df["datetime"].min())
    end_time = to_pydt(df["datetime"].max())
    if start_time is None or end_time is None:
        st.warning("Couldn't parse datetime column — falling back to index-based selection")
        use_index = True
    else:
        use_index = False

    if not use_index:
        default_start = end_time - datetime.timedelta(seconds=window_seconds)
        sel_start, sel_end = st.slider(
            "Select time range (datetime)",
            min_value=start_time,
            max_value=end_time,
            value=(default_start, end_time),
            format="YYYY-MM-DD HH:mm:ss"
        )
        window_df = df[(df["datetime"] >= pd.to_datetime(sel_start)) & (df["datetime"] <= pd.to_datetime(sel_end))]
    else:
        n = len(df)
        end_idx = st.slider("Select index range", 0, n, (max(0, n - int(window_seconds * fs_guess)), n))
        window_df = df.iloc[end_idx[0]:end_idx[1]]
else:
    # fallback to index-based window
    n = len(df)
    end_idx = st.slider("Select index range", 0, n, (max(0, n - int(window_seconds * fs_guess)), n))
    window_df = df.iloc[end_idx[0]:end_idx[1]]

if window_df.empty:
    st.warning("No data in selected window")
    st.stop()


ppg_cols = [c for c in window_df.columns if c.lower().startswith("ppg") or c.lower() == "ppg"]
if "ppg_noisy" in window_df.columns:
    ppg_cols = [c for c in ["ppg_noisy","ppg_clean","ppg_filtered"] if c in window_df.columns]
elif "ppg" in window_df.columns and len(ppg_cols) == 1:
    # single ppg column
    ppg_cols = [ppg_cols[0]]


left, right = st.columns([2,1])

with left:
    st.subheader("IMU: Acc & Gyro magnitude")
    fig_imu = go.Figure()
    if "datetime" in window_df.columns:
        xaxis = window_df["datetime"]
    else:
        xaxis = window_df.index
    if "acc_mag" in window_df.columns:
        fig_imu.add_trace(go.Scatter(x=xaxis, y=window_df["acc_mag"], mode="lines", name="acc_mag"))
    if "gyro_mag" in window_df.columns:
        fig_imu.add_trace(go.Scatter(x=xaxis, y=window_df["gyro_mag"], mode="lines", name="gyro_mag"))
    fig_imu.update_layout(height=250, margin=dict(l=10,r=10,t=30,b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig_imu, use_container_width=True)

    st.markdown("### PPG signals (stacked)")
    if ppg_cols:
        # display each PPG in its own small chart stacked vertically
        for c in ppg_cols:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xaxis, y=window_df[c], mode="lines", name=c))
            fig.update_layout(height=160, margin=dict(l=10,r=10,t=20,b=10), title=c)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No PPG columns found. Subsystem D can generate synthetic PPG (ppg_noisy/ppg_clean/ppg_filtered).")

with right:
    st.subheader("Live gauges & indicators")
    hr_candidates = []
    for c in ["hr","hr_synth","hr_generated"]:
        if c in window_df.columns:
            hr_candidates.append(c)
    hr_val = None
    if hr_candidates:
        for c in hr_candidates:
            non_null = window_df[c].dropna()
            if len(non_null):
                hr_val = float(non_null.iloc[-1])
                hr_label = c
                break

    if hr_val is None:
        hr_val = None
    # HR circular gauge
    if hr_val is not None:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=hr_val,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"HR ({hr_label})", 'font': {'size': 14}},
            delta={'reference': 72, 'valueformat': '.1f'},
            gauge={
                'axis': {'range': [30, 180]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [30, 60], 'color': "lightcyan"},
                    {'range': [60, 100], 'color': "lightgreen"},
                    {'range': [100, 140], 'color': "orange"},
                    {'range': [140, 180], 'color': "red"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': hr_val}}))
        fig_g.update_layout(height=300, margin=dict(l=10,r=10,t=20,b=20))
        st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.info("No HR available in selected window.")

    # Artifact status LED (most frequent label in window)
    st.markdown("**Artifact status (window mode)**")
    if "label" in window_df.columns:
        mode_label = window_df["label"].mode().iloc[0] if not window_df["label"].mode().empty else "unknown"
        color = {"clean":"green", "artifact_light":"orange", "artifact_heavy":"red"}.get(mode_label, "gray")
        st.markdown(f"<div style='display:flex;align-items:center;gap:10px'><div style='width:18px;height:18px;border-radius:50%;background:{color}'></div><div>{mode_label}</div></div>", unsafe_allow_html=True)
    else:
        st.info("No label column present")

    # Motion severity meter (mean motion_score)
    st.markdown("**Motion severity (mean)**")
    ms_mean = float(window_df["motion_score"].mean()) if "motion_score" in window_df.columns else 0.0
    # normalize for progress bar
    cap = max(1.0, ms_mean, 200.0)
    prog = min(1.0, ms_mean / cap)
    st.progress(prog)
    st.caption(f"Mean motion score: {ms_mean:.3f}")

    # SNR meter if ppg_clean & ppg_noisy present
    if set(["ppg_noisy","ppg_clean"]).issubset(window_df.columns):
        noisy = window_df["ppg_noisy"].dropna().to_numpy()
        clean = window_df["ppg_clean"].dropna().to_numpy()
        if noisy.size and clean.size:
            signal_power = np.mean(clean**2)
            noise_power = np.mean((noisy - clean)**2)
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-9))
            st.markdown("**Estimated SNR**")
            st.metric("SNR (dB)", f"{snr_db:.2f} dB")
            # visual horizontal bar 
            snr_norm = (snr_db + 20) / 60.0  
            snr_norm = min(1.0, max(0.0, snr_norm))
            st.progress(snr_norm)
    else:
        st.info("SNR: need ppg_noisy & ppg_clean columns to compute")


st.markdown("---")
st.subheader("Spectral analysis / FFT")
if ppg_cols and show_fft:
    ppg_pick = st.selectbox("Choose PPG signal to FFT", ppg_cols)
    sig = window_df[ppg_pick].dropna().to_numpy()
    if len(sig) > 2:
        xf, yf = compute_fft(sig, fs=fs_guess)
        fig = go.Figure(go.Scatter(x=xf, y=yf, mode="lines"))
        fig.update_layout(title=f"FFT of {ppg_pick}", xaxis_title="Hz", yaxis_title="magnitude", height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough PPG samples for FFT in selected window")
else:
    st.info("No PPG for FFT or FFT disabled")


if show_corr:
    st.markdown("---")
    st.subheader("Correlation matrix (motion vs PPG)")
    corr_cols = []
    for c in ['acc_mag','acc_change','gyro_mag','gyro_change','motion_score']:
        if c in window_df.columns:
            corr_cols.append(c)
    
    if ppg_cols:
        corr_cols += ppg_cols
    corr_cols = [c for i,c in enumerate(corr_cols) if c not in corr_cols[:i]]
    if len(corr_cols) >= 2:
        corr = window_df[corr_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", zmin=-1, zmax=1, color_continuous_scale="RdBu")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns to compute correlation matrix")


if show_metrics:
    st.markdown("---")
    st.subheader("ROC / AUC (optional model)")
    if uploaded_model is None:
        st.info("Upload a trained classifier (.pkl or .joblib) and optionally a scaler to compute ROC/AUC on the selected window.")
    else:
        try:
            model = joblib.load(uploaded_model)
            scaler = None
            if uploaded_scaler is not None:
                try:
                    scaler = joblib.load(uploaded_scaler)
                except Exception as e:
                    st.warning(f"Failed to load scaler: {e}")
            st.success("Model loaded — attempting ROC/AUC")

            if 'label' not in window_df.columns:
                st.warning("No label column present in window — cannot compute ROC/AUC")
            else:
                # prepare features: exclude some known non-feature columns
                exclude = set(["label","datetime","timestamp_ms"])
                feature_cols = [c for c in window_df.columns if c not in exclude and np.issubdtype(window_df[c].dtype, np.number)]
                if not feature_cols:
                    st.warning("No numeric feature columns found to feed model")
                else:
                    X = window_df[feature_cols].fillna(0).to_numpy()
                    if scaler is not None:
                        try:
                            X = scaler.transform(X)
                        except Exception as e:
                            st.warning(f"Scaler transform failed: {e} — proceeding without scaler.")
                    try:
                        # if classifier gives predict_proba
                        if hasattr(model, "predict_proba"):
                            y_score = model.predict_proba(X)[:,1]
                        else:
                            # fallback to decision_function or predict
                            if hasattr(model, "decision_function"):
                                y_score = model.decision_function(X)
                            else:
                                y_score = model.predict(X)
                        y_true = (window_df["label"] == st.text_input("Positive class label for ROC", value="artifact_heavy")).astype(int)
                        auc = roc_auc_score(y_true, y_score)
                        fpr, tpr, _ = roc_curve(y_true, y_score)
                        st.metric("ROC AUC", f"{auc:.3f}")
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                        fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR", height=300)
                        st.plotly_chart(fig_roc, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Failed to compute ROC/AUC: {e}")
        except Exception as e:
            st.error(f"Failed loading model: {e}")


st.markdown("---")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Download selected window CSV**")
    to_download = window_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=to_download, file_name="window_dump.csv", mime="text/csv")
with col_b:
    st.markdown("**Quick statistics (window)**")
    st.write(window_df.describe(include='all'))

st.markdown("---")
