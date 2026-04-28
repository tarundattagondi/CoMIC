# GenAI Traffic Classification

A deep learning based system for detecting **GenAI vs Non-GenAI network traffic** from encrypted packet traces.  
The project uses Wireshark CSV exports, extracts temporal traffic behavior, and classifies traffic using a CNN-LSTM model.

---

## Overview

Modern GenAI applications such as ChatGPT, Gemini, and Grok often communicate over encrypted network channels. Since payload inspection is not possible, this project focuses on identifying GenAI traffic using only behavioral network signals such as:

- Packet timing
- Upload/download direction
- Burst and idle patterns
- Packet size behavior
- DL/UL traffic asymmetry
- Inter-arrival time patterns

The goal is to show that encrypted traffic can still reveal meaningful behavioral fingerprints without accessing packet contents.

---

## Features

- Wireshark CSV based traffic ingestion
- Automatic client IP detection
- Upload/download direction inference
- Time-window based feature extraction
- Sliding sequence generation for temporal modeling
- CNN-LSTM model for traffic classification
- Streamlit dashboard for interactive inference
- Window-level GenAI / Non-GenAI prediction
- Confidence scores and traffic visualizations

---

## Project Pipeline

```text
Wireshark CSV
     ↓
Packet Cleaning
     ↓
Client IP Detection
     ↓
Direction Labeling
     ↓
1-second Time Windowing
     ↓
Feature Extraction
     ↓
Sliding Sequence Creation
     ↓
CNN-LSTM Classification
     ↓
Dashboard Visualization
```
---

## Model Approach

This project uses a CNN-LSTM hybrid architecture.

The CNN layers learn local traffic patterns such as bursts, packet-size changes, and short-term transitions.

The LSTM layer models how these patterns evolve over time, such as:

- Burst → silence → burst
- Continuous streaming behavior
- Long idle gaps followed by activity
- Upload-heavy or download-heavy regions

This makes the model suitable for encrypted traffic where content is hidden but temporal behavior is still visible.

---

## Extracted Features

Each time window is represented using 17 traffic features:

```text
pkt_count
total_bytes
ul_pkt_count
dl_pkt_count
ul_bytes
dl_bytes
mean_pkt_size
std_pkt_size
max_pkt_size
p95_p50_pkt_size
mean_iat
ul_mean_iat
dl_mean_iat
ul_mean_pkt_size
dl_mean_pkt_size
dl_ul_byte_ratio
dl_ul_log_ratio
```
---
## Repository Structure

```
GenAI_Traffic_Classification/
│
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
│
├── artifacts/              # Trained model and scaler
│   ├── cnn_lstm_model.pth
│   └── scaler.pkl
│
├── images/                 # Dashboard or project images
│
└── src/
    ├── config.py           # Feature list and window settings
    ├── preprocessing.py    # Packet cleaning and client IP detection
    ├── features.py         # Window-level feature extraction
    ├── sequence.py         # Time windows and sliding sequences
    ├── model.py            # CNN-LSTM model architecture
    ├── inference.py        # Prediction pipeline
    ├── run_inference.py    # Script-based inference
    └── visualization.py    # Plotly visualizations
```
---

## Tech Stack
```
Python
Pandas
NumPy
PyTorch
Scikit-learn
Joblib
Plotly
Matplotlib
Streamlit
Wireshark CSV exports
```
---

## Installation

Clone the repository:

```Bash
git clone https://github.com/KoushikRama/GenAI_Traffic_Classification.git
cd GenAI_Traffic_Classification
```

Create and activate a virtual environment:

```Bash
python -m venv venv
```
For Windows:
```Bash
venv\Scripts\activate
```
For macOS/Linux:
```Bash
source venv/bin/activate
```
Install dependencies:
```Bash
pip install -r requirements.txt
```
---
## Running the Streamlit App
```Bash
streamlit run app.py
```
Then upload a Wireshark CSV file through the web interface.

The dashboard will show:

- Total analyzed windows
- GenAI percentage
- Non-GenAI percentage
- Window-level predictions
- Confidence scores
- Traffic timeline
- UL/DL packet and byte behavior
- Inter-arrival time behavior
- DL/UL ratio trends
---
## Input Format

The uploaded CSV should be exported from Wireshark and must contain at least these columns:

```
Time
Source
Destination
Length
```
Example:
```
Time,Source,Destination,Length
0.000000,192.168.1.5,34.117.59.81,66
0.002341,34.117.59.81,192.168.1.5,1514
```
---
## Classification Output

The model produces window-level predictions:
```
GenAI
Non_GenAI
```
Each prediction also includes a confidence score.

Example output:
```
window_id | start_sec | end_sec | label    | confidence
0         | 0         | 40      | GenAI    | 0.94
1         | 20        | 60      | GenAI    | 0.91
2         | 40        | 80      | Non_GenAI| 0.87
```
---

## Key Idea

The main idea of this project is that encrypted GenAI traffic still carries behavioral signals.Even without payload access, traffic patterns such as packet timing, direction changes, burst structure, idle gaps, and upload/download imbalance can help distinguish GenAI applications from regular non-GenAI traffic.

---

## Current Limitations

- The model depends on the quality and diversity of collected packet traces.
- Small datasets may reduce generalization across networks and devices.
- Similar traffic behaviors, such as streaming or idle buffering, can overlap with GenAI behavior.
- The current system performs classification at the traffic-window level, not full application attribution.
- Real-world deployment would require more traces from different networks, devices, and GenAI platforms.

---

## Future Improvements

- Add more GenAI and non-GenAI application traces
- Improve session-level aggregation
- Add transformer-based temporal models
- Compare CNN-LSTM with Random Forest, XGBoost, and pure LSTM baselines
- Add explainability for important traffic regions
- Improve robustness across different network conditions
- Deploy dashboard as a hosted web app

---
## Contributers

**Koushik Rama**
**Mahitrinadh Chilkuri**

MS Computer Science, George Mason University

Focus: AI/ML, Network Traffic Analysis, GenAI Systems, and Deep Learning
