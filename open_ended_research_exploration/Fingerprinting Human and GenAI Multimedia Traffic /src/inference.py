import torch
import numpy as np
import joblib
import pandas as pd
from src.model import CNNLSTM
from src.preprocessing import clean_packet_df
from src.sequence import build_test_sequences

device = torch.device("cpu")

model = CNNLSTM(F=17)
model.load_state_dict(torch.load("artifacts/cnn_lstm_model.pth", map_location=device))
model.eval()

scaler = joblib.load("artifacts/scaler.pkl")

def normalize(seqs):
    N,T,F = seqs.shape
    return scaler.transform(seqs.reshape(-1,F)).reshape(N,T,F)

def normalize_seq_full(seq_full):
    N, F = seq_full.shape
    
    seq_full_norm = scaler.transform(seq_full.reshape(-1, F)).reshape(N, F)
    
    return seq_full_norm

def predict(seqs):
    X = torch.tensor(seqs, dtype=torch.float32)
    with torch.no_grad():
        out = model(X)
        probs = torch.softmax(out, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs

def run_pipeline(df):

    df = clean_packet_df(df)

    seqs, windows, seq_full = build_test_sequences(df)

    seqs = normalize(seqs)

    preds, probs = predict(seqs)

    return preds, probs, windows, seq_full, seqs


def summarize(preds: np.ndarray):
    total = len(preds)
    genai = int((preds == 0).sum())
    non_genai = int((preds == 1).sum())

    return {
        "total_windows": total,
        "genai_pct": (genai / total) * 100,
        "non_genai_pct": (non_genai / total) * 100,
    }


def predictions_to_df(preds, probs, windows, seqs, window_size=1.0):
    rows = []
    for i, (p, prob, (s, e), seq) in enumerate(zip(preds, probs, windows, seqs)):
        rows.append({
            "window_id": i,
            "start_sec": s * window_size,
            "end_sec": e * window_size,
            "label": "GenAI" if p == 0 else "Non_GenAI",
            "confidence": float(prob[p])
        })
    return pd.DataFrame(rows)
