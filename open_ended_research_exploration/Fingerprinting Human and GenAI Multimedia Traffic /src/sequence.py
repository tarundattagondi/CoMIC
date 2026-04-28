import pandas as pd
import numpy as np
from src.config import WINDOW_SIZE, SEQ_LEN, STEP, FEATURE_NAMES
from src.features import window_feature_vector
from src.preprocessing import clean_packet_df

# =============================
# TIME WINDOWING
# =============================
def create_time_windows(df):
    start = df["Time"].min()
    end = df["Time"].max()

    df["bin"] = ((df["Time"] - start)//WINDOW_SIZE).astype(int)

    groups = df.groupby("bin")

    all_bins = np.arange(0, int((end - start)//WINDOW_SIZE) + 1)

    full_groups = []

    for b in all_bins:
        if b in groups.groups:
            full_groups.append(groups.get_group(b))
        else:
            full_groups.append(pd.DataFrame(columns=df.columns))

    return full_groups

def build_sequence(df):
    groups = create_time_windows(df)
    return np.array([window_feature_vector(g, FEATURE_NAMES) for g in groups])

# =============================
# SLIDING WINDOWS
# =============================
def create_sliding(seq_full):
    seqs = []
    windows = []
    for i in range(0, len(seq_full) - SEQ_LEN + 1, STEP):
        seqs.append(seq_full[i:i+SEQ_LEN])
        windows.append((i, i + SEQ_LEN))
    return np.array(seqs), np.array(windows)

def build_test_sequences(df):
    df = clean_packet_df(df)

    seq_full = build_sequence(df)

    if len(seq_full) < SEQ_LEN:
        print("Too short")
        return None, None

    seqs, windows = create_sliding(seq_full)

    return seqs, windows, seq_full