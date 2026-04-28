import pandas as pd
import numpy as np
import ipaddress

def is_private_ip(ip):
    try:
        return ipaddress.ip_address(ip).is_private
    except:
        return False

def detect_client_ip(df):
    src_counts = df["Source"].value_counts()
    dst_counts = df["Destination"].value_counts()

    common = set(src_counts.index).intersection(set(dst_counts.index))
    private_common = [ip for ip in common if is_private_ip(ip)]

    if private_common:
        scores = {ip: src_counts.get(ip,0)+dst_counts.get(ip,0) for ip in private_common}
        return max(scores, key=scores.get)

    all_ips = pd.concat([df["Source"], df["Destination"]])
    return all_ips.value_counts().index[0]

def clean_packet_df(df):
    df = df.copy()
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce")

    df = df.dropna().sort_values("Time").reset_index(drop=True)

    client_ip = detect_client_ip(df)
    df["direction"] = np.where(df["Source"] == client_ip, 1, -1)

    df["iat"] = df["Time"].diff().fillna(0.0)

    return df