import numpy as np

def safe_mean(x): return float(x.mean()) if len(x)>0 else 0.0
def safe_std(x): return float(x.std(ddof=1)) if len(x)>1 else 0.0
def safe_max(x): return float(x.max()) if len(x)>0 else 0.0
def safe_percentile(x,q): return float(np.percentile(x,q)) if len(x)>0 else 0.0

def window_feature_vector(wdf, FEATURE_NAMES):
    if wdf.empty:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    ul = wdf[wdf["direction"] == 1]
    dl = wdf[wdf["direction"] == -1]

    ul_bytes = ul["Length"].sum() if len(ul)>0 else 0
    dl_bytes = dl["Length"].sum() if len(dl)>0 else 0

    p95 = safe_percentile(wdf["Length"],95)
    p50 = safe_percentile(wdf["Length"],50)

    return np.array([
        len(wdf),
        wdf["Length"].sum(),
        len(ul),
        len(dl),
        ul_bytes,
        dl_bytes,
        safe_mean(wdf["Length"]),
        safe_std(wdf["Length"]),
        safe_max(wdf["Length"]),
        p95-p50,
        safe_mean(wdf["iat"]),
        safe_mean(ul["iat"]) if len(ul)>0 else 0,
        safe_mean(dl["iat"]) if len(dl)>0 else 0,
        safe_mean(ul["Length"]) if len(ul)>0 else 0,
        safe_mean(dl["Length"]) if len(dl)>0 else 0,
        (dl_bytes / (ul_bytes + 1)) if ul_bytes>0 else 0,
        np.log1p(dl_bytes / (ul_bytes + 1)) if ul_bytes>0 else 0
    ], dtype=np.float32)