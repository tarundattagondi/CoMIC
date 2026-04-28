import numpy as np
import plotly.graph_objects as go


# =========================================================
# HELPER → ADD BACKGROUND REGIONS
# =========================================================
def add_prediction_regions(fig, preds, windows, y_min, y_max, xref="x", yref="y"):
    for p, (s, e) in zip(preds, windows):
        color = "red" if p == 0 else "blue"

        fig.add_shape(
            type="rect",
            x0=s, x1=e,
            y0=y_min, y1=y_max,
            fillcolor=color,
            opacity=0.12,
            line_width=0,
            xref=xref,
            yref=yref
        )

    return fig


# =========================================================
# REGION TIMELINE
# =========================================================
def plot_regions(preds, windows):
    fig = go.Figure()

    for p, (s, e) in zip(preds, windows):
        color = "red" if p == 0 else "blue"
        label = "GenAI" if p == 0 else "Non-GenAI"

        fig.add_trace(go.Scatter(
            x=[s, e],
            y=[1, 1],
            mode="lines",
            line=dict(width=20, color=color),
            name=label,
            hovertemplate=f"{label}<br>{s}-{e}s<extra></extra>"
        ))

    fig.update_layout(
        title="Traffic Classification Timeline",
        xaxis_title="Time (seconds)",
        yaxis=dict(visible=False),
        template="plotly_dark",
        height=200
    )

    return fig


# =========================================================
# UL/DL BYTES
# =========================================================
def plot_ul_dl_bytes(seq_full, preds, windows):
    x = np.arange(len(seq_full))
    ul = seq_full[:, 4]
    dl = seq_full[:, 5]

    fig = go.Figure()

    # background
    fig = add_prediction_regions(fig, preds, windows, min(ul.min(), dl.min()), max(ul.max(), dl.max()))

    fig.add_trace(go.Scatter(x=x, y=ul, name="UL Bytes"))
    fig.add_trace(go.Scatter(x=x, y=dl, name="DL Bytes"))

    fig.update_layout(
        title="UL vs DL Bytes",
        xaxis_title="Time (seconds)",
        yaxis_title="Bytes",
        template="plotly_dark",
        hovermode="x unified"
    )

    return fig


# =========================================================
# UL/DL PACKETS
# =========================================================
def plot_ul_dl_packets(seq_full, preds, windows):
    x = np.arange(len(seq_full))
    ul = seq_full[:, 2]
    dl = seq_full[:, 3]

    fig = go.Figure()

    fig = add_prediction_regions(fig, preds, windows, min(ul.min(), dl.min()), max(ul.max(), dl.max()))

    fig.add_trace(go.Scatter(x=x, y=ul, name="UL Packets"))
    fig.add_trace(go.Scatter(x=x, y=dl, name="DL Packets"))

    fig.update_layout(
        title="UL vs DL Packets",
        xaxis_title="Time (seconds)",
        yaxis_title="Packets",
        template="plotly_dark",
        hovermode="x unified"
    )

    return fig

# =========================================================
# UL/DL IAT
# =========================================================
def plot_ul_dl_iat(seq_full, preds, windows):
    x = np.arange(len(seq_full))
    ul = seq_full[:, 11]
    dl = seq_full[:, 12]

    fig = go.Figure()

    # background
    fig = add_prediction_regions(fig, preds, windows, min(ul.min(), dl.min()), max(ul.max(), dl.max()))

    fig.add_trace(go.Scatter(x=x, y=ul, name="UL IAT"))
    fig.add_trace(go.Scatter(x=x, y=dl, name="DL IAT"))

    fig.update_layout(
        title="UL vs DL IAT",
        xaxis_title="Time (seconds)",
        yaxis_title="IAT",
        template="plotly_dark",
        hovermode="x unified"
    )

    return fig

# =========================================================
# p95-p50
# =========================================================
def plot_p95_p50(seq_full, preds, windows):
    x = np.arange(len(seq_full))
    ratio = seq_full[:, 9]

    fig = go.Figure()

    fig = add_prediction_regions(fig, preds, windows, ratio.min(), ratio.max())

    fig.add_trace(go.Scatter(x=x, y=ratio, name="P95-P50"))

    fig.update_layout(
        title="P95-P50",
        xaxis_title="Time (seconds)",
        yaxis_title="p95-p50",
        template="plotly_dark",
        hovermode="x unified"
    )

    return fig
# =========================================================
# DL/UL RATIO
# =========================================================
def plot_dl_ul_ratio(seq_full, preds, windows):
    x = np.arange(len(seq_full))
    ratio = seq_full[:, 15]

    fig = go.Figure()

    fig = add_prediction_regions(fig, preds, windows, ratio.min(), ratio.max())

    fig.add_trace(go.Scatter(x=x, y=ratio, name="DL/UL Ratio"))

    fig.update_layout(
        title="DL/UL Ratio",
        xaxis_title="Time (seconds)",
        yaxis_title="Ratio",
        template="plotly_dark",
        hovermode="x unified"
    )

    return fig


