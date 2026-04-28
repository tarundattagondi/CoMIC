import streamlit as st
import pandas as pd

from src.inference import (
    run_pipeline,
    summarize,
    predictions_to_df,
    normalize_seq_full
)

from src.visualization import (
    plot_regions,
    plot_ul_dl_bytes,
    plot_ul_dl_packets,
    plot_dl_ul_ratio,
    plot_ul_dl_iat,
    plot_p95_p50
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="GenAI Traffic Classifier", layout="wide")

st.title("GenAI vs Non-GenAI Traffic Classifier")
st.write("Upload a Wireshark CSV file to classify temporal regions.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # ================= PIPELINE =================
        preds, probs, windows, seq_full, seqs = run_pipeline(df)
        summary = summarize(preds)
        pred_df = predictions_to_df(preds, probs, windows, seqs)

        seq_full_norm = normalize_seq_full(seq_full)

        # ================= METRICS =================
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Windows", summary["total_windows"])
        c2.metric("GenAI %", f'{summary["genai_pct"]:.2f}%')
        c3.metric("Non-GenAI %", f'{summary["non_genai_pct"]:.2f}%')

        st.divider()

        # ================= TOGGLE =================
        use_normalized = st.toggle("Use Normalized Data (Model View)", value=False)
        seq_display = seq_full_norm if use_normalized else seq_full

        # ================= TIMELINE =================
        st.subheader("Traffic Timeline")
        st.plotly_chart(plot_regions(preds, windows), use_container_width=True)

        # ================= DASHBOARD =================
        st.subheader("Traffic Dashboard")
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                plot_ul_dl_packets(seq_display, preds, windows),
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                plot_ul_dl_bytes(seq_display, preds, windows),
                use_container_width=True
            )

        col3, col4 = st.columns(2)

        with col3:
            st.plotly_chart(
                plot_ul_dl_iat(seq_display, preds, windows),
                use_container_width=True
            )

        with col4:
            st.plotly_chart(
                plot_p95_p50(seq_display, preds, windows),
                use_container_width=True
            )
        
        st.subheader("DL/UL Ratio")
        st.plotly_chart(
            plot_dl_ul_ratio(seq_display, preds, windows),
            use_container_width=True
        )

        # ================= TABLE =================
        st.subheader("Window Predictions")
        st.dataframe(pred_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")