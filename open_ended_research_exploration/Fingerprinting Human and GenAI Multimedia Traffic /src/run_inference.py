import pandas as pd
from src.inference import run_pipeline, summarize

df = pd.read_csv("test_csv/test1.csv")

preds, probs, windows, seqs = run_pipeline(df)

print(preds)
summarize(preds)