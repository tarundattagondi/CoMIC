#!/usr/bin/env python3
"""Bootstrap confidence intervals for DeViBench accuracy.

For a deterministic inference benchmark, '3-seed error bars' on the same config
produce identical runs. The statistically sound substitute is bootstrap CI:
resample per-sample correctness vector with replacement, recompute accuracy,
repeat B=10000 times, report 95% CI from percentiles.

Usage:
  python3 bootstrap_ci.py <path-to-result.json> [--B 10000]
"""
import argparse
import json
import os
import sys

import numpy as np


def per_sample_correctness(rows):
    rows = [r for r in rows if "answer" in r and "response" in r]
    return np.array([
        1 if str(r["answer"]).strip() == str(r["response"]).strip() else 0
        for r in rows
    ])


def bootstrap(corr, B=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(corr)
    means = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        means[b] = corr[idx].mean()
    lo, mid, hi = np.percentile(means, [2.5, 50, 97.5])
    return lo, mid, hi, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()

    d = json.load(open(args.json_path))
    corr = per_sample_correctness(d)
    if len(corr) == 0:
        print("No valid rows.")
        sys.exit(1)

    acc = corr.mean()
    lo, mid, hi, _ = bootstrap(corr, B=args.B)

    print(f"file:     {os.path.basename(args.json_path)}")
    print(f"n:        {len(corr)}")
    print(f"acc:      {acc*100:.2f}%  ({corr.sum()}/{len(corr)})")
    print(f"95% CI:   [{lo*100:.2f}%, {hi*100:.2f}%]  (bootstrap B={args.B})")
    print(f"CI half-width: ±{((hi - lo) / 2) * 100:.2f} pp")


if __name__ == "__main__":
    main()
