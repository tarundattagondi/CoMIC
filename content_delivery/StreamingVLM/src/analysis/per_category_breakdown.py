#!/usr/bin/env python3
"""Per-category accuracy breakdown for DeViBench result files.

Required by paper Section 5: shows where each method wins/loses (Counting,
Text-Rich Understanding, Action Perception, etc). Pass any number of result
files; first one is treated as the baseline for delta columns.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from glob import glob


def load(p):
    with open(p) as f:
        d = json.load(f)
    return d[1:] if d else []


def per_cat_acc(preds):
    by = defaultdict(lambda: [0, 0])  # cat -> [correct, total]
    for r in preds:
        c = r.get("task", "?")
        ans = str(r.get("answer", "")).strip()
        rsp = str(r.get("response", "")).strip()
        by[c][1] += 1
        if ans == rsp:
            by[c][0] += 1
    out = {c: (corr / max(1, tot), tot) for c, (corr, tot) in by.items()}
    overall_corr = sum(c for c, _ in by.values())
    overall_tot = sum(t for _, t in by.values())
    out["__OVERALL__"] = (overall_corr / max(1, overall_tot), overall_tot)
    return out


def short(label, n=42):
    return label if len(label) <= n else label[: n - 1] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="Result JSON files; first = baseline.")
    ap.add_argument("--csv", help="Optional CSV output path.")
    args = ap.parse_args()

    files = []
    for pattern in args.files:
        if any(c in pattern for c in "*?["):
            files.extend(sorted(glob(pattern)))
        else:
            files.append(pattern)
    if not files:
        sys.exit("No files matched.")

    cat_accs = {}
    for f in files:
        cat_accs[os.path.basename(f)] = per_cat_acc(load(f))

    base_name = os.path.basename(files[0])
    base = cat_accs[base_name]
    cats = sorted(set().union(*[a.keys() for a in cat_accs.values()]),
                  key=lambda c: (c != "__OVERALL__", c))

    # Header
    name_w = 28
    print(f"\nBaseline: {base_name}\n")
    header = f"{'category':<32}{'n':>5}  " + "  ".join(
        f"{short(os.path.basename(f).replace('StreamingVLM_30fps_360_', ''), name_w):<{name_w}}"
        for f in files
    )
    print(header)
    print("-" * len(header))
    for c in cats:
        n = base.get(c, (0, 0))[1]
        cells = []
        base_a = base.get(c, (None,))[0]
        for f in files:
            ca = cat_accs[os.path.basename(f)].get(c)
            if ca is None or ca[1] == 0:
                cells.append(f"{'-':<{name_w}}")
                continue
            a = ca[0]
            if f == files[0]:
                cells.append(f"{a*100:>6.2f}%{' ':<{name_w-7}}")
            else:
                d = (a - base_a) * 100 if base_a is not None else 0.0
                cells.append(f"{a*100:>6.2f}% ({d:+.2f}pp){' ':<{name_w-21}}")
        print(f"{short(c, 30):<32}{n:>5}  " + "  ".join(cells))

    if args.csv:
        with open(args.csv, "w") as fout:
            fout.write("file,category,n,acc\n")
            for f in files:
                ca = cat_accs[os.path.basename(f)]
                for c, (a, n) in ca.items():
                    fout.write(f"{os.path.basename(f)},{c},{n},{a:.6f}\n")
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
