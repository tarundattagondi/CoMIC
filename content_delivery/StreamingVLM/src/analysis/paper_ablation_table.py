#!/usr/bin/env python3
"""Generate the paper's headline ablation table from result JSONs.

Produces a Markdown (and LaTeX, if --latex passed) table of:
  Method | r | n | Acc | Δ vs vanilla STAMP-T r=0.20 | median latency

Layout matches the paper's Table 2 mock-up:
  Plain STAMP-T  (anchor row)
  + N6 repack
  + N7 equal-weight
  + N6 + N7
  + M1 merge
  + M3 FOCUS-enhance
  + M5 MSSAVT
  + TAST
  + 3-seed SC majority vote
  + winning stack
"""
import argparse
import json
import os
import re
import statistics
from glob import glob


RESULTS = os.environ.get(
    "STREAMINGVLM_RESULTS_DIR",
    "results/devibench/30fps/vision_sync",
)


METHOD_ORDER = [
    ("Plain STAMP-T",          "stamptemporal_r{R}_a0.5_l0.3_K10_vit31_c3_noadapt_multi7152331"),
    ("+ N6 (repack pos)",      "stamptemporal_r{R}_a0.5_l0.3_K10_vit31_c3_noadapt_n6repack_multi7152331"),
    ("+ N7 (equal layer w)",   "stamptemporal_r{R}_a0.5_l0.3_K10_vit31_c3_noadapt_n7eqw_multi7152331"),
    ("+ N6 + N7",              "stamptemporal_r{R}_a0.5_l0.3_K10_vit31_c3_noadapt_n6repack_n7eqw_multi7152331"),
    ("+ M1 (DyMU merge)",      "stamptemporal_r{R}_a0.5_l0.3_K10_vit31_c3_noadapt_merge_multi7152331"),
    ("+ M3 (FOCUS-enhance α=0.1)", "stamptemporal_r{R}_a0.5_l0.3_K10_vit31_c3_noadapt_m3fe0.1_multi7152331"),
    ("+ M5 (MSSAVT α=0.15)",   "stamptemporal_r{R}_a0.5_l0.3_K10_vit31_c3_noadapt_m5mssavt0.15_multi7152331"),
    ("+ TAST (32 tok, α=0.2)", "stamptemporal_r{R}_a0.5_l0.3_K10_vit31_c3_noadapt_tast_n32_g0.1_a0.2_multi7152331"),
]


def load(path):
    with open(path) as f:
        d = json.load(f)
    return d[1:] if d else []


def acc_latency(preds):
    if not preds:
        return None, None, 0
    correct = sum(1 for r in preds
                  if str(r.get("answer", "")).strip() == str(r.get("response", "")).strip())
    inf = [r["inference_time"] for r in preds if isinstance(r.get("inference_time"), (int, float))]
    med = statistics.median(inf) if inf else None
    return correct / len(preds), med, len(preds)


def find_match(pattern):
    """Substring match on filename (handles trailing tags)."""
    candidates = sorted(glob(os.path.join(RESULTS, f"StreamingVLM_30fps_360_{pattern}*.json")))
    candidates = [c for c in candidates if "_baseline_backup" not in c]
    return candidates[0] if candidates else None


def sc_voted_acc(r_value, sc_scale, base_ids=None):
    """Majority-vote across all SC seed files at this r and scale."""
    pat = f"StreamingVLM_30fps_360_stamptemporal_r{r_value}_a0.5_l0.3_K10_vit31_c3_noadapt_sc_s{sc_scale}_seed*.json"
    files = sorted(glob(os.path.join(RESULTS, pat)))
    files = [f for f in files if "_baseline_backup" not in f]
    if len(files) < 2:
        return None, None, 0, len(files)
    preds = [load(f) for f in files]
    n = min(len(p) for p in preds)
    voted = []
    for i in range(n):
        votes = [str(sp[i].get("response", "")).strip() for sp in preds]
        from collections import Counter
        winner = Counter(votes).most_common(1)[0][0]
        rec = dict(preds[0][i])
        rec["response"] = winner
        voted.append(rec)
    a, lat, n_v = acc_latency(voted)
    return a, lat, n_v, len(files)


def fmt_latency(ms):
    return f"{ms/1000:.1f}s" if ms else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", default="0.11", help="Pruning ratio screening level (default 0.11).")
    ap.add_argument("--sc-r", default="0.30")
    ap.add_argument("--sc-scale", default="0.05")
    ap.add_argument("--latex", action="store_true", help="Also emit a LaTeX tabular block.")
    args = ap.parse_args()

    rows = []
    for label, pat in METHOD_ORDER:
        f = find_match(pat.format(R=args.r))
        if f:
            a, lat, n = acc_latency(load(f))
            rows.append((label, args.r, n, a, lat, os.path.basename(f)))
        else:
            rows.append((label, args.r, 0, None, None, "(missing)"))

    sc_a, sc_lat, sc_n, sc_seeds = sc_voted_acc(args.sc_r, args.sc_scale)
    rows.append((f"+ SC majority vote ({sc_seeds} seeds)", args.sc_r, sc_n, sc_a, sc_lat, "—"))

    # Anchor for delta column = first row with valid acc
    anchor_acc = next((r[3] for r in rows if r[3] is not None), None)

    # Markdown
    print("\n## Phase 2/3 ablation table\n")
    print(f"| Method | r | n | Acc | Δ vs anchor | median lat |")
    print(f"|---|---|---|---|---|---|")
    for label, r, n, a, lat, src in rows:
        if a is None:
            print(f"| {label} | {r} | — | (pending) | — | — |")
            continue
        d = ((a - anchor_acc) * 100) if anchor_acc else 0.0
        delta = "—" if a == anchor_acc else f"{d:+.2f}pp"
        print(f"| {label} | {r} | {n} | {a*100:.2f}% | {delta} | {fmt_latency(lat)} |")

    if args.latex:
        print("\n```latex")
        print(r"\begin{tabular}{lcccc}")
        print(r"\toprule")
        print(r"Method & $r$ & Acc (\%) & $\Delta$ vs anchor (pp) & Lat (s) \\")
        print(r"\midrule")
        for label, r, n, a, lat, _ in rows:
            if a is None:
                print(f"{label} & {r} & --- & --- & --- \\\\")
                continue
            d = ((a - anchor_acc) * 100) if anchor_acc else 0.0
            print(f"{label} & {r} & {a*100:.2f} & {d:+.2f} & {(lat/1000 if lat else 0):.1f} \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print("```")


if __name__ == "__main__":
    main()
