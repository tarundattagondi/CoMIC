# StreamingVLM

Training-free token-pruning and streaming-state methods for real-time infinite video stream understanding, built on top of the StreamingVLM (MIT HAN Lab) inference stack and Qwen2.5-VL-7B-Instruct.

**Team:** Sri Sashank Potluru, Venkata Akhil Akkineni
**Course:** GMU CoMIC (content delivery track)
**Headline result:** **86.50%** on DeViBench (n=652) on Qwen2.5-VL-7B-Instruct using our M1+TAST stack at r=0.30, **+4.42 pp [+1.82, +6.88]** over StreamingVLM-vanilla 82.08% (paired bootstrap, 95% CI). Training-free, no retraining, ~30% LLM-token compute.

---

## What is in this folder

```
StreamingVLM/
├── README.md                  ← this file
├── LICENSE                    ← MIT (upstream from MIT HAN Lab)
├── paper/
│   ├── draft.pdf              ← 8-page conference draft
│   └── all_artifacts.pdf      ← exhibits bundle (tables, Pareto plot, diagrams)
├── figures/
│   ├── m1_dymu_merge.pdf      ← M1 spatial-merge architecture
│   ├── sc_pipeline.pdf        ← self-consistency pipeline
│   ├── stamp_temporal.pdf     ← STAMP-Temporal architecture (LaTeX render)
│   ├── stamp_temporal-1.png   ← raster version
│   └── pareto_devibench.pdf   ← FLOPs vs accuracy Pareto plot on DeViBench
├── diagrams/
│   ├── stamp_temporal.html    ← interactive STAMP-Temporal diagram
│   ├── focus.html             ← interactive FOCUS diagram
│   └── m1_tast_stack.html     ← interactive headline-architecture diagram
└── src/
    ├── inference/
    │   ├── stamp_temporal.py  ← STAMP-T + TAST + DSTM + M1 merge + Phase-7 hooks
    │   ├── spatial_focus.py   ← FOCUS text-guided spatial pruner
    │   ├── dstm.py            ← Dual Scene + Delta Memory module
    │   ├── stamp.py           ← original STAMP (baseline reference)
    │   ├── streaming_args.py  ← unified CLI flags
    │   ├── inference.py       ← top-level streaming inference loop
    │   ├── qwen2_5/           ← patched Qwen2.5-VL forward passes
    │   └── generate/          ← streaming KV-cache generation
    ├── eval/
    │   ├── DeViBench/eval_devi_360.py
    │   ├── MVBench/evaluate_mvbench.py
    │   └── MLVU/evaluate_mlvu.py
    └── analysis/
        ├── bootstrap_ci.py            ← paired-bootstrap 95% CI
        ├── paper_ablation_table.py    ← ablation-table generator
        └── per_category_breakdown.py  ← per-task accuracy split
```

The full Hopper development tree (logs, SLURM scripts, checkpoints, raw result JSONs, dropped-method code, presentations, training artifacts, etc.) is intentionally **not** included — only files needed to read, reproduce, or extend the surviving methods.

---

## Methods (surviving in the paper)

| Module | File | What it does |
|---|---|---|
| **STAMP-Temporal** | `src/inference/stamp_temporal.py` | Pure-ViT temporal token pruning. Multi-signal scoring (attention salience, frame entropy, cross-frame novelty), short-term EMA momentum, entropy-adaptive keep ratio, top-k selection per chunk. |
| **TAST** | `src/inference/stamp_temporal.py` (`tast_*` paths) | Temporal Accumulative State Tokens. EMA-pooled summary tokens (default 32, γ=0.1) injected alongside kept visual tokens to carry long-horizon context across chunks. |
| **DSTM** | `src/inference/dstm.py` | Dual Scene + Delta Memory. Surprise-gated memory bank that keeps a stable scene token set and a delta-update set per chunk. |
| **M1 (DyMU spatial merge)** | `src/inference/stamp_temporal.py` (`merge_*` paths) | Cosine-bipartite spatial token merging within a frame, executed before temporal scoring. |
| **FOCUS** | `src/inference/spatial_focus.py` | Text-guided per-frame spatial pruning. Scores visual tokens by their cross-attention to the text query (no extra parameters, training-free). |
| **Headline stack: M1 + STAMP-T (r=0.30) + TAST** | composed via `streaming_args.py` flags | The configuration that produces the +4.42 pp cross-model win on Qwen2.5-VL-7B-Instruct (see Results). |

The M1+TAST stack architecture is rendered in `diagrams/m1_tast_stack.html` (open in any browser).

### Phase-7 long-video attempts (negative result, kept for transparency)

`stamp_temporal.py` also contains three training-free long-video variants attempted to close the −7 pp loss on MLVU plotQA. All three failed; see paper §Discussion. CLI flags:

| Variant | Flag set |
|---|---|
| A — Hierarchical TAST | `--tast_hierarchical --tast_gamma_long 0.01 --tast_segment_len 8` |
| B — Adaptive γ | `--tast_adaptive_gamma --tast_gamma_tau 40.0` |
| C — Keep-ratio ramp | `--stamp_ratio_ramp --stamp_ratio_ramp_early 0.15 --stamp_ratio_ramp_late 0.45 --stamp_ratio_ramp_chunks 30` |

Variant A produced byte-identical output to the control (no behavior change), B was within bootstrap CI of zero, and C / A+C regressed by −1.85 pp.

### Dropped methods (intentionally not in this PR)

CRISP, PRISM, STAR, STAMP-T+ (the 14-variant sweep), Video-CDPruner — all underperformed plain STAMP-T at every keep ratio in our sweeps and were cut from the paper. They live in `streaming_vlm/inference/archived_dropped_methods/` on the development cluster but are not shipped here.

---

## Headline results (paired bootstrap, 95% CI)

| Comparison | n | Δ | 95% CI | Verdict |
|---|---|---|---|---|
| **Cross-model DeViBench (Qwen2.5-VL-7B-Instruct)** vs StreamingVLM-vanilla 82.08% | 652 | **+4.42 pp** | [+1.82, +6.88] | **sig. win, training-free transfer (86.50%)** |
| **MVBench paired** stack r=0.30 vs vanilla | 1325 | **+3.55 pp** | [+1.43, +5.58] | **sig. win** |
| StreamingVLM same-model DeViBench stack r=0.30 vs vanilla 82.08% | 652 | +0.13 pp | [−2.94, +3.04] | in noise (compute-equivalent at ~30% LLM tokens) |
| MLVU plotQA stack r=0.30 vs unpruned vanilla 66.00% | 500 | −7.00 pp | [−10.00, −4.20] | sig. loss (long video) |

Single-method DeViBench (n=652, vanilla 82.08%):

| Method | Accuracy | Δ |
|---|---|---|
| STAMP-T r=0.90 multi-layer | 82.98% | +0.90 |
| FOCUS r=0.85 (text-guided) | 82.98% | +0.90 |
| FOCUS + STAMP-T composition | 82.98% | +0.90 (no orthogonal lift) |

See `paper/draft.pdf` §4 for full result tables and `figures/pareto_devibench.pdf` for the FLOPs/accuracy trade-off curve.

---

## Reproducing the headline result

**Hardware:** A100 80 GB (one is enough). All methods are training-free; only inference is run.

**Environment:** Python 3.11, PyTorch 2.6, transformers 4.45+, flash-attn 2.7+, Qwen2.5-VL-7B-Instruct weights from HuggingFace.

**DeViBench (cross-model, headline):**

```bash
python src/eval/DeViBench/eval_devi_360.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --data_dir /path/to/DeViBench \
    --stamp_temporal --stamp_temporal_r 0.30 --stamp_temporal_no_adaptive_r \
    --merge_enabled --merge_threshold 0.85 \
    --tast_enabled --tast_state_tokens 32 --tast_gamma 0.1 \
    --output results/devi_stack_r030_qwen.json
```

**MVBench paired:**

```bash
python src/eval/MVBench/evaluate_mvbench.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --stamp_temporal --stamp_temporal_r 0.30 --stamp_temporal_no_adaptive_r \
    --merge_enabled --merge_threshold 0.85 \
    --tast_enabled --tast_state_tokens 32 --tast_gamma 0.1 \
    --output results/mvbench_stack_r030.json
```

**MLVU plotQA (with Phase-7 variants):**

```bash
python src/eval/MLVU/evaluate_mlvu.py \
    --tasks plotQA \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --stamp_temporal --stamp_temporal_r 0.30 --stamp_temporal_no_adaptive_r \
    --merge_enabled --tast_enabled --tast_state_tokens 32 --tast_gamma 0.1 \
    --tast_hierarchical --tast_gamma_long 0.01 --tast_segment_len 8 \
    --output results/mlvu_plotqa_phase7_A.json
```

**Bootstrap CI on a result JSON:**

```bash
python src/analysis/bootstrap_ci.py \
    --treatment results/devi_stack_r030_qwen.json \
    --control results/devi_vanilla_qwen.json \
    --n_boot 10000
```

---

## Status & roadmap

- **Done:** all 6 paper phases (vanilla ceilings, cheap fixes, major rewrites incl. M1/M3/M5/N7/TAST, test-time scaling at compute parity, cross-benchmark/cross-model robustness, paper assembly).
- **Done (negative):** Phase-7 long-video training-free fixes — none closed the −7 pp MLVU plotQA gap.
- **Deferred to camera-ready:** NextQA, TempCompass.
- **Open:** trainable per-chunk gate or learned-γ TAST for long video; not attempted in the training-free regime.

---

## Contributors

- **Sri Sashank Potluru** — `sashank.potluru22@gmail.com` — GitHub [@Sashankpotluru](https://github.com/Sashankpotluru)
- **Venkata Akhil Akkineni** — co-developer, M1 merge + Phase-7 evaluator wiring

All commits in this PR are co-authored.

---

## License

MIT, inherited from the upstream StreamingVLM project (MIT HAN Lab). See `LICENSE`.
