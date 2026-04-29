class StreamingArgs:
    def __init__(self, pos_mode: str, all_text: bool = False,
                 fastv_k: int = None, fastv_r: float = 0.5,
                 stamp_r1: float = None, stamp_alpha: float = 0.5,
                 stamp_lambda: float = 0.3, stamp_K: int = 10):
        self.pos_mode = pos_mode
        self.all_text = all_text
        assert pos_mode in ["append", "shrink"], "pos_mode must be in ['append', 'shrink']"
        # append mode grows indefinitely
        # shrink mode ensures current kv cache position ids are continuous
        self.input_ids = None    # shrink mode needs complete input ids passed to attention to detect video frames
        self.video_grid_thw = None
        self.second_per_grid_ts = None
        # FastV visual token pruning
        # fastv_k: layer index at which to prune (capture attention at fastv_k-1, prune before fastv_k)
        # fastv_r: fraction of visual tokens to KEEP (e.g. 0.5 = keep top 50% by attention score)
        assert fastv_k is None or fastv_k >= 1, "fastv_k must be >= 1 (need at least one prior layer)"
        self.fastv_k = fastv_k
        self.fastv_r = fastv_r
        # Qwen2.5-VL visual placeholder token IDs (<|video_pad|>=151656, <|image_pad|>=151655)
        self.fastv_visual_token_ids = [151656, 151655]
        # Set at runtime by model_forward so language_forward can identify visual positions
        self.current_input_ids = None

        # ── STAMP: Streaming Temporal Attention Momentum Pruning ──────────────
        # stamp_r1:     Stage 1 keep ratio (None = STAMP disabled)
        # stamp_alpha:  weight mixing momentum vs novelty (alpha*M + (1-alpha)*N)
        # stamp_lambda: EMA decay for attention momentum
        # stamp_K:      keyframe interval — every K chunks, skip Stage 1 entirely
        self.stamp_r1 = stamp_r1
        self.stamp_alpha = stamp_alpha
        self.stamp_lambda = stamp_lambda
        self.stamp_K = stamp_K
        # Runtime state (updated each chunk by stamp_update_state)
        self.chunk_idx = 0                    # total chunks processed so far
        self.attention_momentum = None        # [N_vis] EMA of FastV attention scores
        self.prev_visual_feats = None         # [N_vis, D] vision encoder output, last chunk
        # Intra-chunk scratch (set by model_forward / language_forward, cleared after use)
        self.stamp_curr_visual_feats = None   # [N_vis, D] vision encoder output, current chunk
        self.stamp_last_attn_scores = None    # [N_vis] FastV attention from last chunk
        self.stamp_kept_vis_local_indices = None  # [keep_k] which local vis indices survived Stage 1
        self.stamp_n_vis = None               # N_vis for current chunk
        self.stamp_is_keyframe = False
        # When True, pre-hook skips STAMP reset (used during multi-chunk streaming eval)
        self.stamp_no_reset = False        # whether current chunk is a keyframe

        # ── STAMP Iteration 2: four optional improvements ─────────────────────
        # Idea 1 — Adaptive pruning ratio: adjust r1 based on scene motion
        self.stamp_adaptive_r1 = False        # enable adaptive r1
        self.stamp_adaptive_r1_high = 0.3     # novelty > this → keep r1+0.25 (dynamic scene)
        self.stamp_adaptive_r1_low = 0.1      # novelty < this → keep r1-0.25 (static scene)

        # Idea 2 — Momentum decay for pruned tokens: faster decay when absent
        self.stamp_momentum_decay = False     # enable differential decay
        self.stamp_gamma = 0.5               # decay rate for pruned tokens (< 1-lambda = 0.7)

        # Idea 3 — Adaptive keyframe detection: scene-cut trigger instead of fixed K
        self.stamp_adaptive_kf = False        # enable adaptive keyframe detection
        self.stamp_adaptive_kf_threshold = 0.5  # avg novelty > this → treat as keyframe

        # Idea 4 — Hierarchical momentum: dual-timescale EMA (short + long term)
        self.stamp_hierarchical = False       # enable dual-timescale momentum
        self.stamp_lambda_long = 0.1          # slow EMA decay rate (vs stamp_lambda for fast)
        self.stamp_alpha_short = 0.35         # weight for short-term momentum
        self.stamp_alpha_long = 0.35          # weight for long-term momentum
        # (1 - alpha_short - alpha_long) = weight for temporal novelty = 0.30
        self.attention_momentum_long = None   # [N_vis] slow EMA state

        # ── STAMP Iteration 3: Token merging (replaces hard pruning) ─────
        self.stamp_merge = False              # True = merge pruned tokens; False = hard prune (original)

        # Running normalization stats for stable scoring
        self.stamp_running_M_mean = None      # EMA of momentum mean
        self.stamp_running_M_std = None       # EMA of momentum std
        self.stamp_running_N_mean = None      # EMA of novelty mean
        self.stamp_running_N_std = None       # EMA of novelty std

        # ── STAMP-Temporal: Pure temporal pruning with ViT-sourced attention ──
        # stamp_temporal: enables the new ViT-sourced temporal pruning (mutually exclusive with stamp_r1)
        self.stamp_temporal = False
        # Which ViT layer to extract attention from (default 31 = last global attention layer)
        # fullatt_block_indexes = [7, 15, 23, 31] — only global attention layers are useful
        self.stamp_temporal_vit_layer = 31
        # Multi-layer fusion: extract from multiple global layers and fuse salience
        # Set to list e.g. [7, 15, 23, 31] for multi-layer; None = single-layer mode
        self.stamp_temporal_vit_layers = None
        # Keep ratio (same semantics as stamp_r1)
        self.stamp_temporal_r = None          # None = disabled
        # Score fusion: alpha * M_vit_norm + (1-alpha) * N_norm
        self.stamp_temporal_alpha = 0.5
        # EMA decay for ViT attention momentum
        self.stamp_temporal_lambda = 0.3
        # Keyframe interval
        self.stamp_temporal_K = 10
        # Entropy-adaptive ratio parameters
        self.stamp_temporal_adaptive_r = True  # enable entropy-adaptive keep ratio
        self.stamp_temporal_r_base = 0.85      # base keep ratio
        self.stamp_temporal_beta_r = 0.1       # entropy sensitivity
        self.stamp_temporal_r_min = 0.5        # minimum keep ratio
        self.stamp_temporal_r_max = 1.0        # maximum keep ratio
        # Length-adaptive keep ratio: increase r for longer videos
        # r_effective = r_base + length_boost * min(chunk_idx / length_warmup, 1.0)
        # This preserves more tokens as video gets longer (more unique content)
        self.stamp_temporal_length_adaptive = False
        self.stamp_temporal_length_boost = 0.10    # max boost to r (e.g. 0.85 -> 0.95)
        self.stamp_temporal_length_warmup = 20     # chunks to reach full boost
        # Compress-not-discard: mean-pool pruned tokens into summary
        self.stamp_temporal_compress = False
        # Token merging: merge pruned tokens into nearest kept token instead of dropping
        self.stamp_temporal_merge = False
        # ── STAMP-T+ (A1+A2+A3): rank-percentile score, text-relevance quality, MMR selection ──
        self.stamp_temporal_plus = False                  # master flag
        self.stamp_temporal_plus_text_weight = 0.3        # weight of text-relevance in quality term (A2)
        self.stamp_temporal_plus_mmr_beta = 0.5           # MMR similarity-penalty coefficient (A1)
        self.stamp_temporal_plus_frame_strata = 0        # A4: per-frame stratified top-k. 0=disabled; K>0 splits vis tokens into K contiguous chunks and allocates equal budget per chunk (temporal coverage guard).

        # ── TAST: Temporal Accumulative State Tokens ──
        # Instead of discarding pruned tokens, accumulate them into persistent
        # state tokens that carry temporal context across all frames.
        self.tast_enabled = False
        self.tast_n_tokens = 32      # Number of state tokens (K)
        self.tast_gamma = 0.1        # EMA decay for state token updates
        # TAST-only mode: no pruning, just inject state tokens from previous chunks
        # This tests whether TAST can improve the streaming baseline itself
        self.tast_only = False        # When True: no pruning, inject K state tokens as extra visual tokens

        # ── DSTM: Delta-State Temporal Memory (Architecture 5) ──
        # Dual-memory system: scene tokens (what scene looks like) + delta tokens (what changed)
        # Surprise-gated updates: only accumulate novel information
        self.dstm_enabled = False
        self.dstm_scene_tokens = 16   # K_s: number of scene memory slots
        self.dstm_delta_tokens = 16   # K_d: number of delta memory slots
        self.dstm_gamma_scene = 0.05  # slow EMA for scene state (long-range memory)
        self.dstm_gamma_delta = 0.2   # fast EMA for delta state (recent changes)
        self.dstm_surprise_beta = 5.0 # surprise gate sensitivity
        self.dstm_surprise_tau = 0.3  # surprise gate threshold
        self.dstm_only = False        # True = no pruning, inject memory into full stream
        self.dstm_blend_alpha = 0.2   # Blend ratio: 0=no injection, 1=full replacement, 0.2=20% memory

        # ── TAST blend alpha: additive blending instead of replacement ──
        self.tast_blend_alpha = 0.2   # Blend ratio for TAST injection (0=no inject, 1=replace)

        # ── CRISP: Clustering-based Regional Importance Spatial Pruning ──
        self.crisp_enabled = False
        self.crisp_r = 0.85             # fraction of visual tokens to keep
        self.crisp_grid_size = 4        # grid cells per side (4×4 = 16 clusters)

        # ── FOCUS: Feature-Optimized Conditioned Unification for Spatial selection ──
        self.focus_enabled = False
        self.focus_r = 0.85             # fraction of visual tokens to keep
        self.focus_enhance_alpha = 0.1  # enhancement strength from pruned neighbors
        self.focus_text_weight = 0.6    # weight for text vs ViT salience (0=ViT only, 1=text only)

        # ── Video-CDPruner: joint spatio-temporal DPP conditional diversity ──
        # CDPruner (arxiv 2506.10967) adapted to streaming video VLMs.
        self.video_cdpruner_enabled = False
        self.video_cdpruner_r = 0.20              # fraction of visual tokens to keep
        self.video_cdpruner_text_weight = 0.6     # text vs ViT salience weight for quality term
        self.video_cdpruner_theta = 0.0           # quality-warp strength; 0 disables the exp() warp
        self.video_cdpruner_ablation = "none"     # "none" | "quality_only" | "diversity_only" | "per_frame"
        self.video_cdpruner_n_frames = 3          # only used when ablation == "per_frame"

        # ── PRISM: Progressive Resolution with Integrated Spatial Management ──
        self.prism_enabled = False
        self.prism_r = 0.85            # overall keep ratio
        self.prism_fine_ratio = 0.5    # fraction of budget for fine-grained tokens
        self.prism_pool_size = 2       # pooling factor for coarse level
        self.prism_enhance_alpha = 0.1 # cross-resolution enhancement strength

        # STAR — Salience-Tuned Adaptive Refinement (Temporal Only)
        self.star_enabled = False       # Enable STAR scoring refinement
        self.star_tc_weight = 0.3       # Temporal consistency sharpening weight
        self.star_gamma = 0.2           # Per-layer divergence boost weight
        self.star_salience_var = None   # Runtime: EMA of per-token salience variance

        # Runtime state for STAMP-Temporal
        self.stamp_temporal_vit_momentum = None     # [N_vis] EMA of ViT salience scores
        self.stamp_temporal_vit_salience = None      # [N_vis] ViT salience from current chunk
        self.stamp_temporal_vit_entropy = None       # scalar: frame attention entropy
        self.stamp_temporal_running_H_mean = None    # EMA of entropy mean
        self.stamp_temporal_running_H_std = None     # EMA of entropy std
        self.stamp_temporal_summary_tokens = []      # list of [1, D] summary tensors
