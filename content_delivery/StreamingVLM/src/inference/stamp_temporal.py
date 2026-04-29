"""
STAMP-Temporal — Pure Temporal Pruning with ViT-Sourced Attention + TAST
=========================================================================
Two-part architecture:

PART 1 (STAMP-Temporal): ViT-sourced multi-signal temporal pruning.
  - Attention source: ViT encoder (multi-layer or single layer)
  - Language-bias-free, no cold start
  - Entropy-adaptive keep ratio

PART 2 (TAST — Temporal Accumulative State Tokens): Novel technique.
  Instead of permanently discarding pruned tokens, TAST maintains K persistent
  state tokens that accumulate information from pruned tokens across ALL frames
  via exponential moving average. State tokens are injected into the LLM input,
  providing temporal context that spans the entire video history.

  Key properties:
  - Training-free: EMA accumulation, no learnable parameters
  - State tokens live in the same embedding space as visual features
  - The LLM processes them as regular visual tokens
  - Unlike VideoLLaMB (requires trained bridge layers) or token merging
    (within-frame only), TAST accumulates ACROSS frames into persistent state

Architecture:
    Frame_t → ViT → [Multi-Signal] → [Adaptive Pruning] → Kept (k tokens)
                                            ↓                    ↓
                                       Pruned (N-k)             ↓
                                            ↓                    ↓
                              [TAST: State Token Accumulation]  ↓
                                       ↓                         ↓
                                  State Tokens (K)              ↓
                                       ↓                         ↓
                              [Inject: replace K pruned positions with state tokens]
                                            ↓
                                    LLM Decoder → Answer

Usage:
    streaming_args.stamp_temporal = True
    streaming_args.stamp_temporal_r = 0.85
    streaming_args.tast_enabled = True       # Enable TAST
    streaming_args.tast_n_tokens = 32        # Number of state tokens
    streaming_args.tast_gamma = 0.1          # EMA decay for state update
"""

import math
import torch
import torch.nn.functional as F


def _tast_effective_gamma(sa, base_gamma):
    """
    Phase-7B: Adaptive γ schedule for TAST.
    γ(t) = γ0 · exp(−t/τ) when --tast_adaptive_gamma is set; else flat base_gamma.
    Preserves early plot-setup chunks (high γ) and locks state on late chunks (low γ).
    """
    if not getattr(sa, 'tast_adaptive_gamma', False):
        return base_gamma
    tau = float(getattr(sa, 'tast_gamma_tau', 40.0))
    t = float(getattr(sa, 'chunk_idx', 0) or 0)
    if tau <= 0:
        return base_gamma
    return float(base_gamma) * math.exp(-t / tau)


def _tast_maybe_update_long_pool(sa, device):
    """
    Phase-7A: Hierarchical TAST — update the long-scale EMA pool.
    Every `tast_segment_len` chunks we EMA the short pool into the long pool with
    slow rate γ_long (default 0.01). The long pool therefore carries a coarser
    timescale than the short pool, combating EMA saturation on long videos
    (MLVU plotQA: short pool alone converges to near-uniform after ~60 chunks).
    No-op when --tast_hierarchical is False, preserving existing TAST behaviour.
    """
    if not getattr(sa, 'tast_hierarchical', False):
        return
    if sa.tast_state_tokens is None:
        return
    segment_len = int(getattr(sa, 'tast_segment_len', 8) or 8)
    if segment_len <= 0:
        return
    chunk_idx = int(getattr(sa, 'chunk_idx', 0) or 0)
    last = int(getattr(sa, 'tast_last_long_update_chunk', -1) or -1)
    # Gate by segment boundary: update only when chunk_idx advanced by ≥ segment_len
    if chunk_idx - last < segment_len:
        return
    sa.tast_last_long_update_chunk = chunk_idx

    short = sa.tast_state_tokens.to(device).float()  # [K, D]
    if sa.tast_state_tokens_long is None:
        # Seed long pool from current short pool
        sa.tast_state_tokens_long = short.detach().cpu()
        return
    gamma_long = float(getattr(sa, 'tast_gamma_long', 0.01))
    long_state = sa.tast_state_tokens_long.to(device).float()
    long_state = (1.0 - gamma_long) * long_state + gamma_long * short
    sa.tast_state_tokens_long = long_state.detach().cpu()


def _tast_build_inject_pool(sa, K_inject, device, dtype):
    """
    Phase-7A: Build the [K_inject, D] tensor to inject into pruned positions.
    Non-hierarchical: returns first K_inject rows of the short pool (existing behaviour).
    Hierarchical: interleaves short + long pool tokens (half each) so the LLM sees
    both recency-weighted and long-horizon memory in one pass.
    """
    short_pool = sa.tast_state_tokens[:K_inject].to(device).to(dtype)
    if not getattr(sa, 'tast_hierarchical', False) or sa.tast_state_tokens_long is None:
        return short_pool

    K_long_req = K_inject // 2
    K_short_req = K_inject - K_long_req
    K_long_avail = sa.tast_state_tokens_long.shape[0]
    K_long = min(K_long_req, K_long_avail)
    K_short = K_inject - K_long
    short_part = sa.tast_state_tokens[:K_short].to(device).to(dtype)
    long_part = sa.tast_state_tokens_long[:K_long].to(device).to(dtype)
    # Interleave: [s0, l0, s1, l1, ...] so temporal scales are spatially mixed at inject
    pool = torch.empty((K_inject, short_part.shape[-1]), device=device, dtype=dtype)
    n_inter = min(K_short, K_long)
    pool[0:2 * n_inter:2] = short_part[:n_inter]
    pool[1:2 * n_inter:2] = long_part[:n_inter]
    # Remainder: whichever side had extra
    if K_short > n_inter:
        pool[2 * n_inter:2 * n_inter + (K_short - n_inter)] = short_part[n_inter:]
    if K_long > n_inter:
        pool[2 * n_inter + (K_short - n_inter):] = long_part[n_inter:]
    return pool


def stamp_temporal_reset_state(sa):
    """Reset all STAMP-Temporal runtime state between videos."""
    sa.stamp_temporal_vit_momentum = None
    sa.stamp_temporal_vit_salience = None
    sa.stamp_temporal_vit_entropy = None
    sa.prev_visual_feats = None
    sa.stamp_curr_visual_feats = None
    sa.chunk_idx = 0
    sa.stamp_kept_vis_local_indices = None
    sa.stamp_n_vis = None
    sa.stamp_is_keyframe = False
    sa.stamp_temporal_running_H_mean = None
    sa.stamp_temporal_running_H_std = None
    sa.stamp_running_M_mean = None
    sa.stamp_running_M_std = None
    sa.stamp_running_N_mean = None
    sa.stamp_running_N_std = None
    sa.stamp_temporal_summary_tokens = []
    # TAST state (short-scale, existing)
    sa.tast_state_tokens = None      # [K, D] persistent state tokens
    sa.tast_state_counts = None      # [K] how many tokens accumulated per state
    # Phase-7A: Hierarchical TAST — second long-scale EMA pool
    sa.tast_state_tokens_long = None    # [K, D] slow-EMA pool, updated every segment_len chunks
    sa.tast_last_long_update_chunk = -1 # last chunk idx at which long pool was updated
    # STAR state
    sa.star_salience_var = None       # [N_vis] EMA of salience variance
    # DSTM state (Architecture 5)
    dstm_enabled = getattr(sa, 'dstm_enabled', False)
    if dstm_enabled:
        from streaming_vlm.inference.dstm import dstm_reset_state
        dstm_reset_state(sa)


def stamp_temporal_stage1_prune(inputs_embeds, position_ids, attention_mask,
                                 cache_position, streaming_args):
    """
    Pre-LLM Stage 1 pruning using ViT-sourced attention signals.

    Called at the entry of streaming_language_model_forward, same location as
    the original stamp_stage1_prune. Uses ViT salience (extracted during
    vision encoder forward) instead of LLM attention from the previous chunk.

    Returns:
        inputs_embeds   — [1, new_seq_len, D]
        position_ids    — [3, 1, new_seq_len]
        attention_mask  — [1, new_seq_len]
        cache_position  — [new_seq_len]
    """
    sa = streaming_args

    # Only prune on initial prefill
    if cache_position is None or cache_position[0] != 0:
        return inputs_embeds, position_ids, attention_mask, cache_position

    if sa.current_input_ids is None:
        return inputs_embeds, position_ids, attention_mask, cache_position

    input_ids = sa.current_input_ids  # [1, seq_len]
    device = inputs_embeds.device

    # Find visual token positions
    vis_token_ids = torch.tensor(sa.fastv_visual_token_ids, device=device)
    vis_mask = torch.isin(input_ids[0], vis_token_ids)
    vis_indices = vis_mask.nonzero(as_tuple=True)[0]
    N_vis = vis_indices.shape[0]

    if N_vis == 0:
        return inputs_embeds, position_ids, attention_mask, cache_position

    sa.stamp_n_vis = N_vis

    # ── Keyframe check ────────────────────────────────────────────────────
    is_keyframe = (sa.stamp_temporal_vit_momentum is None) or (sa.chunk_idx % sa.stamp_temporal_K == 0)
    sa.stamp_is_keyframe = is_keyframe

    if is_keyframe:
        sa.stamp_kept_vis_local_indices = torch.arange(N_vis, device=device)

        # TAST keyframe: initialize state tokens from this frame's visual features
        # This happens on the FIRST chunk — seed state tokens from ViT output
        tast_enabled = getattr(sa, 'tast_enabled', False)
        if tast_enabled and sa.tast_state_tokens is None:
            tast_K = getattr(sa, 'tast_n_tokens', 32)
            vis_embeds = inputs_embeds[0, vis_indices, :].float()  # [N_vis, D]
            K_actual = min(tast_K, N_vis)
            # Initialize from evenly-spaced visual tokens
            seed_idx = torch.linspace(0, N_vis - 1, K_actual, device=device).long()
            sa.tast_state_tokens = vis_embeds[seed_idx].detach().cpu()
            sa.tast_state_counts = torch.ones(K_actual)
        elif tast_enabled and sa.tast_state_tokens is not None and sa.chunk_idx > 0:
            # On subsequent keyframes, update state tokens from ALL visual features
            # This is the key: state tokens accumulate even on keyframes (no pruning)
            tast_gamma = _tast_effective_gamma(sa, getattr(sa, 'tast_gamma', 0.1))
            tast_K = sa.tast_state_tokens.shape[0]
            vis_embeds = inputs_embeds[0, vis_indices, :].float()  # [N_vis, D]
            state_on_device = sa.tast_state_tokens.to(device)
            state_norm = F.normalize(state_on_device, dim=-1)
            vis_norm = F.normalize(vis_embeds, dim=-1)
            sim = torch.mm(vis_norm, state_norm.t())  # [N_vis, K]
            assignments = sim.argmax(dim=1)
            for ki in range(tast_K):
                mask_ki = (assignments == ki)
                if mask_ki.any():
                    cluster_mean = vis_embeds[mask_ki].mean(dim=0)
                    state_on_device[ki] = (1.0 - tast_gamma) * state_on_device[ki] + tast_gamma * cluster_mean
            sa.tast_state_tokens = state_on_device.detach().cpu()

            # Phase-7A: Hierarchical TAST — also update long-scale pool on segment boundaries
            _tast_maybe_update_long_pool(sa, device)

        # DSTM: update memory on keyframe
        dstm_enabled = getattr(sa, 'dstm_enabled', False)
        if dstm_enabled:
            from streaming_vlm.inference.dstm import dstm_keyframe_update
            dstm_keyframe_update(inputs_embeds, vis_indices, streaming_args)

        return inputs_embeds, position_ids, attention_mask, cache_position

    # ── DSTM-Only Mode: inject scene+delta memory WITHOUT pruning ────────
    dstm_only = getattr(sa, 'dstm_only', False)
    dstm_enabled_here = getattr(sa, 'dstm_enabled', False)
    if dstm_only and dstm_enabled_here:
        from streaming_vlm.inference.dstm import dstm_process_frame
        dstm_process_frame(inputs_embeds, vis_indices, streaming_args, is_keyframe=False)
        sa.stamp_kept_vis_local_indices = torch.arange(N_vis, device=device)
        return inputs_embeds, position_ids, attention_mask, cache_position

    # ── TAST-Only Mode: inject state tokens WITHOUT pruning ──────────────
    # Replace lowest-salience visual tokens with temporal state tokens
    # This can improve even the unpruned streaming baseline
    tast_only = getattr(sa, 'tast_only', False)
    tast_enabled_here = getattr(sa, 'tast_enabled', False)
    if tast_only and tast_enabled_here and sa.tast_state_tokens is not None:
        tast_K = sa.tast_state_tokens.shape[0]
        tast_gamma = _tast_effective_gamma(sa, getattr(sa, 'tast_gamma', 0.1))
        vis_embeds = inputs_embeds[0, vis_indices, :].float()  # [N_vis, D]

        # Update state tokens from current frame's visual features
        state_on_device = sa.tast_state_tokens.to(device)
        state_norm = F.normalize(state_on_device, dim=-1)
        vis_norm = F.normalize(vis_embeds, dim=-1)
        sim = torch.mm(vis_norm, state_norm.t())  # [N_vis, K]
        assignments = sim.argmax(dim=1)
        for ki in range(tast_K):
            mask_ki = (assignments == ki)
            if mask_ki.any():
                cluster_mean = vis_embeds[mask_ki].mean(dim=0)
                state_on_device[ki] = (1.0 - tast_gamma) * state_on_device[ki] + tast_gamma * cluster_mean
        sa.tast_state_tokens = state_on_device.detach().cpu()

        # Phase-7A: Hierarchical TAST — also update long-scale pool
        _tast_maybe_update_long_pool(sa, device)

        # Inject: blend state tokens into lowest-salience visual tokens
        # Uses additive blending to preserve original ViT features
        vit_salience = sa.stamp_temporal_vit_salience
        if vit_salience is not None and vit_salience.shape[0] == N_vis:
            blend_alpha = getattr(sa, 'tast_blend_alpha', 0.2)
            salience_device = vit_salience.to(device)
            K_inject = min(tast_K, N_vis // 10)  # Reduced from 25% to 10%
            _, bottom_idx = salience_device.topk(K_inject, largest=False)
            inject_seq_pos = vis_indices[bottom_idx]
            state_to_inject = _tast_build_inject_pool(sa, K_inject, device, inputs_embeds.dtype)
            original = inputs_embeds[0, inject_seq_pos, :]
            inputs_embeds[0, inject_seq_pos, :] = (
                (1.0 - blend_alpha) * original + blend_alpha * state_to_inject
            )

        sa.stamp_kept_vis_local_indices = torch.arange(N_vis, device=device)
        return inputs_embeds, position_ids, attention_mask, cache_position

    # ── ViT Salience Momentum Update ─────────────────────────────────────
    # Use ViT salience from the CURRENT chunk (available immediately after ViT forward)
    vit_salience = sa.stamp_temporal_vit_salience  # [N_vis] from vision_forward
    if vit_salience is not None:
        vit_salience = vit_salience.to(device)
        if vit_salience.shape[0] == N_vis:
            M_prev = sa.stamp_temporal_vit_momentum.to(device)
            sa.stamp_temporal_vit_momentum = (
                (1.0 - sa.stamp_temporal_lambda) * M_prev
                + sa.stamp_temporal_lambda * vit_salience
            )
        else:
            # Shape mismatch → treat as keyframe
            sa.stamp_kept_vis_local_indices = torch.arange(N_vis, device=device)
            return inputs_embeds, position_ids, attention_mask, cache_position
    else:
        # No salience available → treat as keyframe
        sa.stamp_kept_vis_local_indices = torch.arange(N_vis, device=device)
        return inputs_embeds, position_ids, attention_mask, cache_position

    M_t = sa.stamp_temporal_vit_momentum.to(device)  # [N_vis]

    # Shape mismatch guard
    if M_t.shape[0] != N_vis:
        sa.stamp_kept_vis_local_indices = torch.arange(N_vis, device=device)
        return inputs_embeds, position_ids, attention_mask, cache_position

    # ── Temporal novelty: N_t = 1 - cosine_sim(V_t[i], V_{t-1}[i]) ──────
    if (
        sa.stamp_curr_visual_feats is not None
        and sa.prev_visual_feats is not None
        and sa.stamp_curr_visual_feats.shape == sa.prev_visual_feats.shape
    ):
        V_t = sa.stamp_curr_visual_feats.float().to(device)
        V_prev = sa.prev_visual_feats.float().to(device)
        cos_sim = F.cosine_similarity(V_t, V_prev, dim=-1)
        N_t = (1.0 - cos_sim).clamp(0.0, 1.0)
    else:
        N_t = torch.ones(N_vis, device=device)

    # ── EMA Z-Score Normalization ────────────────────────────────────────
    norm_ema_rate = 0.3

    def _ema_normalize(x, running_mean, running_std, ema_rate):
        cur_mean = x.mean()
        cur_std = x.std().clamp(min=1e-8)
        if running_mean is None:
            new_mean, new_std = cur_mean, cur_std
        else:
            new_mean = (1 - ema_rate) * running_mean + ema_rate * cur_mean
            new_std = (1 - ema_rate) * running_std + ema_rate * cur_std
        z = (x - new_mean) / new_std.clamp(min=1e-8)
        return torch.sigmoid(z), new_mean.item(), new_std.item()

    M_norm, sa.stamp_running_M_mean, sa.stamp_running_M_std = _ema_normalize(
        M_t, sa.stamp_running_M_mean, sa.stamp_running_M_std, norm_ema_rate
    )
    N_norm, sa.stamp_running_N_mean, sa.stamp_running_N_std = _ema_normalize(
        N_t, sa.stamp_running_N_mean, sa.stamp_running_N_std, norm_ema_rate
    )

    # ── Score Fusion ─────────────────────────────────────────────────────
    alpha = sa.stamp_temporal_alpha
    score_t = alpha * M_norm + (1.0 - alpha) * N_norm

    # ── STAMP-T+ (A1+A2+A3): rank-norm override + text-relevance quality term ──
    # Rationale (see memory project_stampt_retention_gap_analysis.md):
    #   A3: sigmoid(z) compresses the score head into [0.5,1.0]; at aggressive r
    #       the top-k head needs wide dynamic range. Rank-percentile restores it.
    #   A2: visual-only scoring is blind to the question; at aggressive r every
    #       slot must be question-relevant. Inject cross-attn-style text relevance.
    if getattr(sa, 'stamp_temporal_plus', False):
        def _rank_pct(x):
            # Double argsort = rank; normalize to (0,1]
            ranks = torch.argsort(torch.argsort(x)).float()
            return (ranks + 1.0) / float(x.numel())
        M_rank = _rank_pct(M_t)
        N_rank = _rank_pct(N_t)
        score_t = alpha * M_rank + (1.0 - alpha) * N_rank

        tw = getattr(sa, 'stamp_temporal_plus_text_weight', 0.3)
        if tw > 0.0:
            vis_end_plus = vis_indices[-1].item() + 1
            text_embeds_plus = inputs_embeds[0, vis_end_plus:, :].float()
            if text_embeds_plus.shape[0] > 0:
                vis_embeds_plus = inputs_embeds[0, vis_indices, :].float()
                vis_norm_plus = F.normalize(vis_embeds_plus, dim=-1)
                txt_norm_plus = F.normalize(text_embeds_plus, dim=-1)
                # Quality = max cosine over text tokens (FOCUS-style)
                text_rel = torch.mm(vis_norm_plus, txt_norm_plus.t()).max(dim=1).values
                text_rel_rank = _rank_pct(text_rel)
                score_t = (1.0 - tw) * score_t + tw * text_rel_rank

    # ── Entropy-Adaptive Keep Ratio ──────────────────────────────────────
    if sa.stamp_temporal_adaptive_r and sa.stamp_temporal_vit_entropy is not None:
        H_t = sa.stamp_temporal_vit_entropy.item()
        H_mean = sa.stamp_temporal_running_H_mean
        H_std = sa.stamp_temporal_running_H_std

        # Update running entropy stats
        if H_mean is None:
            sa.stamp_temporal_running_H_mean = H_t
            sa.stamp_temporal_running_H_std = 1.0
            H_mean = H_t
            H_std = 1.0
        else:
            sa.stamp_temporal_running_H_mean = (1 - norm_ema_rate) * H_mean + norm_ema_rate * H_t
            sa.stamp_temporal_running_H_std = max(
                (1 - norm_ema_rate) * H_std + norm_ema_rate * abs(H_t - H_mean),
                1e-8
            )
            H_mean = sa.stamp_temporal_running_H_mean
            H_std = sa.stamp_temporal_running_H_std

        r_t = sa.stamp_temporal_r_base + sa.stamp_temporal_beta_r * (H_t - H_mean) / max(H_std, 1e-8)
        r_t = max(sa.stamp_temporal_r_min, min(sa.stamp_temporal_r_max, r_t))
    else:
        r_t = sa.stamp_temporal_r if sa.stamp_temporal_r is not None else 0.85

    # ── Length-Adaptive Keep Ratio ────────────────────────────────────────
    # For longer videos, gradually increase keep ratio to preserve fine details.
    # Long videos have more diverse content (needle-in-haystack, temporal ordering)
    # that gets lost with aggressive pruning.
    if getattr(sa, "stamp_temporal_length_adaptive", False):
        length_boost = getattr(sa, "stamp_temporal_length_boost", 0.10)
        length_warmup = getattr(sa, "stamp_temporal_length_warmup", 20)
        progress = min(sa.chunk_idx / max(length_warmup, 1), 1.0)
        r_t = r_t + length_boost * progress
        r_t = min(r_t, getattr(sa, "stamp_temporal_r_max", 1.0))

    # ── Phase-7C: Keep-ratio ramp (long-video override) ──────────────────
    # Hard override: r_t is linearly interpolated from r_early to r_late over the
    # first `ramp_chunks` chunks, then held at r_late for the remainder. Designed
    # to dedicate budget to plot-critical late context on long-video benchmarks
    # (MLVU plotQA). OVERRIDES entropy-adaptive + length-adaptive paths so that
    # the sweep cleanly isolates ramp effect. OFF by default.
    if getattr(sa, "stamp_ratio_ramp", False):
        r_early = float(getattr(sa, "stamp_ratio_ramp_early", 0.15))
        r_late = float(getattr(sa, "stamp_ratio_ramp_late", 0.45))
        ramp_chunks = int(getattr(sa, "stamp_ratio_ramp_chunks", 30) or 30)
        progress = min(sa.chunk_idx / max(ramp_chunks, 1), 1.0)
        r_t = r_early + (r_late - r_early) * progress

    # ── STAR: Salience-Tuned Adaptive Refinement (Temporal Only) ────────
    # Two purely temporal refinements to the pruning score:
    #   1. Temporal Consistency: tokens with stable salience across frames
    #      are more reliable → sharpen their scores (boost high, penalize low)
    #   2. Per-Layer Divergence: tokens where ViT layers disagree carry
    #      unique multi-scale temporal info → protect from pruning
    if getattr(sa, "star_enabled", False):
        star_tc = getattr(sa, "star_tc_weight", 0.3)
        star_gamma = getattr(sa, "star_gamma", 0.2)

        # Signal 1: Temporal Consistency Weighting
        # Track EMA variance of salience across frames for each token position.
        # Low variance = consistent importance signal → sharpen the score.
        # High variance = volatile → dampen toward the mean (less confident).
        star_salience_var = getattr(sa, "star_salience_var", None)
        current_salience = sa.stamp_temporal_vit_salience.to(device) if sa.stamp_temporal_vit_salience is not None else None
        if current_salience is not None and current_salience.shape[0] == N_vis:
            if star_salience_var is not None and star_salience_var.shape[0] == N_vis:
                sal_var = star_salience_var.to(device)
                # Compute consistency: inverse of normalized variance
                var_norm = sal_var / sal_var.max().clamp(min=1e-8)
                # consistency ∈ [0, 1]: 1 = perfectly consistent, 0 = highly volatile
                consistency = 1.0 - var_norm
                # Sharpen scores for consistent tokens, dampen for volatile ones
                # consistent high-score → boost; consistent low-score → penalize more
                score_mean = score_t.mean()
                deviation = score_t - score_mean
                score_t = score_t + star_tc * consistency * deviation

            # Update running variance: EMA of squared deviation from momentum
            momentum = sa.stamp_temporal_vit_momentum.to(device) if sa.stamp_temporal_vit_momentum is not None else current_salience
            if momentum.shape[0] == N_vis:
                instant_var = (current_salience - momentum).pow(2)
                tc_ema_rate = 0.2
                if star_salience_var is None or star_salience_var.shape[0] != N_vis:
                    sa.star_salience_var = instant_var.detach().cpu()
                else:
                    old_var = star_salience_var.to(device)
                    sa.star_salience_var = ((1 - tc_ema_rate) * old_var + tc_ema_rate * instant_var).detach().cpu()

        # Signal 2: Per-Layer Salience Divergence Boost
        # Tokens where ViT layers disagree carry unique multi-scale info → protect
        per_layer_sal = getattr(sa, "stamp_temporal_per_layer_salience", None)
        if per_layer_sal is not None and per_layer_sal.shape[1] == N_vis:
            per_layer_dev = per_layer_sal.to(device)  # [n_layers, N_vis]
            layer_std = per_layer_dev.std(dim=0)  # [N_vis]
            layer_std_norm = layer_std / layer_std.max().clamp(min=1e-8)
            divergence_boost = 1.0 + star_gamma * layer_std_norm
            score_t = score_t * divergence_boost

    # ── Phase 3: Self-Consistency Stochastic Scoring ────────────────────
    # Inject Gumbel(0, scale) noise into score_t so multiple SC passes
    # (each with a different sc_seed) produce different prunings → different
    # MCQ predictions → majority vote across runs. Deterministic when seed=0.
    sc_scale = float(getattr(sa, 'stamp_temporal_sc_scale', 0.0) or 0.0)
    sc_seed = int(getattr(sa, 'stamp_temporal_sc_seed', 0) or 0)
    if sc_scale > 0.0 and sc_seed > 0:
        gen = torch.Generator(device=device).manual_seed(sc_seed * 9973 + sa.chunk_idx)
        u = torch.rand(score_t.shape, device=device, generator=gen).clamp(min=1e-9, max=1 - 1e-9)
        gumbel = -torch.log(-torch.log(u))
        score_t = score_t + sc_scale * gumbel

    # ── M5 MSSAVT: same-frame (frame-local) cosine-diversity penalty ─────
    # Rationale: A1 global MMR killed cross-frame same-object evidence.
    # MSSAVT only penalizes redundancy *within* one frame (same MRoPE T-channel),
    # using LLM-L0 input_embeds as features. Penalty_i = alpha * max cos-sim
    # with any same-frame peer that has a *higher* score_t.
    if (getattr(sa, 'stamp_temporal_mssavt', False)
            and position_ids is not None
            and position_ids.shape[0] >= 1):
        mssavt_alpha = float(getattr(sa, 'stamp_temporal_mssavt_alpha', 0.15))
        t_pos = position_ids[0, 0, vis_indices].long()  # [N_vis]
        vis_embeds_f = inputs_embeds[0, vis_indices, :].float()
        vis_embeds_n = F.normalize(vis_embeds_f, dim=-1)
        unique_t = torch.unique(t_pos)
        penalty = torch.zeros_like(score_t)
        for t_val in unique_t.tolist():
            idx = (t_pos == t_val).nonzero(as_tuple=True)[0]
            if idx.numel() < 2:
                continue
            emb_g = vis_embeds_n[idx]                    # [m, D]
            sc_g = score_t[idx]                          # [m]
            sim = torch.mm(emb_g, emb_g.t())             # [m, m]
            sim.fill_diagonal_(-1.0)
            higher = (sc_g.unsqueeze(1) < sc_g.unsqueeze(0)).float()  # j strictly higher
            masked = sim * higher + (1 - higher) * (-1.0)
            pen_g, _ = masked.max(dim=1)
            pen_g = pen_g.clamp(min=0.0)
            penalty[idx] = pen_g
        score_t = score_t - mssavt_alpha * penalty

    # ── Top-r_t Selection ────────────────────────────────────────────────
    keep_k = max(1, int(N_vis * r_t))
    frame_strata = int(getattr(sa, 'stamp_temporal_plus_frame_strata', 0) or 0)
    if getattr(sa, 'stamp_temporal_plus', False) and keep_k < N_vis and frame_strata > 1:
        # A4: Per-frame stratified selection.
        # Assumes visual tokens are ordered (frame, row, col). Partition N_vis into
        # frame_strata contiguous chunks, allocate floor(keep_k / S) to each, give
        # remainder to the highest-scoring chunks, then run the selection (MMR or top-k)
        # WITHIN each chunk. Guarantees temporal coverage at aggressive r.
        S = min(frame_strata, N_vis)
        base = keep_k // S
        rem = keep_k - base * S
        # Stratum boundaries (approx equal width)
        bounds = [int(round(i * N_vis / S)) for i in range(S + 1)]
        # Allocate remainder to strata with highest per-stratum max score
        stratum_max = torch.tensor([
            score_t[bounds[i]:bounds[i+1]].max().item() if bounds[i+1] > bounds[i] else -1e9
            for i in range(S)
        ], device=device)
        top_rem = torch.topk(stratum_max, min(rem, S)).indices.tolist() if rem > 0 else []
        use_mmr = getattr(sa, 'stamp_temporal_plus_mmr_beta', 0.5) > 0.0
        if use_mmr:
            vis_embeds_mmr = inputs_embeds[0, vis_indices, :].float()
            vis_norm_mmr = F.normalize(vis_embeds_mmr, dim=-1)
        picked_all = []
        for i in range(S):
            lo, hi = bounds[i], bounds[i+1]
            if hi <= lo:
                continue
            k_i = base + (1 if i in top_rem else 0)
            k_i = min(k_i, hi - lo)
            if k_i <= 0:
                continue
            seg_score = score_t[lo:hi].clone().float()
            if use_mmr:
                beta_mmr = getattr(sa, 'stamp_temporal_plus_mmr_beta', 0.5)
                seg_norm = vis_norm_mmr[lo:hi]
                cur = seg_score.clone()
                for _ in range(k_i):
                    j = int(torch.argmax(cur).item())
                    picked_all.append(lo + j)
                    sim = torch.mv(seg_norm, seg_norm[j]).clamp(min=0.0)
                    cur = cur - beta_mmr * sim
                    cur[j] = float('-inf')
            else:
                _, top_idx = seg_score.topk(k_i)
                picked_all.extend([lo + int(j.item()) for j in top_idx])
        if len(picked_all) < keep_k:
            # Fill shortfall with global top-k of remaining tokens
            already = set(picked_all)
            mask_rem = torch.ones(N_vis, dtype=torch.bool, device=device)
            for j in already:
                mask_rem[j] = False
            remaining_scores = score_t.clone()
            remaining_scores[~mask_rem] = float('-inf')
            _, fill = remaining_scores.topk(keep_k - len(picked_all))
            picked_all.extend([int(j.item()) for j in fill])
        kept_local_indices = torch.tensor(sorted(picked_all[:keep_k]), dtype=torch.long, device=device)
    elif getattr(sa, 'stamp_temporal_plus', False) and keep_k < N_vis:
        # A1: MMR-style selection — iterative greedy with similarity penalty.
        # Each picked token dampens the remaining scores by
        #   Δ = beta * max(cos(v_i, v_picked), 0)
        # which is a relaxed, cheaper proxy for DPP's determinant penalty.
        beta_mmr = getattr(sa, 'stamp_temporal_plus_mmr_beta', 0.5)
        vis_embeds_mmr = inputs_embeds[0, vis_indices, :].float()
        vis_norm_mmr = F.normalize(vis_embeds_mmr, dim=-1)
        current_scores = score_t.clone().float()
        picked = torch.full((keep_k,), -1, dtype=torch.long, device=device)
        for i_pick in range(keep_k):
            idx = int(torch.argmax(current_scores).item())
            picked[i_pick] = idx
            # Dampen unpicked by similarity; picked stays -inf so won't re-pick
            sim_to_pick = torch.mv(vis_norm_mmr, vis_norm_mmr[idx]).clamp(min=0.0)
            current_scores = current_scores - beta_mmr * sim_to_pick
            current_scores[idx] = float('-inf')
        kept_local_indices, _ = picked.sort()
    else:
        _, top_local_idx = score_t.topk(keep_k)
        kept_local_indices, _ = top_local_idx.sort()
    sa.stamp_kept_vis_local_indices = kept_local_indices

    # Build boolean keep mask over the full sequence
    kept_vis_seq_pos = vis_indices[kept_local_indices]
    keep_seq = (~vis_mask).clone()
    keep_seq[kept_vis_seq_pos] = True

    # ── TAST: Temporal Accumulative State Tokens ───────────────────────
    tast_enabled = getattr(sa, 'tast_enabled', False)
    tast_K = getattr(sa, 'tast_n_tokens', 32)
    # Phase-7B: adaptive γ schedule (decays γ with chunk_idx if --tast_adaptive_gamma)
    tast_gamma = _tast_effective_gamma(sa, getattr(sa, 'tast_gamma', 0.1))

    if tast_enabled and keep_k < N_vis:
        vis_embeds = inputs_embeds[0, vis_indices, :].float()  # [N_vis, D]
        pruned_mask = torch.ones(N_vis, dtype=torch.bool, device=device)
        pruned_mask[kept_local_indices] = False
        pruned_indices = pruned_mask.nonzero(as_tuple=True)[0]
        n_pruned = pruned_indices.numel()

        if n_pruned > 0:
            pruned_embeds_tok = vis_embeds[pruned_indices]  # [n_pruned, D]

            if sa.tast_state_tokens is None:
                # Initialize state tokens from pruned tokens via k-means-like assignment
                K_actual = min(tast_K, n_pruned)
                # Seed: pick K evenly spaced pruned tokens
                seed_idx = torch.linspace(0, n_pruned - 1, K_actual, device=device).long()
                sa.tast_state_tokens = pruned_embeds_tok[seed_idx].detach().cpu()
                sa.tast_state_counts = torch.ones(K_actual)

                # Refine with one k-means iteration: assign all pruned to nearest seed, recompute centroids
                state_on_device = sa.tast_state_tokens.to(device)
                state_norm = F.normalize(state_on_device, dim=-1)
                pruned_norm = F.normalize(pruned_embeds_tok, dim=-1)
                sim = torch.mm(pruned_norm, state_norm.t())  # [n_pruned, K]
                assignments = sim.argmax(dim=1)
                for ki in range(K_actual):
                    mask_ki = (assignments == ki)
                    if mask_ki.any():
                        sa.tast_state_tokens[ki] = pruned_embeds_tok[mask_ki].mean(dim=0).detach().cpu()
                        sa.tast_state_counts[ki] = mask_ki.sum().item()
            else:
                # Update existing state tokens with EMA from pruned tokens
                K_actual = sa.tast_state_tokens.shape[0]
                state_on_device = sa.tast_state_tokens.to(device)
                state_norm = F.normalize(state_on_device, dim=-1)
                pruned_norm = F.normalize(pruned_embeds_tok, dim=-1)
                sim = torch.mm(pruned_norm, state_norm.t())  # [n_pruned, K]
                assignments = sim.argmax(dim=1)

                # EMA update: S_k = (1 - γ) * S_k + γ * mean(assigned pruned tokens)
                for ki in range(K_actual):
                    mask_ki = (assignments == ki)
                    if mask_ki.any():
                        cluster_mean = pruned_embeds_tok[mask_ki].mean(dim=0)
                        state_on_device[ki] = (1.0 - tast_gamma) * state_on_device[ki] + tast_gamma * cluster_mean
                        sa.tast_state_counts[ki] += mask_ki.sum().item()

                sa.tast_state_tokens = state_on_device.detach().cpu()

            # Phase-7A: Hierarchical TAST — also refresh long-scale pool
            _tast_maybe_update_long_pool(sa, device)

            # Inject state tokens: blend into pruned positions
            K_inject = min(K_actual, n_pruned)
            inject_pruned_idx = torch.linspace(0, n_pruned - 1, K_inject, device=device).long()
            inject_local_idx = pruned_indices[inject_pruned_idx]  # local indices in vis_indices
            inject_seq_pos = vis_indices[inject_local_idx]        # absolute sequence positions

            # Additive blending: preserve original, add temporal context
            blend_alpha = getattr(sa, 'tast_blend_alpha', 0.2)
            state_to_inject = _tast_build_inject_pool(sa, K_inject, device, inputs_embeds.dtype)
            original = inputs_embeds[0, inject_seq_pos, :]
            inputs_embeds[0, inject_seq_pos, :] = (
                (1.0 - blend_alpha) * original + blend_alpha * state_to_inject
            )

            # Add these positions to the keep mask
            keep_seq[inject_seq_pos] = True

            # Update kept indices to include state token positions
            all_kept = torch.cat([kept_local_indices, inject_local_idx])
            sa.stamp_kept_vis_local_indices, _ = all_kept.sort()

    # ── DSTM with pruning: update scene+delta memory from ALL tokens ────
    elif getattr(sa, 'dstm_enabled', False) and keep_k < N_vis:
        from streaming_vlm.inference.dstm import dstm_process_frame
        dstm_process_frame(inputs_embeds, vis_indices, streaming_args, is_keyframe=False)
        # DSTM injects memory tokens at low-salience positions already in keep_seq
        # We need to ensure injected positions are kept
        K_s = getattr(sa, 'dstm_scene_tokens', 16)
        K_d = getattr(sa, 'dstm_delta_tokens', 16)
        K_total_inject = min(K_s + K_d, N_vis * 3 // 10)
        if K_total_inject > 0 and sa.dstm_scene_state is not None:
            vit_sal = sa.stamp_temporal_vit_salience
            if vit_sal is not None and vit_sal.shape[0] == N_vis:
                sal_dev = vit_sal.to(device)
                _, bottom_idx = sal_dev.topk(K_total_inject, largest=False)
                inject_seq_pos = vis_indices[bottom_idx]
                keep_seq[inject_seq_pos] = True
                all_kept = torch.cat([kept_local_indices, bottom_idx])
                sa.stamp_kept_vis_local_indices, _ = all_kept.sort()

    # ── Token Merging (non-TAST path): merge pruned into nearest kept ──
    elif getattr(sa, 'stamp_temporal_merge', False) and keep_k < N_vis:
        vis_embeds = inputs_embeds[0, vis_indices, :]  # [N_vis, D]
        pruned_mask = torch.ones(N_vis, dtype=torch.bool, device=device)
        pruned_mask[kept_local_indices] = False
        pruned_indices = pruned_mask.nonzero(as_tuple=True)[0]

        if pruned_indices.numel() > 0:
            kept_embeds = vis_embeds[kept_local_indices]  # [k, D]
            pruned_embeds_tok = vis_embeds[pruned_indices]  # [N-k, D]
            kept_norm = F.normalize(kept_embeds.float(), dim=-1)
            pruned_norm = F.normalize(pruned_embeds_tok.float(), dim=-1)
            sim = torch.mm(pruned_norm, kept_norm.t())  # [N-k, k]
            nearest = sim.argmax(dim=1)  # [N-k]
            pruned_scores = score_t[pruned_indices]
            kept_scores = score_t[kept_local_indices]
            for ki in range(keep_k):
                merge_mask = (nearest == ki)
                if merge_mask.any():
                    merge_embeds = pruned_embeds_tok[merge_mask]
                    merge_weights = pruned_scores[merge_mask]
                    w_kept = kept_scores[ki]
                    w_total = w_kept + merge_weights.sum()
                    merged = (w_kept * kept_embeds[ki] + (merge_weights.unsqueeze(-1) * merge_embeds).sum(dim=0)) / w_total
                    inputs_embeds[0, vis_indices[kept_local_indices[ki]], :] = merged.to(inputs_embeds.dtype)

    # ── Compress-Not-Discard (optional) ──────────────────────────────────
    elif getattr(sa, 'stamp_temporal_compress', False) and keep_k < N_vis:
        vis_embeds = inputs_embeds[0, vis_indices, :]  # [N_vis, D]
        pruned_mask = torch.ones(N_vis, dtype=torch.bool, device=device)
        pruned_mask[kept_local_indices] = False
        pruned_embeds = vis_embeds[pruned_mask]
        if pruned_embeds.numel() > 0:
            summary = pruned_embeds.mean(dim=0, keepdim=True)  # [1, D]
            sa.stamp_temporal_summary_tokens.append(summary.detach().cpu())

    # ── M3: FOCUS-style spatial neighbor enhancement on kept tokens ──────
    # Blend a weighted average of nearby pruned tokens into each kept token.
    # Weight = inverse spatial distance (from MRoPE H/W channels) × pruned token score.
    # Works only when MRoPE position_ids carry spatial coordinates (Qwen2.5-VL).
    if (getattr(sa, 'stamp_temporal_focus_enhance', False)
            and keep_k < N_vis
            and position_ids is not None
            and position_ids.shape[0] >= 3):
        focus_alpha = float(getattr(sa, 'stamp_temporal_focus_enhance_alpha', 0.1))
        kept_mask = torch.zeros(N_vis, dtype=torch.bool, device=device)
        kept_mask[kept_local_indices] = True
        pruned_idx = (~kept_mask).nonzero(as_tuple=True)[0]
        if pruned_idx.numel() > 0:
            h_pos = position_ids[1, 0, vis_indices].float()
            w_pos = position_ids[2, 0, vis_indices].float()
            kept_h = h_pos[kept_local_indices]
            kept_w = w_pos[kept_local_indices]
            pruned_h = h_pos[pruned_idx]
            pruned_w = w_pos[pruned_idx]
            dh = kept_h.unsqueeze(1) - pruned_h.unsqueeze(0)
            dw = kept_w.unsqueeze(1) - pruned_w.unsqueeze(0)
            dist_sq = dh ** 2 + dw ** 2 + 1e-8
            spatial_w = 1.0 / dist_sq
            pruned_scores = score_t[pruned_idx]
            wts = spatial_w * pruned_scores.unsqueeze(0)
            wts = wts / (wts.sum(dim=1, keepdim=True) + 1e-8)
            vis_embeds_fe = inputs_embeds[0, vis_indices, :].float()
            enhancement = torch.mm(wts, vis_embeds_fe[pruned_idx])
            kept_embeds_fe = vis_embeds_fe[kept_local_indices]
            blended = (1.0 - focus_alpha) * kept_embeds_fe + focus_alpha * enhancement
            inputs_embeds[0, vis_indices[kept_local_indices], :] = blended.to(inputs_embeds.dtype)

    # ── Prune all sequence-length tensors ────────────────────────────────
    pruned_embeds = inputs_embeds[:, keep_seq, :]
    pruned_pos = position_ids[:, :, keep_seq] if position_ids is not None else None
    pruned_attn_mask = attention_mask[:, keep_seq] if attention_mask is not None else None
    pruned_cache = cache_position[keep_seq] if cache_position is not None else None

    # N6: re-pack position IDs to contiguous 0..K-1 across all 3 MRoPE channels.
    # Fixes RoPE position-gap degradation at aggressive r (PPE paper 2510.22936).
    if getattr(sa, 'stamp_temporal_repack_pos', False) and pruned_pos is not None:
        new_seq_len = pruned_pos.shape[-1]
        new_ids = torch.arange(new_seq_len, device=device, dtype=pruned_pos.dtype)
        pruned_pos = new_ids.view(1, 1, -1).expand(pruned_pos.shape[0], pruned_pos.shape[1], -1).contiguous()
    if getattr(sa, 'stamp_temporal_repack_pos', False) and pruned_cache is not None:
        pruned_cache = torch.arange(pruned_cache.shape[0], device=device, dtype=pruned_cache.dtype)

    sa.current_input_ids = sa.current_input_ids[:, keep_seq] if sa.current_input_ids is not None else None

    return pruned_embeds, pruned_pos, pruned_attn_mask, pruned_cache


def stamp_temporal_update_state(streaming_args, device):
    """
    Called after the LLM forward pass to update temporal state for the next chunk.

    Unlike original STAMP, this does NOT need LLM attention scores — momentum
    is updated using ViT salience which is already available. This function
    handles:
        - Initializing momentum on first chunk
        - Storing visual features for next chunk's novelty
        - TAST: updating state tokens from kept visual features (post-LLM)
        - Incrementing chunk counter
    """
    sa = streaming_args
    N_vis = getattr(sa, 'stamp_n_vis', None)
    if N_vis is None:
        return

    # Initialize momentum on first chunk (keyframe path)
    if sa.stamp_temporal_vit_momentum is None:
        vit_sal = sa.stamp_temporal_vit_salience
        if vit_sal is not None:
            sa.stamp_temporal_vit_momentum = vit_sal.detach().cpu()
        else:
            sa.stamp_temporal_vit_momentum = torch.zeros(N_vis)

    # Store current visual feats for next chunk's novelty
    if sa.stamp_curr_visual_feats is not None:
        sa.prev_visual_feats = sa.stamp_curr_visual_feats.detach().cpu()
        sa.stamp_curr_visual_feats = None

    sa.chunk_idx += 1
