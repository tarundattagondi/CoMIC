"""
STAMP — Streaming Temporal Attention Momentum Pruning
======================================================
Stage 1 (pre-LLM, before layer 0):
    score_t(i) = alpha * M_t(i)  +  (1 - alpha) * N_t(i)
    where:
        M_t(i) = (1-lambda) * M_{t-1}(i)  +  lambda * A_{t-1}(i)   # attention momentum EMA
        N_t(i) = 1 - cosine_sim(V_t[i], V_{t-1}[i])                  # temporal novelty
    Keep top-r1 visual tokens. Pruned tokens never enter any LLM layer.

Stage 2 (inside LLM, at layer fastv_k):
    Standard FastV spatial pruning using live attention scores from layer fastv_k-1.
    Keep top-r2 tokens (controlled by fastv_r in streaming_args).

Keyframe refresh (every stamp_K chunks):
    Skip Stage 1 entirely — keep ALL visual tokens.
    Stage 2 still runs. Momentum is fully refreshed with the complete attention signal.
    Prevents the "death spiral" where pruned tokens decay permanently.

Iteration 2 improvements (all optional, controlled by streaming_args flags):
    Idea 1 — Adaptive r1: adjust keep ratio based on scene motion (avg temporal novelty).
    Idea 2 — Momentum decay: pruned tokens decay at rate stamp_gamma instead of (1-lambda).
    Idea 3 — Adaptive keyframe: detect scene cuts from avg novelty instead of fixed K.
    Idea 4 — Hierarchical momentum: dual EMA (fast + slow timescales) for richer scoring.

Usage:
    streaming_args = StreamingArgs(
        pos_mode="shrink",
        fastv_k=2, fastv_r=1.0,    # Stage 2 disabled (r=1.0 = keep all)
        stamp_r1=0.5,               # Stage 1 keep ratio
        stamp_alpha=0.5,            # weight: alpha*momentum + (1-alpha)*novelty
        stamp_lambda=0.3,           # EMA decay rate for momentum
        stamp_K=10,                 # keyframe interval
    )
"""

import torch
import torch.nn.functional as F


def stamp_stage1_prune(inputs_embeds, position_ids, attention_mask, cache_position, streaming_args):
    """
    Pre-LLM Stage 1 pruning. Called at the entry of streaming_language_model_forward,
    after position_ids are finalised but before causal_mask is computed.

    Only fires on the initial prefill of each chunk (cache_position[0] == 0)
    and only when visual tokens are present in the sequence.

    Returns:
        inputs_embeds   — [1, new_seq_len, D]   (shorter if pruning occurred)
        position_ids    — [3, 1, new_seq_len]   (or None)
        attention_mask  — [1, new_seq_len]      (or None)
        cache_position  — [new_seq_len]         (or None)
    """
    sa = streaming_args

    # Only prune on initial prefill (cache_position[0] == 0).
    if cache_position is None or cache_position[0] != 0:
        return inputs_embeds, position_ids, attention_mask, cache_position

    if sa.current_input_ids is None:
        return inputs_embeds, position_ids, attention_mask, cache_position

    input_ids = sa.current_input_ids  # [1, seq_len]
    device = inputs_embeds.device

    # Find visual token positions in the current sequence
    vis_token_ids = torch.tensor(sa.fastv_visual_token_ids, device=device)
    vis_mask = torch.isin(input_ids[0], vis_token_ids)   # [seq_len] bool
    vis_indices = vis_mask.nonzero(as_tuple=True)[0]     # [N_vis] absolute positions
    N_vis = vis_indices.shape[0]

    if N_vis == 0:
        return inputs_embeds, position_ids, attention_mask, cache_position

    # Store N_vis for stamp_update_state() to use after Stage 2
    sa.stamp_n_vis = N_vis

    # ── Keyframe check (fixed interval) ───────────────────────────────────────
    is_keyframe = (sa.attention_momentum is None) or (sa.chunk_idx % sa.stamp_K == 0)
    sa.stamp_is_keyframe = is_keyframe

    if is_keyframe:
        sa.stamp_kept_vis_local_indices = torch.arange(N_vis, device=device)
        return inputs_embeds, position_ids, attention_mask, cache_position

    # ── Idea 2: Momentum update with differential decay for pruned tokens ─────
    if sa.stamp_last_attn_scores is not None:
        prev_attn = sa.stamp_last_attn_scores.to(device)
        if prev_attn.shape[0] == N_vis:
            if sa.stamp_momentum_decay and sa.stamp_kept_vis_local_indices is not None:
                # Build boolean mask of tokens kept in the PREVIOUS chunk
                kept_mask = torch.zeros(N_vis, dtype=torch.bool, device=device)
                prev_kept = sa.stamp_kept_vis_local_indices
                valid = prev_kept[prev_kept < N_vis]
                kept_mask[valid] = True
                M_prev = sa.attention_momentum.to(device)
                # Kept tokens: normal EMA update; pruned tokens: faster decay
                sa.attention_momentum = torch.where(
                    kept_mask,
                    (1.0 - sa.stamp_lambda) * M_prev + sa.stamp_lambda * prev_attn,
                    sa.stamp_gamma * M_prev,
                )
                # Idea 4 (hierarchical): also update long-term momentum
                if sa.stamp_hierarchical and sa.attention_momentum_long is not None:
                    if sa.attention_momentum_long.shape[0] == N_vis:
                        M_long = sa.attention_momentum_long.to(device)
                        sa.attention_momentum_long = torch.where(
                            kept_mask,
                            (1.0 - sa.stamp_lambda_long) * M_long + sa.stamp_lambda_long * prev_attn,
                            sa.stamp_gamma * M_long,
                        )
            else:
                # Standard EMA update (original behaviour)
                M_prev = sa.attention_momentum.to(device)
                sa.attention_momentum = (
                    (1.0 - sa.stamp_lambda) * M_prev + sa.stamp_lambda * prev_attn
                )
                # Idea 4 (hierarchical): update long-term momentum
                if sa.stamp_hierarchical and sa.attention_momentum_long is not None:
                    if sa.attention_momentum_long.shape[0] == N_vis:
                        M_long = sa.attention_momentum_long.to(device)
                        sa.attention_momentum_long = (
                            (1.0 - sa.stamp_lambda_long) * M_long + sa.stamp_lambda_long * prev_attn
                        )

    M_t = sa.attention_momentum.to(device)  # [N_vis]

    # Shape mismatch guard (e.g. resolution switch) → treat as keyframe
    if M_t.shape[0] != N_vis:
        sa.stamp_kept_vis_local_indices = torch.arange(N_vis, device=device)
        return inputs_embeds, position_ids, attention_mask, cache_position

    # ── Temporal novelty: N_t = 1 - cosine_sim(V_t[i], V_{t-1}[i]) ──────────
    if (
        sa.stamp_curr_visual_feats is not None
        and sa.prev_visual_feats is not None
        and sa.stamp_curr_visual_feats.shape == sa.prev_visual_feats.shape
    ):
        V_t    = sa.stamp_curr_visual_feats.float().to(device)   # [N_vis, D]
        V_prev = sa.prev_visual_feats.float().to(device)          # [N_vis, D]
        cos_sim = F.cosine_similarity(V_t, V_prev, dim=-1)        # [N_vis]
        N_t = (1.0 - cos_sim).clamp(0.0, 1.0)
    else:
        N_t = torch.ones(N_vis, device=device)

    # ── Idea 3: Adaptive keyframe detection from scene novelty ────────────────
    avg_novelty = N_t.mean().item()
    if sa.stamp_adaptive_kf and avg_novelty > sa.stamp_adaptive_kf_threshold:
        # Scene cut detected → treat as keyframe, skip pruning
        sa.stamp_is_keyframe = True
        sa.stamp_kept_vis_local_indices = torch.arange(N_vis, device=device)
        return inputs_embeds, position_ids, attention_mask, cache_position

    # ── Normalize scores using running EMA statistics ─────────────────────
    norm_ema_rate = 0.3  # how fast running stats adapt

    def _ema_normalize(x, running_mean, running_std, ema_rate):
        """Z-score normalize using running EMA of mean/std, then sigmoid to [0,1]."""
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

    # ── Idea 4: Hierarchical scoring using short + long momentum ─────────
    if (
        sa.stamp_hierarchical
        and sa.attention_momentum_long is not None
        and sa.attention_momentum_long.shape[0] == N_vis
    ):
        M_long = sa.attention_momentum_long.to(device)
        M_long_norm = (M_long - M_long.min()) / (M_long.max() - M_long.min() + 1e-8)
        score_t = (
            sa.stamp_alpha_short * M_norm
            + sa.stamp_alpha_long * M_long_norm
            + (1.0 - sa.stamp_alpha_short - sa.stamp_alpha_long) * N_norm
        )
    else:
        # Original scoring
        score_t = sa.stamp_alpha * M_norm + (1.0 - sa.stamp_alpha) * N_norm

    # ── Idea 1: Adaptive pruning ratio based on scene motion ──────────────────
    if sa.stamp_adaptive_r1:
        if avg_novelty > sa.stamp_adaptive_r1_high:
            r1 = min(sa.stamp_r1 + 0.25, 0.9)   # dynamic scene → keep more tokens
        elif avg_novelty < sa.stamp_adaptive_r1_low:
            r1 = max(sa.stamp_r1 - 0.25, 0.1)   # static scene → prune more aggressively
        else:
            r1 = sa.stamp_r1
    else:
        r1 = sa.stamp_r1

    # ── Keep top-r1 visual tokens ─────────────────────────────────────────────
    keep_k = max(1, int(N_vis * r1))
    _, top_local_idx = score_t.topk(keep_k)
    kept_local_indices, _ = top_local_idx.sort()     # preserve original token order
    sa.stamp_kept_vis_local_indices = kept_local_indices  # [keep_k]

    # Build boolean keep mask over the full sequence
    kept_vis_seq_pos = vis_indices[kept_local_indices]  # absolute positions of kept vis tokens
    keep_seq = (~vis_mask).clone()                      # True for all non-visual tokens
    keep_seq[kept_vis_seq_pos] = True                   # True for kept visual tokens

    # ── Token merging: merge pruned tokens into nearest kept neighbor ─────
    if getattr(sa, 'stamp_merge', False) and keep_k < N_vis:
        # Extract visual embeddings [N_vis, D]
        vis_embeds = inputs_embeds[0, vis_indices, :]  # [N_vis, D]
        kept_embeds = vis_embeds[kept_local_indices]    # [keep_k, D]

        # Find pruned token indices (local, within visual tokens)
        all_local = torch.arange(N_vis, device=device)
        pruned_mask = torch.ones(N_vis, dtype=torch.bool, device=device)
        pruned_mask[kept_local_indices] = False
        pruned_local_indices = all_local[pruned_mask]   # [N_vis - keep_k]

        if pruned_local_indices.numel() > 0:
            pruned_embeds = vis_embeds[pruned_local_indices]  # [N_pruned, D]

            # Cosine similarity: each pruned token vs all kept tokens
            pruned_norm = F.normalize(pruned_embeds.float(), dim=-1)  # [N_pruned, D]
            kept_norm = F.normalize(kept_embeds.float(), dim=-1)      # [keep_k, D]
            sim = torch.mm(pruned_norm, kept_norm.t())                # [N_pruned, keep_k]
            nearest_kept = sim.argmax(dim=1)                          # [N_pruned] — index into kept

            # Score-weighted merge: w = score_pruned / (score_pruned + score_kept_neighbor)
            pruned_scores = score_t[pruned_local_indices]
            kept_neighbor_scores = score_t[kept_local_indices[nearest_kept]]
            w = (pruned_scores / (pruned_scores + kept_neighbor_scores + 1e-8)).unsqueeze(-1)  # [N_pruned, 1]

            # Weighted average: kept_new = (1-w)*kept + w*pruned
            merged = kept_embeds.clone().float()
            for p_idx in range(pruned_local_indices.numel()):
                k_idx = nearest_kept[p_idx]
                weight = w[p_idx]
                merged[k_idx] = merged[k_idx] * (1 - weight) + pruned_embeds[p_idx].float() * weight

            # Write merged embeddings back into inputs_embeds
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[0, kept_vis_seq_pos, :] = merged.to(inputs_embeds.dtype)

    # ── Prune all sequence-length tensors ─────────────────────────────────────
    pruned_embeds = inputs_embeds[:, keep_seq, :]

    pruned_pos = position_ids[:, :, keep_seq] if position_ids is not None else None

    pruned_attn_mask = attention_mask[:, keep_seq] if attention_mask is not None else None

    pruned_cache = cache_position[keep_seq] if cache_position is not None else None

    # Update current_input_ids so FastV Stage 2 finds correct visual positions
    sa.current_input_ids = sa.current_input_ids[:, keep_seq] if sa.current_input_ids is not None else None

    return pruned_embeds, pruned_pos, pruned_attn_mask, pruned_cache


def stamp_update_state(streaming_args, fastv_attn_scores, device):
    """
    Called immediately after Stage 2 (FastV) fires at layer K.

    Updates:
        streaming_args.stamp_last_attn_scores  — full [N_vis] vector (0 for pruned tokens)
        streaming_args.attention_momentum      — initialised on first real chunk
        streaming_args.attention_momentum_long — Idea 4: initialised on first real chunk
        streaming_args.prev_visual_feats       — V_t stored for next chunk's novelty
        streaming_args.chunk_idx               — incremented

    Args:
        fastv_attn_scores: [N_vis_kept_stage1] mean attention scores for visual tokens
                           that survived Stage 1, in kept-token order.
        device: torch device.
    """
    sa = streaming_args
    N_vis = getattr(sa, 'stamp_n_vis', None)
    if N_vis is None:
        return

    kept = getattr(sa, 'stamp_kept_vis_local_indices', None)

    # Reconstruct full-length attention vector: pruned positions get 0
    full_attn = torch.zeros(N_vis, device=device)
    if kept is not None:
        n = min(kept.shape[0], fastv_attn_scores.shape[0])
        full_attn[kept[:n]] = fastv_attn_scores[:n].float().to(device)
    else:
        n = min(N_vis, fastv_attn_scores.shape[0])
        full_attn[:n] = fastv_attn_scores[:n].float().to(device)

    sa.stamp_last_attn_scores = full_attn

    # Initialise momentum on very first chunk (momentum was None → keyframe path)
    if sa.attention_momentum is None or sa.attention_momentum.shape[0] != N_vis:
        sa.attention_momentum = full_attn.clone()

    # Idea 4: initialise long-term momentum on first chunk
    if sa.stamp_hierarchical:
        if sa.attention_momentum_long is None or sa.attention_momentum_long.shape[0] != N_vis:
            sa.attention_momentum_long = full_attn.clone()

    # Store current visual feats as "previous" for next chunk's novelty signal
    if sa.stamp_curr_visual_feats is not None:
        sa.prev_visual_feats = sa.stamp_curr_visual_feats.detach().cpu()
        sa.stamp_curr_visual_feats = None

    sa.chunk_idx += 1
