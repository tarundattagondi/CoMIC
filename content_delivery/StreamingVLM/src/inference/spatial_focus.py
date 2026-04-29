"""
FOCUS — Feature-Optimized Conditioned Unification for Spatial selection
========================================================================
Training-free text-query-guided spatial token selection. Uses the text
portion of the input to compute relevance scores for each visual token,
keeping the most query-relevant tokens and enhancing survivors with
context from pruned neighbors.

Key novelty vs prior work:
  - TEXT-CONDITIONED spatial pruning: no prior training-free method uses
    the text query to guide which visual tokens to keep.
  - RELEVANCE ENHANCEMENT: surviving tokens are enriched with weighted
    context from their pruned spatial neighbors, compensating for lost detail.
  - Training-free: uses cosine similarity in the embedding space, no
    learned parameters.

Insertion point: Pre-LLM (language_forward.py ~line 558).

Architecture:
    Frame → ViT → Visual Tokens [N_vis, D]     Text → Embed → [N_txt, D]
                        ↓                                ↓
              [Cross-Attention: vis × text^T → relevance scores]
                        ↓
         ┌──────────────┴──────────────┐
    Top-r tokens                 Bottom (1-r) tokens
    (keep)                       (pruned)
         ↓                              ↓
    [Spatial Neighbor Enhancement]      ↓
    (weighted avg of pruned neighbors) ↓
         ↓                              ↓
    Enhanced tokens              (discarded)
         └──────────────────────────────┘
                   → LLM
"""

import torch
import torch.nn.functional as F


def focus_spatial_prune(inputs_embeds, position_ids, attention_mask,
                        cache_position, streaming_args):
    """
    FOCUS spatial pruning: text-query-guided token selection with
    spatial neighbor enhancement.

    Args: same as stamp_temporal_stage1_prune
    Returns: (inputs_embeds, position_ids, attention_mask, cache_position)
    """
    sa = streaming_args

    # Only on initial prefill
    if cache_position is None or cache_position[0] != 0:
        return inputs_embeds, position_ids, attention_mask, cache_position

    if sa.current_input_ids is None:
        return inputs_embeds, position_ids, attention_mask, cache_position

    device = inputs_embeds.device

    # Find visual and text token positions
    vis_token_ids = torch.tensor(sa.fastv_visual_token_ids, device=device)
    input_ids = sa.current_input_ids[0]
    vis_mask = torch.isin(input_ids, vis_token_ids)
    vis_indices = vis_mask.nonzero(as_tuple=True)[0]
    N_vis = vis_indices.shape[0]

    if N_vis == 0:
        return inputs_embeds, position_ids, attention_mask, cache_position

    focus_r = getattr(sa, 'focus_r', 0.85)  # fraction of visual tokens to keep
    focus_enhance_alpha = getattr(sa, 'focus_enhance_alpha', 0.1)  # enhancement strength
    focus_text_weight = getattr(sa, 'focus_text_weight', 0.6)  # weight for text vs ViT score

    # ── Extract visual and text embeddings ───────────────────────────────
    vis_embeds = inputs_embeds[0, vis_indices, :].float()  # [N_vis, D]

    # Text tokens: everything after visual tokens
    vis_end = vis_indices[-1].item() + 1
    seq_len = inputs_embeds.shape[1]
    text_embeds = inputs_embeds[0, vis_end:, :].float()  # [N_txt, D]
    N_txt = text_embeds.shape[0]

    if N_txt == 0:
        # No text tokens — fallback to ViT salience only
        return _fallback_vit_prune(inputs_embeds, position_ids, attention_mask,
                                   cache_position, sa, vis_indices, N_vis, focus_r)

    # ── Compute text-conditioned relevance scores ────────────────────────
    # Cosine similarity between each visual token and each text token
    vis_norm = F.normalize(vis_embeds, dim=-1)  # [N_vis, D]
    txt_norm = F.normalize(text_embeds, dim=-1)  # [N_txt, D]

    # Relevance = max similarity to any text token (captures query-alignment)
    cross_sim = torch.mm(vis_norm, txt_norm.t())  # [N_vis, N_txt]
    text_relevance = cross_sim.max(dim=1).values  # [N_vis]

    # ── Combine with ViT salience if available ───────────────────────────
    vit_salience = getattr(sa, 'stamp_temporal_vit_salience', None)
    if vit_salience is not None and vit_salience.shape[0] == N_vis:
        vit_scores = vit_salience.to(device).float()
        # Normalize both to [0, 1]
        text_rel_norm = (text_relevance - text_relevance.min()) / (text_relevance.max() - text_relevance.min() + 1e-8)
        vit_norm_score = (vit_scores - vit_scores.min()) / (vit_scores.max() - vit_scores.min() + 1e-8)
        # Weighted combination
        combined_score = focus_text_weight * text_rel_norm + (1 - focus_text_weight) * vit_norm_score
    else:
        combined_score = text_relevance

    # ── Select top-r tokens ──────────────────────────────────────────────
    keep_k = max(1, int(N_vis * focus_r))
    if keep_k >= N_vis:
        return inputs_embeds, position_ids, attention_mask, cache_position

    topk_result = combined_score.topk(keep_k)
    kept_local = topk_result.indices.sort().values  # sorted local indices within vis_indices

    # Pruned indices
    all_local = torch.arange(N_vis, device=device)
    pruned_mask = torch.ones(N_vis, dtype=torch.bool, device=device)
    pruned_mask[kept_local] = False
    pruned_local = all_local[pruned_mask]

    # ── Spatial Neighbor Enhancement ─────────────────────────────────────
    # For each kept token, find its nearest pruned neighbors in spatial
    # coordinates and blend their information
    if focus_enhance_alpha > 0 and pruned_local.shape[0] > 0:
        h_pos = position_ids[1, 0, vis_indices].float()  # [N_vis]
        w_pos = position_ids[2, 0, vis_indices].float()  # [N_vis]

        kept_h = h_pos[kept_local]  # [keep_k]
        kept_w = w_pos[kept_local]
        pruned_h = h_pos[pruned_local]  # [n_pruned]
        pruned_w = w_pos[pruned_local]

        # Pairwise spatial distance: [keep_k, n_pruned]
        dh = kept_h.unsqueeze(1) - pruned_h.unsqueeze(0)
        dw = kept_w.unsqueeze(1) - pruned_w.unsqueeze(0)
        dist_sq = dh ** 2 + dw ** 2 + 1e-8

        # Inverse-distance weights (spatial proximity)
        spatial_weights = 1.0 / dist_sq  # [keep_k, n_pruned]
        # Also weight by importance of pruned token
        pruned_scores = combined_score[pruned_local]  # [n_pruned]
        importance_weights = spatial_weights * pruned_scores.unsqueeze(0)  # [keep_k, n_pruned]

        # Normalize weights per kept token
        importance_weights = importance_weights / (importance_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Compute enhancement: weighted average of pruned embeddings
        pruned_embeds = vis_embeds[pruned_local]  # [n_pruned, D]
        enhancement = torch.mm(importance_weights, pruned_embeds)  # [keep_k, D]

        # Blend enhancement into kept tokens
        kept_embeds = vis_embeds[kept_local]  # [keep_k, D]
        enhanced_embeds = (1.0 - focus_enhance_alpha) * kept_embeds + focus_enhance_alpha * enhancement
        enhanced_embeds = enhanced_embeds.to(inputs_embeds.dtype)
    else:
        enhanced_embeds = inputs_embeds[0, vis_indices[kept_local], :]

    # ── Reconstruct sequence: [prefix | enhanced_vis | suffix] ───────────
    vis_start = vis_indices[0].item()
    vis_end_pos = vis_indices[-1].item() + 1

    prefix_embeds = inputs_embeds[0, :vis_start, :]
    suffix_embeds = inputs_embeds[0, vis_end_pos:, :]

    new_embeds = torch.cat([prefix_embeds, enhanced_embeds, suffix_embeds], dim=0)
    new_embeds = new_embeds.unsqueeze(0)

    new_len = new_embeds.shape[1]
    prefix_len = vis_start
    suffix_len = seq_len - vis_end_pos

    # Position IDs
    prefix_pos = position_ids[:, :, :vis_start]
    kept_pos = position_ids[:, :, vis_indices[kept_local]]
    suffix_pos = position_ids[:, :, vis_end_pos:]
    new_position_ids = torch.cat([prefix_pos, kept_pos, suffix_pos], dim=2)

    # Attention mask & cache position
    if attention_mask is not None:
        new_attn_mask = torch.ones(1, new_len, device=device, dtype=attention_mask.dtype)
    else:
        new_attn_mask = None
    new_cache_position = torch.arange(new_len, device=device)

    # Store stats
    sa._focus_n_kept = keep_k
    sa._focus_n_pruned = N_vis - keep_k

    return new_embeds, new_position_ids, new_attn_mask, new_cache_position


def _fallback_vit_prune(inputs_embeds, position_ids, attention_mask,
                        cache_position, sa, vis_indices, N_vis, keep_ratio):
    """Fallback: prune using ViT salience only (no text tokens available)."""
    device = inputs_embeds.device
    vit_salience = getattr(sa, 'stamp_temporal_vit_salience', None)
    if vit_salience is None or vit_salience.shape[0] != N_vis:
        return inputs_embeds, position_ids, attention_mask, cache_position

    keep_k = max(1, int(N_vis * keep_ratio))
    if keep_k >= N_vis:
        return inputs_embeds, position_ids, attention_mask, cache_position

    scores = vit_salience.to(device).float()
    kept_local = scores.topk(keep_k).indices.sort().values

    vis_start = vis_indices[0].item()
    vis_end = vis_indices[-1].item() + 1
    seq_len = inputs_embeds.shape[1]

    prefix = inputs_embeds[0, :vis_start, :]
    kept = inputs_embeds[0, vis_indices[kept_local], :]
    suffix = inputs_embeds[0, vis_end:, :]
    new_embeds = torch.cat([prefix, kept, suffix], dim=0).unsqueeze(0)

    new_len = new_embeds.shape[1]
    prefix_pos = position_ids[:, :, :vis_start]
    kept_pos = position_ids[:, :, vis_indices[kept_local]]
    suffix_pos = position_ids[:, :, vis_end:]
    new_pos = torch.cat([prefix_pos, kept_pos, suffix_pos], dim=2)

    new_attn = torch.ones(1, new_len, device=device, dtype=attention_mask.dtype) if attention_mask is not None else None
    new_cache = torch.arange(new_len, device=device)

    return new_embeds, new_pos, new_attn, new_cache
