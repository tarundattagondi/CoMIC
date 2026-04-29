"""
DSTM — Delta-State Temporal Memory (Architecture 5)
====================================================
The crown jewel: a training-free temporal memory system that separates
WHAT the scene looks like (scene state) from WHAT IS CHANGING (delta state),
with surprise-gated updates that only accumulate novel information.

Key innovations over TAST (Architecture 4):
  1. DUAL MEMORY: Scene tokens (static context) + Delta tokens (temporal dynamics)
  2. SURPRISE GATING: Only update memory when incoming tokens carry new information
     g_k = σ(cosine_distance(incoming, memory_slot_k) - threshold)
     High surprise → large update; low surprise → near-zero update
  3. SALIENCE-WEIGHTED UPDATE: Important tokens (high ViT salience) contribute
     more to memory than background tokens
  4. MULTI-SCALE TEMPORAL DECAY: Scene tokens decay slowly (preserve long-range),
     delta tokens decay fast (capture recent changes)
  5. ADAPTIVE INJECTION: Number of injected tokens scales with temporal complexity
     (more changes → more delta tokens injected)

Mathematical formulation:
    Scene State:  S_k^{t} = (1 - γ_s · g_k) · S_k^{t-1} + γ_s · g_k · μ_k^t
    Delta State:  D_k^{t} = (1 - γ_d · g_k) · D_k^{t-1} + γ_d · g_k · Δ_k^t
    where:
        g_k = σ(β · (1 - cos_sim(μ_k^t, S_k^{t-1})) - τ)   (surprise gate)
        μ_k^t = Σ(w_i · x_i) / Σ(w_i) for tokens assigned to slot k  (salience-weighted mean)
        Δ_k^t = μ_k^t - μ_k^{t-1}                            (temporal difference)
        γ_s = 0.05 (slow scene decay)
        γ_d = 0.2  (fast delta decay)
        β = 5.0    (surprise sensitivity)
        τ = 0.3    (surprise threshold)

Architecture:
    Frame_t → ViT → [Multi-Layer Salience] → [STAMP-Temporal Pruning] → Kept tokens
                                                      ↓
                                                 ALL tokens (or pruned)
                                                      ↓
                                          ┌───────────┴───────────┐
                                    [Assign to nearest         [Compute temporal
                                     scene slot via             difference per slot]
                                     cosine similarity]              ↓
                                          ↓                     Delta tokens D_k
                                    Scene tokens S_k                 ↓
                                          ↓               ┌─────────┴─────────┐
                                    [Surprise gate:       [Surprise gate:
                                     update only if        update only if
                                     novel content]        novel change]
                                          ↓                          ↓
                                    ┌─────┴─────┐           ┌───────┴───────┐
                               [Scene Memory]          [Delta Memory]
                                    └─────┬─────┘           └───────┬───────┘
                                          ↓                          ↓
                                    [Inject at low-salience positions in LLM input]
                                          ↓
                                    LLM Decoder → Answer

Novelty claim:
    "To the best of our knowledge, DSTM is the first training-free method that
    separates persistent scene state from temporal change state with surprise-gated
    updates, enabling selective memory accumulation across video frames without
    any learned parameters."

Usage:
    streaming_args.dstm_enabled = True
    streaming_args.dstm_scene_tokens = 16     # K_s scene memory slots
    streaming_args.dstm_delta_tokens = 16     # K_d delta memory slots
    streaming_args.dstm_gamma_scene = 0.05    # slow scene EMA
    streaming_args.dstm_gamma_delta = 0.2     # fast delta EMA
    streaming_args.dstm_surprise_beta = 5.0   # surprise gate sensitivity
    streaming_args.dstm_surprise_tau = 0.3    # surprise gate threshold
    streaming_args.dstm_only = False          # True = no pruning, inject into full stream
"""

import torch
import torch.nn.functional as F


def dstm_reset_state(sa):
    """Reset all DSTM runtime state between videos."""
    sa.dstm_scene_state = None       # [K_s, D] persistent scene memory
    sa.dstm_delta_state = None       # [K_d, D] persistent delta memory
    sa.dstm_prev_slot_means = None   # [K_s, D] previous frame's per-slot means (for delta computation)
    sa.dstm_scene_counts = None      # [K_s] assignment counts for diagnostics
    sa.dstm_delta_counts = None      # [K_d] assignment counts for diagnostics
    sa.dstm_temporal_complexity = 0.0  # running estimate of how much is changing


def _compute_surprise_gate(incoming, memory, beta, tau):
    """
    Compute per-slot surprise gate values.

    g_k = σ(β · (1 - cos_sim(incoming_k, memory_k)) - τ)

    Args:
        incoming: [K, D] — new slot means from current frame
        memory:   [K, D] — existing memory state
        beta:     float  — sensitivity (higher = sharper gate)
        tau:      float  — threshold (higher = more selective)

    Returns:
        gates: [K] in (0, 1) — per-slot update weights
    """
    cos_sim = F.cosine_similarity(incoming, memory, dim=-1)  # [K]
    # cosine distance: 1 - cos_sim ∈ [0, 2], typically [0, 1] for similar embeddings
    surprise = 1.0 - cos_sim  # [K]
    gates = torch.sigmoid(beta * surprise - tau)  # [K]
    return gates


def _assign_tokens_to_slots(token_embeds, slot_embeds):
    """
    Assign each token to its nearest memory slot via cosine similarity.

    Args:
        token_embeds: [N, D]
        slot_embeds:  [K, D]

    Returns:
        assignments: [N] — index of nearest slot per token
        similarities: [N, K] — full similarity matrix (for soft weighting if needed)
    """
    tok_norm = F.normalize(token_embeds, dim=-1)
    slot_norm = F.normalize(slot_embeds, dim=-1)
    sim = torch.mm(tok_norm, slot_norm.t())  # [N, K]
    assignments = sim.argmax(dim=1)  # [N]
    return assignments, sim


def _salience_weighted_slot_means(token_embeds, assignments, salience_weights, K):
    """
    Compute salience-weighted mean embedding per slot.

    Args:
        token_embeds:    [N, D]
        assignments:     [N] — slot index per token
        salience_weights: [N] — per-token importance weight from ViT attention
        K:               int — number of slots

    Returns:
        slot_means: [K, D] — weighted mean per slot
        slot_has_tokens: [K] — bool, whether slot received any tokens
    """
    D = token_embeds.shape[1]
    device = token_embeds.device
    slot_means = torch.zeros(K, D, device=device, dtype=token_embeds.dtype)
    slot_has_tokens = torch.zeros(K, dtype=torch.bool, device=device)

    for ki in range(K):
        mask = (assignments == ki)
        if mask.any():
            w = salience_weights[mask].unsqueeze(-1)  # [n_i, 1]
            weighted = token_embeds[mask] * w  # [n_i, D]
            slot_means[ki] = weighted.sum(dim=0) / w.sum().clamp(min=1e-8)
            slot_has_tokens[ki] = True

    return slot_means, slot_has_tokens


def dstm_process_frame(inputs_embeds, vis_indices, streaming_args, is_keyframe=False):
    """
    Core DSTM processing for one frame/chunk.

    Called from stamp_temporal_stage1_prune (after pruning decision but before
    sequence compression) or in DSTM-only mode (no pruning).

    Args:
        inputs_embeds:  [1, seq_len, D] — full sequence embeddings
        vis_indices:    [N_vis] — positions of visual tokens in sequence
        streaming_args: StreamingArgs — contains DSTM state and config
        is_keyframe:    bool — first chunk flag

    Modifies:
        inputs_embeds in-place: injects scene + delta state tokens at low-salience positions
        streaming_args.dstm_*: updates all DSTM memory state
    """
    sa = streaming_args
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    N_vis = vis_indices.shape[0]

    # Config
    K_s = getattr(sa, 'dstm_scene_tokens', 16)
    K_d = getattr(sa, 'dstm_delta_tokens', 16)
    gamma_s = getattr(sa, 'dstm_gamma_scene', 0.05)
    gamma_d = getattr(sa, 'dstm_gamma_delta', 0.2)
    beta = getattr(sa, 'dstm_surprise_beta', 5.0)
    tau = getattr(sa, 'dstm_surprise_tau', 0.3)

    K_total = K_s + K_d  # total memory slots
    vis_embeds = inputs_embeds[0, vis_indices, :].float()  # [N_vis, D]
    D = vis_embeds.shape[1]

    # Get salience weights for weighted updates
    vit_salience = getattr(sa, 'stamp_temporal_vit_salience', None)
    if vit_salience is not None and vit_salience.shape[0] == N_vis:
        salience = vit_salience.to(device).float()
        # Normalize to [0, 1] for weighting
        s_min, s_max = salience.min(), salience.max()
        if s_max > s_min:
            salience = (salience - s_min) / (s_max - s_min)
        # Add floor so low-salience tokens still contribute somewhat
        salience = salience * 0.8 + 0.2  # range [0.2, 1.0]
    else:
        salience = torch.ones(N_vis, device=device)

    # ─── KEYFRAME: Initialize memory ────────────────────────────────────
    if is_keyframe and sa.dstm_scene_state is None:
        K_s_actual = min(K_s, N_vis // 2)
        K_d_actual = min(K_d, N_vis // 2)

        # Initialize scene tokens from evenly-spaced high-salience visual tokens
        if vit_salience is not None and vit_salience.shape[0] == N_vis:
            # Pick top tokens by salience, then subsample evenly
            sal_dev = vit_salience.to(device)
            _, top_idx = sal_dev.topk(min(N_vis, K_s_actual * 4))
            top_idx_sorted, _ = top_idx.sort()
            seed_idx = top_idx_sorted[torch.linspace(0, len(top_idx_sorted) - 1, K_s_actual, device=device).long()]
        else:
            seed_idx = torch.linspace(0, N_vis - 1, K_s_actual, device=device).long()

        scene_init = vis_embeds[seed_idx].clone()  # [K_s, D]

        # Refine with one k-means iteration
        assignments, _ = _assign_tokens_to_slots(vis_embeds, scene_init)
        for ki in range(K_s_actual):
            mask = (assignments == ki)
            if mask.any():
                w = salience[mask].unsqueeze(-1)
                scene_init[ki] = (vis_embeds[mask] * w).sum(0) / w.sum().clamp(min=1e-8)

        sa.dstm_scene_state = scene_init.detach().cpu()
        sa.dstm_scene_counts = torch.ones(K_s_actual)

        # Initialize delta tokens to zero (no change yet on first frame)
        sa.dstm_delta_state = torch.zeros(K_d_actual, D, dtype=torch.float32)
        sa.dstm_delta_counts = torch.zeros(K_d_actual)

        # Store current slot means for next frame's delta computation
        slot_means, _ = _salience_weighted_slot_means(vis_embeds, assignments, salience, K_s_actual)
        sa.dstm_prev_slot_means = slot_means.detach().cpu()

        sa.dstm_temporal_complexity = 0.0
        return  # Don't inject on keyframe — no history yet

    # ─── NON-KEYFRAME: Update memory with surprise gating ───────────────
    if sa.dstm_scene_state is None:
        return  # Safety: shouldn't happen but guard anyway

    K_s_actual = sa.dstm_scene_state.shape[0]
    K_d_actual = sa.dstm_delta_state.shape[0]
    scene_on_device = sa.dstm_scene_state.to(device)  # [K_s, D]
    delta_on_device = sa.dstm_delta_state.to(device)   # [K_d, D]

    # 1. Assign current visual tokens to scene slots
    assignments, _ = _assign_tokens_to_slots(vis_embeds, scene_on_device)

    # 2. Compute salience-weighted slot means for current frame
    slot_means, slot_has_tokens = _salience_weighted_slot_means(
        vis_embeds, assignments, salience, K_s_actual
    )

    # 3. Compute surprise gates for scene memory
    scene_gates = _compute_surprise_gate(slot_means, scene_on_device, beta, tau)  # [K_s]
    # Zero out gates for slots with no tokens this frame
    scene_gates = scene_gates * slot_has_tokens.float()

    # 4. Surprise-gated scene update: S_k = (1 - γ_s · g_k) · S_k + γ_s · g_k · μ_k
    gate_expanded = (gamma_s * scene_gates).unsqueeze(-1)  # [K_s, 1]
    scene_on_device = (1.0 - gate_expanded) * scene_on_device + gate_expanded * slot_means

    # 5. Compute temporal deltas: Δ_k = μ_k^t - μ_k^{t-1}
    if sa.dstm_prev_slot_means is not None:
        prev_means = sa.dstm_prev_slot_means.to(device)  # [K_s, D]
        # Delta is the CHANGE in each slot's representation
        raw_deltas = slot_means - prev_means  # [K_s, D]

        # Map scene slot deltas to delta memory slots
        # If K_d == K_s, 1:1 mapping. If K_d < K_s, take top-K_d by delta magnitude.
        delta_magnitudes = raw_deltas.norm(dim=-1)  # [K_s]
        if K_d_actual >= K_s_actual:
            delta_incoming = raw_deltas[:K_d_actual]
            delta_gates = scene_gates[:K_d_actual]
        else:
            _, top_delta_idx = delta_magnitudes.topk(K_d_actual)
            delta_incoming = raw_deltas[top_delta_idx]  # [K_d, D]
            delta_gates = scene_gates[top_delta_idx]

        # 6. Surprise-gated delta update: D_k = (1 - γ_d · g_k) · D_k + γ_d · g_k · Δ_k
        delta_gate_expanded = (gamma_d * delta_gates).unsqueeze(-1)  # [K_d, 1]
        delta_on_device = (1.0 - delta_gate_expanded) * delta_on_device + delta_gate_expanded * delta_incoming

        # Track temporal complexity (how much is changing across the video)
        sa.dstm_temporal_complexity = (
            0.9 * sa.dstm_temporal_complexity +
            0.1 * delta_magnitudes.mean().item()
        )

    # 7. Save updated state
    sa.dstm_scene_state = scene_on_device.detach().cpu()
    sa.dstm_delta_state = delta_on_device.detach().cpu()
    sa.dstm_prev_slot_means = slot_means.detach().cpu()

    # ─── INJECT: Blend memory tokens into low-salience visual tokens ────
    # CRITICAL FIX: Use additive blending instead of replacement.
    # Replacement destroys fine-grained spatial info (especially text reading).
    # Blending preserves original ViT features while adding temporal context.
    if vit_salience is None or vit_salience.shape[0] != N_vis:
        return  # Can't inject without salience ranking

    blend_alpha = getattr(sa, 'dstm_blend_alpha', 0.2)  # how much memory to blend in
    salience_for_ranking = vit_salience.to(device)

    # Inject into fewer tokens: at most 10% of visual tokens (was 30%)
    K_s_inject = min(K_s_actual, N_vis // 10)
    complexity_scale = min(1.0, sa.dstm_temporal_complexity / 0.5)
    K_d_inject = max(1, int(min(K_d_actual, N_vis // 10) * max(0.25, complexity_scale)))
    K_inject = K_s_inject + K_d_inject
    K_inject = min(K_inject, N_vis // 10)  # hard cap at 10%
    K_s_inject = min(K_s_inject, max(1, K_inject * K_s_actual // (K_s_actual + K_d_actual)))
    K_d_inject = K_inject - K_s_inject

    if K_inject <= 0:
        return

    # Find the K_inject lowest-salience visual token positions
    _, bottom_idx = salience_for_ranking.topk(K_inject, largest=False)
    inject_seq_pos = vis_indices[bottom_idx]

    # Prepare injection tokens: [K_s_inject scene tokens, K_d_inject delta tokens]
    scene_to_inject = sa.dstm_scene_state[:K_s_inject].to(device).to(dtype)
    delta_to_inject = sa.dstm_delta_state[:K_d_inject].to(device).to(dtype)
    inject_tokens = torch.cat([scene_to_inject, delta_to_inject], dim=0)  # [K_inject, D]

    # Additive blending: preserve original token, add temporal context
    original = inputs_embeds[0, inject_seq_pos[:K_inject], :]
    inputs_embeds[0, inject_seq_pos[:K_inject], :] = (
        (1.0 - blend_alpha) * original + blend_alpha * inject_tokens[:K_inject]
    )


def dstm_keyframe_update(inputs_embeds, vis_indices, streaming_args):
    """
    Update DSTM memory on a keyframe chunk (no pruning, but still accumulate).
    Called when is_keyframe=True and memory already exists.
    """
    sa = streaming_args
    if sa.dstm_scene_state is None:
        # First keyframe — initialize
        dstm_process_frame(inputs_embeds, vis_indices, streaming_args, is_keyframe=True)
        return

    device = inputs_embeds.device
    N_vis = vis_indices.shape[0]
    vis_embeds = inputs_embeds[0, vis_indices, :].float()

    K_s_actual = sa.dstm_scene_state.shape[0]
    scene_on_device = sa.dstm_scene_state.to(device)

    # Get salience
    vit_salience = getattr(sa, 'stamp_temporal_vit_salience', None)
    if vit_salience is not None and vit_salience.shape[0] == N_vis:
        salience = vit_salience.to(device).float()
        s_min, s_max = salience.min(), salience.max()
        if s_max > s_min:
            salience = (salience - s_min) / (s_max - s_min)
        salience = salience * 0.8 + 0.2
    else:
        salience = torch.ones(N_vis, device=device)

    # Assign and update with full (ungated) EMA — keyframes are always important
    assignments, _ = _assign_tokens_to_slots(vis_embeds, scene_on_device)
    slot_means, slot_has_tokens = _salience_weighted_slot_means(
        vis_embeds, assignments, salience, K_s_actual
    )

    # Stronger update on keyframes (gamma_s * 3, capped at 0.5)
    gamma_kf = min(0.5, getattr(sa, 'dstm_gamma_scene', 0.05) * 3.0)
    for ki in range(K_s_actual):
        if slot_has_tokens[ki]:
            scene_on_device[ki] = (1.0 - gamma_kf) * scene_on_device[ki] + gamma_kf * slot_means[ki]

    sa.dstm_scene_state = scene_on_device.detach().cpu()
    sa.dstm_prev_slot_means = slot_means.detach().cpu()
