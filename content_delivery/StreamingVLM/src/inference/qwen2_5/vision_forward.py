from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from typing import Optional, Tuple
from torch.nn import functional as F
from types import MethodType
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import logger,apply_rotary_pos_emb_flashatt,flash_attn_varlen_func
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import logger,apply_rotary_pos_emb_flashatt, apply_rotary_pos_emb_vision
import math

def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 
                                dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
    """Fallback implementation using standard PyTorch attention"""
    batch_size = len(cu_seqlens_q) - 1
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    
    # Convert varlen to batched format
    q_batched = torch.zeros(batch_size, max_seqlen_q, num_heads, head_dim, 
                            dtype=q.dtype, device=q.device)
    k_batched = torch.zeros(batch_size, max_seqlen_k, num_heads, head_dim,
                            dtype=k.dtype, device=k.device)
    v_batched = torch.zeros(batch_size, max_seqlen_k, num_heads, head_dim,
                            dtype=v.dtype, device=v.device)
    
    for i in range(batch_size):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
        q_batched[i, :q_end-q_start] = q[q_start:q_end]
        k_batched[i, :k_end-k_start] = k[k_start:k_end]
        v_batched[i, :k_end-k_start] = v[k_start:k_end]
    
    # Standard attention
    q_batched = q_batched.transpose(1, 2)
    k_batched = k_batched.transpose(1, 2)
    v_batched = v_batched.transpose(1, 2)
    
    scale = softmax_scale if softmax_scale else (1.0 / (head_dim ** 0.5))
    attn_output = F.scaled_dot_product_attention(q_batched, k_batched, v_batched, 
                                                    dropout_p=dropout_p, scale=scale)
    attn_output = attn_output.transpose(1, 2)
    
    # Convert back to varlen
    output = torch.zeros_like(q)
    for i in range(batch_size):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        output[q_start:q_end] = attn_output[i, :q_end-q_start]
    
    return output

def streaming_visual_attention_forward_with_weights(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vision attention forward that extracts per-token salience and entropy.
        Used only at the ViT layer selected for STAMP-Temporal.

        Memory-efficient: never materializes the full [heads, seq, seq] attention matrix.
        Instead, processes per-segment and accumulates only [seq_len] salience and entropy.

        Returns:
            attn_output: [seq_length, hidden_size]
            salience:    [seq_length] — how much attention each token receives (averaged over heads)
            entropy:     scalar — mean attention entropy across all heads and query positions
        """
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)

        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)  # [seq_len, num_heads, head_dim]
        k = k.squeeze(0)

        head_dim = q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)
        num_segments = len(cu_seqlens) - 1

        # Accumulate per-token salience and entropy WITHOUT storing full attention matrix
        salience = torch.zeros(seq_length, dtype=torch.float32, device=q.device)
        entropy_sum = torch.zeros(1, dtype=torch.float32, device=q.device)
        total_queries = 0
        attn_output = torch.zeros(seq_length, self.num_heads, head_dim,
                                  dtype=q.dtype, device=q.device)

        for seg in range(num_segments):
            s = cu_seqlens[seg].item()
            e = cu_seqlens[seg + 1].item()
            seg_len = e - s
            if seg_len <= 0:
                continue
            # q_seg: [seg_len, num_heads, head_dim] → [num_heads, seg_len, head_dim]
            q_seg = q[s:e].permute(1, 0, 2)
            k_seg = k[s:e].permute(1, 0, 2)
            v_seg = v[s:e].permute(1, 0, 2)

            # [num_heads, seg_len, seg_len] — only for this segment (small window)
            scores = torch.matmul(q_seg, k_seg.transpose(-2, -1)) * scale
            weights = torch.softmax(scores, dim=-1).float()

            # Salience: sum of attention received per key token, averaged over heads
            # weights[:, j, i] = attention from query j to key i
            # sum over queries (dim=1) → [num_heads, seg_len], mean over heads → [seg_len]
            seg_salience = weights.sum(dim=1).mean(dim=0)  # [seg_len]
            salience[s:e] = seg_salience

            # Entropy: -sum(p * log(p)) per query, per head
            eps = 1e-10
            seg_entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)  # [num_heads, seg_len]
            entropy_sum += seg_entropy.sum()
            total_queries += self.num_heads * seg_len

            # Output: standard attention output
            out_seg = torch.matmul(weights.to(v_seg.dtype), v_seg)
            attn_output[s:e] = out_seg.permute(1, 0, 2)

        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        frame_entropy = entropy_sum / max(total_queries, 1)
        return attn_output, salience.detach(), frame_entropy.detach()


def streaming_visual_attention_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        else:
            cos, sin = position_embeddings
        # q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q, k = apply_rotary_pos_emb_vision(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

def streaming_visual_block_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states),
        cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb,
        position_embeddings=position_embeddings,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))

    return hidden_states


def streaming_visual_encoder_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """Vision backbone: patch-embed → window reordering → multi-layer local/global attention → merge

    STAMP-Temporal extraction modes:
      - Single layer: self._stamp_temporal_extract_layer = int (e.g. 31)
      - Multi-layer fusion: self._stamp_temporal_extract_layers = list (e.g. [7, 15, 23, 31])
        Extracts salience at each global attention layer and fuses via weighted average.
        Later layers get exponentially higher weight (captures information flow depth).
    """

    # 1. Patch → Token
    hidden_states = self.patch_embed(hidden_states)

    # 2. Rotary position encoding
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    # 3. window indexing
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens, device=hidden_states.device, dtype=torch.int32
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)  # Remove duplicate window start points

    # 4. Rearrange tokens within the same window to adjacent memory blocks
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit,
                                          self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index]
    hidden_states = hidden_states.reshape(seq_len, -1)

    # Position encoding is also reordered
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit,
                                            self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index].reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    # 5. Global cu_seqlens
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                         grid_thw[:, 0]).cumsum(dim=0).to(torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0))

    # STAMP-Temporal: extraction config
    extract_layer = getattr(self, '_stamp_temporal_extract_layer', None)
    extract_layers = getattr(self, '_stamp_temporal_extract_layers', None)  # multi-layer mode
    # Determine which layers need attention extraction
    extract_set = set()
    if extract_layers is not None:
        extract_set = set(extract_layers)
    elif extract_layer is not None:
        extract_set = {extract_layer}

    # Multi-layer salience accumulator
    layer_saliences = {}  # layer_idx → [seq_len] salience
    layer_entropies = {}  # layer_idx → scalar entropy

    # 6. Swin blocks
    for idx, blk in enumerate(self.blocks):
        cu_now = cu_seqlens if idx in self.fullatt_block_indexes else cu_window_seqlens

        if idx in extract_set:
            # At this layer, use the attention-weight-capturing forward
            normed = blk.norm1(hidden_states)

            attn_out, salience, frame_entropy = streaming_visual_attention_forward_with_weights(
                blk.attn, normed, cu_seqlens=cu_now, position_embeddings=position_embeddings,
            )
            hidden_states = hidden_states + attn_out
            hidden_states = hidden_states + blk.mlp(blk.norm2(hidden_states))

            layer_saliences[idx] = salience
            layer_entropies[idx] = frame_entropy
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_now, position_embeddings=position_embeddings)

    # 7. Merge + restore token order
    hidden_states = self.merger(hidden_states)
    reverse_idx = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_idx]

    # STAMP-Temporal: compute post-merger salience scores
    if layer_saliences:
        smu = self.spatial_merge_unit  # typically 4

        def _merge_salience(raw_sal):
            """Average salience within merge groups and undo window reordering."""
            n_merged = raw_sal.shape[0] // smu
            sal_grouped = raw_sal[:n_merged * smu].reshape(n_merged, smu)
            merged = sal_grouped.mean(dim=1)  # [n_merged]
            return merged[reverse_idx]

        if len(layer_saliences) == 1:
            # Single-layer mode (backward compatible)
            layer_idx = next(iter(layer_saliences))
            merged_salience = _merge_salience(layer_saliences[layer_idx])
            final_entropy = layer_entropies[layer_idx]
        else:
            # Multi-layer fusion: weighted average with exponential layer weighting
            # Later layers get higher weight: w_i = 2^(position_in_list)
            sorted_layers = sorted(layer_saliences.keys())
            fused = torch.zeros(layer_saliences[sorted_layers[0]].shape[0] // smu,
                                dtype=torch.float32,
                                device=layer_saliences[sorted_layers[0]].device)
            fused = fused[reverse_idx]  # pre-allocate in unwindowed order
            fused = torch.zeros_like(fused)
            weight_sum = 0.0
            entropy_weighted = 0.0
            # N7: when flag is set, weight full-attention layers equally instead of 1:2:4:8
            equal_w = getattr(self, '_stamp_temporal_equal_layer_weights', False)
            for pos, lidx in enumerate(sorted_layers):
                w = 1.0 if equal_w else (2.0 ** pos)  # N7 uses equal weights
                merged = _merge_salience(layer_saliences[lidx])
                # Normalize each layer's salience to [0,1] before fusion
                s_min = merged.min()
                s_max = merged.max()
                if s_max > s_min:
                    merged = (merged - s_min) / (s_max - s_min)
                fused = fused + w * merged
                entropy_weighted += w * layer_entropies[lidx]
                weight_sum += w
            merged_salience = fused / weight_sum
            final_entropy = entropy_weighted / weight_sum

        self._stamp_temporal_vit_salience = merged_salience.detach().cpu()
        self._stamp_temporal_vit_entropy = final_entropy.cpu() if isinstance(final_entropy, torch.Tensor) else torch.tensor(final_entropy)

        # STAR: store per-layer normalized saliences for divergence computation
        if len(layer_saliences) > 1:
            per_layer_norm = []
            for lidx in sorted(layer_saliences.keys()):
                m = _merge_salience(layer_saliences[lidx])
                s_min, s_max = m.min(), m.max()
                if s_max > s_min:
                    m = (m - s_min) / (s_max - s_min)
                per_layer_norm.append(m)
            self._stamp_temporal_per_layer_salience = torch.stack(per_layer_norm, dim=0).detach().cpu()  # [n_layers, N_vis]
        else:
            self._stamp_temporal_per_layer_salience = None

    return hidden_states

