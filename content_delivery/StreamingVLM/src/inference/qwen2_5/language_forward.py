from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor                                           
from qwen_vl_utils import process_vision_info
import torch                                                                                                                        
from typing import Optional, Tuple, List, Union                                                                                     
from torch.nn import functional as F                                                                                                
from types import MethodType                                                                                                        
  # from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import logger,rotate_half,Cache,BaseModelOutputWithPast,repeat_kv,_flash_attention_forward, StaticCache, SlidingWindowCache, AttentionMaskConverter, make_flex_block_causal_mask, BlockMask                                                                      
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import logger,rotate_half,Cache,BaseModelOutputWithPast,repeat_kv,StaticCache, SlidingWindowCache, AttentionMaskConverter                                                                             
from streaming_vlm.inference.generate.streaming_cache import StreamingCache
from streaming_vlm.inference.streaming_args import StreamingArgs                                                                    
import math                                                                                                                         
import torch.nn as nn
                                                                                                                                      
                                              
  # Try to import _flash_attention_forward                                                                                            
try:                                                                                                                                
      from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import _flash_attention_forward                                         
except ImportError:                                                                                                                 
      try:                                                                                                                            
          from transformers.modeling_flash_attention_utils import _flash_attention_forward
      except ImportError:                                                                                                             
          # Provide a fallback implementation
          import torch.nn.functional as F                                                                                             
                                                                                                                                      
          def _flash_attention_forward(   
              query_states,                                                                                                           
              key_states,                                                                                                             
              value_states,
              attention_mask,                                                                                                         
              query_length,
              dropout=0.0,                    
              softmax_scale=None,         
              is_causal=False,
              **kwargs                                                                                                                
          ):                                  
              """Fallback to standard attention"""                                                                                    
              if softmax_scale is None:       
                  softmax_scale = query_states.size(-1) ** -0.5                                                                       
   
              attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * softmax_scale                                 
                                          
              if attention_mask is not None:                                                                                          
                  attn_weights = attn_weights + attention_mask                                                                        
                                              
              attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)                              
                                          
              if dropout > 0.0:                                                                                                       
                  attn_weights = F.dropout(attn_weights, p=dropout)                                                                   
                                                                                                                                      
              attn_output = torch.matmul(attn_weights, value_states)                                                                  
                  
              return attn_output, attn_weights                                                                                        
                                          
                                                                                                                                      
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):                                                
   
      """Modified to calculate Rope position during decode attention when Q length is not equal to position embedding length          
      Right aligned                       
      Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors                                         
  (https://qwenlm.github.io/blog/qwen2-vl/).                                                                                          
                                          
      Explanation:                                                                                                                    
          Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding                
          sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
          vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.               
          Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
          For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,             
          height and width) of text embedding is always the same, so the text embedding rotary position embedding has no              
          difference with modern LLMs.                                                                                                
                                                                                                                                      
      Args:                                                                                                                           
          q (`torch.Tensor`): The query tensor.                                                                                       
          k (`torch.Tensor`): The key tensor.                                                                                         
          cos (`torch.Tensor`): The cosine part of the rotary embedding.                                                              
          sin (`torch.Tensor`): The sine part of the rotary embedding.                                                                
          position_ids (`torch.Tensor`):  
              The position indices of the tokens corresponding to the query and key tensors. For example, this can be
              used to pass offsetted position ids when working with a KV-cache.                                                       
          mrope_section(`List(int)`):         
              Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.                     
          unsqueeze_dim (`int`, *optional*, defaults to 1):                                                                           
              The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and                     
              sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note              
              that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and             
              k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes                             
              cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have              
              the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.                                             
      Returns:                                                                                                                        
          `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.                  
      """                                                                                                                             
      mrope_section = mrope_section * 2                                                                                               
      if q.shape[2] != k.shape[2]:                                                                                                    
          q_cos = cos[:, :, -q.shape[2]:]                                                                                             
          q_sin = sin[:, :, -q.shape[2]:]     
          q_cos = torch.cat([m[i % 3] for i, m in enumerate(q_cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(                  
              unsqueeze_dim                   
          )                                                                                                                           
          q_sin = torch.cat([m[i % 3] for i, m in enumerate(q_sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(                  
              unsqueeze_dim                                                                                                           
          )                                                                                                                           
          q_embed = (q * q_cos) + (rotate_half(q) * q_sin)                                                                            
                                                                                                                                      
      cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(                          
          unsqueeze_dim                                                                                                               
      )                                                                                                                               
      sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(                          
          unsqueeze_dim                                                                                                               
      )                                                                                                                               
      if q.shape[2] == k.shape[2]:
          q_embed = (q * cos) + (rotate_half(q) * sin)                                                                                
      k_embed = (k * cos) + (rotate_half(k) * sin)
      return q_embed, k_embed                                                                                                         
                  
def streaming_text_eager_attn_forward(                                                                                              
          self,                           
          hidden_states: torch.Tensor,                                                                                                
          attention_mask: Optional[torch.Tensor] = None,                                                                              
          position_ids: Optional[torch.LongTensor] = None,
          past_key_value: Optional[Cache] = None,                                                                                     
          output_attentions: bool = False,    
          use_cache: bool = False,                                                                                                    
          cache_position: Optional[torch.LongTensor] = None,
          position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC                 
          streaming_args: StreamingArgs = None,
      ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:                                                
          bsz, q_len, _ = hidden_states.size()
                                                                                                                                      
          query_states = self.q_proj(hidden_states)                                                                                   
          key_states = self.k_proj(hidden_states)                                                                                     
          value_states = self.v_proj(hidden_states)                                                                                   
                                                                                                                                      
          query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)                                             
          key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)                                                 
          value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
                                                                                                                                      
          cos, sin = position_embeddings  
                                                                                                                                      
          if streaming_args.pos_mode == "append":                                                                                     
              # In append mode, pass in position embedding corresponding to newly added input_ids
              query_states, key_states = apply_multimodal_rotary_pos_emb(                                                             
                  query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]                                              
              )                               
                                                                                                                                      
          if past_key_value is not None:                                                                                              
              cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models                    
              key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, position_ids, cache_kwargs)  
                                                                                                                                      
          if streaming_args.pos_mode == "shrink":
              # In shrink mode, pass in position embedding corresponding to entire input_ids, need to add after update kv cache       
              k_len = key_states.shape[2]                                                                                             
              cos_len = cos.shape[2]                                                                                                  
              if cos_len >= k_len:                                                                                                    
                  # cos/sin covers all key positions — trim to match                                                                  
                  cos_k = cos[:, :, :k_len]                                                                                           
                  sin_k = sin[:, :, :k_len]                                                                                           
                  query_states, key_states = apply_multimodal_rotary_pos_emb(query_states, key_states, cos_k, sin_k,                  
  self.rope_scaling["mrope_section"])                                                                                                 
              else:                                                                                                                   
                  # KV cache larger than cos/sin (multi-chunk streaming) — only rotate the new (last cos_len) tokens                  
                  old_keys = key_states[:, :, :-cos_len, :]                                                                           
                  new_keys = key_states[:, :, -cos_len:, :]                                                                           
                  query_states, new_keys = apply_multimodal_rotary_pos_emb(query_states, new_keys, cos, sin,                          
  self.rope_scaling["mrope_section"])                                                                                                 
                  key_states = torch.cat([old_keys, new_keys], dim=2)                                                                 
                                                                                                                                      
          # repeat k/v heads if n_kv_heads < n_heads                                                                                  
          key_states = repeat_kv(key_states, self.num_key_value_groups)
          value_states = repeat_kv(value_states, self.num_key_value_groups)                                                           
                                                                                                                                      
          if output_attentions and q_len > 1:
              # FastV capture during prefill: avoid O(n^2) full attention matrix.                                                     
              # Use scaled_dot_product_attention for the actual output, then compute                                                  
              # only the last query row manually (FastV only uses attention[-1]).                                                     
              attn_mask_sdpa = attention_mask[:, :, :, :key_states.shape[-2]] if attention_mask is not None else None                 
              attn_output = F.scaled_dot_product_attention(                                                                           
                  query_states, key_states, value_states,                                                                             
                  attn_mask=attn_mask_sdpa,                                                                                           
                  dropout_p=self.attention_dropout if self.training else 0.0,                                                         
              )                                                                                                                       
              # Compute last-row attention for FastV scoring                                                                          
              last_q = query_states[:, :, -1:, :]                                                                                     
              last_attn = torch.matmul(last_q, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)                                 
              if attention_mask is not None:                                                                                          
                  last_attn = last_attn + attention_mask[:, :, -1:, :key_states.shape[-2]]
              if query_states.dtype == torch.float16:                                                                                 
                  last_attn = torch.where(torch.isinf(last_attn), torch.zeros_like(last_attn), last_attn)                             
              last_attn = nn.functional.softmax(last_attn, dim=-1, dtype=torch.float32).to(query_states.dtype)
              # Return only the last row [bsz, heads, 1, k_len] — FastV only uses [-1]
              attn_weights = last_attn                                                                                  
          else:                                                                                                                       
              attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)                        
                                                                                                                                      
              if attention_mask is not None:  # no matter the length, we just slice it                                                
                  causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                  attn_weights = attn_weights + causal_mask                                                                           
                                                                                                                                      
              # Fix precision issues in Qwen2-VL float16 inference                                                                    
              if query_states.dtype == torch.float16:                                                                                 
                  attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)
                                                                                                                                      
              # upcast attention to fp32      
              attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)                  
              attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)                    
              attn_output = torch.matmul(attn_weights, value_states)                                                                  
                                                                                                                                      
          if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):                                                       
              raise ValueError(                                                                                                       
                  f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"                            
                  f" {attn_output.size()}"                                                                                            
              )                           
                                                                                                                                      
          attn_output = attn_output.transpose(1, 2).contiguous()                                                                      
          attn_output = attn_output.reshape(bsz, q_len, -1)                                                                           
                                                                                                                                      
          attn_output = self.o_proj(attn_output)                                                                                      
                                          
          if not output_attentions:                                                                                                   
              attn_weights = None                                                                                                     
          return attn_output, attn_weights, past_key_value
                                                                                                                                      
                                          
def streaming_text_flash_attn_forward(
          self,                                                                                                                       
          hidden_states: torch.Tensor,
          attention_mask: Optional[torch.Tensor] = None,                                                                              
          position_ids: Optional[torch.LongTensor] = None,
          past_key_value: Optional[Cache] = None,
          output_attentions: bool = False,                                                                                            
          use_cache: bool = False,        
          cache_position: Optional[torch.LongTensor] = None,                                                                          
          position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC                 
          streaming_args: StreamingArgs = None,
      ):                                                                                                                              
          if output_attentions:
              # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.          
              logger.warning_once(            
                  "Streaming Qwen is using Flash Attention, but `flash attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
              )                                                                                                                       
              return streaming_text_eager_attn_forward(
                  self,                                                                                                               
                  hidden_states=hidden_states,
                  attention_mask=attention_mask,                                                                                      
                  position_ids=position_ids,                                                                                          
                  past_key_value=past_key_value,
                  output_attentions=output_attentions,                                                                                
                  use_cache=use_cache,                                                                                                
                  cache_position=cache_position,
                  position_embeddings=position_embeddings,                                                                            
                  streaming_args=streaming_args,
              )
                                                                                                                                      
          bsz, q_len, _ = hidden_states.size()
                                                                                                                                      
          query_states = self.q_proj(hidden_states)
          key_states = self.k_proj(hidden_states)                                                                                     
          value_states = self.v_proj(hidden_states)
                                                                                                                                      
          query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
          key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
          value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
          # Because the input can be padded, the absolute sequence length depends on the max position id.                             
          cos, sin = position_embeddings      
          if streaming_args.pos_mode == "append":                                                                                     
              # In append mode, pass in position embedding corresponding to newly added input_ids
              query_states, key_states = apply_multimodal_rotary_pos_emb(                                                             
                  query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
              )                                                                                                                       
                                          
          if past_key_value is not None:                                                                                              
              cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models                    
              key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                                                                                                                                      
          if streaming_args.pos_mode == "shrink":
              # In shrink mode, pass in position embedding corresponding to entire input_ids, need to add after update kv cache       
              k_len = key_states.shape[2]
              cos_len = cos.shape[2]                                                                                                  
              if cos_len >= k_len:            
                  # cos/sin covers all key positions — trim to match                                                                  
                  cos_k = cos[:, :, :k_len]
                  sin_k = sin[:, :, :k_len]                                                                                           
                  query_states, key_states = apply_multimodal_rotary_pos_emb(query_states, key_states, cos_k, sin_k,
  self.rope_scaling["mrope_section"])                                                                                                 
              else:                       
                  # KV cache larger than cos/sin (multi-chunk streaming) — only rotate the new (last cos_len) tokens                  
                  old_keys = key_states[:, :, :-cos_len, :]                                                                           
                  new_keys = key_states[:, :, -cos_len:, :]                                                                           
                  query_states, new_keys = apply_multimodal_rotary_pos_emb(query_states, new_keys, cos, sin,                          
  self.rope_scaling["mrope_section"])         
                  key_states = torch.cat([old_keys, new_keys], dim=2)                                                                 
   
                                                                                                                                      
          # repeat k/v heads if n_kv_heads < n_heads
          key_states = repeat_kv(key_states, self.num_key_value_groups)                                                               
          value_states = repeat_kv(value_states, self.num_key_value_groups)                                                           
          dropout_rate = 0.0 if not self.training else self.attention_dropout
                                                                                                                                      
          # In PEFT, usually we cast the layer norms in float32 for training stability reasons
          # therefore the input hidden states gets silently casted in float32. Hence, we need                                         
          # cast them back in float16 just to be sure everything works as expected.
          input_dtype = query_states.dtype                                                                                            
          if input_dtype == torch.float32:
              if torch.is_autocast_enabled():                                                                                         
                  target_dtype = torch.get_autocast_gpu_dtype()                                                                       
              # Handle the case where the model is quantized                                                                          
              elif hasattr(self.config, "_pre_quantization_dtype"):                                                                   
                  target_dtype = self.config._pre_quantization_dtype
              else:                                                                                                                   
                  target_dtype = self.q_proj.weight.dtype                                                                             
                                                                                                                                      
              logger.warning_once(                                                                                                    
                  f"The input hidden states seems to be silently casted in float32, this might be related to"                         
                  f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"            
                  f" {target_dtype}."         
              )                                                                                                                       
                                                                                                                                      
              query_states = query_states.to(target_dtype)                                                                            
              key_states = key_states.to(target_dtype)                                                                                
              value_states = value_states.to(target_dtype)                                                                            
                                          
          # Reashape to the expected shape for Flash Attention
          query_states = query_states.transpose(1, 2)                                                                                 
          key_states = key_states.transpose(1, 2)
          value_states = value_states.transpose(1, 2)                                                                                 
                                          
          if (                                                                                                                        
              self.config.use_sliding_window                                                                                          
              and getattr(self.config, "sliding_window", None) is not None
              and self.layer_idx >= self.config.max_window_layers                                                                     
          ):                                  
              sliding_window = self.config.sliding_window                                                                             
          else:
              sliding_window = None                                                                                                   
                  
          attn_output = _flash_attention_forward(                                                                                     
              query_states,               
              key_states,
              value_states,                                                                                                           
              None, # Modified here       
              q_len,                                                                                                                  
              dropout=dropout_rate,                                                                                                   
              sliding_window=sliding_window,
              is_causal=self.is_causal,                                                                                               
              use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
          )                                                                                                                           
                                          
          attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()                                                              
          attn_output = self.o_proj(attn_output)                                                                                      
                                                                                                                                      
          if not output_attentions:                                                                                                   
              attn_weights = None
                                                                                                                                      
          return attn_output, attn_weights, past_key_value
                                                                                                                                      
def streaming_text_decoder_layer_forward(
          self,                                                                                                                       
          hidden_states: torch.Tensor,
          attention_mask: Optional[torch.Tensor] = None,                                                                              
          position_ids: Optional[torch.LongTensor] = None,
          past_key_value: Optional[Tuple[torch.Tensor]] = None,
          output_attentions: Optional[bool] = False,                                                                                  
          use_cache: Optional[bool] = False,                                                                                          
          cache_position: Optional[torch.LongTensor] = None,                                                                          
          position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,                                                    
          streaming_args: StreamingArgs = None,
          **kwargs,                                                                                                                   
      ):                                                                                                                              
                                              
      residual = hidden_states                                                                                                        
      hidden_states = self.input_layernorm(hidden_states)
                                                                                                                                      
      hidden_states, self_attn_weights, present_key_value = self.self_attn(
          hidden_states=hidden_states,                                                                                                
          attention_mask=attention_mask,      
          position_ids=position_ids,                                                                                                  
          past_key_value=past_key_value,
          output_attentions=output_attentions,                                                                                        
          use_cache=use_cache,
          cache_position=cache_position,                                                                                              
          position_embeddings=position_embeddings,
          streaming_args=streaming_args,
      )                                                                                                                               
      hidden_states = residual + hidden_states
                                                                                                                                      
      # ---------------- FFN (MLP) ----------------
                                          
      residual = hidden_states                
      hidden_states = self.post_attention_layernorm(hidden_states)                                                                    
      hidden_states = self.mlp(hidden_states)
      hidden_states = residual + hidden_states                                                                                        
                  
      # ---------------- Remaining (others) ----------------                                                                          
      outputs = (hidden_states,)          
      if output_attentions:
          outputs += (self_attn_weights,)                                                                                             
      if use_cache:                           
          outputs += (present_key_value,)                                                                                             
      return outputs
                                                                                                                                      
  # This modification is for using both flashattention and eager methods when visualizing attention map.
def _update_causal_mask(                                                                                                            
          self,   
          attention_mask,                                                                                                             
          input_tensor: torch.Tensor,         
          cache_position: torch.Tensor,                                                                                               
          past_key_values: Cache,
          output_attentions: bool = False,                                                                                            
      ):                                  
          consider_as_flash_attn = (self.config._attn_implementation == "flash_attention_2") and (output_attentions == False)         
          if consider_as_flash_attn:                                                                                                  
              if attention_mask is not None and past_key_values is not None:
                  is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]                                     
                  if is_padding_right:                                                                                                
                      raise ValueError(   
                          "You are attempting to perform batched generation with padding_side='right'"                                
                          " this may lead to unexpected behaviour for Flash Attention version of Qwen2_5_VL. Make sure to "           
                          " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "                                    
                      )                                                                                                               
              if attention_mask is not None and 0.0 in attention_mask:                                                                
                  return attention_mask                                                                                               
              return None                                                                                                             
          if self.config._attn_implementation == "flex_attention":                                                                    
              if isinstance(attention_mask, torch.Tensor):                                                                            
                  try:                                                                                                                
                      from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import make_flex_block_causal_mask                      
                      attention_mask = make_flex_block_causal_mask(attention_mask)                                                    
                  except ImportError:                                                                                                 
                      pass                                                                                                            
              return attention_mask                                                                                                   
                                                                                                                                      
          # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in                 
          # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail               
          # to infer the attention mask.      
          past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0                                   
          using_static_cache = isinstance(past_key_values, StaticCache)
          using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)                                                
                                                                                                                                      
          # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward             
          if (                                                                                                                        
              not consider_as_flash_attn                                                                                              
              and not (using_static_cache or using_sliding_window_cache)                                                              
              and not output_attentions                                                                                               
          ):                                                                                                                          
              if AttentionMaskConverter._ignore_causal_mask_sdpa(
                  attention_mask,                                                                                                     
                  inputs_embeds=input_tensor,
                  past_key_values_length=past_seen_tokens,                                                                            
                  sliding_window=self.config.sliding_window,                                                                          
                  is_training=self.training,  
              ):                                                                                                                      
                  return None
                                                                                                                                      
          dtype = input_tensor.dtype          
          min_dtype = torch.finfo(dtype).min                                                                                          
          sequence_length = input_tensor.shape[1]
          # SlidingWindowCache or StaticCache                                                                                         
          if using_sliding_window_cache or using_static_cache:
              target_length = past_key_values.get_max_cache_shape()                                                                   
          # StreamingCache or no cache                                                                                                
          else:                               
              target_length = (                                                                                                       
                  attention_mask.shape[-1]                                                                                            
                  if isinstance(attention_mask, torch.Tensor)                                                                         
                  else past_seen_tokens + sequence_length + 1                                                                         
              )                                                                                                                       
                                                                                                                                      
          # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
          causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(                                                   
              attention_mask,                                                                                                         
              sequence_length=sequence_length,                                                                                        
              target_length=target_length,                                                                                            
              dtype=dtype,                                                                                                            
              device=input_tensor.device,                                                                                             
              cache_position=cache_position,                                                                                          
              batch_size=input_tensor.shape[0],                                                                                       
              config=self.config,                                                                                                     
              past_key_values=past_key_values,                                                                                        
          )                                                                                                                           
                                                                                                                                      
          if (                                                                                                                        
              not consider_as_flash_attn                                                                                              
              and attention_mask is not None                                                                                          
              and attention_mask.device.type in ["cuda", "xpu", "npu"]                                                                
              and not output_attentions   
          ):                                  
              # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when                
              # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
              # Details: https://github.com/pytorch/pytorch/issues/110213                                                             
              causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
                                                                                                                                      
          return causal_mask                                                                                                          
                                                                                                                                      
def streaming_language_model_forward(                                                                                               
      self,                                                                                                                           
      input_ids: Optional[torch.LongTensor] = None,                                                                                   
      attention_mask: Optional[torch.Tensor] = None,                                                                                  
      position_ids: Optional[torch.LongTensor] = None,
      past_key_values: Optional[List[torch.FloatTensor]] = None,                                                                      
      inputs_embeds: Optional[torch.FloatTensor] = None,
      use_cache: Optional[bool] = None,                                                                                               
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,                                                                                    
      return_dict: Optional[bool] = None,                                                                                             
      cache_position: Optional[torch.LongTensor] = None,
      streaming_args: StreamingArgs = None,                                                                                           
  ) -> Union[Tuple, BaseModelOutputWithPast]:

      # Fallback to model-attached streaming_args for eval mode (Trainer doesn't pass streaming_args)
      if streaming_args is None and hasattr(self, '_streaming_args'):
          streaming_args = self._streaming_args

      output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions                       
      output_hidden_states = (                                                                                                        
          output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
      )                                                                                                                               
      use_cache = use_cache if use_cache is not None else self.config.use_cache
                                                                                                                                      
      return_dict = return_dict if return_dict is not None else self.config.use_return_dict                                           
                                          
      if (input_ids is None) ^ (inputs_embeds is not None):                                                                           
          raise ValueError("You must specify exactly one of input_ids or inputs_embeds")                                              
                                              
      if self.gradient_checkpointing and self.training:                                                                               
          if use_cache:
              logger.warning_once(                                                                                                    
                  "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
              )                                                                                                                       
              use_cache = False                                                                                                       
   
      # torch.jit.trace() doesn't support cache objects in the output                                                                 
      if use_cache and past_key_values is None and not torch.jit.is_tracing():
          past_key_values = StreamingCache()                                                                                          
                                                                                                                                      
      if inputs_embeds is None:               
          inputs_embeds = self.embed_tokens(input_ids)                                                                                
                  
      if cache_position is None:                                                                                                      
          past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
          cache_position = torch.arange(                                                                                              
              past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
          )                                   
                                                                                                                                      
      # the hard coded `3` is for temporal, height and width.
      if position_ids is None:                                                                                                        
          position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
      elif position_ids.dim() == 2:                                                                                                   
          position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
                                                                                                                                      
      # ── STAMP Stage 1: pre-LLM predictive pruning (before layer 0) ──────────
      # Only fires when stamp_r1 is set and visual tokens are present.
      # Prunes inputs_embeds / position_ids / attention_mask / cache_position
      # BEFORE causal_mask is computed so everything stays consistent.
      if streaming_args is not None and streaming_args.stamp_r1 is not None:
          from streaming_vlm.inference.stamp import stamp_stage1_prune
          inputs_embeds, position_ids, attention_mask, cache_position = stamp_stage1_prune(
              inputs_embeds, position_ids, attention_mask, cache_position, streaming_args
          )

      # ── STAMP-Temporal: pre-LLM pruning using ViT-sourced attention ────────
      if streaming_args is not None and getattr(streaming_args, 'stamp_temporal', False) and streaming_args.stamp_temporal_r is not None:
          from streaming_vlm.inference.stamp_temporal import stamp_temporal_stage1_prune
          inputs_embeds, position_ids, attention_mask, cache_position = stamp_temporal_stage1_prune(
              inputs_embeds, position_ids, attention_mask, cache_position, streaming_args
          )

      # ── FOCUS: text-guided spatial pre-filter (after temporal, before causal mask) ──
      if streaming_args is not None and getattr(streaming_args, 'focus_enabled', False):
          from streaming_vlm.inference.spatial_focus import focus_spatial_prune
          inputs_embeds, position_ids, attention_mask, cache_position = focus_spatial_prune(
              inputs_embeds, position_ids, attention_mask, cache_position, streaming_args
          )

      # ── Video-CDPruner: joint spatio-temporal DPP conditional-diversity pruning ──
      if streaming_args is not None and getattr(streaming_args, 'video_cdpruner_enabled', False):
          from streaming_vlm.inference.video_cdpruner import video_cdpruner_prune
          inputs_embeds, position_ids, attention_mask, cache_position = video_cdpruner_prune(
              inputs_embeds, position_ids, attention_mask, cache_position, streaming_args
          )

      causal_mask = self._update_causal_mask(
          attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
      )                                                                                                                               
                                                                                                                                      
      hidden_states = inputs_embeds                                                                                                   
                                                                                                                                      
      # create position embeddings to be shared across the decoder layers
      position_embeddings = self.rotary_emb(hidden_states, position_ids)

      # decoder layers                                                                                                                
      all_hidden_states = () if output_hidden_states else None
      all_self_attns = () if output_attentions else None                                                                              
      next_decoder_cache = None           
                                              
      # ---- FastV setup (visual tokens at positions 35-611) ----
      bsz, q_len, _ = hidden_states.size()
      FASTV_K = streaming_args.fastv_k if (streaming_args is not None and streaming_args.fastv_k is not None) else None
      FASTV_RATIO = streaming_args.fastv_r if (streaming_args is not None and streaming_args.fastv_r is not None) else 0.5
      # is_prefill must be True only for the INITIAL prefill (cache_position[0]==0).
      # Subsequent chunk prefills have cache_position[0]>0 and must NOT run FastV,
      # because the video tokens are no longer at the hard-coded positions 35..611.
      is_prefill = (FASTV_K is not None) and (cache_position is not None and cache_position[0] == 0)
      # ---------------------                                                                                                         
                                          
      for layer_idx, decoder_layer in enumerate(self.layers):                                                                         
          if output_hidden_states:                                                                                                    
              all_hidden_states += (hidden_states,)
                                                                                                                                      
          # FastV: prune visual tokens at layer K using attention captured from layer K-1
          if is_prefill and layer_idx == FASTV_K and hasattr(self, 'last_attention') and self.last_attention is not None:
              device = hidden_states.device
              cur_len = hidden_states.shape[1]
              # Dynamically find visual token positions from current input_ids
              video_token_id = 151656  # <|video_pad|>
              input_ids_flat = streaming_args.current_input_ids[0]
              visual_positions = (input_ids_flat == video_token_id).nonzero(as_tuple=True)[0]
              vis_start = visual_positions[0].item()
              vis_end = visual_positions[-1].item() + 1
              num_visual = vis_end - vis_start
              image_attention_score = self.last_attention.mean(dim=1)[0][-1][vis_start:vis_end]
              n_available = image_attention_score.shape[0]
              keep_k = min(int(num_visual * FASTV_RATIO), n_available)
              if keep_k < n_available:
                  # Stage 2: prune visual tokens spatially
                  top_attention_rank_index = image_attention_score.topk(keep_k).indices + vis_start
                  keep_indexs = torch.cat((
                      torch.arange(vis_start, device=device),
                      top_attention_rank_index,
                      torch.arange(vis_end, cur_len, device=device)
                  )).sort().values
                  hidden_states = hidden_states[:, keep_indexs, :]
                  if position_ids is not None:
                      position_ids = position_ids[:, :, keep_indexs]
                  if cache_position is not None:
                      cache_position = cache_position[keep_indexs]
                  if causal_mask is not None:
                      new_len = hidden_states.shape[1]
                      causal_mask = causal_mask[:, :, :new_len, :new_len]
                  position_embeddings = self.rotary_emb(hidden_states, position_ids)
              # fastv_r=1.0: no pruning, layer computation proceeds normally
              self.last_attention = None
              # STAMP: update momentum state using the attention scores just captured
              if streaming_args is not None and streaming_args.stamp_r1 is not None:
                  from streaming_vlm.inference.stamp import stamp_update_state
                  # image_attention_score is [N_vis_kept_stage1] mean attention per visual token
                  stamp_update_state(streaming_args, image_attention_score.detach(), hidden_states.device)
              # STAMP-Temporal: update state (doesn't need LLM attention — uses ViT salience)
              if streaming_args is not None and getattr(streaming_args, 'stamp_temporal', False) and streaming_args.stamp_temporal_r is not None:
                  from streaming_vlm.inference.stamp_temporal import stamp_temporal_update_state
                  stamp_temporal_update_state(streaming_args, hidden_states.device)                                                                                              
                                          
          # Capture attention at layer K-1 for FastV                                                                                  
          capture_for_fastv = is_prefill and (layer_idx == FASTV_K - 1)                                                               
          this_layer_output_attn = output_attentions or capture_for_fastv                                                             
                                                                                                                                      
          if self.gradient_checkpointing and self.training:
              layer_outputs = self._gradient_checkpointing_func(                                                                      
                  decoder_layer.__call__,                                                                                             
                  hidden_states,                                                                                                      
                  causal_mask,                                                                                                        
                  position_ids,                                                                                                       
                  past_key_values,            
                  output_attentions,      
                  use_cache,
                  cache_position,                                                                                                     
                  position_embeddings,
              )                                                                                                                       
          else:                               
              layer_outputs = decoder_layer(
                  hidden_states,
                  attention_mask=causal_mask,                                                                                         
                  position_ids=position_ids,
                  past_key_value=past_key_values,                                                                                     
                  output_attentions=this_layer_output_attn,
                  use_cache=use_cache,    
                  cache_position=cache_position,
                  position_embeddings=position_embeddings,                                                                            
                  streaming_args=streaming_args,
              )                                                                                                                       
                  
          hidden_states = layer_outputs[0]                                                                                            
   
          if capture_for_fastv:                                                                                                       
              self.last_attention = layer_outputs[1]
                                              
          if use_cache:                   
              next_decoder_cache = layer_outputs[2 if this_layer_output_attn else 1]
          if output_attentions:                                                                                                       
              all_self_attns += (layer_outputs[1],)
                                                                                                                                      
      hidden_states = self.norm(hidden_states)                                                                                        
   
      # add hidden states from the last decoder layer                                                                                 
      if output_hidden_states:            
          all_hidden_states += (hidden_states,)
                                                                                                                                      
      next_cache = next_decoder_cache if use_cache else None
                                                                                                                                      
      if streaming_args is not None and streaming_args.pos_mode == "shrink":
          # For each generated token, need to pad input_ids, otherwise position embedding and kv cache lengths will be inconsistent
          streaming_args.input_ids = F.pad(streaming_args.input_ids, (0, 1), 'constant', 0)
                                                                                                                                      
      if not return_dict:                     
          return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)                    
      return BaseModelOutputWithPast(                                                                                                 
          last_hidden_state=hidden_states,                                                                                            
          past_key_values=next_cache,                                                                                                 
          hidden_states=all_hidden_states,                                                                                            
          attentions=all_self_attns,          
      )   