from streaming_vlm.inference.qwen2_5.vision_forward import streaming_visual_attention_forward,streaming_visual_block_forward, streaming_visual_encoder_forward
# from streaming_vlm.inference.qwen2_5.language_forward import streaming_language_model_forward,streaming_text_flash_attn_forward,streaming_text_decoder_layer_forward,_update_causal_mask
from streaming_vlm.inference.qwen2_5.language_forward import streaming_language_model_forward,streaming_text_decoder_layer_forward,_update_causal_mask, streaming_text_flash_attn_forward
from streaming_vlm.inference.qwen2_5.model_forward import model_forward,qwen2_5_vl_forward,prepare_inputs_for_streaming_generation
from streaming_vlm.inference.qwen2_5.pos_emb import get_rope_index
from transformers import Qwen2_5_VLForConditionalGeneration
from types import MethodType
from streaming_vlm.inference.generate.streaming_generate_qwen import streaming_generate,_sample
from streaming_vlm.inference.generate.prepare_generation import prepare_multiturn_multimodal_inputs_for_generation
import os
os.makedirs('./output_image', exist_ok=True) 

DURATION_SEC = 4
WINDOW_SIZE = 4 # Ensure WINDOW_SIZE is divisible by DURATION_SEC
OUT_FPS = 1
MAX_TOKEN_PER_DURATION = 30
MAX_PIXELS = 360 * 640

def convert_qwen2_5_to_streaming(model: Qwen2_5_VLForConditionalGeneration):
    # Patch vision encoder / blocks and decoder layers
    model.generate = MethodType(streaming_generate, model)
    model.prepare_inputs_for_generation = MethodType(prepare_multiturn_multimodal_inputs_for_generation, model)
    model._sample = MethodType(_sample, model)

    model.forward = MethodType(qwen2_5_vl_forward, model)
    model.model.forward = MethodType(model_forward, model.model)
    try:
        # transformers < 4.50: visual + language_model nested inside model.model
        model.model.language_model.forward = MethodType(streaming_language_model_forward, model.model.language_model)
        model.model.language_model._update_causal_mask = MethodType(_update_causal_mask, model.model.language_model)
        for layer in model.model.language_model.layers:
            layer.forward = MethodType(streaming_text_decoder_layer_forward, layer)
            layer.self_attn.forward = MethodType(streaming_text_flash_attn_forward, layer.self_attn)
        model.model.visual.forward = MethodType(streaming_visual_encoder_forward, model.model.visual)
        for blk in model.model.visual.blocks:
            blk.forward = MethodType(streaming_visual_block_forward, blk)
            blk.attn.forward = MethodType(streaming_visual_attention_forward, blk.attn)
        model.model.get_rope_index = MethodType(get_rope_index, model.model)
    except AttributeError:
        # transformers 4.50+: model.model IS the LLM; visual moved to model.visual
        model.model._update_causal_mask = MethodType(_update_causal_mask, model.model)
        for layer in model.model.layers:
            layer.forward = MethodType(streaming_text_decoder_layer_forward, layer)
            layer.self_attn.forward = MethodType(streaming_text_flash_attn_forward, layer.self_attn)
        model.visual.forward = MethodType(streaming_visual_encoder_forward, model.visual)
        for blk in model.visual.blocks:
            blk.forward = MethodType(streaming_visual_block_forward, blk)
            blk.attn.forward = MethodType(streaming_visual_attention_forward, blk.attn)
        model.model.get_rope_index = MethodType(get_rope_index, model.model)
    return model 
