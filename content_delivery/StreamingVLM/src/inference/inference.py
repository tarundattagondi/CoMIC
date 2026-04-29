#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streaming video description + WebVTT subtitle generation
Run example:
    python streaming_inference.py \
        --model_path Qwen/Qwen2-VL-7B-Instruct \
        --video_path demo/sources/howto_fix_laptop_mute_1080p.mp4 \
        --output_dir generated_subtitles.vtt
"""
import sys
import os
current_file = os.path.abspath(__file__)
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, grandparent_dir)
os.environ["FLASH_ATTENTION_AVAILABLE"] = "0"


import torch, functools, os, argparse
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2VLProcessor, AutoProcessor
from streaming_vlm.inference.streaming_args import StreamingArgs
from streaming_vlm.utils.get_qwen_range import *
from qwen_vl_utils.vision_process import (
    FORCE_QWENVL_VIDEO_READER, VIDEO_TOTAL_PIXELS, FPS_MAX_FRAMES, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, FRAME_FACTOR, IMAGE_FACTOR, FPS,
    smart_nframes, smart_resize
)
import sys
import json
from streaming_vlm.inference.qwen2_5.patch_model import convert_qwen2_5_to_streaming
from streaming_vlm.inference.qwen2.patch_model import convert_qwen2_to_streaming
from contextlib import contextmanager          
from transformers import set_seed
set_seed(42)   
from streaming_vlm.data.lmm_dataset import LMMDataset
from livecc_utils import  get_smart_resized_video_reader
from livecc_utils.video_process_patch import _read_video_decord_plus, _spatial_resize_video
import time
from streaming_vlm.utils.vtt_utils import open_vtt, sec2ts


import decord


# -----------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------
TOTAL_VIDEO_DURATION = 6000  # Total video duration to process (seconds)
DEFAULT_CHUNK_DURATION = 1       # Duration of each video chunk (seconds)
DEFAULT_WINDOW_SIZE = 16
DEFAULT_TEXT_ROUND = 16
DEFAULT_TEXT_SINK = 512
DEFAULT_TEXT_SLIDING_WINDOW = 512
DEFAULT_TEMPERATURE = 0.9
DEFAULT_REPETITION_PENALTY = 1.05

NFRAMES = FPS * DEFAULT_WINDOW_SIZE
MAX_PIXELS = max(min(VIDEO_MAX_PIXELS, VIDEO_TOTAL_PIXELS / NFRAMES * FRAME_FACTOR), int(VIDEO_MIN_PIXELS * 1.05))
MAX_TOKEN_PER_DURATION = 20

debug_quiet = False

# -----------------------------------------------------------------
# Helper: KV cache pruning
# -----------------------------------------------------------------
def prune_id_and_kv_cache(input_ids, past_key_values, start_index, end_index):
    # Closed interval
    input_ids = torch.cat([input_ids[:,:start_index], input_ids[:,end_index+1:]],dim=1)

    for i, (k_layer, v_layer) in enumerate(past_key_values):
        seq_len = k_layer.shape[2]
        # Clamp to valid range: FastV-pruned layers (K+) may be shorter than input_ids
        eff_start = min(start_index, seq_len)
        eff_end = min(end_index, seq_len - 1)
        indices_to_keep = list(range(eff_start)) + list(range(eff_end + 1, seq_len))
        if not indices_to_keep:
            continue
        indices_tensor = torch.tensor(indices_to_keep, device=k_layer.device)
        past_key_values.key_cache[i] = torch.index_select(k_layer, 2, indices_tensor)
        past_key_values.value_cache[i] = torch.index_select(v_layer, 2, indices_tensor)

    return input_ids, past_key_values

def contiguous_id_and_kv(input_ids, past_key_values):
    input_ids = input_ids.contiguous()
    for i, (k_layer, v_layer) in enumerate(past_key_values):
        past_key_values.key_cache[i] = k_layer.contiguous()
        past_key_values.value_cache[i] = v_layer.contiguous()
    return input_ids, past_key_values

def load_model_and_processor(model_path, model_base = 'Qwen2_5'):
    if model_base == 'Qwen2_5':
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     model_path, torch_dtype="auto", device_map="cuda",
        #     attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        # )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="cuda"
        )
        model = convert_qwen2_5_to_streaming(model)
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    elif model_base == 'Qwen2':
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="cuda",
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        model = convert_qwen2_to_streaming(model)
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    return model, processor

def process_past_kv(past_key_values, i, text_round, visual_round, full_conversation_history, prev_generated_ids,
                    assistant_start_bias, assistant_end_bias, recent_video_window_clips, recent_pixel_values_videos,
                    text_sink,text_sliding_window):

    if i>=text_round:
        # Move the assistant text generated later into "previous text"
        if full_conversation_history[0]['role'] != 'previous text':
            breakpoint()
        assert full_conversation_history[0]['role'] == 'previous text'
        assert full_conversation_history[-2*text_round]['role'] == 'user'
        assert full_conversation_history[-(2*text_round-1)]['role'] == 'assistant' # The last pattern must be ... User Assistant User Assistant; from the end, find the first assistant
        full_conversation_history[0]['content'] += full_conversation_history[-(2*text_round-1)]['content'][:-4] # Remove the trailing " ..."

        def resort_id_and_kv(input_ids, past_key_values, src_start_idx, src_end_idx, dst_idx):
            # Move from src_start_idx..src_end_idx to after dst_idx
            assert  dst_idx < src_start_idx <= src_end_idx
            input_ids = torch.cat([input_ids[:,:dst_idx+1], input_ids[:,src_start_idx:src_end_idx+1], input_ids[:,dst_idx+1:src_start_idx], input_ids[:,src_end_idx+1:]],dim=1)
            
            for i, (k_layer, v_layer) in enumerate(past_key_values):
                past_key_values.key_cache[i] = torch.cat([k_layer[:,:,:dst_idx+1], k_layer[:,:,src_start_idx:src_end_idx+1], k_layer[:,:,dst_idx+1:src_start_idx], k_layer[:,:,src_end_idx+1:]],dim=2)
                past_key_values.value_cache[i] = torch.cat([v_layer[:,:,:dst_idx+1], v_layer[:,:,src_start_idx:src_end_idx+1], v_layer[:,:,dst_idx+1:src_start_idx], v_layer[:,:,src_end_idx+1:]],dim=2)
            return input_ids, past_key_values

        assistant_text_start_idx, assistant_text_end_idx = get_qwen_range(prev_generated_ids, 'assistant', 0)

        previous_text_start_idx, previous_text_end_idx = get_qwen_range(prev_generated_ids, 'previous text', 0, contain_lf=False)

        if prev_generated_ids[0,assistant_text_end_idx] == TOKEN_IDS["\n"]:
            src_start_idx = assistant_text_start_idx+assistant_start_bias
            src_end_idx = assistant_text_end_idx-assistant_end_bias-1
        else:
            src_start_idx = assistant_text_start_idx+assistant_start_bias
            src_end_idx = assistant_text_end_idx-assistant_end_bias
        if src_start_idx <= src_end_idx:
            prev_generated_ids,past_key_values = resort_id_and_kv(prev_generated_ids, past_key_values, src_start_idx, src_end_idx, previous_text_end_idx-1)

        # Prune: delete user_text and assistant formatting
        for k,item in enumerate(full_conversation_history[-2*text_round]['content']):
            if item['type'] == 'text':
                del full_conversation_history[-2*text_round]['content'][k] # In livecc format, the first text is "time xx-xx"; just delete the first one to avoid removing the first query
                break
        
        del full_conversation_history[-(2*text_round-1)]
        if visual_round > text_round:
            # If more visual rounds are kept than text rounds, delete this text round separately;
            # otherwise, delete the entire user at the end
            user_text_start_idx, user_text_end_idx = get_qwen_range(prev_generated_ids, 'user_text', -text_round, contain_lf=False)
            prev_generated_ids,past_key_values = prune_id_and_kv_cache(prev_generated_ids, past_key_values, user_text_start_idx, user_text_end_idx)
        try:
            assistant_text_start_idx, assistant_text_end_idx = get_qwen_range(prev_generated_ids, 'assistant', -text_round)
        except Exception as e:
            breakpoint()
        prev_generated_ids,past_key_values = prune_id_and_kv_cache(prev_generated_ids, past_key_values, assistant_text_start_idx, assistant_text_end_idx)

    if i >= visual_round :
        recent_video_window_clips.pop(0)
        recent_pixel_values_videos.pop(0)
        if visual_round < text_round:
            # Similarly:
            # If more text rounds are kept than visual rounds, delete this visual round separately;
            # otherwise, delete the entire user at the end
            if visual_round >= text_round:
                full_conversation_history[1]['content'] = [item for item in full_conversation_history[1]['content'] if item['type']!='video']
            else:
                full_conversation_history[-2*visual_round]['content'] = [item for item in full_conversation_history[-2*visual_round]['content'] if item['type']!='video']
            
            video_token_start_index, video_token_end_index = get_qwen_range(prev_generated_ids, 'vision', 0)
            prev_generated_ids, past_key_values = prune_id_and_kv_cache(prev_generated_ids, past_key_values, video_token_start_index, video_token_end_index)
    
    if i >= max(visual_round,text_round):
        # If both the vision and text of this round have been deleted, then delete the entire block
        del full_conversation_history[1]
        user_start_idx, user_end_idx = get_qwen_range(prev_generated_ids, 'user', 0)
        prev_generated_ids, past_key_values = prune_id_and_kv_cache(prev_generated_ids, past_key_values, user_start_idx, user_end_idx)

    if i > 0: 
        if text_sink is not None or text_sliding_window is not None:
            previous_text_start_idx, previous_text_end_idx = get_qwen_range(prev_generated_ids, 'previous text', 0)
            cut_start_idx = previous_text_start_idx + text_sink + 4 if text_sink is not None else previous_text_start_idx
            cut_end_idx = previous_text_end_idx - text_sliding_window - 1 if text_sliding_window is not None else previous_text_end_idx
            # 4 and 1 are magic numbers because the structure of "previous text" cannot be broken. (<|im_start|>previous text\n ... <|im_end|>)
            if cut_start_idx <= cut_end_idx:
                prev_generated_ids, past_key_values = prune_id_and_kv_cache(prev_generated_ids, past_key_values, cut_start_idx, cut_end_idx)
        prev_generated_ids, past_key_values = contiguous_id_and_kv(prev_generated_ids, past_key_values)
    
    return past_key_values, prev_generated_ids, recent_video_window_clips, recent_pixel_values_videos

def printq(*args, quiet=False, **kwargs): 
    """Use like print; when quiet=True, suppress output.""" 
    if not quiet: 
        print(*args, **kwargs)
# -----------------------------------------------------------------
# Main logic
# -----------------------------------------------------------------
def streaming_inference(model_path="",
                        video_path="", output_dir=None,
                        model_base = 'Qwen2_5',
                        model = None,
                        processor = None,
                        window_size = DEFAULT_WINDOW_SIZE,
                        chunk_duration = DEFAULT_CHUNK_DURATION,
                        text_round = DEFAULT_TEXT_ROUND,
                        previous_text = "",
                        test_data_json = None,
                        test_data_idx = None,
                        pos_mode = "shrink",
                        all_text = False, # All PEs are 1D to adapt to livecc's buggy training code
                        skip_first_chunk = 0,
                        recompute = False,
                        gt_json = None,
                        gt_idx = None,
                        text_sink = None,
                        text_sliding_window = None,
                        temperature = DEFAULT_TEMPERATURE,
                        duration = TOTAL_VIDEO_DURATION,
                        query = "Commentate on this match",
                        repetition_penalty=DEFAULT_REPETITION_PENALTY,
                        quiet=False,
                        emit_json=False,
                        time_test = False,
                        fastv_k = None,   # FastV: layer at which to prune (None = disabled)
                        fastv_r = 0.5,    # FastV: fraction of visual tokens to keep
                        stamp_r1 = None,  # STAMP Stage 1 keep ratio (None = STAMP disabled)
                        stamp_alpha = 0.5,  # STAMP: weight mixing momentum vs novelty
                        stamp_lambda = 0.3, # STAMP: EMA decay for attention momentum
                        stamp_K = 10,       # STAMP: keyframe interval (every K chunks)
                        ):
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    NUM_WINDOW_TO_KEEP = window_size // chunk_duration
    assert NUM_WINDOW_TO_KEEP * chunk_duration == window_size, \
        "window_size must be divisible by chunk_duration"

    printq(f"DEBUG: What is the chunk_duration? {chunk_duration}", quiet=quiet)

    # Load model
    device = model.device if model is not None else "cuda" if torch.cuda.is_available() else "cpu"
    streaming_args = StreamingArgs(pos_mode=pos_mode, all_text=all_text,
                                   fastv_k=fastv_k, fastv_r=fastv_r,
                                   stamp_r1=stamp_r1, stamp_alpha=stamp_alpha,
                                   stamp_lambda=stamp_lambda, stamp_K=stamp_K)

    if model is None or processor is None:
        model, processor = load_model_and_processor(model_path, model_base)
    else: 
        if model_base == 'Qwen2_5':
            model = convert_qwen2_5_to_streaming(model)
        elif model_base == 'Qwen2':
            model = convert_qwen2_to_streaming(model)
    
    assistant_start_bias = len(processor(text="<|im_start|>assistant\n")['input_ids'][0])
    assistant_end_bias = len(processor(text=" ...<|im_end|>")['input_ids'][0])

    # Load GT
    use_gt = False
    if gt_json is not None:
        use_gt = True
        with open(gt_json, 'r') as f:
            for i,line in enumerate(f):
                if i == gt_idx:
                    gt_dict = json.loads(line)
                    break
    
                # f[gt_idx] 
    # Load from dataset #########################################################
    from_dataset = False
    if test_data_json is not None:
        assert test_data_idx is not None
        dataset = LMMDataset(
            annotation_paths=[
               test_data_json
            ], 
            processor=processor, # Pass in the initialized processor
            with_context=False, # Do not use context
            return_conversation=True,
        )[test_data_idx]
        from_dataset = True

    if not from_dataset:
        print("os.getenv('DATASET_PATH', ", os.getenv('DATASET_PATH', ""), ")")
        print("os.path.join(os.getenv('DATASET_PATH', ""), video_path)", os.path.join(os.getenv('DATASET_PATH', ""), video_path))

        # Try different path combinations to find the video
        video_reader = None
        if os.path.exists(video_path):
            video_reader, resized_height, resized_width = get_smart_resized_video_reader(video_path, MAX_PIXELS)
        elif os.path.exists(os.path.join(os.getenv('DATASET_PATH', ""), video_path)):
            video_path = os.path.join(os.environ['DATASET_PATH'], video_path)
            video_reader, resized_height, resized_width = get_smart_resized_video_reader(video_path, MAX_PIXELS)
        elif os.path.exists(os.path.join(os.getenv('EVAL_DATASET_PATH', ""), video_path)):
            video_path = os.path.join(os.environ['EVAL_DATASET_PATH'], video_path)
            video_reader, resized_height, resized_width = get_smart_resized_video_reader(video_path, MAX_PIXELS)
        elif os.getenv('UPLOAD_DATASET_PATH') and os.path.exists(os.path.join(os.getenv('UPLOAD_DATASET_PATH', ""), video_path)):
            # Handle uploaded videos with UPLOAD_DATASET_PATH
            video_path = os.path.join(os.environ['UPLOAD_DATASET_PATH'], video_path)
            video_reader, resized_height, resized_width = get_smart_resized_video_reader(video_path, MAX_PIXELS)
        else:
            # If none of the above paths work, raise an error with debug info
            raise FileNotFoundError(
                f"Video file not found. Tried the following paths:\n"
                f"  1. {video_path} (exists: {os.path.exists(video_path)})\n"
                f"  2. {os.path.join(os.getenv('DATASET_PATH', ''), video_path)} (exists: {os.path.exists(os.path.join(os.getenv('DATASET_PATH', ''), video_path))})\n"
                f"  3. {os.path.join(os.getenv('EVAL_DATASET_PATH', ''), video_path)} (exists: {os.path.exists(os.path.join(os.getenv('EVAL_DATASET_PATH', ''), video_path))})\n"
                f"  4. {os.path.join(os.getenv('UPLOAD_DATASET_PATH', ''), video_path) if os.getenv('UPLOAD_DATASET_PATH') else 'UPLOAD_DATASET_PATH not set'}"
            )

        video_reader.get_frame_timestamp(0) # array([0.0, 0.01668333], dtype=float32)
        # video_pts = torch.from_numpy(video_reader._frame_pts[:, 1]) # End timestamp for each frame
        printq(f"Give me the resized dimensions", quiet=quiet)
        printq(f"resized_height: {resized_height}, resized_width: {resized_width}", quiet=quiet)
    ##############################################################################
    # Prepare subtitle file header ##############################################
    if output_dir is not None:
        if os.path.exists(output_dir):
            os.remove(output_dir)
        with open_vtt(output_dir):  # Write WEBVTT header
            pass
        printq(f"Subtitles will be written to: {output_dir}", quiet=quiet)
    ##############################################################################

    past_key_values = None
    full_conversation_history = []
    prev_generated_ids = None  # Position embeddings must be recomputed every round, so we must keep this
    recent_video_window_clips = [] # Qwen: video before passing through the vision tower
    recent_pixel_values_videos = [] # 
    num_chunks = int((duration + chunk_duration - 1) // chunk_duration)

    responses = []
    ground_truths = []
    printq(f"num_chunks: {num_chunks}", quiet=quiet)
    if time_test:
        time_results = []
    for i in range(num_chunks):
        _sync()
        loop_start = time.perf_counter()
        section_time = {k: 0.0 for k in ['PKV', 'CHECK', 'VIDEO', 'INPUT', 'GEN', 'POST']}
    
        start_time = (i + skip_first_chunk) * chunk_duration 

        ########################## Handle past_key_values ###################################
        _sync(); _t = time.perf_counter()

        past_key_values, prev_generated_ids, recent_video_window_clips, recent_pixel_values_videos = process_past_kv(past_key_values, i, 
                                                                                         text_round=text_round, visual_round=window_size, 
                                                                                         full_conversation_history=full_conversation_history, 
                                                                                         prev_generated_ids=prev_generated_ids, 
                                                                                         assistant_start_bias=assistant_start_bias, 
                                                                                         assistant_end_bias=assistant_end_bias, 
                                                                                         recent_video_window_clips=recent_video_window_clips,
                                                                                         recent_pixel_values_videos=recent_pixel_values_videos,
                                                                                         text_sink=text_sink,
                                                                                         text_sliding_window=text_sliding_window)
        _sync();section_time['PKV'] += (time.perf_counter() - _t)

        _sync(); _t = time.perf_counter()

        try:
            if from_dataset:
                current_video_chunk = dataset['conversation'][2*i+1]['content'][1]['video'] 
                video_start = dataset['start_timestamp']  # Start time of this segment itself
                start_time = video_start + start_time
                ground_truths.append({'ground_truth':dataset['conversation'][2*i+2]['content'][0]['text'] , 'start_time':start_time, 'end_time':start_time+chunk_duration})
            else:
                    ele = {'type': 'video', 'video': video_path, 'video_start': start_time, 'video_end': (start_time+chunk_duration)}
                    current_video_chunk, sample_fps, clip_pts = _read_video_decord_plus(ele,  return_pts=True, strict_fps=True, only_get_last_frame=int(chunk_duration*FPS), vr=video_reader)

                    printq(f"DEBUG: current_video_chunk shape 1? {current_video_chunk.shape}", quiet=debug_quiet)
                    
                    current_video_chunk = _spatial_resize_video(current_video_chunk)

                    # vr = decord.VideoReader(current_video_chunk, num_threads=2)
                    # video_fps   = vr.get_avg_fps()
                    printq(f"DEBUG: current_video_chunk shape 2? {current_video_chunk.shape}", quiet=debug_quiet) # shape: (T, C, H, W) => (frames, channels, height, width)


        except Exception as e:
            print(f"Error in streaming_inference: {e}")
            break
        recent_video_window_clips.append(current_video_chunk)
        _sync();section_time['VIDEO'] += (time.perf_counter() - _t)

        ########################## Build input ##############################################
        _sync(); _t = time.perf_counter()
        if i == 0:
            prompt = f'Time={start_time:.1f}-{start_time+chunk_duration:.1f}s'
            if from_dataset: # Load from livecc dataset
                previous_text = dataset['conversation'][0]['content']
                user_content = [
                    {"type": "text", "text": prompt},
                    dataset['conversation'][1]['content'][1], # video
                    {"type": "text", "text": query},
                ]
            else: # Load from local file
                user_content = [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": video_path },
                    {"type": "text", "text": query},
                ]
            full_conversation_history = [
                {"role": "previous text", "content": previous_text},
                {"role": "user", "content": user_content}
                ]
            text = processor.apply_chat_template(full_conversation_history, tokenize=False, add_generation_prompt=True)
            
        else:
            prompt = f'Time={start_time:.1f}-{start_time+chunk_duration:.1f}s'

            if from_dataset:
                user_content = [
                    {"type": "text", "text": prompt},
                    dataset['conversation'][2*i+1]['content'][1], # video
                ]
            else:
                user_content = [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": video_path, "start": start_time,
                    "duration": chunk_duration},
                ]
            full_conversation_history.append({"role": "user", "content": user_content})
            text = processor.apply_chat_template([{"role": "user", "content": user_content}], tokenize=False, add_generation_prompt=True)
            text = '\n' + text[SYSTEM_PROMPT_OFFSET:]
        
        printq(f"DEBUG: recent_video_window_clips shape? {len(recent_video_window_clips)}", quiet=debug_quiet)
        printq(f"DEBUG: recent_video_window_clips[-1] shape? {recent_video_window_clips[-1].shape}", quiet=debug_quiet)

        inputs = processor(
            text=[text],
            videos=recent_video_window_clips[-1],
            padding=True,
            return_tensors="pt",
        ).to(device)
        

        printq(f"DEBUG: Autoprocessor configuration {processor.image_processor}", quiet=debug_quiet)

        printq(f"DEBUG: Autoprocessor video_grid_thw shape? {inputs['video_grid_thw']}", quiet=debug_quiet)
        printq(f"DEBUG: Autoprocessor inputs? {inputs}", quiet=debug_quiet)
     
        if prev_generated_ids is not None:
            # Because the model's last generated token is <|im_end|>, a new round will add an extra "\n".
            # We must keep that "\n", hence the -1.
            # Special case: (when only one round of text is kept) if the last assistant was removed,
            # then the last token of the previous user will be "\n"
            if prev_generated_ids[:,-1].item()!=TOKEN_IDS["\n"]:
                inputs['input_ids'] = torch.cat([prev_generated_ids,inputs['input_ids']],dim=1) 
            else:
                inputs['input_ids'] = torch.cat([prev_generated_ids,inputs['input_ids'][:,1:]],dim=1) 
            inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])

        recent_pixel_values_videos.append(inputs['pixel_values_videos'])
        if streaming_args.pos_mode == "shrink" or recompute:
            streaming_args.input_ids = inputs['input_ids']
            if i==0:
                streaming_args.video_grid_thw = inputs['video_grid_thw']
                streaming_args.second_per_grid_ts = inputs['second_per_grid_ts'] if 'second_per_grid_ts' in inputs else None
            else:
                streaming_args.video_grid_thw = torch.cat([streaming_args.video_grid_thw,inputs['video_grid_thw']],dim=0)
                streaming_args.second_per_grid_ts = torch.cat([streaming_args.second_per_grid_ts,inputs['second_per_grid_ts']],dim=0) if 'second_per_grid_ts' in inputs else None

        current_input_len = inputs['input_ids'].shape[1]

        _sync();section_time['INPUT'] += (time.perf_counter() - _t)
        ########################## Model generation (commentary) ############################
        _sync(); _t = time.perf_counter()
        if recompute:
            if i>0:
                inputs['pixel_values_videos'] = torch.cat(recent_pixel_values_videos,dim=0)
                inputs['video_grid_thw'] = streaming_args.video_grid_thw[-len(recent_pixel_values_videos):]
                inputs['second_per_grid_ts'] = streaming_args.second_per_grid_ts[-len(recent_pixel_values_videos):]
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKEN_PER_DURATION,
                use_cache=True,
                return_dict_in_generate=True,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                streaming_args=streaming_args,
                pad_token_id=151645,
                temperature=temperature,
            )
        else:
            outputs = model.generate(
                **inputs,
                past_key_values=past_key_values,
                max_new_tokens=MAX_TOKEN_PER_DURATION,
                use_cache=True,
                return_dict_in_generate=True,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                streaming_args=streaming_args,
                pad_token_id=151645,
                temperature=temperature,
            )
        _sync();section_time['GEN'] += (time.perf_counter() - _t)

        ########################## Post-process input_ids ##################################
        _sync(); _t = time.perf_counter()
        generated_ids = outputs.sequences
        if generated_ids[0,-1].item() != 151645: # <|im_end|>
            # If the last token is not <|im_end|>, append it to keep the format consistent
            generated_ids = torch.cat([generated_ids, torch.tensor([[151645]], device = device)],dim=1)
        newly_generated_ids = generated_ids[:, current_input_len:]

        response = processor.batch_decode(newly_generated_ids, skip_special_tokens=True)[0]
        responses.append({'response':response[:-4], 'start_time':start_time, 'end_time':start_time+chunk_duration})

        time_key = f'Time={start_time:.1f}-{start_time+chunk_duration:.1f}s'
        past_key_values = outputs.past_key_values
        hms_start = time.strftime('%H:%M:%S', time.gmtime(int(start_time)))
        hms_end = time.strftime('%H:%M:%S', time.gmtime(int(start_time + chunk_duration)))
        printq(f"Time={hms_start}-{hms_end}: \033[1m\033[34m{response}\033[0m", f'past_key_values: {past_key_values.get_seq_length() if past_key_values is not None else 0}', flush=True, quiet=quiet)
        if emit_json:
            try:
                if emit_json:
                    sys.stdout.write(json.dumps({
                        "type": "segment",
                        "start": float(start_time),
                        "end": float(start_time + chunk_duration),
                        "text": response[:-4]
                    }, ensure_ascii=False) + "\n")
                    sys.stdout.flush()
            except Exception as _e:
                pass
        prev_generated_ids = generated_ids.clone()
        if use_gt and gt_dict[time_key]['phrase'] != response:
            printq(f"Decoded text [{response}] is incorrect. Use ground truth [{gt_dict[time_key]['phrase']}] instead", quiet=quiet)
            prune_id_and_kv_cache(prev_generated_ids,past_key_values,current_input_len,past_key_values.get_seq_length()-1)
            response = gt_dict[time_key]['phrase']
            prev_generated_ids = torch.cat([inputs['input_ids'], torch.tensor(processor(text=[gt_dict[time_key]['phrase']+"<|im_end|>\n"])['input_ids'],device = device)],dim=1)
        assistant_turn = {"role": "assistant", "content": response}
        full_conversation_history.append(assistant_turn)
            

        _sync();section_time['POST'] += (time.perf_counter() - _t)

        # ------------------------- Print profiling results -----------------
        _sync()
        loop_total = time.perf_counter() - loop_start
        printq(
            f"[Loop {i}] total={loop_total:.3f}s | "
            f"PKV={section_time['PKV']:.3f}s | "
            f"CHECK={section_time['CHECK']:.3f}s | "
            f"VIDEO={section_time['VIDEO']:.3f}s | "
            f"INPUT={section_time['INPUT']:.3f}s | "
            f"GEN={section_time['GEN']:.3f}s | "
            f"POST={section_time['POST']:.3f}s",
            flush=True,
            quiet=quiet,
        )
        if time_test:
            time_results.append(section_time)
        # ============================================================

        # ★ Write WebVTT
        ts_start = sec2ts(start_time)
        ts_end = sec2ts(start_time + chunk_duration)
        if output_dir is not None:
            with open_vtt(output_dir) as vf:
                vf.write(f"{ts_start} --> {ts_end}\n Infer Time: {loop_total:.3f}s\n {response}\n\n")
    if output_dir is not None:
        printq(f"\n✅ Subtitles saved to: {output_dir}\n", quiet=quiet)
    if time_test:
        return time_results
    return responses


if __name__ == "__main__":
    default_video = "Baidu_NBA_EN/NBA.2015.06.09.Warriors.vs.Cavaliers.720p.HDTV.60fps.x264.mp4"
    from huggingface_hub import hf_hub_download
    default_video = hf_hub_download(
        repo_id="mit-han-lab/Inf-Stream-Eval",
        filename="Baidu_NBA_EN/6.4_17-18SeasonNBAFinalG2_WarriorsVSCavaliers_720P.mp4",
        repo_type="dataset"
    )

    args = argparse.ArgumentParser()
    args.add_argument("--pos_mode", type=str, default="shrink", choices=["append", "shrink"])
    args.add_argument("--all_text", action="store_true", default=False)
    args.add_argument("--model_path", type=str, default="mit-han-lab/StreamingVLM")
    args.add_argument("--model_base", type=str, choices=["Qwen2_5", "Qwen2", "VILA"], default="Qwen2_5")
    args.add_argument("--video_path", type=str, default=default_video)
    args.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE)
    args.add_argument("--chunk_duration", type=int, default=DEFAULT_CHUNK_DURATION)
    args.add_argument("--text_round", type=int, default=DEFAULT_TEXT_ROUND)
    args.add_argument("--previous_text", type=str, default="THis is a video with title 'Golden State Warriors’ BEST PLAYS from the 2024-25 NBA Season'. This is Highlight of Warriors. Curry is going to score a 3-pointer")
    args.add_argument("--skip_first_chunk", type=int, default=0)
    args.add_argument("--recompute", action='store_true')
    args.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    # Both None: no truncation
    # One None and the other not: treat None as 0 (keep nothing), so both are applied
    # Both non-None: apply both
    args.add_argument("--text_sink", type=int, default=DEFAULT_TEXT_SINK)
    args.add_argument("--text_sliding_window", type=int, default=DEFAULT_TEXT_SLIDING_WINDOW)

    args.add_argument("--output_dir", type=str)
    args.add_argument("--emit_json", action="store_true", help="逐秒输出 JSON 行到 stdout")

    args.add_argument("--test_data_json", type=str, default=None)
    args.add_argument("--test_data_idx", type=int, default=None)

    args.add_argument("--gt_json", type=str, default=None)
    args.add_argument("--gt_idx", type=int, default=0)

    # FastV visual token pruning
    args.add_argument("--fastv_k", type=int, default=None,
                      help="Layer index at which to prune visual tokens (capture attention at fastv_k-1). None = disabled.")
    args.add_argument("--fastv_r", type=float, default=0.5,
                      help="Fraction of visual tokens to KEEP after pruning (default 0.5 = keep top 50%%).")

    # STAMP: Streaming Temporal Attention Momentum Pruning
    args.add_argument("--stamp_r1", type=float, default=None,
                      help="STAMP Stage 1 keep ratio (None = STAMP disabled). E.g. 0.5 = keep top 50%% pre-LLM.")
    args.add_argument("--stamp_alpha", type=float, default=0.5,
                      help="STAMP: weight of momentum vs novelty in combined score (default 0.5).")
    args.add_argument("--stamp_lambda", type=float, default=0.3,
                      help="STAMP: EMA decay rate for attention momentum (default 0.3).")
    args.add_argument("--stamp_K", type=int, default=10,
                      help="STAMP: keyframe interval — every K chunks all visual tokens are kept (default 10).")

    args = args.parse_args()

    if args.output_dir is None:
        os.makedirs("output", exist_ok=True)
        args.output_dir = f"output/{args.model_path.replace('/','_')}_viswin{args.window_size}_txtwin{args.text_round}_prvsink{args.text_sink}_prvwin{args.text_sliding_window}_tprt{args.temperature}.vtt"
    streaming_inference(**args.__dict__)
