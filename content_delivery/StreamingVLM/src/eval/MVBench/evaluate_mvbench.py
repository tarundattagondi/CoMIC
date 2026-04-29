"""
MVBench Streaming Evaluation with STAMP-Temporal Support
=========================================================
Evaluates Qwen2.5-VL on MVBench (4000 samples, 20 temporal tasks)
with optional STAMP-Temporal pruning, TAST, and DSTM.

Uses the same multi-chunk streaming approach as DeViBench/MLVU.
"""

import json, os, torch, tqdm, sys, argparse, glob, time
import multiprocessing as mp
import decord
from torch.utils.data import Dataset
from transformers import logging, Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils.vision_process import process_vision_info, smart_nframes, FPS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

logger = logging.get_logger(__name__)


def _read_video_decord(video_path, start=None, end=None):
    """Read video using decord, optionally trimming to [start, end] seconds."""
    vr = decord.VideoReader(video_path, num_threads=2)
    fps_native = vr.get_avg_fps()
    total_frames = len(vr)

    if start is not None and end is not None:
        start_frame = max(0, int(start * fps_native))
        end_frame = min(total_frames, int(end * fps_native))
    else:
        start_frame = 0
        end_frame = total_frames

    # Sample at target FPS (from env var or default)
    target_fps = float(os.environ.get('QWENVL_FPS', '2.0'))
    duration = (end_frame - start_frame) / fps_native
    n_frames = max(1, int(duration * target_fps))
    n_frames = min(n_frames, 512)  # cap to avoid OOM on very long videos

    indices = torch.linspace(start_frame, end_frame - 1, n_frames).long().tolist()
    frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
    video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
    return video_tensor, target_fps


def load_mvbench_data(data_dir):
    """
    Load MVBench dataset from HuggingFace-downloaded directory.
    MVBench uses json/ subdirectory with per-task JSON files.
    Each JSON is a list of dicts: {video, question, candidates, answer}.
    Returns list of dicts with keys: task, video, question, candidates, answer.
    """
    samples = []
    json_dir = os.path.join(data_dir, 'json')

    if not os.path.isdir(json_dir):
        # Fallback: try data_dir directly
        json_dir = data_dir

    json_files = sorted(glob.glob(os.path.join(json_dir, '*.json')))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")

    for jf in json_files:
        task_name = os.path.splitext(os.path.basename(jf))[0]
        with open(jf) as f:
            task_data = json.load(f)

        for item in task_data:
            sample = {
                'task': task_name,
                'video': item.get('video', ''),
                'question': item.get('question', ''),
                'candidates': item.get('candidates', []),
                'answer': item.get('answer', ''),
            }
            if 'start' in item:
                sample['start'] = item['start']
            if 'end' in item:
                sample['end'] = item['end']
            samples.append(sample)

    return samples


def save_function_print(function, save_path, *args, **kwargs):
    """Capture function print output to file."""
    import io
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    function(*args, **kwargs)
    sys.stdout = old_stdout
    output = buffer.getvalue()
    print(output)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(output)


def evaluate_mvbench_results(results):
    """Evaluate MVBench results by task."""
    task_to_counts = {}
    total_correct = 0
    total = 0

    for r in results:
        task = r['task']
        if task not in task_to_counts:
            task_to_counts[task] = {'correct': 0, 'total': 0}
        task_to_counts[task]['total'] += 1
        total += 1

        if r['correct']:
            task_to_counts[task]['correct'] += 1
            total_correct += 1

    for task in sorted(task_to_counts.keys()):
        c = task_to_counts[task]
        print(f"  {task:30s}: {c['correct']:4d}/{c['total']:4d} = {100*c['correct']/max(c['total'],1):.1f}%")

    print(f"\n  OVERALL: {total_correct}/{total} = {100*total_correct/max(total,1):.2f}%")


_VIDEO_INDEX = {}  # cached {basename: full_path} for fast lookup

def _build_video_index(video_dir):
    """Build a filename→path index for all video files under video_dir."""
    global _VIDEO_INDEX
    if _VIDEO_INDEX:
        return
    print(f"Building video index from {video_dir}...")
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            if f.endswith(('.mp4', '.webm', '.avi', '.mkv')):
                _VIDEO_INDEX[f] = os.path.join(root, f)
                # Also index with relative path from video_dir
                rel = os.path.relpath(os.path.join(root, f), video_dir)
                _VIDEO_INDEX[rel] = os.path.join(root, f)
    print(f"Indexed {len(_VIDEO_INDEX)} video files")


def _find_video(video_dir, video_ref):
    """Find a video file given its reference (may be basename or relative path)."""
    # Direct path
    direct = os.path.join(video_dir, video_ref)
    if os.path.exists(direct):
        return direct
    # Search index
    if not _VIDEO_INDEX:
        _build_video_index(video_dir)
    # Try exact match
    if video_ref in _VIDEO_INDEX:
        return _VIDEO_INDEX[video_ref]
    # Try basename only
    basename = os.path.basename(video_ref)
    if basename in _VIDEO_INDEX:
        return _VIDEO_INDEX[basename]
    return direct  # fallback — will raise FileNotFoundError


def stamp_streaming_mvbench_predict(model, processor, samples, video_dir, options, n_chunks=3,
                                     question_prefix='', question_postfix='\nPlease select the correct answer.',
                                     answer_prefix='The answer is:\n', abcd_previous_str='\n'):
    """Multi-chunk streaming eval for MVBench with STAMP-Temporal."""
    from streaming_vlm.inference.stamp_temporal import stamp_temporal_reset_state

    strict_option_ids = [
        processor.tokenizer(f'{abcd_previous_str}{opt}').input_ids[-1] for opt in options
    ]

    model.eval()
    device = next(model.parameters()).device
    sa = model._streaming_args

    def stamp_reset_state(sa_arg):
        stamp_temporal_reset_state(sa_arg)
        if hasattr(sa_arg, 'stamp_last_attn_scores'):
            sa_arg.stamp_last_attn_scores = None

    results = []

    with torch.no_grad():
        for i in tqdm.tqdm(range(len(samples)), desc="MVBench streaming eval"):
            sample = samples[i]

            # Reset state
            stamp_reset_state(sa)
            sa.stamp_no_reset = True
            sa.input_ids = None

            t_start = time.perf_counter()

            # Load video — MVBench videos are in various subdirectories
            video_path = _find_video(video_dir, sample['video'])

            try:
                start_t = sample.get('start', None)
                end_t = sample.get('end', None)
                video, _ = _read_video_decord(video_path, start=start_t, end=end_t)
                video = _spatial_resize_video(video)
            except Exception as e:
                print(f"Skipping sample {i}: {e}")
                continue

            # Context chunks (warm up temporal state)
            for _ctx_idx in range(n_chunks - 1):
                ctx_conv = [{"role": "user", "content": [
                    {"type": "video", "video": video},
                    {"type": "text", "text": "Watch this video segment."},
                ]}]
                ctx_text = processor.apply_chat_template(ctx_conv, tokenize=False, add_generation_prompt=False)
                ctx_inputs = processor(
                    text=[ctx_text], videos=[video],
                    return_tensors="pt", padding=True,
                ).to(device)
                _ = model(**ctx_inputs)

            # Final chunk: video + question + options
            candidates = sample['candidates']
            if isinstance(candidates, list) and len(candidates) > 0:
                # Format as A/B/C/D options
                option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
                options_text = '\n'.join(f"{option_letters[j]}. {c}" for j, c in enumerate(candidates))
                query = question_prefix + sample['question'] + '\n' + options_text + question_postfix
            else:
                query = question_prefix + sample['question'] + question_postfix

            final_conv = [{"role": "user", "content": [
                {"type": "video", "video": video},
                {"type": "text", "text": query},
            ]}]
            final_text = processor.apply_chat_template(final_conv, tokenize=False, add_generation_prompt=True)
            final_text = final_text + answer_prefix

            final_inputs = processor(
                text=[final_text], videos=[video],
                return_tensors="pt", padding=True,
            ).to(device)

            final_outputs = model(**final_inputs)
            last_logits = final_outputs.logits[0, -1, :]

            # Get prediction from option logits
            option_logits = last_logits[torch.tensor(strict_option_ids, device=device)]
            pred_idx = option_logits.argmax().item()

            # Map prediction to answer
            pred_text = options[pred_idx]

            # Check correctness: match against answer text or option letter
            answer = sample['answer']
            correct = False
            if isinstance(candidates, list) and len(candidates) > 0:
                # Find which option letter the answer corresponds to
                option_letters = ['A', 'B', 'C', 'D', 'E', 'F']
                for j, c in enumerate(candidates):
                    if c == answer and j < len(option_letters):
                        if pred_text == option_letters[j]:
                            correct = True
                        break
                # Also check direct text match
                if not correct and pred_idx < len(candidates):
                    if candidates[pred_idx] == answer:
                        correct = True
            else:
                correct = (pred_text == answer)

            t_end = time.perf_counter()

            results.append({
                'task': sample['task'],
                'question': sample['question'],
                'answer': answer,
                'response': pred_text,
                'correct': correct,
                'inference_time': round((t_end - t_start) * 1000, 2),
            })

            sa.stamp_no_reset = False

            # Checkpoint every 200 samples
            if len(results) % 200 == 0:
                print(f"  Checkpoint: {len(results)} samples, "
                      f"acc={100*sum(r['correct'] for r in results)/len(results):.2f}%")

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_memory_mb = peak_bytes / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats()

    return results, peak_memory_mb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MVBench streaming evaluation.")

    parser.add_argument("--data_dir", type=str, required=True, help="Path to MVBench dataset dir.")
    parser.add_argument("--video_dir", type=str, default=None, help="Path to video files (if separate).")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model dir.")
    parser.add_argument("--n_chunks", type=int, default=3, help="Number of chunks for streaming eval.")

    # STAMP-Temporal args
    parser.add_argument("--stamp_temporal", action="store_true", default=False)
    parser.add_argument("--stamp_temporal_r", type=float, default=None)
    parser.add_argument("--stamp_temporal_alpha", type=float, default=0.5)
    parser.add_argument("--stamp_temporal_lambda", type=float, default=0.3)
    parser.add_argument("--stamp_temporal_K", type=int, default=10)
    parser.add_argument("--stamp_temporal_vit_layer", type=int, default=31)
    parser.add_argument("--stamp_temporal_adaptive_r", action="store_true", default=True)
    parser.add_argument("--stamp_temporal_no_adaptive_r", action="store_true", default=False)
    parser.add_argument("--stamp_temporal_r_base", type=float, default=0.85)
    parser.add_argument("--stamp_temporal_vit_layers", type=str, default=None)
    parser.add_argument("--stamp_temporal_compress", action="store_true", default=False)
    parser.add_argument("--stamp_temporal_merge", action="store_true", default=False)
    # Phase 2A / 2B / 3 flags (mirrored from DeViBench eval)
    parser.add_argument("--stamp_temporal_repack_pos", action="store_true", default=False)
    parser.add_argument("--stamp_temporal_equal_layer_weights", action="store_true", default=False)
    parser.add_argument("--stamp_temporal_focus_enhance", action="store_true", default=False)
    parser.add_argument("--stamp_temporal_focus_enhance_alpha", type=float, default=0.1)
    parser.add_argument("--stamp_temporal_mssavt", action="store_true", default=False)
    parser.add_argument("--stamp_temporal_mssavt_alpha", type=float, default=0.15)
    parser.add_argument("--stamp_temporal_sc_scale", type=float, default=0.0)
    parser.add_argument("--stamp_temporal_sc_seed", type=int, default=0)

    # TAST
    parser.add_argument("--tast", action="store_true", default=False)
    parser.add_argument("--tast_n_tokens", type=int, default=32)
    parser.add_argument("--tast_gamma", type=float, default=0.1)
    parser.add_argument("--tast_only", action="store_true", default=False)
    parser.add_argument("--tast_blend_alpha", type=float, default=0.2)
    # Phase-7A: Hierarchical TAST
    parser.add_argument("--tast_hierarchical", action="store_true", default=False)
    parser.add_argument("--tast_gamma_long", type=float, default=0.01)
    parser.add_argument("--tast_segment_len", type=int, default=8)
    # Phase-7B: Adaptive γ
    parser.add_argument("--tast_adaptive_gamma", action="store_true", default=False)
    parser.add_argument("--tast_gamma_tau", type=float, default=40.0)
    # Phase-7C: Keep-ratio ramp
    parser.add_argument("--stamp_ratio_ramp", action="store_true", default=False)
    parser.add_argument("--stamp_ratio_ramp_early", type=float, default=0.15)
    parser.add_argument("--stamp_ratio_ramp_late", type=float, default=0.45)
    parser.add_argument("--stamp_ratio_ramp_chunks", type=int, default=30)

    # DSTM
    parser.add_argument("--dstm", action="store_true", default=False)
    parser.add_argument("--dstm_only", action="store_true", default=False)
    parser.add_argument("--dstm_scene_tokens", type=int, default=16)
    parser.add_argument("--dstm_delta_tokens", type=int, default=16)
    parser.add_argument("--dstm_gamma_scene", type=float, default=0.05)
    parser.add_argument("--dstm_gamma_delta", type=float, default=0.2)
    parser.add_argument("--dstm_surprise_beta", type=float, default=5.0)
    parser.add_argument("--dstm_surprise_tau", type=float, default=0.3)
    parser.add_argument("--dstm_blend_alpha", type=float, default=0.2)

    # Spatial pruning: CRISP / FOCUS / PRISM
    parser.add_argument("--crisp", action="store_true", help="Enable CRISP spatial pruning.")
    parser.add_argument("--crisp_r", type=float, default=0.85)
    parser.add_argument("--crisp_grid_size", type=int, default=4)
    parser.add_argument("--focus", action="store_true", help="Enable FOCUS spatial pruning.")
    parser.add_argument("--focus_r", type=float, default=0.85)
    parser.add_argument("--focus_enhance_alpha", type=float, default=0.1)
    parser.add_argument("--focus_text_weight", type=float, default=0.6)
    parser.add_argument("--prism", action="store_true", help="Enable PRISM spatial pruning.")
    parser.add_argument("--prism_r", type=float, default=0.85)
    parser.add_argument("--prism_fine_ratio", type=float, default=0.5)
    parser.add_argument("--prism_pool_size", type=int, default=2)
    parser.add_argument("--prism_enhance_alpha", type=float, default=0.1)

    args = parser.parse_args()

    if args.stamp_temporal_no_adaptive_r:
        args.stamp_temporal_adaptive_r = False
    if args.stamp_temporal_vit_layers:
        args.stamp_temporal_vit_layers = [int(x.strip()) for x in args.stamp_temporal_vit_layers.split(',')]

    # Spatial modules need STAMP-Temporal pipeline for ViT salience extraction
    if (args.crisp or args.focus or args.prism) and not args.stamp_temporal:
        print("Spatial pruning enabled: auto-enabling --stamp_temporal --stamp_temporal_r 1.0")
        args.stamp_temporal = True
        if args.stamp_temporal_r is None:
            args.stamp_temporal_r = 1.0
        if not args.stamp_temporal_vit_layers:
            args.stamp_temporal_vit_layers = [7, 15, 23, 31]

    # Auto-enable FastV pipeline for STAMP-Temporal
    fastv_k = None
    fastv_r = 1.0
    if args.stamp_temporal:
        fastv_k = 2
        fastv_r = 1.0

    mp.set_start_method('spawn', force=True)

    model_path = args.model_path
    is_quantized = any(q in model_path.upper() for q in ['AWQ', 'GPTQ', 'INT4', 'INT8'])
    load_kwargs = dict(torch_dtype="auto", attn_implementation='flash_attention_2')
    if is_quantized:
        load_kwargs['device_map'] = 'auto'
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    except:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)

    processor = AutoProcessor.from_pretrained(model_path, padding_side='left')

    # Load MVBench data
    samples = load_mvbench_data(args.data_dir)
    print(f"Loaded {len(samples)} MVBench samples across {len(set(s['task'] for s in samples))} tasks")

    options = ['No', 'Yes', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E']

    if args.stamp_temporal:
        from streaming_vlm.inference.qwen2_5.patch_model import convert_qwen2_5_to_streaming
        from streaming_vlm.inference.streaming_args import StreamingArgs

        model = convert_qwen2_5_to_streaming(model)

        streaming_args = StreamingArgs(pos_mode="shrink", fastv_k=fastv_k, fastv_r=fastv_r)
        streaming_args.stamp_temporal = args.stamp_temporal
        streaming_args.stamp_temporal_r = args.stamp_temporal_r
        streaming_args.stamp_temporal_alpha = args.stamp_temporal_alpha
        streaming_args.stamp_temporal_lambda = args.stamp_temporal_lambda
        streaming_args.stamp_temporal_K = args.stamp_temporal_K
        streaming_args.stamp_temporal_vit_layer = args.stamp_temporal_vit_layer
        streaming_args.stamp_temporal_adaptive_r = args.stamp_temporal_adaptive_r
        streaming_args.stamp_temporal_r_base = args.stamp_temporal_r_base
        streaming_args.stamp_temporal_compress = args.stamp_temporal_compress
        streaming_args.stamp_temporal_vit_layers = args.stamp_temporal_vit_layers
        streaming_args.stamp_temporal_merge = args.stamp_temporal_merge
        streaming_args.stamp_temporal_repack_pos = args.stamp_temporal_repack_pos
        streaming_args.stamp_temporal_equal_layer_weights = args.stamp_temporal_equal_layer_weights
        streaming_args.stamp_temporal_focus_enhance = args.stamp_temporal_focus_enhance
        streaming_args.stamp_temporal_focus_enhance_alpha = args.stamp_temporal_focus_enhance_alpha
        streaming_args.stamp_temporal_mssavt = args.stamp_temporal_mssavt
        streaming_args.stamp_temporal_mssavt_alpha = args.stamp_temporal_mssavt_alpha
        streaming_args.stamp_temporal_sc_scale = args.stamp_temporal_sc_scale
        streaming_args.stamp_temporal_sc_seed = args.stamp_temporal_sc_seed
        # TAST
        streaming_args.tast_enabled = args.tast
        streaming_args.tast_n_tokens = args.tast_n_tokens
        streaming_args.tast_gamma = args.tast_gamma
        streaming_args.tast_only = args.tast_only
        streaming_args.tast_blend_alpha = args.tast_blend_alpha
        # Phase-7A / 7B / 7C
        streaming_args.tast_hierarchical = args.tast_hierarchical
        streaming_args.tast_gamma_long = args.tast_gamma_long
        streaming_args.tast_segment_len = args.tast_segment_len
        streaming_args.tast_adaptive_gamma = args.tast_adaptive_gamma
        streaming_args.tast_gamma_tau = args.tast_gamma_tau
        streaming_args.stamp_ratio_ramp = args.stamp_ratio_ramp
        streaming_args.stamp_ratio_ramp_early = args.stamp_ratio_ramp_early
        streaming_args.stamp_ratio_ramp_late = args.stamp_ratio_ramp_late
        streaming_args.stamp_ratio_ramp_chunks = args.stamp_ratio_ramp_chunks
        # DSTM
        streaming_args.dstm_enabled = args.dstm
        streaming_args.dstm_only = args.dstm_only
        streaming_args.dstm_scene_tokens = args.dstm_scene_tokens
        streaming_args.dstm_delta_tokens = args.dstm_delta_tokens
        streaming_args.dstm_gamma_scene = args.dstm_gamma_scene
        streaming_args.dstm_gamma_delta = args.dstm_gamma_delta
        streaming_args.dstm_surprise_beta = args.dstm_surprise_beta
        streaming_args.dstm_surprise_tau = args.dstm_surprise_tau
        streaming_args.dstm_blend_alpha = args.dstm_blend_alpha
        # Spatial pruning: CRISP / FOCUS / PRISM
        streaming_args.crisp_enabled = args.crisp
        streaming_args.crisp_r = args.crisp_r
        streaming_args.crisp_grid_size = args.crisp_grid_size
        streaming_args.focus_enabled = args.focus
        streaming_args.focus_r = args.focus_r
        streaming_args.focus_enhance_alpha = args.focus_enhance_alpha
        streaming_args.focus_text_weight = args.focus_text_weight
        streaming_args.prism_enabled = args.prism
        streaming_args.prism_r = args.prism_r
        streaming_args.prism_fine_ratio = args.prism_fine_ratio
        streaming_args.prism_pool_size = args.prism_pool_size
        streaming_args.prism_enhance_alpha = args.prism_enhance_alpha

        model._streaming_args = streaming_args
        model.model._streaming_args = streaming_args

        def _setup_streaming_args_hook(module, args_in, kwargs):
            sa = module._streaming_args
            if (getattr(sa, 'stamp_temporal', False)) and not sa.stamp_no_reset:
                from streaming_vlm.inference.stamp_temporal import stamp_temporal_reset_state
                stamp_temporal_reset_state(sa)
            input_ids = kwargs.get('input_ids', None)
            if input_ids is None and len(args_in) > 0 and isinstance(args_in[0], torch.Tensor):
                input_ids = args_in[0]
            if input_ids is not None:
                sa.input_ids = input_ids
                sa.current_input_ids = input_ids
            video_grid_thw = kwargs.get('video_grid_thw', None)
            if video_grid_thw is not None:
                sa.video_grid_thw = video_grid_thw
            sa.second_per_grid_ts = kwargs.get('second_per_grid_ts', None)

        model.register_forward_pre_hook(_setup_streaming_args_hook, with_kwargs=True)

        if not is_quantized:
            model = model.cuda()
        video_dir = args.video_dir or args.data_dir
        results, peak_mem = stamp_streaming_mvbench_predict(
            model=model, processor=processor, samples=samples,
            video_dir=video_dir, options=options, n_chunks=args.n_chunks,
        )
    else:
        # Non-streaming baseline — TODO if needed
        raise NotImplementedError("Non-streaming MVBench eval not yet implemented. Use --stamp_temporal.")

    # Save results
    extras = []
    if args.stamp_temporal_vit_layers:
        extras.append('multi' + ''.join(str(l) for l in args.stamp_temporal_vit_layers))
    if args.tast:
        extras.append(f'tast{args.tast_n_tokens}g{args.tast_gamma}')
    if args.tast_only:
        extras.append('tast_only')
    if args.dstm:
        extras.append(f'dstm_s{args.dstm_scene_tokens}d{args.dstm_delta_tokens}_ba{args.dstm_blend_alpha}')
    if args.dstm_only:
        extras.append('dstm_only')
    extras_str = ('_' + '_'.join(extras)) if extras else ''
    r_str = args.stamp_temporal_r if args.stamp_temporal_r else '1.0'

    save_json_path = f'results/mvbench/mvbench_stamptemporal_r{r_str}_c{args.n_chunks}{extras_str}.json'
    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    json.dump(results, open(save_json_path, 'w'))
    print(f"\nResults saved to: {save_json_path}")

    save_txt_path = save_json_path.replace('.json', '.txt')
    save_function_print(evaluate_mvbench_results, save_txt_path, results)

    print(f"\nPeak GPU Memory: {peak_mem:.2f} MB")
