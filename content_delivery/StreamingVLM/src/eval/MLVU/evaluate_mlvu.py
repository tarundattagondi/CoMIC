"""
MLVU Benchmark Evaluation for STAMP
====================================
Evaluates STAMP on the MLVU (Multi-task Long Video Understanding) Dev set.
Videos range from 3 minutes to 2+ hours — ideal for temporal pruning.

MLVU Dev tasks (MCQ with 4 options):
  - plotQA:          Plot understanding (movies, shows)
  - needle:          Needle-in-a-haystack (find specific detail)
  - ego:             Egocentric video understanding
  - count:           Action/object counting
  - order:           Temporal ordering
  - anomaly_reco:    Anomaly recognition (surveillance)
  - topic_reasoning: Topic-based reasoning

Usage:
  python3 streaming_vlm/eval/MLVU/evaluate_mlvu.py \
    --data_dir data/mlvu/json \
    --video_dir data/mlvu/video \
    --model_path mit-han-lab/StreamingVLM \
    --fastv_k 2 --fastv_r 1.0 \
    --stamp_r1 0.75 --stamp_alpha 0.5 --stamp_lambda 0.3 --stamp_K 10 \
    --n_chunks 5
"""

import json, os, torch, tqdm, sys, argparse, time
import numpy as np
import decord

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    logging,
)
from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils.vision_process import process_vision_info, smart_nframes, FPS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

logger = logging.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# MLVU Dataset
# ═══════════════════════════════════════════════════════════════════

# Dev set tasks (MCQ only — skip sub_scene and summary which are open-ended)
MLVU_TASKS = {
    "plotQA":         ("1_plotQA.json",         "plotQA",         "video"),
    "needle":         ("2_needle.json",         "needle",         "video"),
    "ego":            ("3_ego.json",            "ego",            "video"),
    "count":          ("4_count.json",          "count",          "video"),
    "order":          ("5_order.json",          "order",          "video"),
    "anomaly_reco":   ("6_anomaly_reco.json",   "anomaly_reco",   "video"),
    "topic_reasoning":("7_topic_reasoning.json","topic_reasoning", "video"),
}


class MLVUDataset:
    """Load MLVU Dev MCQ data from either JSON or Parquet format.

    JSON format (from GitHub repo):
        data_dir/1_plotQA.json  — each entry has {video, question, candidates, answer, duration}
        video names like "movie101_66.mp4"

    Parquet format (from HuggingFace sy1998/MLVU_dev):
        data_dir/test-00000-of-00001.parquet
        columns: video_name, duration, question, candidates, answer (letter), task_type, question_id
        video names like "needle_32.mp4", "surveil_0.mp4"
    """

    def __init__(self, data_dir, video_dir, tasks=None):
        self.video_dir = video_dir
        self.samples = []

        # Auto-detect format: prefer JSON (matches downloaded video names)
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        if json_files:
            self._load_json(data_dir, tasks)
        else:
            parquet_path = os.path.join(data_dir, 'test-00000-of-00001.parquet')
            self._load_parquet(parquet_path, tasks)

        # Filter to only samples whose video files exist
        available = []
        missing = 0
        for s in self.samples:
            vpath = os.path.join(self.video_dir, s['video'])
            if os.path.exists(vpath):
                available.append(s)
            else:
                missing += 1
        if missing > 0:
            print(f"Warning: {missing} videos not found, {len(available)} available")
        self.samples = available
        print(f"Dataset ready: {len(self.samples)} samples")

    def _load_parquet(self, parquet_path, tasks):
        """Load from HuggingFace parquet format."""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            print("pyarrow not available, falling back to JSON loading")
            return self._load_json(os.path.dirname(parquet_path), tasks)

        table = pq.read_table(parquet_path)
        data = table.to_pydict()
        task_filter = set(tasks) if tasks else None

        for i in range(len(data['video_name'])):
            task = data['task_type'][i]
            if task_filter and task not in task_filter:
                continue

            # Parquet answer is a letter (A/B/C/D)
            answer_letter = data['answer'][i]
            candidates = data['candidates'][i]

            # Map task to video subdirectory
            task_to_subdir = {
                'plotQA': 'plotQA', 'needle': 'needle', 'ego': 'ego',
                'count': 'count', 'order': 'order',
                'anomaly_reco': 'anomaly_reco', 'topic_reasoning': 'topic_reasoning',
            }
            subdir = task_to_subdir.get(task, task)

            self.samples.append({
                'task_type': task,
                'video': data['video_name'][i],
                'question': data['question'][i],
                'candidates': candidates,
                'answer_letter': answer_letter,
                'answer_idx': ord(answer_letter) - ord('A'),
                'duration': data['duration'][i],
            })
        print(f"Loaded {len(self.samples)} samples from parquet")

    def _load_json(self, data_dir, tasks):
        """Load from GitHub JSON format."""
        task_list = tasks or list(MLVU_TASKS.keys())
        for task_name in task_list:
            if task_name not in MLVU_TASKS:
                print(f"Warning: unknown task '{task_name}', skipping")
                continue
            json_file, video_subdir, _ = MLVU_TASKS[task_name]
            json_path = os.path.join(data_dir, json_file)
            if not os.path.exists(json_path):
                print(f"Warning: {json_path} not found, skipping task '{task_name}'")
                continue
            with open(json_path, 'r') as f:
                entries = json.load(f)
            for entry in entries:
                # JSON answer is the text of the correct option
                answer_idx = -1
                for idx, c in enumerate(entry['candidates']):
                    if c == entry['answer']:
                        answer_idx = idx
                        break
                self.samples.append({
                    'task_type': task_name,
                    'video': entry['video'],
                    'question': entry['question'],
                    'candidates': entry['candidates'],
                    'answer_letter': chr(ord('A') + answer_idx) if answer_idx >= 0 else '?',
                    'answer_idx': answer_idx,
                    'duration': entry.get('duration', 0),
                })
        print(f"Loaded {len(self.samples)} samples from JSON")

    def format_mcq(self, sample):
        """Format question + options in MLVU style. Returns (question_text, correct_index)."""
        q = f"Question: {sample['question']}\nOptions:\n"
        for idx, c in enumerate(sample['candidates']):
            q += f"({chr(ord('A') + idx)}) {c}\n"
        q = q.rstrip()
        return q, sample['answer_idx']


def stamp_reset_state(sa):
    """Reset all STAMP temporal state for a new video sample."""
    sa.chunk_idx = 0
    sa.attention_momentum = None
    sa.attention_momentum_long = None
    sa.prev_visual_feats = None
    sa.stamp_last_attn_scores = None
    sa.stamp_curr_visual_feats = None
    sa.stamp_kept_vis_local_indices = None
    sa.stamp_n_vis = None
    sa.stamp_is_keyframe = False
    sa.stamp_running_M_mean = None
    sa.stamp_running_M_std = None
    sa.stamp_running_N_mean = None
    sa.stamp_running_N_std = None
    # STAMP-Temporal state
    if getattr(sa, 'stamp_temporal', False):
        from streaming_vlm.inference.stamp_temporal import stamp_temporal_reset_state
        stamp_temporal_reset_state(sa)


def load_video(video_path, max_frames=None):
    """Load video using decord, with optional frame limit for very long videos."""
    vr = decord.VideoReader(video_path, num_threads=2)
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = total_frames / fps

    if max_frames and total_frames > max_frames:
        # Uniformly sample max_frames
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        frames = vr.get_batch(indices.tolist())
    else:
        frames = vr.get_batch(list(range(total_frames)))

    # Convert to [T, C, H, W] tensor
    video = torch.from_numpy(frames.asnumpy()).permute(0, 3, 1, 2).float()
    return video, fps, duration


# ═══════════════════════════════════════════════════════════════════
# STAMP Streaming Eval for MLVU
# ═══════════════════════════════════════════════════════════════════

def stamp_mlvu_eval(
    model, processor, dataset, n_chunks=5, max_frames_per_chunk=None,
    checkpoint_path=None, resume_results=None,
):
    """
    Multi-chunk streaming eval on MLVU videos.

    For long videos (3 min - 2 hours), we use MORE chunks than DeViBench
    to properly exercise STAMP's temporal signals:
      - Chunks 0..n_chunks-2: context (warm up momentum + novelty)
      - Chunk n_chunks-1: video + question → prediction

    Unlike DeViBench (short 4-9s videos), MLVU videos are long enough
    that temporal novelty (N_t) will be meaningful — different chunks
    see genuinely different content.
    """
    # Token IDs for answer options (A, B, C, D)
    options = ['A', 'B', 'C', 'D']
    option_ids = [
        processor.tokenizer(f'\n{opt}').input_ids[-1] for opt in options
    ]

    model.eval()
    device = next(model.parameters()).device
    sa = getattr(model, '_streaming_args', None)

    results = []
    acc_dict = {}
    start_idx = 0
    torch.cuda.reset_peak_memory_stats()

    # Resume from checkpoint if provided
    if resume_results is not None:
        results = resume_results
        start_idx = len(results)
        # Rebuild acc_dict from resumed results
        for r in results:
            t = r['task']
            if t not in acc_dict:
                acc_dict[t] = [0, 0]
            acc_dict[t][1] += 1
            if r['correct']:
                acc_dict[t][0] += 1
        print(f"Resumed from checkpoint: {start_idx} samples already done")

    with torch.no_grad():
        for i in tqdm.tqdm(range(start_idx, len(dataset.samples)), desc="MLVU Eval",
                           initial=start_idx, total=len(dataset.samples)):
            sample = dataset.samples[i]
            task = sample['task_type']
            if task not in acc_dict:
                acc_dict[task] = [0, 0]  # [correct, total]
            acc_dict[task][1] += 1

            # ── Reset STAMP state ──────────────────────────────────────
            if sa is not None:
                stamp_reset_state(sa)
                sa.stamp_no_reset = True
                sa.input_ids = None

            t_start = time.perf_counter()

            # ── Load video ─────────────────────────────────────────────
            video_path = os.path.join(dataset.video_dir, sample['video'])
            try:
                video, _ = _read_video_decord_plus({
                    'video': video_path,
                })
                video = _spatial_resize_video(video)  # [T, C, H, W]
            except Exception as e:
                print(f'Skipping sample {i} ({sample["video"]}): {e}')
                continue

            T = video.shape[0]

            # ── Context chunks: warm up STAMP ──────────────────────────
            # Pass full video each time (same as DeViBench methodology).
            # This builds temporal state: momentum from attention,
            # novelty from visual feature changes.
            for _ctx_idx in range(n_chunks - 1):
                ctx_conv = [{"role": "user", "content": [
                    {"type": "video", "video": video},
                    {"type": "text", "text": "Watch this video carefully."},
                ]}]
                ctx_text = processor.apply_chat_template(
                    ctx_conv, tokenize=False, add_generation_prompt=False
                )
                ctx_inputs = processor(
                    text=[ctx_text], videos=[video],
                    return_tensors="pt", padding=True,
                ).to(device)
                _ = model(**ctx_inputs)

            # ── Final chunk: video + MCQ question ──────────────────────
            mcq_text, correct_idx = dataset.format_mcq(sample)
            question_prompt = (
                "Carefully watch this video and pay attention to every detail. "
                "Based on your observations, select the best option that accurately "
                "addresses the question.\n\n"
                + mcq_text
                + "\nPlease select the correct answer."
            )

            final_conv = [{"role": "user", "content": [
                {"type": "video", "video": video},
                {"type": "text", "text": question_prompt},
            ]}]
            final_text = processor.apply_chat_template(
                final_conv, tokenize=False, add_generation_prompt=True
            )
            final_text = final_text + "The answer is:\n"

            final_inputs = processor(
                text=[final_text], videos=[video],
                return_tensors="pt", padding=True,
            ).to(device)

            final_outputs = model(**final_inputs)
            last_logits = final_outputs.logits[0, -1, :]
            opt_logits = last_logits[torch.tensor(option_ids, device=device)]

            # Only consider valid options (A..D based on num candidates)
            n_opts = len(sample['candidates'])
            pred_idx = opt_logits[:n_opts].argmax().item()
            pred_letter = chr(ord('A') + pred_idx)
            is_correct = (pred_idx == correct_idx)

            if is_correct:
                acc_dict[task][0] += 1

            t_end = time.perf_counter()

            results.append({
                'task': task,
                'video': sample['video'],
                'duration': sample['duration'],
                'question': sample['question'],
                'answer': sample['answer_letter'],
                'prediction': sample['candidates'][pred_idx] if pred_idx < len(sample['candidates']) else 'INVALID',
                'pred_letter': pred_letter,
                'correct': is_correct,
                'inference_time_ms': round((t_end - t_start) * 1000, 2),
            })

            if sa is not None:
                sa.stamp_no_reset = False

            # Checkpoint every 100 samples
            if checkpoint_path and len(results) % 100 == 0:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                with open(checkpoint_path, 'w') as f:
                    json.dump(results, f)
                print(f"  Checkpoint saved: {len(results)} samples → {checkpoint_path}")

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_memory_gb = peak_bytes / (1024 ** 3)

    return results, acc_dict, peak_memory_gb


def print_results(results, acc_dict, peak_memory_gb, args):
    """Print formatted results."""
    print("\n" + "=" * 60)
    print("MLVU STAMP Evaluation Results")
    print("=" * 60)

    total_correct = 0
    total_count = 0
    for task in sorted(acc_dict.keys()):
        correct, total = acc_dict[task]
        total_correct += correct
        total_count += total
        pct = correct / total * 100 if total > 0 else 0
        print(f"  {task:20s}: {correct:4d}/{total:4d} = {pct:.2f}%")

    overall = total_correct / total_count * 100 if total_count > 0 else 0
    print(f"\n  {'OVERALL':20s}: {total_correct:4d}/{total_count:4d} = {overall:.2f}%")
    print(f"\n  Peak GPU Memory: {peak_memory_gb:.2f} GB")

    # Duration breakdown
    durations = [r['duration'] for r in results]
    if durations:
        print(f"\n  Video Duration Stats:")
        print(f"    Min: {min(durations):.0f}s  Max: {max(durations):.0f}s  Avg: {np.mean(durations):.0f}s")

    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLVU Benchmark Eval with STAMP")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to MLVU JSON annotation dir (containing 1_plotQA.json etc.)")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Path to MLVU video root dir (containing plotQA/, needle/, etc.)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace model path")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names to evaluate (default: all)")

    # FastV / STAMP args (same as DeViBench eval)
    parser.add_argument("--fastv_k", type=int, default=None)
    parser.add_argument("--fastv_r", type=float, default=0.5)
    parser.add_argument("--stamp_r1", type=float, default=None)
    parser.add_argument("--stamp_alpha", type=float, default=0.5)
    parser.add_argument("--stamp_lambda", type=float, default=0.3)
    parser.add_argument("--stamp_K", type=int, default=10)
    parser.add_argument("--n_chunks", type=int, default=5,
                        help="Number of chunks for STAMP streaming (default 5 for long videos)")

    # STAMP iteration 2 flags
    parser.add_argument("--stamp_adaptive_r1", action="store_true", default=False)
    parser.add_argument("--stamp_adaptive_r1_high", type=float, default=0.3)
    parser.add_argument("--stamp_adaptive_r1_low", type=float, default=0.1)
    parser.add_argument("--stamp_momentum_decay", action="store_true", default=False)
    parser.add_argument("--stamp_gamma", type=float, default=0.5)
    parser.add_argument("--stamp_adaptive_kf", action="store_true", default=False)
    parser.add_argument("--stamp_adaptive_kf_threshold", type=float, default=0.5)
    parser.add_argument("--stamp_hierarchical", action="store_true", default=False)
    parser.add_argument("--stamp_lambda_long", type=float, default=0.1)
    parser.add_argument("--stamp_alpha_short", type=float, default=0.35)
    parser.add_argument("--stamp_alpha_long", type=float, default=0.35)
    parser.add_argument("--stamp_merge", action="store_true", default=False)

    # STAMP-Temporal args
    parser.add_argument("--stamp_temporal", action="store_true", default=False,
                        help="Enable STAMP-Temporal (ViT-sourced attention)")
    parser.add_argument("--stamp_temporal_r", type=float, default=0.85,
                        help="STAMP-Temporal: keep ratio (default 0.85)")
    parser.add_argument("--stamp_temporal_alpha", type=float, default=0.5)
    parser.add_argument("--stamp_temporal_lambda", type=float, default=0.3)
    parser.add_argument("--stamp_temporal_K", type=int, default=10)
    parser.add_argument("--stamp_temporal_vit_layer", type=int, default=31)
    parser.add_argument("--stamp_temporal_vit_layers", type=str, default=None,
                        help="Comma-separated ViT layers for multi-layer fusion (e.g. '7,15,23,31')")
    parser.add_argument("--stamp_temporal_adaptive_r", action="store_true", default=True)
    parser.add_argument("--stamp_temporal_no_adaptive_r", action="store_true", default=False)
    parser.add_argument("--stamp_temporal_r_base", type=float, default=0.85)
    parser.add_argument("--stamp_temporal_compress", action="store_true", default=False)
    parser.add_argument("--stamp_temporal_merge", action="store_true", default=False)
    # TAST: Temporal Accumulative State Tokens
    parser.add_argument("--tast", action="store_true", default=False,
                        help="Enable TAST: accumulate pruned tokens into persistent state tokens.")
    parser.add_argument("--tast_n_tokens", type=int, default=32,
                        help="TAST: number of state tokens (default 32).")
    parser.add_argument("--tast_gamma", type=float, default=0.1,
                        help="TAST: EMA decay for state token updates (default 0.1).")
    parser.add_argument("--tast_only", action="store_true", default=False,
                        help="TAST-only mode: no pruning, inject state tokens.")
    # DSTM (Architecture 5) arguments
    parser.add_argument("--dstm", action="store_true", default=False,
                        help="Enable DSTM: delta-state temporal memory with surprise gating.")
    parser.add_argument("--dstm_only", action="store_true", default=False,
                        help="DSTM-only mode: no pruning, inject scene+delta memory.")
    parser.add_argument("--dstm_scene_tokens", type=int, default=16,
                        help="DSTM: number of scene memory slots (default 16).")
    parser.add_argument("--dstm_delta_tokens", type=int, default=16,
                        help="DSTM: number of delta memory slots (default 16).")
    parser.add_argument("--dstm_gamma_scene", type=float, default=0.05,
                        help="DSTM: scene state EMA decay (default 0.05).")
    parser.add_argument("--dstm_gamma_delta", type=float, default=0.2,
                        help="DSTM: delta state EMA decay (default 0.2).")
    parser.add_argument("--dstm_surprise_beta", type=float, default=5.0,
                        help="DSTM: surprise gate sensitivity (default 5.0).")
    parser.add_argument("--dstm_surprise_tau", type=float, default=0.3,
                        help="DSTM: surprise gate threshold (default 0.3).")
    parser.add_argument("--dstm_blend_alpha", type=float, default=0.2,
                        help="DSTM: blend ratio for memory injection (0=none, 1=replace, default 0.2).")
    parser.add_argument("--tast_blend_alpha", type=float, default=0.2,
                        help="TAST: blend ratio for state token injection (0=none, 1=replace, default 0.2).")

    # ── Phase-7A: Hierarchical TAST ──
    parser.add_argument("--tast_hierarchical", action="store_true", default=False,
                        help="Phase-7A: maintain a second slow-EMA TAST pool and interleave at inject time.")
    parser.add_argument("--tast_gamma_long", type=float, default=0.01,
                        help="Phase-7A: EMA decay of long-scale TAST pool (default 0.01).")
    parser.add_argument("--tast_segment_len", type=int, default=8,
                        help="Phase-7A: chunks between long-pool updates (default 8).")
    # ── Phase-7B: Adaptive γ schedule ──
    parser.add_argument("--tast_adaptive_gamma", action="store_true", default=False,
                        help="Phase-7B: γ(t)=γ0·exp(-t/τ); preserves early state, locks late chunks.")
    parser.add_argument("--tast_gamma_tau", type=float, default=40.0,
                        help="Phase-7B: τ in chunks for γ(t) decay (default 40).")
    # ── Phase-7C: Keep-ratio ramp ──
    parser.add_argument("--stamp_ratio_ramp", action="store_true", default=False,
                        help="Phase-7C: override r_t with linear ramp r_early→r_late over ramp_chunks.")
    parser.add_argument("--stamp_ratio_ramp_early", type=float, default=0.15,
                        help="Phase-7C: keep ratio at chunk 0 (default 0.15).")
    parser.add_argument("--stamp_ratio_ramp_late", type=float, default=0.45,
                        help="Phase-7C: keep ratio after ramp_chunks (default 0.45).")
    parser.add_argument("--stamp_ratio_ramp_chunks", type=int, default=30,
                        help="Phase-7C: number of chunks to ramp over (default 30).")

    # ── Spatial pruning: CRISP / FOCUS / PRISM ──
    parser.add_argument("--crisp", action="store_true", help="Enable CRISP spatial pruning.")
    parser.add_argument("--crisp_r", type=float, default=0.85, help="CRISP: keep ratio.")
    parser.add_argument("--crisp_grid_size", type=int, default=4, help="CRISP: grid cells per side.")
    parser.add_argument("--focus", action="store_true", help="Enable FOCUS spatial pruning.")
    parser.add_argument("--focus_r", type=float, default=0.85, help="FOCUS: keep ratio.")
    parser.add_argument("--focus_enhance_alpha", type=float, default=0.1, help="FOCUS: enhancement strength.")
    parser.add_argument("--focus_text_weight", type=float, default=0.6, help="FOCUS: text vs ViT weight.")
    parser.add_argument("--prism", action="store_true", help="Enable PRISM spatial pruning.")
    parser.add_argument("--prism_r", type=float, default=0.85, help="PRISM: keep ratio.")
    parser.add_argument("--prism_fine_ratio", type=float, default=0.5, help="PRISM: fine token budget ratio.")
    parser.add_argument("--prism_pool_size", type=int, default=2, help="PRISM: pooling factor.")
    parser.add_argument("--prism_enhance_alpha", type=float, default=0.1, help="PRISM: cross-resolution enhancement.")

    parser.add_argument("--resume", type=str, default=None,
                        help="Path to partial results JSON to resume from")

    args = parser.parse_args()

    # Handle STAMP-Temporal flags
    if args.stamp_temporal_no_adaptive_r:
        args.stamp_temporal_adaptive_r = False
    if args.stamp_temporal_vit_layers:
        args.stamp_temporal_vit_layers = [int(x.strip()) for x in args.stamp_temporal_vit_layers.split(',')]
    else:
        args.stamp_temporal_vit_layers = None

    # Spatial modules need STAMP-Temporal pipeline for ViT salience extraction
    if (args.crisp or args.focus or args.prism) and not args.stamp_temporal:
        print("Spatial pruning enabled: auto-enabling --stamp_temporal --stamp_temporal_r 1.0")
        args.stamp_temporal = True
        if args.stamp_temporal_r is None:
            args.stamp_temporal_r = 1.0
        if not args.stamp_temporal_vit_layers:
            args.stamp_temporal_vit_layers = [7, 15, 23, 31]

    # Auto-enable FastV if STAMP is on
    if args.stamp_r1 is not None and args.fastv_k is None:
        print("STAMP enabled: auto-setting --fastv_k 2 --fastv_r 1.0")
        args.fastv_k = 2
        args.fastv_r = 1.0
    if args.stamp_temporal and args.fastv_k is None:
        print("STAMP-Temporal enabled: auto-setting --fastv_k 2 --fastv_r 1.0")
        args.fastv_k = 2
        args.fastv_r = 1.0

    # ── Load model ─────────────────────────────────────────────────
    model_path = args.model_path
    is_quantized = any(q in model_path.upper() for q in ['AWQ', 'GPTQ', 'INT4', 'INT8'])
    load_kwargs = dict(torch_dtype="auto", attn_implementation='flash_attention_2')
    if is_quantized:
        load_kwargs['device_map'] = 'auto'
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs
        )
    except Exception:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs
        )

    processor = AutoProcessor.from_pretrained(model_path, padding_side='left')

    # ── Patch model with STAMP/FastV ───────────────────────────────
    if args.fastv_k is not None:
        from streaming_vlm.inference.qwen2_5.patch_model import convert_qwen2_5_to_streaming
        from streaming_vlm.inference.streaming_args import StreamingArgs

        model = convert_qwen2_5_to_streaming(model)

        streaming_args = StreamingArgs(
            pos_mode="shrink", fastv_k=args.fastv_k, fastv_r=args.fastv_r,
            stamp_r1=args.stamp_r1, stamp_alpha=args.stamp_alpha,
            stamp_lambda=args.stamp_lambda, stamp_K=args.stamp_K,
        )
        # Iteration 2 flags
        streaming_args.stamp_adaptive_r1 = args.stamp_adaptive_r1
        streaming_args.stamp_adaptive_r1_high = args.stamp_adaptive_r1_high
        streaming_args.stamp_adaptive_r1_low = args.stamp_adaptive_r1_low
        streaming_args.stamp_momentum_decay = args.stamp_momentum_decay
        streaming_args.stamp_gamma = args.stamp_gamma
        streaming_args.stamp_adaptive_kf = args.stamp_adaptive_kf
        streaming_args.stamp_adaptive_kf_threshold = args.stamp_adaptive_kf_threshold
        streaming_args.stamp_hierarchical = args.stamp_hierarchical
        streaming_args.stamp_lambda_long = args.stamp_lambda_long
        streaming_args.stamp_alpha_short = args.stamp_alpha_short
        streaming_args.stamp_alpha_long = args.stamp_alpha_long
        streaming_args.stamp_merge = args.stamp_merge

        # STAMP-Temporal fields
        streaming_args.stamp_temporal = args.stamp_temporal
        streaming_args.stamp_temporal_r = args.stamp_temporal_r
        streaming_args.stamp_temporal_alpha = args.stamp_temporal_alpha
        streaming_args.stamp_temporal_lambda = args.stamp_temporal_lambda
        streaming_args.stamp_temporal_K = args.stamp_temporal_K
        streaming_args.stamp_temporal_vit_layer = args.stamp_temporal_vit_layer
        streaming_args.stamp_temporal_vit_layers = args.stamp_temporal_vit_layers
        streaming_args.stamp_temporal_adaptive_r = args.stamp_temporal_adaptive_r
        streaming_args.stamp_temporal_r_base = args.stamp_temporal_r_base
        streaming_args.stamp_temporal_compress = args.stamp_temporal_compress
        streaming_args.stamp_temporal_merge = args.stamp_temporal_merge
        # TAST
        streaming_args.tast_enabled = args.tast
        streaming_args.tast_n_tokens = args.tast_n_tokens
        streaming_args.tast_gamma = args.tast_gamma
        streaming_args.tast_only = args.tast_only
        # DSTM (Architecture 5)
        streaming_args.dstm_enabled = args.dstm
        streaming_args.dstm_only = args.dstm_only
        streaming_args.dstm_scene_tokens = args.dstm_scene_tokens
        streaming_args.dstm_delta_tokens = args.dstm_delta_tokens
        streaming_args.dstm_gamma_scene = args.dstm_gamma_scene
        streaming_args.dstm_gamma_delta = args.dstm_gamma_delta
        streaming_args.dstm_surprise_beta = args.dstm_surprise_beta
        streaming_args.dstm_surprise_tau = args.dstm_surprise_tau
        streaming_args.dstm_blend_alpha = args.dstm_blend_alpha
        streaming_args.tast_blend_alpha = args.tast_blend_alpha
        # Phase-7A: Hierarchical TAST
        streaming_args.tast_hierarchical = args.tast_hierarchical
        streaming_args.tast_gamma_long = args.tast_gamma_long
        streaming_args.tast_segment_len = args.tast_segment_len
        # Phase-7B: Adaptive γ
        streaming_args.tast_adaptive_gamma = args.tast_adaptive_gamma
        streaming_args.tast_gamma_tau = args.tast_gamma_tau
        # Phase-7C: Keep-ratio ramp
        streaming_args.stamp_ratio_ramp = args.stamp_ratio_ramp
        streaming_args.stamp_ratio_ramp_early = args.stamp_ratio_ramp_early
        streaming_args.stamp_ratio_ramp_late = args.stamp_ratio_ramp_late
        streaming_args.stamp_ratio_ramp_chunks = args.stamp_ratio_ramp_chunks
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

        # Pre-hook
        def _setup_streaming_args_hook(module, args_in, kwargs):
            sa = module._streaming_args
            if (sa.stamp_r1 is not None or sa.stamp_temporal) and not sa.stamp_no_reset:
                stamp_reset_state(sa)
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

    # ── Load dataset ───────────────────────────────────────────────
    tasks = args.tasks.split(',') if args.tasks else None
    dataset = MLVUDataset(args.data_dir, args.video_dir, tasks=tasks)

    if len(dataset.samples) == 0:
        print("No samples loaded. Check --data_dir and --video_dir paths.")
        sys.exit(1)

    # ── Resume from checkpoint if provided ───────────────────────
    resume_results = None
    if args.resume and os.path.exists(args.resume):
        with open(args.resume, 'r') as f:
            resume_results = json.load(f)
        print(f"Loaded {len(resume_results)} results from {args.resume}")

    # ── Build checkpoint path ──────────────────────────────────────
    if args.stamp_temporal:
        extras = []
        if args.stamp_temporal_adaptive_r: extras.append('adaptive')
        if args.stamp_temporal_merge: extras.append('merge')
        if args.stamp_temporal_vit_layers: extras.append('multi' + ''.join(str(l) for l in args.stamp_temporal_vit_layers))
        if args.tast: extras.append(f'tast{args.tast_n_tokens}g{args.tast_gamma}')
        if args.tast_hierarchical: extras.append(f'tasthier_gl{args.tast_gamma_long}_seg{args.tast_segment_len}')
        if args.tast_adaptive_gamma: extras.append(f'tastadagam_tau{args.tast_gamma_tau}')
        if args.stamp_ratio_ramp: extras.append(f'ramp{args.stamp_ratio_ramp_early}to{args.stamp_ratio_ramp_late}c{args.stamp_ratio_ramp_chunks}')
        if args.dstm: extras.append(f'dstm_s{args.dstm_scene_tokens}d{args.dstm_delta_tokens}')
        if args.dstm_only: extras.append('dstm_only')
        extras_str = ('_' + '_'.join(extras)) if extras else ''
        vit_str = f'vit{args.stamp_temporal_vit_layer}' if not args.stamp_temporal_vit_layers else ''
        ckpt_suffix = f'_stamptemporal_r{args.stamp_temporal_r}_{vit_str}_c{args.n_chunks}{extras_str}'
    elif args.stamp_r1 is not None:
        ckpt_suffix = f'_stamp_r1{args.stamp_r1}_a{args.stamp_alpha}_l{args.stamp_lambda}_K{args.stamp_K}_c{args.n_chunks}'
    elif args.fastv_k is not None:
        ckpt_suffix = f'_fastv_k{args.fastv_k}_r{args.fastv_r}'
    else:
        ckpt_suffix = '_baseline'
    checkpoint_path = f'results/mlvu/mlvu{ckpt_suffix}_checkpoint.json'

    # ── Run evaluation ─────────────────────────────────────────────
    if not is_quantized:
        model = model.cuda()
    results, acc_dict, peak_memory_gb = stamp_mlvu_eval(
        model=model, processor=processor, dataset=dataset,
        n_chunks=args.n_chunks,
        checkpoint_path=checkpoint_path,
        resume_results=resume_results,
    )

    # ── Print and save results ─────────────────────────────────────
    print_results(results, acc_dict, peak_memory_gb, args)

    # Save results
    suffix = ckpt_suffix  # reuse the suffix built earlier

    save_dir = 'results/mlvu'
    os.makedirs(save_dir, exist_ok=True)

    # JSON with full results
    json_path = os.path.join(save_dir, f'mlvu{suffix}.json')
    config = {
        'stamp_r1': args.stamp_r1,
        'stamp_alpha': args.stamp_alpha,
        'stamp_lambda': args.stamp_lambda,
        'stamp_K': args.stamp_K,
        'fastv_k': args.fastv_k,
        'fastv_r': args.fastv_r,
        'n_chunks': args.n_chunks,
    }
    if args.stamp_temporal:
        config.update({
            'stamp_temporal': True,
            'stamp_temporal_r': args.stamp_temporal_r,
            'stamp_temporal_vit_layer': args.stamp_temporal_vit_layer,
            'stamp_temporal_vit_layers': args.stamp_temporal_vit_layers,
            'stamp_temporal_adaptive_r': args.stamp_temporal_adaptive_r,
            'stamp_temporal_merge': args.stamp_temporal_merge,
        })
    json.dump({
        'config': config,
        'acc_dict': acc_dict,
        'peak_memory_gb': peak_memory_gb,
        'results': results,
    }, open(json_path, 'w'), indent=2)

    # Summary text file
    txt_path = os.path.join(save_dir, f'mlvu{suffix}.txt')
    with open(txt_path, 'w') as f:
        total_c = sum(v[0] for v in acc_dict.values())
        total_n = sum(v[1] for v in acc_dict.values())
        for task in sorted(acc_dict.keys()):
            c, n = acc_dict[task]
            f.write(f"{task}: {c}/{n}={c/n:.4f}\n")
        f.write(f"\nAccuracy: {total_c}/{total_n}={round(total_c/total_n*100, 2)}\n")
        f.write(f"Peak Memory: {peak_memory_gb:.2f} GB\n")

    print(f"\nResults saved to: {json_path}")
    print(f"Summary saved to: {txt_path}")
