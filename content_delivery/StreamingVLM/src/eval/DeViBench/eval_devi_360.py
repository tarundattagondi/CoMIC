import json, os, torch, functools, tqdm, random, sys, argparse
import multiprocessing as mp
import numpy as np
import decord
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, TrainerCallback, logging, Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from livecc_utils import _read_video_decord_plus, _spatial_resize_video
from qwen_vl_utils.vision_process import process_vision_info, smart_nframes, FPS
import functools
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

logger = logging.get_logger(__name__)
# HF-style logger

debug_quiet = False

def printq(*args, quiet=debug_quiet, **kwargs): 
    """Use like print; when quiet=True, suppress output.""" 
    if not quiet: 
        print(*args, **kwargs)

def _read_may1fps_video_decord(ele: dict):
    """read video using decord.VideoReader. can handle more cases compared to _read_video_decord.

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
        sample_fps
        clip_pts if return_pts=True
    """
    video_path = ele["video"]

    if os.path.exists(video_path):
        vr = decord.VideoReader(video_path, num_threads=2)
    else:
        raise ValueError(f'video_path {video_path} not found')

    video_start = ele.get('video_start', None)
    video_end = ele.get('video_end', None)

    video_fps = vr.get_avg_fps()

    clip_idxs, clip_pts = None, None

    if video_start is not None or video_end is not None:
        vr.get_frame_timestamp(0)
        video_pts = vr._frame_pts[:,1]
        video_start = video_pts[0] if not video_start else video_start
        video_end = video_pts[-1] if not video_end else video_end

        video_start = min(max(video_pts[0], video_start), video_pts[-1])
        video_end = min(max(video_pts[0], video_end), video_pts[-1])

        video_end = max(video_start + 1, video_end)

        clip_idxs = ((video_start <= video_pts) & (video_pts <= video_end)).nonzero()[0]

        total_frames = len(clip_idxs)
    else:
        total_frames = len(vr)

    total_frames_for_smart_nframes = total_frames
    video_fps_for_smart_nframes = video_fps

    if total_frames < 2:
        total_frames_for_smart_nframes = 2

    if video_fps < FPS:
        total_frames_for_smart_nframes = int(total_frames * FPS / video_fps)
        video_fps_for_smart_nframes = FPS

    nframes = smart_nframes(ele, total_frames=total_frames_for_smart_nframes, video_fps=video_fps_for_smart_nframes) 

    nframes_idxs = np.linspace(0, total_frames - 1, nframes).round().astype(int)

    clip_idxs = nframes_idxs if clip_idxs is None else clip_idxs[nframes_idxs]

    clip = torch.from_numpy(vr.get_batch(clip_idxs).asnumpy()).permute(0, 3, 1, 2)  # Convert to TCHW format

    sample_fps = len(clip_idxs) / max(total_frames, 1e-6) * video_fps

    return clip, sample_fps


def save_function_print(function: callable, save_path: str, *args, **kwargs):
    original_stdout = sys.stdout
    try:
        with open(save_path, 'w') as f:
            sys.stdout = f  
            function(*args, **kwargs)          
    finally:
        sys.stdout = original_stdout 


## Callback for computing inference times
class PerSampleTimingCallback(TrainerCallback):
    def __init__(self):
        self.timings = []
        self.start_time = None

    def on_prediction_step(self, args, state, control, **kwargs):
        """
        Event called after each prediction step.
        We calculate the time elapsed since the last step started.
        """
        # Record the end time of the current step
        end_time = time.perf_counter()
        
        if self.start_time is not None:
            # Calculate duration
            duration = end_time - self.start_time
            self.timings.append(duration)
        
        # Reset start time for the next step
        self.start_time = time.perf_counter()



class DeViBenchMCQDataset(Dataset):
    def __init__(self, path, question_prefix, question_postfix, answer_prefix, sample: int = None, device=None):
        lines = open(path).readlines()

        if sample is not None:
            random.seed(42)
            lines = random.sample(lines, sample)

        self.datums = [json.loads(line) for line in tqdm.tqdm(lines, desc='load datums')]

        if isinstance(self.datums[0], str):
            self.datums = [json.loads(datum) for datum in tqdm.tqdm(self.datums, desc='load datumsx2')]

        self.question_prefix = question_prefix
        self.question_postfix = question_postfix
        self.answer_prefix = answer_prefix

        self.data_dir = os.path.dirname(path)

        
    def __len__(self):
        return len(self.datums)

    def __getitem__(self, i):
        datum = self.datums[i]
        conversation = [{"role": "user", "content": []}]

        video_inputs = None

        if datum['task'] in ['REC', 'SSR', 'CRR']:  # 'REC', 'SSR', 'CRR' have already been chunked
            query = datum['question']
        else:
            query = self.question_prefix + datum['question'] + '\n' + '\n'.join(datum['options']) + self.question_postfix

        video, _ = _read_may1fps_video_decord({
            'video': os.path.join(self.data_dir, datum['video']), 
            'video_start': datum['video_start'], 
            'video_end': datum['video_end']
        })

        video = _spatial_resize_video(video)

        conversation[0]['content'].append({"type": "video", "video": video})

        video_inputs = [video]

        conversation[0]['content'].append({"type": "text", "text": query})

        if video_inputs is None:
            for _ in range(10):
                try:
                    _, video_inputs = process_vision_info(conversation)
                    break
                except:
                    print(f"{_}-th process_vision_info failed. retry...")
        return conversation, video_inputs[0]

    def data_collator(self, batch, processor):
        conversations, video_inputs = zip(*batch)

        texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)

        texts = [text + self.answer_prefix for text in texts]

        inputs = processor(
            text=texts,
            images=None,
            videos=list(video_inputs),
            padding=True,
            return_tensors="pt",
        )

        return inputs


def preprocess_logits_for_metrics(logits, labels, strict_option_ids): 
    return torch.stack([logit[(logit[:, 0] != -100).nonzero().squeeze()[-1], strict_option_ids] for logit in logits]).argmax(dim=-1)


def mcq_predict(
    model,
    processor,
    benchmark_path: str,
    options: list[str],
    question_prefix: str = '',
    question_postfix: str = '\nPlease select the correct answer.',
    answer_prefix: str = 'Answer:',
    abcd_previous_str: str = ': ',
    use_liger_kernel: bool = True,
    per_device_eval_batch_size: int = 1,
    dataloader_num_workers: int = 4,
    max_samples: int = None,
):
    strict_option_ids = [processor.tokenizer(f'{abcd_previous_str}{_}').input_ids[-1] for _ in options] 

    dataset = DeViBenchMCQDataset(benchmark_path, question_prefix=question_prefix, question_postfix=question_postfix, answer_prefix=answer_prefix, sample=max_samples)

    # Instantiate the timing callback
    timing_callback = PerSampleTimingCallback()

    trainer = Trainer(
        model=model, 
        args=TrainingArguments(
            output_dir='outputs/', do_predict=True, 
            per_device_eval_batch_size=per_device_eval_batch_size, 
            dataloader_num_workers=dataloader_num_workers, 
            report_to='none', use_liger_kernel=use_liger_kernel
        ), 
        data_collator=functools.partial(dataset.data_collator, processor=processor),
        processing_class=processor,
        preprocess_logits_for_metrics=functools.partial(preprocess_logits_for_metrics, strict_option_ids=strict_option_ids),
        callbacks=[timing_callback],
    )

    timing_callback.start_time = time.perf_counter()

    letter_idxs_predictions = trainer.predict(dataset, ignore_keys=['past_key_values', 'hidden_states', 'attentions', 'rope_deltas']).predictions

    # Obtain system metrics (latency/peak memory)
    inference_times = timing_callback.timings
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_memory_usage = peak_bytes / (1024 * 1024)

    torch.cuda.reset_peak_memory_stats()

    return letter_idxs_predictions, dataset.datums, trainer.args.process_index, inference_times, peak_memory_usage


def evaluate_ovobench_results(results: list):
    task_to_counts = {}
    total_answer = 0
    correct_answer = 0
    for result in results:
        total_answer += 1
        task = result['task']
        if task not in task_to_counts:
            task_to_counts[task] = {'correct': 0, 'total': 0}
        task_to_counts[task]['total'] += 1
        if result['response'][:len(result['answer'])] == result['answer']:
            correct_answer += 1
            task_to_counts[task]['correct'] += 1

    rt_accs, bt_accs, fr_accs = [], [], []
    for task, counts in task_to_counts.items():
        print(f'{task}: {counts["correct"]}/{counts["total"]}={counts["correct"]/counts["total"]}')
        # if task in ['OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR']:
        #     rt_accs.append(counts['correct']/counts['total'])
        # elif task in ['EPM', 'ASI', 'HLD']:
        #     bt_accs.append(counts['correct']/counts['total'])
        # else:
        #     fr_accs.append(counts['correct']/counts['total'])
    
    print(f'\nAccuracy: {correct_answer}/{total_answer}={round((correct_answer/total_answer) * 100, 2)}')

    # if rt_accs:
    #     print(f'Real-Time Visual Perception avg.: {sum(rt_accs)}/{len(rt_accs)}={sum(rt_accs)/len(rt_accs)}')
    # if bt_accs:
    #     print(f'Backward Tracing avg.: {sum(bt_accs)}/{len(bt_accs)}={sum(bt_accs)/len(bt_accs)}')
    # if fr_accs:
    #     print(f'Forward Tracing avg.: {sum(fr_accs)}/{len(fr_accs)}={sum(fr_accs)/len(fr_accs)}')


def stamp_reset_state(sa):
    """Reset all STAMP temporal state for a new video sample."""
    sa.chunk_idx = 0
    sa.attention_momentum = None
    sa.attention_momentum_long = None   # Idea 4: hierarchical long-term momentum
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


def stamp_streaming_mcq_predict(
    model, processor, benchmark_path, options,
    question_prefix='', question_postfix='\nPlease select the correct answer.',
    answer_prefix='The answer is:\n', abcd_previous_str='\n',
    n_chunks=3,
    max_samples: int = None,
):
    """
    Multi-chunk streaming eval for STAMP.

    Each video is split into n_chunks sequential chunks:
      - Chunks 0 .. n_chunks-2: context only (no question).
        STAMP processes these to warm up attention_momentum and prev_visual_feats.
      - Chunk n_chunks-1: video + MCQ question → prediction.

    STAMP state persists ACROSS chunks within one sample (stamp_no_reset=True).
    STAMP state is fully reset BETWEEN samples.
    """
    strict_option_ids = [
        processor.tokenizer(f'{abcd_previous_str}{opt}').input_ids[-1] for opt in options
    ]

    dataset = DeViBenchMCQDataset(
        benchmark_path,
        question_prefix=question_prefix,
        question_postfix=question_postfix,
        answer_prefix=answer_prefix,
        sample=max_samples,
    )

    model.eval()
    device = next(model.parameters()).device
    sa = model._streaming_args
    all_predictions = []
    all_timings = []
    evaluated_datums = []   # only datums that were actually predicted (skipped ones excluded)
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataset.datums))):
            datum = dataset.datums[i]

            # ── Reset STAMP state for new sample ─────────────────────────────
            stamp_reset_state(sa)
            sa.stamp_no_reset = True   # block pre-hook from resetting between chunks
            sa.input_ids = None  # ensure clean input_ids for each new video

            t_start = time.perf_counter()

            # ── Load video ───────────────────────────────────────────────────
            try:
                video, _ = _read_may1fps_video_decord({
                    'video': os.path.join(dataset.data_dir, datum['video']),
                    'video_start': datum['video_start'],
                    'video_end': datum['video_end'],
                })
                video = _spatial_resize_video(video)   # [T, C, H, W]
            except Exception as e:
                print(f'Skipping sample {i}: {e}')
                continue
            T = video.shape[0]

            # ── Context chunks: warm up STAMP momentum ──────────────────────
            # Each chunk processes the FULL video so the model retains full
            # visual context.  Chunking exists only to build temporal state
            # (attention momentum + novelty).  N_vis is identical across
            # chunks → no shape-mismatch issues.
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
                _ = model(**ctx_inputs)   # STAMP momentum updated inside forward

            # ── Final chunk: full video + question + options ────────────────
            if datum['task'] in ['REC', 'SSR', 'CRR']:
                query = datum['question']
            else:
                query = question_prefix + datum['question'] + '\n' + '\n'.join(datum['options']) + question_postfix

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
            last_logits = final_outputs.logits[0, -1, :]   # [vocab_size]
            option_logits = last_logits[torch.tensor(strict_option_ids, device=device)]
            pred_idx = option_logits.argmax().item()
            all_predictions.append(pred_idx)
            evaluated_datums.append(datum)   # track only datums that were actually predicted

            t_end = time.perf_counter()
            all_timings.append(t_end - t_start)   # store seconds; line 508 converts to ms

            sa.stamp_no_reset = False   # restore for cleanliness

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_memory_mb = peak_bytes / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats()

    return all_predictions, evaluated_datums, 0, all_timings, peak_memory_mb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Format OVO-Bench dataset JSONL file.")

    parser.add_argument("--benchmark_dir", type=str, required=True, help="Path to ovobench dir.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model dir.")
    parser.add_argument("--fastv_k", type=int, default=None, help="FastV: layer index to prune at (capture attn at k-1, prune before k).")
    parser.add_argument("--fastv_r", type=float, default=0.5, help="FastV: fraction of visual tokens to keep (default 0.5).")
    # STAMP args
    parser.add_argument("--stamp_r1", type=float, default=None, help="STAMP Stage 1 keep ratio (None = STAMP disabled).")
    parser.add_argument("--stamp_alpha", type=float, default=0.5, help="STAMP: weight of momentum vs novelty (default 0.5).")
    parser.add_argument("--stamp_lambda", type=float, default=0.3, help="STAMP: EMA decay rate for momentum (default 0.3).")
    parser.add_argument("--stamp_K", type=int, default=10, help="STAMP: keyframe interval (default 10).")
    # STAMP Iteration 2 args
    parser.add_argument("--stamp_adaptive_r1", action="store_true", default=False,
                        help="Idea 1: adapt r1 based on scene novelty (dynamic→keep more, static→prune more).")
    parser.add_argument("--stamp_adaptive_r1_high", type=float, default=0.3,
                        help="Idea 1: novelty threshold for r1+0.25 (default 0.3).")
    parser.add_argument("--stamp_adaptive_r1_low", type=float, default=0.1,
                        help="Idea 1: novelty threshold for r1-0.25 (default 0.1).")
    parser.add_argument("--stamp_momentum_decay", action="store_true", default=False,
                        help="Idea 2: faster momentum decay for pruned tokens (rate=stamp_gamma).")
    parser.add_argument("--stamp_gamma", type=float, default=0.5,
                        help="Idea 2: decay rate for pruned tokens (default 0.5, vs 1-lambda=0.7).")
    parser.add_argument("--stamp_adaptive_kf", action="store_true", default=False,
                        help="Idea 3: detect scene cuts from avg novelty instead of fixed K.")
    parser.add_argument("--stamp_adaptive_kf_threshold", type=float, default=0.5,
                        help="Idea 3: avg novelty > this triggers a keyframe refresh (default 0.5).")
    parser.add_argument("--stamp_hierarchical", action="store_true", default=False,
                        help="Idea 4: dual-timescale momentum (short+long EMA) for richer scoring.")
    parser.add_argument("--stamp_lambda_long", type=float, default=0.1,
                        help="Idea 4: slow EMA decay rate (default 0.1 vs stamp_lambda=0.3).")
    parser.add_argument("--stamp_alpha_short", type=float, default=0.35,
                        help="Idea 4: weight for short-term momentum in score (default 0.35).")
    parser.add_argument("--stamp_alpha_long", type=float, default=0.35,
                        help="Idea 4: weight for long-term momentum in score (default 0.35).")
    parser.add_argument("--stamp_merge", action="store_true", default=False,
                        help="Iter3: merge pruned tokens into nearest kept neighbor instead of hard pruning.")
    parser.add_argument("--n_chunks", type=int, default=3,
                        help="Number of chunks for STAMP streaming eval (default 3).")

    # STAMP-Temporal args (mutually exclusive with stamp_r1)
    parser.add_argument("--stamp_temporal", action="store_true", default=False,
                        help="Enable STAMP-Temporal: ViT-sourced temporal pruning (replaces stamp_r1).")
    parser.add_argument("--stamp_temporal_r", type=float, default=None,
                        help="STAMP-Temporal keep ratio (None = disabled).")
    parser.add_argument("--stamp_temporal_alpha", type=float, default=0.5,
                        help="STAMP-Temporal: weight of ViT momentum vs novelty (default 0.5).")
    parser.add_argument("--stamp_temporal_lambda", type=float, default=0.3,
                        help="STAMP-Temporal: EMA decay rate for ViT momentum (default 0.3).")
    parser.add_argument("--stamp_temporal_K", type=int, default=10,
                        help="STAMP-Temporal: keyframe interval (default 10).")
    parser.add_argument("--stamp_temporal_vit_layer", type=int, default=31,
                        help="STAMP-Temporal: ViT layer to extract attention from (default 31 = last global attn).")
    parser.add_argument("--stamp_temporal_adaptive_r", action="store_true", default=True,
                        help="STAMP-Temporal: enable entropy-adaptive keep ratio.")
    parser.add_argument("--stamp_temporal_no_adaptive_r", action="store_true", default=False,
                        help="STAMP-Temporal: disable entropy-adaptive keep ratio.")
    parser.add_argument("--stamp_temporal_r_base", type=float, default=0.85,
                        help="STAMP-Temporal: base keep ratio for adaptive mode (default 0.85).")
    parser.add_argument("--stamp_temporal_compress", action="store_true", default=False,
                        help="STAMP-Temporal: compress pruned tokens into summary (default False).")
    parser.add_argument("--stamp_temporal_vit_layers", type=str, default=None,
                        help="STAMP-Temporal: comma-separated list of ViT layers for multi-layer fusion (e.g. '7,15,23,31').")
    parser.add_argument("--stamp_temporal_merge", action="store_true", default=False,
                        help="STAMP-Temporal: merge pruned tokens into nearest kept token (default False).")
    # ── STAMP-T+ (aggressive-regime variant) ──
    parser.add_argument("--stamp_temporal_plus", action="store_true", default=False,
                        help="STAMP-T+: rank-percentile score + text-relevance quality + MMR selection. Requires --stamp_temporal.")
    parser.add_argument("--stamp_temporal_plus_text_weight", type=float, default=0.3,
                        help="STAMP-T+: weight of text-relevance in quality term (default 0.3).")
    parser.add_argument("--stamp_temporal_plus_mmr_beta", type=float, default=0.5,
                        help="STAMP-T+: MMR similarity penalty coefficient (default 0.5).")
    parser.add_argument("--stamp_temporal_plus_frame_strata", type=int, default=0,
                        help="STAMP-T+: per-frame stratified top-k; 0=disabled, K>0 partitions N_vis into K contiguous chunks with equal budget (A4 temporal coverage guard).")
    # ── N6 / N7 cheap structural fixes (Phase 2A) ──
    parser.add_argument("--stamp_temporal_repack_pos", action="store_true", default=False,
                        help="N6: re-pack pruned-token position IDs to contiguous 0..K-1, fixing RoPE position-gap degradation at aggressive r.")
    parser.add_argument("--stamp_temporal_equal_layer_weights", action="store_true", default=False,
                        help="N7: weight ViT salience layers equally (1:1:1:1) instead of exponential 1:2:4:8, suppresses windowed-layer salience leakage.")
    # ── M3: FOCUS-style spatial neighbor enhancement (Phase 2B) ──
    parser.add_argument("--stamp_temporal_focus_enhance", action="store_true", default=False,
                        help="M3: blend pruned-neighbor info into each kept token via inverse-spatial-distance weighting (FOCUS enhancement on STAMP-T kept set).")
    parser.add_argument("--stamp_temporal_focus_enhance_alpha", type=float, default=0.1,
                        help="M3: blend strength for neighbor enhancement (default 0.1).")
    parser.add_argument("--stamp_temporal_mssavt", action="store_true", default=False,
                        help="M5: frame-local cosine-diversity penalty (same MRoPE T-channel only).")
    parser.add_argument("--stamp_temporal_mssavt_alpha", type=float, default=0.15,
                        help="M5: strength of same-frame redundancy penalty (default 0.15).")
    # ── Phase 3: Stochastic top-k for self-consistency ──
    parser.add_argument("--stamp_temporal_sc_scale", type=float, default=0.0,
                        help="Phase 3: Gumbel noise scale on score_t before topk; >0 enables stochastic pruning for SC.")
    parser.add_argument("--stamp_temporal_sc_seed", type=int, default=0,
                        help="Phase 3: SC pass seed; pair distinct seeds with sc_scale>0 to get diverse prunings.")
    # TAST: Temporal Accumulative State Tokens
    parser.add_argument("--tast", action="store_true", default=False,
                        help="Enable TAST: accumulate pruned tokens into persistent state tokens.")
    parser.add_argument("--tast_n_tokens", type=int, default=32,
                        help="TAST: number of state tokens (default 32).")
    parser.add_argument("--tast_gamma", type=float, default=0.1,
                        help="TAST: EMA decay for state token updates (default 0.1).")
    parser.add_argument("--tast_only", action="store_true", default=False,
                        help="TAST-only mode: no pruning, inject state tokens into unpruned stream.")
    # DSTM (Architecture 5) arguments
    parser.add_argument("--dstm", action="store_true", default=False,
                        help="Enable DSTM: delta-state temporal memory with surprise gating.")
    parser.add_argument("--dstm_only", action="store_true", default=False,
                        help="DSTM-only mode: no pruning, inject scene+delta memory into unpruned stream.")
    parser.add_argument("--dstm_scene_tokens", type=int, default=16,
                        help="DSTM: number of scene memory slots (default 16).")
    parser.add_argument("--dstm_delta_tokens", type=int, default=16,
                        help="DSTM: number of delta memory slots (default 16).")
    parser.add_argument("--dstm_gamma_scene", type=float, default=0.05,
                        help="DSTM: scene state EMA decay (default 0.05, slow).")
    parser.add_argument("--dstm_gamma_delta", type=float, default=0.2,
                        help="DSTM: delta state EMA decay (default 0.2, fast).")
    parser.add_argument("--dstm_surprise_beta", type=float, default=5.0,
                        help="DSTM: surprise gate sensitivity (default 5.0).")
    parser.add_argument("--dstm_surprise_tau", type=float, default=0.3,
                        help="DSTM: surprise gate threshold (default 0.3).")
    parser.add_argument("--dstm_blend_alpha", type=float, default=0.2,
                        help="DSTM: blend ratio for memory injection (0=none, 1=replace, default 0.2).")
    parser.add_argument("--tast_blend_alpha", type=float, default=0.2,
                        help="TAST: blend ratio for state token injection (0=none, 1=replace, default 0.2).")
    # ── Phase-7A: Hierarchical TAST (two EMA timescales) ──
    parser.add_argument("--tast_hierarchical", action="store_true", default=False,
                        help="Phase-7A: maintain a second slow-EMA TAST pool and interleave at inject time.")
    parser.add_argument("--tast_gamma_long", type=float, default=0.01,
                        help="Phase-7A: EMA decay of long-scale TAST pool (default 0.01, 10× slower).")
    parser.add_argument("--tast_segment_len", type=int, default=8,
                        help="Phase-7A: how many chunks between long-pool EMA updates (default 8).")
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
    parser.add_argument("--crisp_r", type=float, default=0.85, help="CRISP: keep ratio (default 0.85).")
    parser.add_argument("--crisp_grid_size", type=int, default=4, help="CRISP: grid cells per side (default 4).")
    parser.add_argument("--focus", action="store_true", help="Enable FOCUS spatial pruning.")
    parser.add_argument("--focus_r", type=float, default=0.85, help="FOCUS: keep ratio (default 0.85).")
    parser.add_argument("--focus_enhance_alpha", type=float, default=0.1, help="FOCUS: enhancement strength.")
    parser.add_argument("--focus_text_weight", type=float, default=0.6, help="FOCUS: text vs ViT weight.")
    parser.add_argument("--prism", action="store_true", help="Enable PRISM spatial pruning.")
    parser.add_argument("--prism_r", type=float, default=0.85, help="PRISM: keep ratio (default 0.85).")
    parser.add_argument("--prism_fine_ratio", type=float, default=0.5, help="PRISM: fine token budget ratio.")
    parser.add_argument("--prism_pool_size", type=int, default=2, help="PRISM: pooling factor.")
    parser.add_argument("--prism_enhance_alpha", type=float, default=0.1, help="PRISM: cross-resolution enhancement.")
    parser.add_argument("--star", action="store_true", help="Enable STAR scoring refinement.")
    parser.add_argument("--star_tc_weight", type=float, default=0.3, help="STAR: temporal consistency sharpening weight.")
    parser.add_argument("--star_gamma", type=float, default=0.2, help="STAR: per-layer divergence boost weight.")

    # ── Video-CDPruner: joint spatio-temporal DPP conditional diversity ──
    parser.add_argument("--video_cdpruner", action="store_true",
                        help="Enable Video-CDPruner (joint spatio-temporal DPP pruning).")
    parser.add_argument("--video_cdpruner_r", type=float, default=0.20,
                        help="Video-CDPruner: fraction of visual tokens to keep (default 0.20 → drop 80%).")
    parser.add_argument("--video_cdpruner_text_weight", type=float, default=0.6,
                        help="Video-CDPruner: text vs ViT weight for quality term.")
    parser.add_argument("--video_cdpruner_theta", type=float, default=0.0,
                        help="Video-CDPruner: quality-warp strength (0 disables warp).")
    parser.add_argument("--video_cdpruner_ablation", type=str, default="none",
                        choices=["none", "quality_only", "diversity_only", "per_frame"],
                        help="Video-CDPruner ablation mode (B5): none=full DPP, quality_only=drop diversity, "
                             "diversity_only=drop quality, per_frame=chunk-by-chunk DPP.")
    parser.add_argument("--video_cdpruner_n_frames", type=int, default=3,
                        help="Video-CDPruner: number of frames per chunk (only for --video_cdpruner_ablation=per_frame).")

    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional: evaluate only N randomly-sampled benchmark rows (seed=42). "
                             "Used for fast diagnostic runs.")

    args = parser.parse_args()

    # Handle --stamp_temporal_no_adaptive_r flag
    if args.stamp_temporal_no_adaptive_r:
        args.stamp_temporal_adaptive_r = False

    # Parse multi-layer fusion string
    if args.stamp_temporal_vit_layers:
        args.stamp_temporal_vit_layers = [int(x.strip()) for x in args.stamp_temporal_vit_layers.split(',')]
    else:
        args.stamp_temporal_vit_layers = None

    # Spatial modules need STAMP-Temporal pipeline for ViT salience extraction
    if (args.crisp or args.focus or args.prism or args.video_cdpruner) and not args.stamp_temporal:
        print("Spatial pruning enabled: auto-enabling --stamp_temporal --stamp_temporal_r 1.0 (for ViT salience)")
        args.stamp_temporal = True
        if args.stamp_temporal_r is None:
            args.stamp_temporal_r = 1.0  # No temporal pruning, just extract ViT salience
        if not args.stamp_temporal_vit_layers:
            args.stamp_temporal_vit_layers = [7, 15, 23, 31]

    # STAMP requires FastV for Stage 2 — auto-enable with default if not set
    if args.stamp_r1 is not None and args.fastv_k is None:
        print("STAMP enabled: auto-setting --fastv_k 2 --fastv_r 0.5 for Stage 2")
        args.fastv_k = 2
        args.fastv_r = 0.5
    # STAMP-Temporal also needs FastV pipeline (for capturing attention at Layer 1, even if r=1.0)
    if args.stamp_temporal and args.fastv_k is None:
        print("STAMP-Temporal enabled: auto-setting --fastv_k 2 --fastv_r 1.0 (no spatial pruning)")
        args.fastv_k = 2
        args.fastv_r = 1.0
    benchmark_path = os.path.join(args.benchmark_dir, 'devibench_formatted.jsonl')

    mp.set_start_method('spawn', force=True)
    
    model_path = args.model_path
    # AWQ/GPTQ quantized models need device_map="auto"
    is_quantized = any(q in model_path.upper() for q in ['AWQ', 'GPTQ', 'INT4', 'INT8'])
    load_kwargs = dict(torch_dtype="auto", attn_implementation='flash_attention_2')
    if is_quantized:
        load_kwargs['device_map'] = 'auto'
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    except:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, **load_kwargs)

    processor = AutoProcessor.from_pretrained(model_path, padding_side='left')

    # FastV: patch model with visual token pruning if requested
    if args.fastv_k is not None:
        from streaming_vlm.inference.qwen2_5.patch_model import convert_qwen2_5_to_streaming
        from streaming_vlm.inference.streaming_args import StreamingArgs
        from types import MethodType

        model = convert_qwen2_5_to_streaming(model)

        streaming_args = StreamingArgs(pos_mode="shrink", fastv_k=args.fastv_k, fastv_r=args.fastv_r,
                                       stamp_r1=args.stamp_r1, stamp_alpha=args.stamp_alpha,
                                       stamp_lambda=args.stamp_lambda, stamp_K=args.stamp_K)
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
        streaming_args.stamp_temporal_adaptive_r = args.stamp_temporal_adaptive_r
        streaming_args.stamp_temporal_r_base = args.stamp_temporal_r_base
        streaming_args.stamp_temporal_compress = args.stamp_temporal_compress
        streaming_args.stamp_temporal_vit_layers = args.stamp_temporal_vit_layers
        streaming_args.stamp_temporal_merge = args.stamp_temporal_merge
        # STAMP-T+
        streaming_args.stamp_temporal_plus = args.stamp_temporal_plus
        streaming_args.stamp_temporal_plus_text_weight = args.stamp_temporal_plus_text_weight
        streaming_args.stamp_temporal_plus_mmr_beta = args.stamp_temporal_plus_mmr_beta
        streaming_args.stamp_temporal_plus_frame_strata = args.stamp_temporal_plus_frame_strata
        # N6 / N7
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
        streaming_args.star_enabled = args.star
        streaming_args.star_tc_weight = args.star_tc_weight
        streaming_args.star_gamma = args.star_gamma
        # Video-CDPruner
        streaming_args.video_cdpruner_enabled = args.video_cdpruner
        streaming_args.video_cdpruner_r = args.video_cdpruner_r
        streaming_args.video_cdpruner_text_weight = args.video_cdpruner_text_weight
        streaming_args.video_cdpruner_theta = args.video_cdpruner_theta
        streaming_args.video_cdpruner_ablation = args.video_cdpruner_ablation
        streaming_args.video_cdpruner_n_frames = args.video_cdpruner_n_frames
        # Store on model and inner model so patched forwards can find it via self._streaming_args
        model._streaming_args = streaming_args
        model.model._streaming_args = streaming_args

        # Pre-hook: populate streaming_args fields from each batch before forward
        def _setup_streaming_args_hook(module, args_in, kwargs):
            sa = module._streaming_args
            # Reset STAMP state between samples (skipped in multi-chunk streaming eval)
            if (sa.stamp_r1 is not None or getattr(sa, 'stamp_temporal', False)) and not sa.stamp_no_reset:
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

    options = ['No', 'Yes', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E']

    if args.stamp_r1 is not None or args.stamp_temporal:
        if not is_quantized:
            model = model.cuda()   # stamp_streaming_mcq_predict bypasses Trainer, must move manually
        # Multi-chunk streaming eval: each video processed as n_chunks sequential chunks
        # STAMP/STAMP-Temporal momentum warms up on context chunks, prunes intelligently on the final chunk
        letter_idxs_predictions, benchmark_datums, process_index, inference_times, peak_memory_usage = stamp_streaming_mcq_predict(
            model=model, processor=processor, benchmark_path=benchmark_path,
            options=options, n_chunks=args.n_chunks,
            answer_prefix='The answer is:\n',
            abcd_previous_str='\n',
            max_samples=args.max_samples,
        )
    else:
        letter_idxs_predictions, benchmark_datums, process_index, inference_times, peak_memory_usage = mcq_predict(
            model=model, processor=processor, benchmark_path=benchmark_path,
            options=options, use_liger_kernel='LiveCC' in model_path,
            answer_prefix='The answer is:\n',
            abcd_previous_str='\n',
            max_samples=args.max_samples,
        )

    if process_index == 0:
        results = []
        results.append({"Peak memory usage": peak_memory_usage})
        for datum, letter_idx_prediction, inference_time in zip(benchmark_datums, letter_idxs_predictions, inference_times):
            results.append({
                "task": datum['task'],
                "question": datum['question'],
                "answer": datum['standard_answer'],
                "response": options[letter_idx_prediction],
                "inference_time": round(inference_time*1000, 2)
            })

        fps_str = os.path.basename(os.path.dirname(args.benchmark_dir))  # e.g. "10_FPS"
        fps_int = int(fps_str.split('_')[0])  # e.g. 10
        if args.stamp_temporal:
            extras = []
            if args.stamp_temporal_adaptive_r: extras.append('adaptive')
            else: extras.append('noadapt')
            if args.stamp_temporal_plus: extras.append(f'plus_tw{args.stamp_temporal_plus_text_weight}_b{args.stamp_temporal_plus_mmr_beta}')
            if args.stamp_temporal_plus and args.stamp_temporal_plus_frame_strata > 0: extras.append(f'fs{args.stamp_temporal_plus_frame_strata}')
            if args.stamp_temporal_repack_pos: extras.append('n6repack')
            if args.stamp_temporal_equal_layer_weights: extras.append('n7eqw')
            if args.stamp_temporal_focus_enhance: extras.append(f'm3fe{args.stamp_temporal_focus_enhance_alpha}')
            if args.stamp_temporal_mssavt: extras.append(f'm5mssavt{args.stamp_temporal_mssavt_alpha}')
            if args.stamp_temporal_sc_scale > 0.0 or args.stamp_temporal_sc_seed > 0:
                extras.append(f'sc_s{args.stamp_temporal_sc_scale}_seed{args.stamp_temporal_sc_seed}')
            if args.stamp_temporal_compress:    extras.append('compress')
            if args.stamp_temporal_merge:       extras.append('merge')
            if args.stamp_temporal_vit_layers:  extras.append('multi' + ''.join(str(l) for l in args.stamp_temporal_vit_layers))
            if args.tast:                       extras.append(f'tast{args.tast_n_tokens}g{args.tast_gamma}')
            if args.tast_hierarchical:          extras.append(f'tasthier_gl{args.tast_gamma_long}_seg{args.tast_segment_len}')
            if args.tast_adaptive_gamma:        extras.append(f'tastadagam_tau{args.tast_gamma_tau}')
            if args.stamp_ratio_ramp:           extras.append(f'ramp{args.stamp_ratio_ramp_early}to{args.stamp_ratio_ramp_late}c{args.stamp_ratio_ramp_chunks}')
            if args.dstm:                       extras.append(f'dstm_s{args.dstm_scene_tokens}d{args.dstm_delta_tokens}_ba{args.dstm_blend_alpha}')
            if args.dstm_only:                  extras.append('dstm_only')
            if args.tast and args.tast_blend_alpha != 1.0: extras.append(f'tba{args.tast_blend_alpha}')
            if args.crisp:                       extras.append(f"crisp_r{args.crisp_r}_g{args.crisp_grid_size}")
            if args.focus:                       extras.append(f"focus_r{args.focus_r}_tw{args.focus_text_weight}")
            if args.prism:                       extras.append(f"prism_r{args.prism_r}_p{args.prism_pool_size}")
            if args.star:                        extras.append(f"star_tc{args.star_tc_weight}_g{args.star_gamma}")
            if args.video_cdpruner:
                vcdp_tag = f"vcdp_r{args.video_cdpruner_r}_tw{args.video_cdpruner_text_weight}_th{args.video_cdpruner_theta}"
                if args.video_cdpruner_ablation != "none":
                    vcdp_tag += f"_abl{args.video_cdpruner_ablation}"
                extras.append(vcdp_tag)
            # Distinguish diagnostic (sub-sampled) runs from full runs so
            # they don't overwrite each other's result JSONs.
            if args.max_samples is not None:
                extras.append(f"maxN{args.max_samples}")
            extras_str = ('_' + '_'.join(extras)) if extras else ''
            fastv_suffix = f'_stamptemporal_r{args.stamp_temporal_r}_a{args.stamp_temporal_alpha}_l{args.stamp_temporal_lambda}_K{args.stamp_temporal_K}_vit{args.stamp_temporal_vit_layer}_c{args.n_chunks}{extras_str}'
        elif args.stamp_r1 is not None:
            ideas = []
            if args.stamp_merge:          ideas.append('merge')
            if args.stamp_adaptive_r1:    ideas.append('i1_ar1')
            if args.stamp_momentum_decay: ideas.append('i2_md')
            if args.stamp_adaptive_kf:    ideas.append('i3_akf')
            if args.stamp_hierarchical:   ideas.append('i4_hier')
            ideas_str = ('_' + '_'.join(ideas)) if ideas else ''
            fastv_suffix = f'_stamp_r1{args.stamp_r1}_a{args.stamp_alpha}_l{args.stamp_lambda}_K{args.stamp_K}_fastv_k{args.fastv_k}_r{args.fastv_r}_c{args.n_chunks}{ideas_str}'
        elif args.fastv_k is not None:
            fastv_suffix = f'_fastv_k{args.fastv_k}_r{args.fastv_r}'
        else:
            fastv_suffix = ''
        save_json_path = f'results/devibench/{fps_int}fps/vision_sync/{os.path.basename(model_path)}_{fps_int}fps_360{fastv_suffix}.json'
        os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
        json.dump(results, open(save_json_path, 'w'))

        save_txt_path = save_json_path.replace('.json', '.txt')
        save_function_print(
            evaluate_ovobench_results,
            save_txt_path,
            results[1:]
        )
