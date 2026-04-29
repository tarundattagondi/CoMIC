import torch
import inspect
from transformers.generation.utils import Cache, logging

logger = logging.get_logger(__name__)

# modified from transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLForConditionalGeneration.prepare_inputs_for_generation
def prepare_multiturn_multimodal_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    **kwargs,
):

    model_inputs = {}
    if self._supports_cache_class:
        model_inputs["cache_position"] = cache_position
    elif cache_position is None:
        past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

    if past_key_values is not None:
        model_inputs["past_key_values"] = past_key_values
        if hasattr(self, '_cache_dependant_input_preparation'):
            inputs_embeds, input_ids = self._cache_dependant_input_preparation(
                input_ids, inputs_embeds, cache_position
            )
        else:
            # transformers 4.50+: trim manually
            if inputs_embeds is not None:
                inputs_embeds = inputs_embeds[:, cache_position]
            else:
                input_ids = input_ids[:, cache_position]

    input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
    if not self.config.is_encoder_decoder:
        if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
            model_inputs[input_ids_key] = None
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
            model_inputs["inputs_embeds"] = None
    else:
        model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

    encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
    attention_mask = (
        kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
    )
    attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
    position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
    if (
        attention_mask is not None
        and kwargs.get(position_ids_key) is None
        and position_ids_key in set(inspect.signature(self.forward).parameters.keys())
    ):
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        kwargs[position_ids_key] = position_ids

    for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
        model_input = kwargs.get(model_input_name)
        if model_input is not None:
            if past_key_values is not None:
                current_input_length = (
                    model_inputs["inputs_embeds"].shape[1]
                    if model_inputs.get("inputs_embeds") is not None
                    else model_inputs[input_ids_key].shape[1]
                )
                model_input = model_input[:, -current_input_length:]
                model_input = model_input.clone(memory_format=torch.contiguous_format)
            model_inputs[model_input_name] = model_input

    if (
        isinstance(past_key_values, Cache)
        and past_key_values.is_compileable
        and attention_mask is not None
        and attention_mask.ndim == 2
    ):
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
        else:
            batch_size, sequence_length = model_inputs[input_ids_key].shape[:2]

        base_model = getattr(self, self.base_model_prefix, self)
        decoder = base_model.get_decoder() if hasattr(base_model, "get_decoder") else None
        causal_mask_creation_function = getattr(
            base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
        )
        if causal_mask_creation_function is None and decoder is not None:
            causal_mask_creation_function = getattr(
                decoder, "_prepare_4d_causal_attention_mask_with_cache_position", None
            )
        if causal_mask_creation_function is None:
            logger.warning_once(
                f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                "writing code, see Llama for an example implementation. If you're a user, please report this "
                "issue on GitHub."
            )
        else:
            attention_mask = causal_mask_creation_function(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_cache_shape(),
                dtype=self.dtype,
                cache_position=cache_position,
                batch_size=batch_size,
                config=self.config,
                past_key_values=past_key_values,
            )
    if attention_mask is not None:
        model_inputs[attention_mask_key] = attention_mask

    if encoder_attention_mask is not None:
        model_inputs["attention_mask"] = encoder_attention_mask

    for key, value in kwargs.items():
        if key not in model_inputs:
            model_inputs[key] = value
    model_inputs["use_cache"] = use_cache
    model_inputs["pixel_values"] = pixel_values
    model_inputs["pixel_values_videos"] = pixel_values_videos
    model_inputs["image_grid_thw"] = image_grid_thw
    model_inputs["video_grid_thw"] = video_grid_thw

    model_inputs.pop("labels", None)

    model_inputs["position_ids"] = None
    if model_inputs["cache_position"][0] != 0 and (model_inputs['input_ids'] != self.config.video_token_id).all():
        model_inputs["pixel_values"] = None
        model_inputs["pixel_values_videos"] = None

    return model_inputs

def prepare_omni_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        input_features=None,
        feature_attention_mask=None,
        use_audio_in_video=False,
        video_second_per_grid=None,
        **kwargs,
    ):
        model_inputs = super(type(self), self).prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )

        model_inputs["position_ids"] = None
        if cache_position[0] != 0 and (model_inputs['input_ids'] != self.config.video_token_index).all():
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs