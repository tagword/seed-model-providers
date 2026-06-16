"""
seed-model-providers — Unified model provider catalog and protocol resolution.

Originally extracted from ``seed.core.model_providers``.

Use::

    from seed_model_providers import (
        resolve_provider_for_preset,
        resolve_chat_protocol,
        list_provider_catalog,
        call_image_generations,
        ...
    )
"""

from seed_model_providers.model_providers import (
    # Catalog / metadata
    USE_TYPE_LABELS,
    PROVIDER_CATALOG,
    list_provider_catalog,
    provider_requires_api_key,
    list_models_for_provider,
    get_provider_spec,
    # Provider / preset resolution
    normalize_provider_id,
    infer_provider_from_url,
    resolve_provider_for_preset,
    resolve_chat_protocol,
    resolve_image_protocol,
    resolve_music_protocol,
    resolve_video_protocol,
    uses_full_reasoning_content_echo,
    uses_deepseek_chat_protocol,
    # Preset mapping / UI helpers
    materialize_preset_from_form,
    enrich_presets_for_ui,
    enrich_preset_defaults,
    preset_display_name,
    preset_display_fields,
    preset_auto_name,
    preset_auto_id,
    infer_preset_use_type,
    infer_use_type_for_provider_model,
    model_label_for_provider,
    normalize_deepseek_chat_model,
    normalize_volcengine_image_model,
    # Chat protocol helpers
    apply_chat_thinking_extra_body,
    apply_chat_stream_options,
    apply_provider_chat_headers,
    should_send_reasoning_content,
    normalize_reasoning_effort,
    normalize_chat_usage,
    default_max_request_body_bytes,
    # Image generation
    call_image_generations,
    normalize_image_size,
    # Music generation
    call_music_generations,
    call_minimax_music_generation,
    # Video generation
    call_video_generations,
    call_agnes_video_generation,
    call_minimax_video_generation,
    normalize_video_num_frames,
)

__all__ = [
    "USE_TYPE_LABELS",
    "PROVIDER_CATALOG",
    "list_provider_catalog",
    "provider_requires_api_key",
    "list_models_for_provider",
    "get_provider_spec",
    "normalize_provider_id",
    "infer_provider_from_url",
    "resolve_provider_for_preset",
    "resolve_chat_protocol",
    "resolve_image_protocol",
    "resolve_music_protocol",
    "resolve_video_protocol",
    "uses_deepseek_chat_protocol",
    "uses_full_reasoning_content_echo",
    "materialize_preset_from_form",
    "enrich_presets_for_ui",
    "enrich_preset_defaults",
    "preset_display_name",
    "preset_display_fields",
    "preset_auto_name",
    "preset_auto_id",
    "infer_preset_use_type",
    "infer_use_type_for_provider_model",
    "model_label_for_provider",
    "normalize_deepseek_chat_model",
    "normalize_volcengine_image_model",
    "apply_chat_thinking_extra_body",
    "apply_chat_stream_options",
    "apply_provider_chat_headers",
    "should_send_reasoning_content",
    "normalize_reasoning_effort",
    "normalize_chat_usage",
    "default_max_request_body_bytes",
    "call_image_generations",
    "normalize_image_size",
    "call_music_generations",
    "call_minimax_music_generation",
    "call_video_generations",
    "call_agnes_video_generation",
    "call_minimax_video_generation",
    "normalize_video_num_frames",
]

__version__ = "1.0.0"
