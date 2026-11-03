"""Training adapter for Qwen3.5 multimodal MoE models."""

from typing import Any

from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from verl_omni.pipelines.model_base import OmniModelBase


@OmniModelBase.register("Qwen3_5MoeForConditionalGeneration", stage="all")
@OmniModelBase.register("Qwen3_5MoeForConditionalGeneration", stage="thinker")
class Qwen35MoeAdapter(OmniModelBase):
    """Adapt Qwen3.5-35B-A3B to verl-omni's single-stage AR contract."""

    auto_model_class = AutoModelForImageTextToText

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        return []

    @classmethod
    def configure_model(cls, module, model_config):
        module = super().configure_model(module, model_config)
        module._no_split_modules = ["Qwen3_5MoeDecoderLayer", "Qwen3_5MoeVisionBlock"]
        return module

    @classmethod
    def configure_processor(cls, model_path: str, model_config) -> Any:
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config) -> Any:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=model_config.trust_remote_code)
