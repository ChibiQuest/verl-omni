"""FSDPTurbo engine for multimodal autoregressive models."""

from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.fsdp_turbo_impl import FSDPTurboEngineWithLMHead

from verl_omni.workers.engine.fsdp.omni_impl import OmniFSDPEngine


@EngineRegistry.register(model_type="omni_model", backend="fsdp_turbo", device=["cuda", "npu"])
class OmniFSDPTurboEngine(FSDPTurboEngineWithLMHead, OmniFSDPEngine):
    """Compose verl's FSDPTurbo parallelism with verl-omni model adapters.

    FSDPTurbo owns meshes, parallelization, clipping, and optimizer steps.
    Its build delegates to OmniFSDPEngine for model loading, replay inputs,
    and vLLM weight export.
    """
