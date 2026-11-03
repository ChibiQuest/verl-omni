# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Native vLLM rollout support for verl-omni model configurations."""

import ray
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

from verl_omni.workers.config import OmniModelConfig


class OmniAwareVLLMHttpServer(vLLMHttpServer):
    """Use OmniModelConfig when native vLLM serves an omni model."""

    def _init_model_config(self, model_config):
        if model_config.get("model_type", "language_model") == "omni_model":
            config = omega_conf_to_dataclass(model_config, dataclass_type=OmniModelConfig)
            self._eager_register_npu_model(config.architecture)
            return config
        return super()._init_model_config(model_config)

    @staticmethod
    def _eager_register_npu_model(architecture: str) -> None:
        """Avoid vLLM's subprocess inspection, which is unsafe with the Ascend plugin."""
        from vllm.model_executor.models import ModelRegistry
        from vllm.platforms import current_platform

        if current_platform.device_type != "npu":
            return
        registered_model = ModelRegistry.models.get(architecture)
        if registered_model is None:
            return
        ModelRegistry.register_model(architecture, registered_model.load_model_cls())


class OmniAwareVLLMReplica(vLLMReplica):
    """Native vLLM replica whose server understands OmniModelConfig."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(OmniAwareVLLMHttpServer)