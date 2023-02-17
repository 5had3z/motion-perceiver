from dataclasses import dataclass
from typing import Any, Dict

from konductor.modules.models import MODEL_REGISTRY, ModelConfig, ExperimentInitConfig

from .motion_perceiver import MotionPerceiver


@dataclass
@MODEL_REGISTRY.register_module("motion-perceiver")
class MotionPerceiverConfig(ModelConfig):
    encoder: Dict[str, Any]
    decoder: Dict[str, Any]

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, *args, **kwargs) -> Any:
        return cls(**config.model.args)

    def get_instance(self, *args, **kwargs) -> Any:
        return MotionPerceiver(self.encoder, self.decoder)
