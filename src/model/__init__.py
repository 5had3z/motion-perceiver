from dataclasses import dataclass
from typing import Any, Dict

from konductor.modules.data import get_dataset_properties
from konductor.modules.models import MODEL_REGISTRY, ExperimentInitConfig
from konductor.modules.models._pytorch import TorchModelConfig

from .motion_perceiver import MotionPerceiver


@dataclass
@MODEL_REGISTRY.register_module("motion-perceiver")
class MotionPerceiverConfig(TorchModelConfig):
    encoder: Dict[str, Any]
    decoder: Dict[str, Any]

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0) -> Any:
        props = get_dataset_properties(config)
        model_cfg = config.model[idx].args

        if "roadmap_size" in props:
            sz = props["roadmap_size"]
            sz = [1, sz, sz]
            model_cfg["encoder"]["roadgraph_ia"]["args"]["image_shape"] = sz

        sz = props["occupancy_size"]
        model_cfg["decoder"]["adapter"]["args"]["image_shape"] = [sz, sz]

        return super().from_config(config)

    def get_instance(self, *args, **kwargs) -> Any:
        return self._apply_extra(MotionPerceiver(self.encoder, self.decoder))
