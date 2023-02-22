from dataclasses import dataclass
from typing import Any, Dict

from konductor.modules.models import MODEL_REGISTRY, ExperimentInitConfig, DatasetConfig
from konductor.modules.models._pytorch import TorchModelConfig

from .motion_perceiver import MotionPerceiver


@dataclass
@MODEL_REGISTRY.register_module("motion-perceiver")
class MotionPerceiverConfig(TorchModelConfig):
    encoder: Dict[str, Any]
    decoder: Dict[str, Any]

    @classmethod
    def from_config(
        cls, config: ExperimentInitConfig, dataset_config: DatasetConfig
    ) -> Any:
        if "roadmap_size" in dataset_config.properties:
            sz = dataset_config.properties["roadmap_size"]
            sz = [1, sz, sz]
            config.model.args["encoder"]["roadgraph_ia"]["args"]["image_shape"] = sz

        sz = dataset_config.properties["occupancy_size"]
        config.model.args["decoder"]["adapter"]["args"]["image_shape"] = [sz, sz]

        return cls(**config.model.args)

    def get_instance(self, *args, **kwargs) -> Any:
        return self.apply_extra(MotionPerceiver(self.encoder, self.decoder))
