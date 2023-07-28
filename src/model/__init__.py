from dataclasses import dataclass
from typing import Any, Dict

from konductor.data import get_dataset_properties
from konductor.models import MODEL_REGISTRY, ExperimentInitConfig
from konductor.models._pytorch import TorchModelConfig

from .motion_perceiver import MotionPerceiver, MotionPercieverWSignals


@dataclass
@MODEL_REGISTRY.register_module("motion-perceiver")
class MotionPerceiverConfig(TorchModelConfig):
    encoder: Dict[str, Any]
    decoder: Dict[str, Any]
    signal_decoder: bool = False

    @classmethod
    def from_config(cls, config: ExperimentInitConfig, idx: int = 0) -> Any:
        props = get_dataset_properties(config)
        model_cfg = config.model[idx].args

        if rg_ia := model_cfg["encoder"].get("roadgraph_ia", False):
            sz = props["roadmap_size"]
            if rg_ia["type"] == "image":
                rg_ia["args"]["image_shape"] = [1, sz, sz]

        # Number of "extra features" is number of vehicle features minus xyt and class
        _skip_labels = {"x", "y", "bbox_yaw", "class"}
        model_cfg["encoder"]["adapter"]["args"]["n_extra_features"] = len(
            [x for x in props["vehicle_features"] if x not in _skip_labels]
        )

        sz = props["occupancy_size"]
        model_cfg["decoder"]["adapter"]["args"]["image_shape"] = [sz, sz]
        model_cfg["decoder"]["position_encoding_limit"] = props["occupancy_roi"]

        # Handle standardization of position encoding relative to map size
        max_freq = props["map_normalize"] * 2
        model_cfg["decoder"]["max_frequency"] = max_freq
        model_cfg["encoder"]["adapter"]["args"]["map_max_freq"] = max_freq
        if "signal_ia" in model_cfg["encoder"]:
            model_cfg["encoder"]["signal_ia"]["args"]["max_frequency"] = max_freq
        if "roadgraph_ia" in model_cfg["encoder"]:
            model_cfg["encoder"]["roadgraph_ia"]["args"]["max_frequency"] = max_freq

        return super().from_config(config)

    def get_instance(self, *args, **kwargs) -> Any:
        model_ = MotionPercieverWSignals if self.signal_decoder else MotionPerceiver
        return self._apply_extra(model_(self.encoder, self.decoder))
