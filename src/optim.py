from konductor.optimizers._pytorch import PG_REGISTRY
from torch import nn


@PG_REGISTRY.register_module("scale-non-flow")
def _non_flow_multiplier(model: nn.Module, multiplier: float, lr: float, **ignore):
    """
    Add multiplication factor to all parameters that aren't associated
    with the historical flow portion of the model. Usually this is a value
    less than 1 to slow learning rate when fine-tuning for flow.
    """
    param_grps = [
        {"params": [], "lr": lr},
        {"params": [], "lr": multiplier * lr},
    ]
    flow_keys = ["decoder.cross_attention2", "decoder.output_adapter.linear2"]
    for name, param in model.named_parameters():
        if any(str_ in name for str_ in flow_keys):
            param_grps[1]["params"].append(param)
        else:
            param_grps[0]["params"].append(param)

    return param_grps
