from typing import Any, Iterable, Mapping
import numpy as np
import torch
from torch import nn
from .models32 import get_model
from .grudense import GRUDense
from .frn import FilterResponseNorm
from .resnet224 import resnet50
from ..swag import SWAG
from ..mcdropout import MCDropout

# standard save model function
def savemodel(
    to,
    modelname: str,
    modelargs: Iterable[Any],
    modelkwargs: Mapping[str, Any],
    model: nn.Module,
    **kwargs
) -> None:
    dic = {
        "modelname": modelname,
        "modelargs": tuple(modelargs),
        "modelkwargs": {k: modelkwargs[k] for k in modelkwargs},
        "modelstates": model.state_dict(),
        **kwargs,
    }
    torch.save(dic, to)


# standard load model function
def loadmodel(fromfile, device=torch.device("cpu")):
    dic = torch.load(fromfile, map_location=device)
    model = globals()[dic["modelname"]](
        *dic["modelargs"], **dic.get("modelkwargs", {})
    ).to(device)
    model.load_state_dict(dic.pop("modelstates"))
    return model, dic


def resnet20(outclass: int, input_size: int = 32) -> torch.nn.Module:
    return get_model(
        "resnet20_frn",
        data_info={"num_classes": outclass, "input_size": input_size},
        activation=torch.nn.Identity,
    )


def resnet20_mcdrop(
    outclass: int, input_size: int = 32, p: float = 0.05
) -> torch.nn.Module:
    return get_model(
        "resnet20_frn",
        data_info={"num_classes": outclass, "input_size": input_size},
        activation=lambda: MCDropout(p),
    )


def softplus_inv(x: float) -> float:
    return x + np.log(-np.expm1(-x))


def resnet20_bbb(
    outclass: int,
    input_size: int = 32,
    prior_precision: float = 1.0,
    std_init: float = 0.05,
    bnn_type: str = "Reparameterization",
) -> torch.nn.Module:
    from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
    bnn_options = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0 / np.sqrt(prior_precision),
        "posterior_mu_init": 0.0,
        "posterior_rho_init": softplus_inv(std_init),
        "type": bnn_type,
        "moped_enable": False,
    }
    model = resnet20(outclass, input_size)
    dnn_to_bnn(model, bnn_options)
    return model


def resnet20_swag(
    outclass: int, input_size: int = 32, max_rank: int = 20
) -> SWAG:
    return SWAG(resnet20(outclass, input_size), max_rank)


def preresnet110(outclass: int, input_size: int = 32) -> torch.nn.Module:
    return get_model(
        "preresnet110_frn",
        data_info={"num_classes": outclass, "input_size": input_size},
        activation=torch.nn.Identity,
    )


def resnet18wide(outclass: int, input_size: int = 32) -> torch.nn.Module:
    return get_model(
        "resnet18",
        data_info={"num_classes": outclass, "input_size": input_size},
    )


def densenet121(outclass: int, input_size: int = 32) -> torch.nn.Module:
    return get_model(
        "densenet121",
        data_info={"num_classes": outclass, "input_size": input_size},
    )


def gru_dense(vocab_size: int, num_classes: int, padding_idx: int) -> GRUDense:
    return GRUDense(vocab_size, num_classes, padding_idx)


def resnet50_imagenet(outclass: int, input_size: int = 224) -> torch.nn.Module:
    return resnet50(activation=nn.Identity, norm_layer=FilterResponseNorm, num_classes = outclass)

# available models
STANDARDMODELS = {
    "resnet20": resnet20,
    "resnet18wide": resnet18wide,
    "preresnet110": preresnet110,
    "densenet121": densenet121,
    "resnet50_imagenet" : resnet50_imagenet,
}
MCDROPMODELS = {
    "resnet20_mcdrop": resnet20_mcdrop,
}
BBBMODELS = {
    "resnet20_bbb": resnet20_bbb,
}
SWAGMODELS = {
    "resnet20_swag": resnet20_swag,
}

MODELS = {**STANDARDMODELS, **MCDROPMODELS, **BBBMODELS, **SWAGMODELS }
