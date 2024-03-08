# Modified from https://github.com/izmailovpavel/understandingbdl
# which is by Pavel Izmailov released under BSD 2-Clause License
import copy
import torch
from torch import nn, Tensor, LongTensor


@torch.no_grad()
def flatten_params_to_cpu(model: nn.Module) -> Tensor:
    return torch.cat([param.cpu().flatten() for param in model.parameters()])


class SWAG(torch.nn.Module):
    def __init__(self, basemodel: nn.Module, max_rank: int = 20):
        assert max_rank >= 0
        super(SWAG, self).__init__()

        self._basemodel = basemodel
        self.basedevice = next(basemodel.parameters()).device
        self.max_rank = max_rank
        self.num_parameters = sum(
            param.numel() for param in self.basemodel.parameters()
        )
        self.num_layers = self.count_layers()
        self.num_channels = self.count_channels()
        # 1st and 2nd moments
        self.mean: Tensor
        self.sq_mean: Tensor
        self.n_models: LongTensor
        self.deviations: Tensor
        self.register_buffer("mean", torch.zeros(self.num_parameters))
        self.register_buffer("sq_mean", torch.zeros(self.num_parameters))
        self.register_buffer("n_models", torch.zeros((), dtype=torch.long))
        # deviations for computing low rank approx of cov mat
        self.register_buffer(
            "deviations", torch.empty((0, self.num_parameters))
        )

    # count layers with params
    def count_layers(self) -> int:
        count = 0
        for m in self.basemodel.modules():
            if hasattr(m, "reset_parameters"):
                count += 1
        return count

    # count total channels
    def count_channels(self) -> int:
        count = 0
        for m in self.basemodel.modules():
            count += self.get_channel(m)
        return count

    @staticmethod
    def get_channel(m: nn.Module) -> int:
        num_channel = set(
            p.size(0) for p in m._parameters.values() if p is not None
        )
        if len(num_channel) == 0:
            return 0
        elif len(num_channel) == 1:
            return next(iter(num_channel))
        else:
            raise ValueError(f"incompatible channel count: {num_channel}")

    @property
    def rank(self) -> int:
        return self.deviations.size(0)

    @property
    def basemodel(self) -> nn.Module:
        return self._basemodel

    def reduce_rank(self, rank: int, step: int = 1) -> None:
        assert step > 0
        if rank > 0:
            self.deviations = self.deviations[-rank * step + (step - 1) : step]
        elif rank == 0:  # SWAG diagonal case
            self.deviations = torch.empty((0, self.num_parameters))
        else:
            raise ValueError(f"rank must be non-negative, received {rank}")

    def to(self, *args, **kwargs):
        self.basemodel.to(*args, **kwargs)
        (device, dtype, non_blocking) = torch._C._nn._parse_to(
            *args, **kwargs
        )[:3]
        self.basedevice = device
        # buffers stay on cpu
        for b in self.buffers(recurse=False):
            b.to(dtype=dtype, non_blocking=non_blocking)
        return self

    def forward(self, *args, **kwargs):
        return self.basemodel.forward(*args, **kwargs)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.basemodel.named_parameters(prefix, recurse)

    @torch.no_grad()
    def collect_model(self, base_model: nn.Module = None) -> None:
        if base_model is None:
            base_model = self.basemodel
        n_models = self.n_models.item()
        w = flatten_params_to_cpu(base_model)
        # update 1st and 2nd moments
        self.mean.mul_(n_models / (n_models + 1.0))
        self.mean.add_(w / (n_models + 1.0))
        self.sq_mean.mul_(n_models / (n_models + 1.0))
        self.sq_mean.add_(w**2 / (n_models + 1.0))
        self.n_models.add_(1)
        # update deviations
        dev = w - self.mean
        if self.rank + 1 > self.max_rank:
            self.deviations = self.deviations[1:, :]
        self.deviations = torch.cat((self.deviations, dev.unsqueeze(0)), dim=0)

    @property
    def variance(self):
        return torch.clamp(self.sq_mean - self.mean**2, torch.finfo().tiny)

    @torch.no_grad()
    def _model_with_weights(self, vector: Tensor, model=None) -> nn.Module:
        offset = 0
        vector = vector.to(self.basedevice)
        if model is None:
            model = copy.deepcopy(self.basemodel)
        for param in model.parameters():
            pnumel, psize = param.numel(), param.size()
            param.data = vector[offset : offset + pnumel].view(psize)
            offset += pnumel
        return model

    # apply swa weight to base model
    def averaged_model(self, model=None) -> nn.Module:
        # return base model with averaged weight
        return self._model_with_weights(self.mean, model)

    sample_mode = ("modelwise", "layerwise", "channelwise")

    # sample weight for base model
    def sampled_model(self, model=None, mode="modelwise") -> nn.Module:
        if mode == "modelwise":
            return self._sampled_model_modelwise(model)
        elif mode == "layerwise":
            return self._sampled_model_layerwise(model)
        elif mode == "channelwise":
            return self._sampled_model_channelwise(model)
        else:
            raise ValueError(
                f"mode should be in {self.sample_mode}, received {mode}"
            )

    def _sampled_model_modelwise(self, model=None) -> nn.Module:
        mean, var, rank, dev = (
            self.mean,
            self.variance,
            self.rank,
            self.deviations,
        )
        if rank > 1:
            sample = mean + torch.sqrt(var / 2.0) * torch.randn_like(var)
            z2 = torch.randn(rank, dtype=dev.dtype)
            sample = sample + z2 @ dev / ((2.0 * (rank - 1)) ** 0.5)
        else:
            sample = mean + torch.sqrt(var) * torch.randn_like(var)
        # return base model with sampled weight
        return self._model_with_weights(sample, model)

    @torch.no_grad()
    def _model_with_weights_layerwise(
        self, sample: Tensor, model=None
    ) -> nn.Module:
        offset = 0
        layerpos = 0
        rank, dev = self.rank, self.deviations
        z2 = torch.randn(self.num_layers, rank, dtype=dev.dtype)
        if model is None:
            model = copy.deepcopy(self.basemodel)
        for m in model.modules():
            if not hasattr(m, "reset_parameters"):
                continue
            for param in m._parameters.values():
                if param is None:
                    continue
                pnumel, psize = param.numel(), param.size()
                pdata = sample[offset : offset + pnumel] + z2[layerpos] @ dev[
                    :, offset : offset + pnumel
                ] / ((2.0 * (rank - 1)) ** 0.5)
                param.data = pdata.view(psize).to(self.basedevice)
                offset += pnumel
            layerpos += 1
        return model

    # sample weight for base model, layerwise decomposition
    def _sampled_model_layerwise(self, model=None) -> nn.Module:
        var = self.variance
        if self.rank > 1:
            sample = self.mean + torch.sqrt(var / 2.0) * torch.randn_like(var)
            return self._model_with_weights_layerwise(sample, model)
        else:
            sample = self.mean + torch.sqrt(var) * torch.randn_like(var)
            return self._model_with_weights(sample, model)

    @torch.no_grad()
    def _model_with_weights_channelwise(
        self, sample: Tensor, model=None
    ) -> nn.Module:
        offset = 0
        coffset = 0
        rank, dev = self.rank, self.deviations
        z2 = torch.randn(rank, self.num_channels, dtype=dev.dtype)
        if model is None:
            model = copy.deepcopy(self.basemodel)
        for m in model.modules():
            c = self.get_channel(m)
            if c == 0:
                continue
            mz = z2[:, coffset : coffset + c]
            for param in m._parameters.values():
                if param is None:
                    continue
                pnumel, psize = param.numel(), param.size()
                param.data = sample[offset : offset + pnumel].view(psize)
                pdev = dev[:, offset : offset + pnumel].view(-1, *psize)
                param.data += torch.einsum("ij...,ij->j...", pdev, mz) / (
                    (2.0 * (rank - 1)) ** 0.5
                )
                param.data = param.data.to(self.basedevice)
                offset += pnumel
            coffset += c
        return model

    # sample weight for base model, channelwise decomposition
    def _sampled_model_channelwise(self, model=None) -> nn.Module:
        var = self.variance
        if self.rank > 1:
            sample = self.mean + (torch.sqrt(var / 2.0)) * torch.randn_like(
                var
            )
            return self._model_with_weights_channelwise(sample, model)
        else:
            sample = self.mean + torch.sqrt(var) * torch.randn_like(var)
            return self._model_with_weights(sample, model)

    # adapt size of self.deviations before loading
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        devsize = state_dict[prefix + "deviations"].size()
        self.deviations = self.deviations.new_empty(devsize)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
