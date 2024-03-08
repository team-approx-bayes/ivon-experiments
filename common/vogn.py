from typing import Callable, Tuple, Dict, Optional
from contextlib import contextmanager
from math import sqrt
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

LossFnType = Callable[[Tensor, Tensor], Tensor]
ClosureType = Callable[[], Tuple[Tensor, Tensor]]


def grad_output_aux(fn):
    from torch.func import grad_and_value
    def _new_fn(*args, **kwargs):
        grad, (output, aux) = grad_and_value(fn, has_aux=True)(*args, **kwargs)
        return grad, output, aux

    return _new_fn


def dict_of_mean(d: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {k: v.mean(dim=0) for k, v in d.items()}


class VOGNClosure:
    def __init__(self, model: nn.Module, lossfn: LossFnType):
        self.model = model
        self.lossfn = lossfn
        self.inputs = None
        self.targets = None

    def update_minibatch(self, inputs: Tensor, targets: Tensor):
        self.inputs = inputs
        self.targets = targets

    def __call__(self):
        from torch.func import vmap,grad_and_value, functional_call
        assert self.inputs is not None
        assert self.targets is not None

        model = self.model

        for p in model.parameters():
            p.grad = None
            p.grad_sq = None

        def per_sample_loss_and_logit(
            params, buffers, input_sample, target_sample
        ) -> Tuple[Tensor, Tensor]:
            prediction = functional_call(
                model, (params, buffers), (input_sample.unsqueeze(0),)
            )
            loss = self.lossfn(prediction, target_sample.unsqueeze(0))
            return loss, prediction.squeeze(0).detach()

        params = {k: v.detach() for k, v in model.named_parameters()}
        buffers = {k: v.detach() for k, v in model.named_buffers()}
        grads, (losses, predictions) = vmap(
            grad_and_value(per_sample_loss_and_logit, has_aux=True),
            in_dims=(None, None, 0, 0),
        )(params, buffers, self.inputs, self.targets)
        for k, p in model.named_parameters():
            per_sample_grad = grads.pop(k)
            p.grad = per_sample_grad.mean(dim=0)
            p.grad_sq = per_sample_grad.square().mean(dim=0)
        return losses.mean(), predictions


class VOGN(Optimizer):
    def __init__(
        self,
        params,
        lr,
        data_size: int,
        mc_samples: int = 1,
        momentum_grad: float = 0.9,
        momentum_hess: Optional[float] = None,
        prior_precision: float = 1.0,
        dampening: float = 0.0,
        temperature: float = 1.0,
        std_init: Optional[float] = None,
    ):
        assert lr > 0.0
        assert data_size >= 1
        assert mc_samples >= 1
        assert prior_precision > 0.0
        assert dampening >= 0.0
        assert temperature >= 0
        if momentum_hess is None:
            momentum_hess = 1.0 - lr  # default follows theoretical derivation
        self.mc_samples = mc_samples
        defaults = dict(
            lr=lr,
            data_size=data_size,
            momentum_grad=momentum_grad,
            momentum_hess=momentum_hess,
            prior_precision=prior_precision,
            dampening=dampening,
            temperature=temperature,
        )
        super().__init__(params, defaults)
        self._init_buffers(std_init)
        self._reset_param_and_grad_samples()

    def _init_buffers(self, std_init):
        for group in self.param_groups:
            if std_init is None:
                std_init = 1.0
            else:
                pass
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["momentum_grad"] = torch.zeros_like(p)
                    self.state[p]["log_std"] = torch.ones_like(p) * torch.log(
                        torch.as_tensor(std_init)
                    )

    def _reset_param_and_grad_samples(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["grad_samples"] = []
                    self.state[p]["grad_sq_samples"] = []

    @torch.no_grad()
    def step(self, closure: VOGNClosure = None):
        if closure is None:
            raise ValueError("VON optimizer requires VOGNClosure instance.")

        self._stash_param_averages()

        losses = []
        outputs = []

        for _ in range(self.mc_samples):
            self._sample_weight()
            with torch.enable_grad():
                loss, output = closure()
            losses.append(loss.detach())
            outputs.append(output.detach())
            self._collect_grad_samples()

        self._update()

        self._restore_param_averages()
        self._reset_param_and_grad_samples()
        avg_loss = torch.mean(torch.stack(losses, dim=0), dim=0)
        avg_output = torch.mean(torch.stack(outputs, dim=0), dim=0)
        return avg_loss, avg_output

    def _stash_param_averages(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["param_average"] = p.data

    def _sample_weight(self):
        for group in self.param_groups:
            rsqrt_n = 1.0 / sqrt(group["data_size"])
            for p in group["params"]:
                if p.requires_grad:
                    normal_sample = torch.randn_like(p)
                    p.data = self._get_weight(p, rsqrt_n, normal_sample)

    def _get_weight(self, p, rsqrt_n, normal_sample):
        std = torch.exp(self.state[p]["log_std"])
        p_avg = self.state[p]["param_average"]
        return p_avg + rsqrt_n * std * normal_sample

    def _collect_grad_samples(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    self.state[p]["grad_samples"].append(p.grad)
                    self.state[p]["grad_sq_samples"].append(p.grad_sq)

    def _update(self):
        for group in self.param_groups:
            lr = group["lr"]
            lamb = group["prior_precision"]
            n = group["data_size"]
            d = group["dampening"]
            m = group["momentum_grad"]
            h = group["momentum_hess"]
            temperature = group["temperature"]
            for p in group["params"]:
                if p.requires_grad:
                    log_std = self.state[p]["log_std"]
                    self._update_momentum_grads(p, lamb, n, m)
                    f = self._compute_f(p, log_std, lamb, n, d, temperature)
                    self.update_log_std_buffers(p, h, f)
                    self._update_param_averages(p, log_std, lr)

    def _restore_param_averages(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    p.data = self.state[p]["param_average"]
                    self.state[p]["param_average"] = None

    def _update_momentum_grads(self, p, lamb, n, m):
        m_grad = self.state[p]["momentum_grad"]
        p_avg = self.state[p]["param_average"]
        grad_avg = torch.mean(
            torch.stack(self.state[p]["grad_samples"], dim=0), dim=0
        )
        self.state[p]["momentum_grad"] = m * m_grad + (1 - m) * (
            (lamb / n) * p_avg + grad_avg
        )

    def _compute_f(self, p, log_std, lamb, n, d, temperature):
        mean_grad_sq = torch.mean(
            torch.stack(self.state[p]["grad_sq_samples"], dim=0), dim=0
        )
        return ((lamb / n) + d + mean_grad_sq) * torch.exp(
            2 * log_std
        ) - temperature

    def _update_param_averages(self, p, log_std, lr):
        p_avg = self.state[p]["param_average"]
        m_grad = self.state[p]["momentum_grad"]
        self.state[p]["param_average"] = p_avg - lr * m_grad * torch.exp(
            2 * log_std
        )

    def update_log_std_buffers(self, p, h, f):
        log_std = self.state[p]["log_std"]
        self.state[p]["log_std"] = log_std - 0.5 * torch.log1p((1 - h) * f)

    @contextmanager
    def sampled_params(self):
        self._stash_param_averages()
        self._sample_weight()

        yield

        self._restore_param_averages()
