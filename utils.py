from argparse import Namespace
from enum import Enum
from pathlib import Path
from PIL import ImageFilter
from tqdm import tqdm
from typing import List, Optional
import yaml

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmt = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmt.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmt = ''
        elif self.summary_type is Summary.AVERAGE:
            fmt = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmt = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmt = '{name} {count:.3f}'
        else:
            raise ValueError(f"Invalid summary type: {self.summary_type}")

        return fmt.format(**self.__dict__)


# xxx(okachaiev): integrate this tracker with summary writer for tensorboard
class ProgressTracker:

    def __init__(self, num_batches: int, meters: Optional[List[AverageMeter]] = None, prefix=''):
        self.batch_fmt = self._get_batch_fmt(num_batches)
        self.meters = meters or []
        self.prefix = prefix

    def _entries(self, batch):
        yield self.prefix + self.batch_fmt.format(batch)
        yield from map(str, self.meters)

    def display(self, batch):
        return '\t'.join(self._entries(batch))

    def _summary_entries(self):
        yield ' *'
        for meter in self.meters:
            yield meter.summary()

    def display_summary(self) -> str:
        return ' '.join(self._summary_entries())

    def _get_batch_fmt(self, num_batches):
        num_digits = len(str(num_batches // 1))+1
        fmt = '{:0' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def create_meter(self, *args, **kwargs) -> AverageMeter:
        meter = AverageMeter(*args, **kwargs)
        self.meters.append(meter)
        return meter

    def reset(self, prefix=''):
        self.prefix = prefix
        for meter in self.meters:
            meter.reset()


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    bs = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0, keepdim=True).mul_(1./bs) for k in topk]


#
# LARSWrapper from solo-learn repo.
#
class LARS(Optimizer):
    """
    Layer-wise adaptive rate scaling
    - Converted from Tensorflow to Pytorch from:
    https://github.com/google-research/simclr/blob/master/lars_optimizer.py
    - Based on:
    https://github.com/noahgolmant/pytorch-lars
    params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): base learning rate
        lr (int): Length / Number of layers we want to apply weight decay, else do not compute
        momentum (float, optional): momentum factor (default: 0.9)
        use_nesterov (bool, optional): flag to use nesterov momentum (default: False)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0.0)
            ("\beta")
        eta (float, optional): LARS coefficient (default: 0.001)
    - Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    - Large Batch Training of Convolutional Networks:
        https://arxiv.org/abs/1708.03888
    """

    def __init__(self, params, lr, len_reduced, momentum=0.9, use_nesterov=False, weight_decay=0.0, classic_momentum=True, eta=0.001):

        self.epoch = 0
        defaults = dict(
            lr=lr,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            classic_momentum=classic_momentum,
            eta=eta,
            len_reduced=len_reduced
        )

        super(LARS, self).__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_nesterov = use_nesterov
        self.classic_momentum = classic_momentum
        self.eta = eta
        self.len_reduced = len_reduced

    def step(self, epoch=None, closure=None):

        loss = None

        if closure is not None:
            loss = closure()

        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            momentum = group['momentum']
            learning_rate = group['lr']

            # TODO: Hacky
            counter = 0
            for p in group['params']:
                if p.grad is None:
                    continue

                param = p.data
                grad = p.grad.data

                param_state = self.state[p]

                # TODO: This really hacky way needs to be improved.
                # Note Excluded are passed at the end of the list to are ignored
                if counter < self.len_reduced:
                    grad += self.weight_decay * param

                # Create parameter for the momentum
                if "momentum_var" not in param_state:
                    next_v = param_state["momentum_var"] = torch.zeros_like(
                        p.data
                    )
                else:
                    next_v = param_state["momentum_var"]

                if self.classic_momentum:
                    trust_ratio = 1.0

                    # TODO: implementation of layer adaptation
                    w_norm = torch.norm(param)
                    g_norm = torch.norm(grad)

                    device = g_norm.get_device()

                    trust_ratio = torch.where(w_norm.ge(0), torch.where(
                        g_norm.ge(0), (self.eta * w_norm / g_norm), torch.Tensor([1.0]).to(device)),
                                              torch.Tensor([1.0]).to(device)).item()

                    scaled_lr = learning_rate * trust_ratio

                    grad_scaled = scaled_lr*grad
                    next_v.mul_(momentum).add_(grad_scaled)

                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (scaled_lr * grad)
                    else:
                        update = next_v

                    p.data.add_(-update)

                # Not classic_momentum
                else:

                    next_v.mul_(momentum).add_(grad)

                    if self.use_nesterov:
                        update = (self.momentum * next_v) + (grad)

                    else:
                        update = next_v

                    trust_ratio = 1.0

                    # TODO: implementation of layer adaptation
                    w_norm = torch.norm(param)
                    v_norm = torch.norm(update)

                    device = v_norm.get_device()

                    trust_ratio = torch.where(w_norm.ge(0), torch.where(
                        v_norm.ge(0), (self.eta * w_norm / v_norm), torch.Tensor([1.0]).to(device)),
                                              torch.Tensor([1.0]).to(device)).item()

                    scaled_lr = learning_rate * trust_ratio

                    p.data.add_(-scaled_lr * update)

                counter += 1

        return loss


class LARSWrapper:
    def __init__(
        self,
        optimizer: Optimizer,
        eta: float = 1e-3,
        clip: bool = False,
        eps: float = 1e-8,
        exclude_bias_n_norm: bool = False,
    ):
        """Wrapper that adds LARS scheduling to any optimizer.
        This helps stability with huge batch sizes.

        Args:
            optimizer (Optimizer): torch optimizer.
            eta (float, optional): trust coefficient. Defaults to 1e-3.
            clip (bool, optional): clip gradient values. Defaults to False.
            eps (float, optional): adaptive_lr stability coefficient. Defaults to 1e-8.
            exclude_bias_n_norm (bool, optional): exclude bias and normalization layers from lars.
                Defaults to False.
        """

        self.optim = optimizer
        self.eta = eta
        self.eps = eps
        self.clip = clip
        self.exclude_bias_n_norm = exclude_bias_n_norm

        # transfer optim methods
        self.state_dict = self.optim.state_dict
        self.load_state_dict = self.optim.load_state_dict
        self.zero_grad = self.optim.zero_grad
        self.add_param_group = self.optim.add_param_group

        self.__setstate__ = self.optim.__setstate__  # type: ignore
        self.__getstate__ = self.optim.__getstate__  # type: ignore
        self.__repr__ = self.optim.__repr__  # type: ignore

    @property
    def defaults(self):
        return self.optim.defaults

    @defaults.setter
    def defaults(self, defaults):
        self.optim.defaults = defaults

    @property  # type: ignore
    def __class__(self):
        return Optimizer

    @property
    def state(self):
        return self.optim.state

    @state.setter
    def state(self, state):
        self.optim.state = state

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    @torch.no_grad()
    def step(self, closure=None):
        weight_decays = []

        for group in self.optim.param_groups:
            weight_decay = group.get("weight_decay", 0)
            weight_decays.append(weight_decay)

            # reset weight decay
            group["weight_decay"] = 0

            # update the parameters
            for p in group["params"]:
                if p.grad is not None and (p.ndim != 1 or not self.exclude_bias_n_norm):
                    self.update_p(p, group, weight_decay)

        # update the optimizer
        self.optim.step(closure=closure)

        # return weight decay control to optimizer
        for group_idx, group in enumerate(self.optim.param_groups):
            group["weight_decay"] = weight_decays[group_idx]

    def update_p(self, p, group, weight_decay):
        # calculate new norms
        p_norm = torch.norm(p.data)
        g_norm = torch.norm(p.grad.data)

        if p_norm != 0 and g_norm != 0:
            # calculate new lr
            new_lr = (self.eta * p_norm) / (g_norm + p_norm * weight_decay + self.eps)

            # clip lr
            if self.clip:
                new_lr = min(new_lr / group["lr"], 1)

            # update params with clipped lr
            p.grad.data += weight_decay * p.data
            p.grad.data *= new_lr


class GBlur:

    def __init__(self, p, seed=0):
        self.p = p
        self.rng = torch.Generator(device='cpu')
        self.rng.manual_seed(seed)

    def __call__(self, img):
        if torch.rand(1, generator=self.rng).item() < self.p:
            sigma = torch.rand(1, generator=self.rng).item() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def cleanup_old_checkpoints(exp_dir: Path, keep: int = 1, no_prompt: bool = False) -> None:
    old_checkpoints = {}
    total_size = 0
    for folder in exp_dir.glob('*/checkpoints'):
        checkpoints = sorted(int(file.name.replace(".pt", "")) for file in folder.glob('*.pt'))
        if len(checkpoints) > keep:
            to_remove = [folder / f"{chkp}.pt" for chkp in checkpoints[:-keep]]
            old_checkpoints.update(to_remove)
            total_size += sum(f.stat().st_size for f in to_remove)

    if not old_checkpoints:
        print('* Nothing to remove')
        return

    if not no_prompt:
        print(f"Cleanup is about to delete {len(old_checkpoints)} files, "
                f"total size: {human_readable_size(total_size)}. Proceed? [Y/n] ")
        choice = input()
    else:
        choice = "Y"

    if choice.strip() != "Y":
        print('* OK, bailing out')
        return

    cleaned_space = 0
    print(f"===> Removing old checkpoints...")
    for checkpoint_file in tqdm(old_checkpoints):
        file_size = checkpoint_file.stat().st_size
        try:
            checkpoint_file.unlink()
        except Exception as exp:
            print(f"Failed to remove {checkpoint_file}, caused by {exp}")
        else:
            cleaned_space += file_size
    print(f"Cleaned up: {human_readable_size(cleaned_space)}")


def load_config_into(config_file: Path, args: Namespace) -> None:
    with open(config_file, 'r') as fd:
        settings = yaml.safe_load(fd)
    for k, v in settings['params'].items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)


def log_exp_config(config_file: Path, args: Namespace, force_override: bool = False) -> None:
    if force_override or not config_file.exists():
        with open(config_file, 'w') as fd:
            yaml.dump({'params': vars(args)}, fd, default_flow_style=False, allow_unicode=True)
