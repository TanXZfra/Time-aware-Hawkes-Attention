import os
import random

import numpy as np
import torch

from models.utils.import_utils import is_torch_mps_available


def set_seed(seed=1029):
    """Setup random seed.

    Args:
        seed (int, optional): random seed. Defaults to 1029.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_device(gpu=-1):
    """Setup the device.

    Args:
        gpu (int, optional): num of GPU to use. Defaults to -1 (not use GPU, i.e., use CPU).
    """
    if gpu >= 0:
        if torch.cuda.is_available():
            device = torch.device("cuda:" + str(gpu))
        elif is_torch_mps_available():
            device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def set_optimizer(optimizer, params, lr, trainer_config):
    """Setup the optimizer.

    Args:
        optimizer (str): name of the optimizer.
        params (dict): dict of params for the optimizer.
        lr (float): learning rate.

    Raises:
        NotImplementedError: if the optimizer's name is wrong or the optimizer is not supported,
        we raise error.

    Returns:
        torch.optim: torch optimizer.
    """

    named_params = list(params)

    phi_params = [
        p for n, p in named_params
        if 'phi_dict' in n and p.requires_grad
    ]
    other_params = [
        p for n, p in named_params
        if 'phi_dict' not in n and p.requires_grad
    ]
    
    phi_decay = trainer_config.l2_coef if hasattr(trainer_config, 'l2_coef') else 0.001

    param_groups = [
        {'params': phi_params, 'weight_decay': phi_decay},
        {'params': other_params, 'weight_decay': 0.0},
    ]

    print(f"[Optimizer] batch_size: {trainer_config.batch_size}")
    print(f"[Optimizer] learning rate: {lr}")
    print(f"[Optimizer] phi_l2_coef: {phi_decay}")
    print(f"[Optimizer] phi_params total elements: {sum(p.numel() for p in phi_params)}")
    print(f"[Optimizer] other_params total elements: {sum(p.numel() for p in other_params)}")


    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = "Adam"
    try:
        optimizer = getattr(torch.optim, optimizer)(param_groups, lr=lr)
    except Exception:
        raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
    return optimizer

    # if isinstance(optimizer, str):
    #     if optimizer.lower() == "adam":
    #         optimizer = "Adam"
    # try:
    #     optimizer = getattr(torch.optim, optimizer)(params, lr=lr)
    # except Exception:
    #     raise NotImplementedError("optimizer={} is not supported.".format(optimizer))
    # return optimizer


def count_model_params(model):
    """Count the number of params of the model.

    Args:
        model (torch.nn.Moduel): a torch model.

    Returns:
        int: total num of the parameters.
    """
    return sum(p.numel() for p in model.parameters())
