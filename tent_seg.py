import torch
import torch.nn as nn


def softmax_entropy_seg(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, C, H, W]
    return: scalar entropy loss
    """
    prob = torch.softmax(logits, dim=1)
    log_prob = torch.log_softmax(logits, dim=1)
    entropy = -(prob * log_prob).sum(dim=1)  # [B, H, W]
    return entropy.mean()


def configure_model_for_tent(model: nn.Module) -> nn.Module:
    """
    TENT official spirit:
    - model.train()
    - freeze all parameters
    - only enable BN affine parameters
    - force BN to use batch statistics
    """
    model.train()
    model.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

    return model


def collect_bn_affine_params(model: nn.Module):
    params = []
    names = []

    for module_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in ["weight", "bias"]:
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")

    return params, names


@torch.enable_grad()
def tent_forward_and_adapt(images, model, optimizer, steps=1):
    """
    images: target test images, no labels used
    """
    logits = None
    loss = None

    for _ in range(steps):
        logits = model(images)
        loss = softmax_entropy_seg(logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return logits, loss