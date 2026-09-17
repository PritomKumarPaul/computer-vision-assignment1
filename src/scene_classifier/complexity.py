from __future__ import annotations

import torch
from torch import nn


def profile_model(
    model: nn.Module,
    input_shape: tuple[int, int, int, int],
    device: torch.device,
) -> dict:
    """Estimate parameter storage and Conv/Linear MACs for one input image.

    FLOPs use the convention of two floating-point operations for every
    multiply-accumulate. Pooling, activations, and data loading are excluded,
    so these values are architecture estimates rather than hardware benchmarks.
    """
    macs = 0
    activation_elements = 0
    hooks = []

    def hook(module: nn.Module, inputs, output) -> None:
        nonlocal macs, activation_elements
        del inputs
        if isinstance(output, torch.Tensor):
            activation_elements += output.numel()
        if isinstance(module, nn.Conv2d):
            kernel_operations = (
                module.kernel_size[0]
                * module.kernel_size[1]
                * module.in_channels
                // module.groups
            )
            macs += output.numel() * kernel_operations
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            model(torch.zeros(input_shape, device=device))
    finally:
        for registered_hook in hooks:
            registered_hook.remove()
        model.train(was_training)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )
    return {
        "input_shape": list(input_shape),
        "total_parameters": parameters,
        "trainable_parameters": trainable,
        "parameter_and_buffer_size_mib": (parameter_bytes + buffer_bytes) / (1024**2),
        "estimated_macs_per_image": macs,
        "estimated_flops_per_image": 2 * macs,
        "counted_activation_elements": activation_elements,
        "flop_counting_note": (
            "Conv2d and Linear only; FLOPs=2*MACs. Pooling and activations excluded."
        ),
    }
