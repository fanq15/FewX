# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
helper class that supports empty tensors on some nn functions.
Ideally, add support directly in PyTorch to empty tensors in
those functions.
This can be removed once https://github.com/pytorch/pytorch/issues/12013
is implemented
"""

import math
import torch
from torch.nn.modules.utils import _ntuple
from torch import nn

class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        """
        Returns a new_shape.

        Args:
            ctx: (todo): write your description
            x: (todo): write your description
            new_shape: (int): write your description
        """
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        """
        Compute a tensor for a tensor.

        Args:
            ctx: (todo): write your description
            grad: (array): write your description
        """
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    """
    Interpolate a tensor.

    Args:
        input: (array): write your description
        size: (int): write your description
        scale_factor: (float): write your description
        mode: (str): write your description
        align_corners: (array): write your description
    """
    if input.numel() > 0:
        return torch.nn.functional.interpolate(
            input, size, scale_factor, mode, align_corners
        )

    def _check_size_scale_factor(dim):
        """
        Check that the scale scale factor is scale.

        Args:
            dim: (int): write your description
        """
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
        if (
            scale_factor is not None
            and isinstance(scale_factor, tuple)
            and len(scale_factor) != dim
        ):
            raise ValueError(
                "scale_factor shape must match input shape. "
                "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
            )

    def _output_size(dim):
        """
        Returns the size of a dimension.

        Args:
            dim: (int): write your description
        """
        _check_size_scale_factor(dim)
        if size is not None:
            return size
        scale_factors = _ntuple(dim)(scale_factor)
        # math.floor might return float in py2.7
        return [
            int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
        ]

    output_shape = tuple(_output_size(2))
    output_shape = input.shape[:-2] + output_shape
    return _NewEmptyTensorOp.apply(input, output_shape)
